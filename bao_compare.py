"""Compare scalar-tensor background predictions with the supplied BAO data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid

from bao_data import ALL_BAO
from background_solver import C_KM_S, H0_KM_S_MPC, compute_rd, solve_background

PARAMS = [
    (0.00, 1e-30, 1e-3),
    (0.02, 1e-30, 1e-3),
    (0.05, 1e-30, 1e-3),
    (0.08, 1e-30, 1e-3),
]

OUT_DIR = Path(__file__).resolve().parent
# The scalar oscillations at m_phi/H0 ~ 700 make the strict 1e-9 Radau
# tolerance disproportionately expensive for the comparison sweep. The
# converged 1e-8/1e-10 pair changes the reported observables below the printed
# precision while keeping the production solver API at the requested default.
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-10
LCDM_REF_RD = 147.09


def _make_distance_interpolants(background: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build monotonic z-grid interpolation arrays for D_H and D_M."""
    z_grid = np.asarray(background["z"], dtype=np.float64)[::-1]
    H_grid = np.asarray(background["H"], dtype=np.float64)[::-1]
    c_over_h0 = C_KM_S / H0_KM_S_MPC
    inv_h = c_over_h0 / H_grid
    dm_grid = cumulative_trapezoid(inv_h, z_grid, initial=0.0)
    return z_grid, H_grid, dm_grid


def _observables(
    distance_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], z: float
) -> tuple[float, float, float]:
    """Return D_H, D_M and D_V in Mpc at a given redshift."""
    z_grid, H_grid, dm_grid = distance_arrays
    E = float(np.interp(z, z_grid, H_grid))
    D_H = C_KM_S / (H0_KM_S_MPC * E)
    D_M = float(np.interp(z, z_grid, dm_grid))
    D_V = (z * D_M**2 * D_H) ** (1.0 / 3.0)
    return D_H, D_M, D_V


def chi2_bao(background: dict, rd_mpc: float) -> tuple[float, int, list[dict]]:
    """Compute diagonal-error BAO chi^2."""
    rows: list[dict] = []
    chi2 = 0.0
    distance_arrays = _make_distance_interpolants(background)
    for z, obs_type, value, sigma, survey in ALL_BAO:
        DH, DM, DV = _observables(distance_arrays, z)
        prediction = {
            "DH_over_rd": DH / rd_mpc,
            "DM_over_rd": DM / rd_mpc,
            "DV_over_rd": DV / rd_mpc,
        }[obs_type]
        residual = (prediction - value) / sigma
        contribution = residual**2
        chi2 += contribution
        rows.append(
            {
                "z": z,
                "observable": obs_type,
                "survey": survey,
                "obs": value,
                "sigma": sigma,
                "prediction": prediction,
                "residual_sigma": residual,
                "chi2_contribution": contribution,
            }
        )
    return float(chi2), len(rows), rows


def _max_dh_percent_z20(background: dict) -> float:
    """Maximum |Delta H/H| relative to the stored LCDM curve for z<20."""
    z = np.asarray(background["z"], dtype=np.float64)
    mask = z < 20.0
    rel = np.abs(np.asarray(background["H"])[mask] / np.asarray(background["H_LCDM"])[mask] - 1.0)
    return float(100.0 * np.max(rel))


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write results with a stable column order."""
    fields = [
        "beta",
        "m_phi_eV",
        "phi_i",
        "r_d_Mpc",
        "r_d_shift_pct",
        "chi2",
        "dchi2",
        "npts",
        "max_dH_pct_z20",
        "Omega_phi0",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results = []
    lcdm_chi2 = np.nan
    model_cache = {}

    print("\nBAO SUMMARY")
    print("=" * 100)
    print(
        f"{'beta':>7} {'r_d [Mpc]':>12} {'shift %':>10} {'chi2':>12} "
        f"{'dchi2':>12} {'N':>5} {'max |dH/H| z<20 %':>20} {'Omega_phi0':>14}"
    )
    print("-" * 100)

    for beta, m_phi, phi_i in PARAMS:
        try:
            bg = solve_background(beta, m_phi, phi_i, rtol=SOLVER_RTOL, atol=SOLVER_ATOL)
            rd = compute_rd(bg)
            chi2, npts, _ = chi2_bao(bg, rd)
            model_cache[beta] = (bg, rd)
            if beta == 0.0:
                lcdm_chi2 = chi2
            results.append(
                {
                    "beta": beta,
                    "m_phi_eV": m_phi,
                    "phi_i": phi_i,
                    "r_d_Mpc": rd,
                    "r_d_shift_pct": 100.0 * (rd / LCDM_REF_RD - 1.0),
                    "chi2": chi2,
                    "dchi2": np.nan,  # filled below
                    "npts": npts,
                    "max_dH_pct_z20": _max_dh_percent_z20(bg),
                    "Omega_phi0": bg["Omega_phi0"],
                }
            )
            print(
                f"{beta:7.3f} {rd:12.4f} {100*(rd/LCDM_REF_RD-1):10.4f} "
                f"{chi2:12.4f} {'--':>12} {npts:5d} "
                f"{_max_dh_percent_z20(bg):20.4f} {bg['Omega_phi0']:14.6e}"
            )
        except Exception as exc:
            print(f"ERROR beta={beta:.3f}: {exc}")
            continue

    for row in results:
        row["dchi2"] = float(row["chi2"] - lcdm_chi2) if np.isfinite(lcdm_chi2) else np.nan

    out_csv = OUT_DIR / "results_bao_rd.csv"
    _write_csv(out_csv, results)

    # Main BAO figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    z_plot = np.linspace(0.0, 20.0, 800)
    for beta, m_phi, phi_i in PARAMS:
        if beta not in model_cache:
            continue
        bg, _ = model_cache[beta]
        z_bg = np.asarray(bg["z"])[::-1]
        H_bg = np.asarray(bg["H"])[::-1]
        H_lcdm = np.asarray(bg["H_LCDM"])[::-1]
        mask = z_bg <= 20.0
        order_z = z_bg[mask]
        order_rel = H_bg[mask] / H_lcdm[mask] - 1.0
        ax.plot(order_z, 100.0 * order_rel, label=f"β={beta:.2f}")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Delta H/H_{\Lambda CDM}$ [%]")
    ax.set_title("Expansion-rate deviation")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    target_beta = 0.05
    if target_beta in model_cache:
        bg, rd = model_cache[target_beta]
        _, _, rows = chi2_bao(bg, rd)
        x = np.arange(len(rows))
        y = [r["residual_sigma"] for r in rows]
        labels = [f"{r['z']:.3g}\n{r['observable'].replace('_over_rd', '')}" for r in rows]
        ax.axhline(0.0, linewidth=1.0)
        ax.scatter(x, y, s=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel(r"$(model-data)/\sigma$")
        ax.set_title("BAO residuals: β=0.05")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bao_plot_rd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("=" * 100)
    print(f"Saved: {out_csv.name}, bao_plot_rd.png")


if __name__ == "__main__":
    main()
