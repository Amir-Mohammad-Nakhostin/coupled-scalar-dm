"""Linear dark-matter growth in the scalar-tensor background."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from background_solver import EV_TO_MPC_INV
from rsd_data import RSD_DATA


def Geff_over_G(k: float, a: float, beta: float, m_phi_eV: float) -> float:
    """Return the quasi-static effective gravitational coupling ratio.

    Parameters
    ----------
    k:
        Comoving wavenumber in Mpc^-1.
    a:
        Scale factor.
    beta:
        Dimensionless dark-matter coupling.
    m_phi_eV:
        Scalar mass in eV.
    """
    if k < 0.0 or a <= 0.0 or m_phi_eV < 0.0:
        raise ValueError("Require k >= 0, a > 0 and m_phi_eV >= 0.")
    m_mpc_inv = m_phi_eV * EV_TO_MPC_INV
    k2 = np.float64(k) ** 2
    denominator = k2 + (a * m_mpc_inv) ** 2
    if denominator == 0.0:
        # k=m=0 is the GR limit of the expression by continuity.
        return 1.0
    return float(1.0 + 2.0 * beta**2 * k2 / denominator)


def _background_interpolants(background: dict):
    """Create stable interpolation helpers on monotonically increasing N."""
    N = np.asarray(background["N"], dtype=np.float64)
    H = np.asarray(background["H"], dtype=np.float64)
    rho_dm = np.asarray(background["rho_DM"], dtype=np.float64)
    rho_b = np.asarray(background.get("rho_b", 0.049 * np.exp(-3.0 * N)), dtype=np.float64)
    return N, H, rho_b, rho_dm


def solve_growth(
    background: dict,
    k: float,
    beta: float,
    m_phi_eV: float,
    N_i: float = np.log(1e-3),
    N_f: float = 0.0,
    method: str = "Radau",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    n_eval: int = 2500,
) -> dict:
    """Solve the linear growth equation from a_i to today.

    The conservative default is Radau at 1e-8/1e-10. The optimized MCMC path
    may request LSODA with the same or looser tolerances because the growth
    equation is smooth once the background has been interpolated.
    """
    if N_i < float(np.min(background["N"])) or N_f > float(np.max(background["N"])):
        raise ValueError("Growth interval must lie inside the background interval.")
    if N_f <= N_i:
        raise ValueError("N_f must be greater than N_i.")

    N_bg, H_bg, rho_b_bg, rho_dm_bg = _background_interpolants(background)

    def rhs(N: float, state: np.ndarray) -> np.ndarray:
        D, Dp = state
        H = float(np.interp(N, N_bg, H_bg))
        rho_b = float(np.interp(N, N_bg, rho_b_bg))
        rho_dm = float(np.interp(N, N_bg, rho_dm_bg))
        a = np.exp(N)
        omega_m = (rho_b + rho_dm) / H**2

        # H'/H from the same background equation used in background_solver.
        phi = float(np.interp(N, N_bg, background["phi"]))
        y = float(np.interp(N, N_bg, background["y"]))
        rho_r = 9.2e-5 * a ** (-4.0)
        hp_over_h = -0.5 * y**2 - 1.5 * (rho_b + rho_dm) / H**2 - 2.0 * rho_r / H**2
        geff = Geff_over_G(k, a, beta, m_phi_eV)
        Dpp = -(2.0 + hp_over_h) * Dp + 1.5 * omega_m * geff * D
        return np.array([Dp, Dpp], dtype=np.float64)

    a_i = np.exp(N_i)
    y0 = np.array([a_i, a_i], dtype=np.float64)
    if int(n_eval) < 16:
        raise ValueError("n_eval must be at least 16.")
    t_eval = np.linspace(N_i, N_f, int(n_eval), dtype=np.float64)

    try:
        sol = solve_ivp(
            rhs,
            (N_i, N_f),
            y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Growth solve_ivp failed for beta={beta}, m_phi_eV={m_phi_eV}, k={k}."
        ) from exc

    if not sol.success:
        raise RuntimeError(f"Growth integration failed: {sol.message}")

    N = sol.t.astype(np.float64)
    D, Dp = sol.y.astype(np.float64)
    a = np.exp(N)
    z = 1.0 / a - 1.0
    f = Dp / D

    return {
        "N": N,
        "a": a,
        "z": z,
        "D": D,
        "f": f,
        "k": float(k),
        "beta": float(beta),
        "m_phi_eV": float(m_phi_eV),
    }


def compute_fsigma8(growth_result: dict, sigma8_0: float = 0.811) -> dict:
    """Compute f*sigma8 and sigma8(z) from a growth solution."""
    if sigma8_0 <= 0.0:
        raise ValueError("sigma8_0 must be positive.")
    D = np.asarray(growth_result["D"], dtype=np.float64)
    D0 = float(D[-1])
    sigma8 = sigma8_0 * D / D0
    fsigma8 = growth_result["f"] * sigma8
    return {
        "z": np.asarray(growth_result["z"], dtype=np.float64),
        "fsigma8": np.asarray(fsigma8, dtype=np.float64),
        "f": np.asarray(growth_result["f"], dtype=np.float64),
        "sigma8": sigma8,
        "D": D,
    }


def chi2_rsd(growth_result: dict, k_ref: float = 0.1) -> tuple:
    """Return diagonal-error chi^2 against the supplied RSD compilation.

    ``k_ref`` is retained in the public signature as a consistency check; the
    supplied growth solution is expected to have been evaluated at this scale.
    """
    if not np.isclose(float(growth_result["k"]), float(k_ref)):
        raise ValueError(
            f"Growth result uses k={growth_result['k']}, but chi2_rsd requested k_ref={k_ref}."
        )
    fs = compute_fsigma8(growth_result)
    z_model = fs["z"][::-1]
    fs_model = fs["fsigma8"][::-1]

    rows = []
    chi2 = 0.0
    for z_obs, value, sigma, survey in RSD_DATA:
        pred = float(np.interp(z_obs, z_model, fs_model))
        residual = (pred - value) / sigma
        contribution = residual**2
        chi2 += contribution
        rows.append(
            {
                "z": z_obs,
                "survey": survey,
                "obs": value,
                "sigma": sigma,
                "prediction": pred,
                "residual_sigma": residual,
                "chi2_contribution": contribution,
            }
        )
    return float(chi2), len(rows), rows
