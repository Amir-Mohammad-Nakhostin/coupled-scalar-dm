"""Background evolution for a scalar-tensor dark-matter coupled cosmology.

The solver uses N = ln(a) as the independent variable and the dimensionless
normalisation H0 = 1, M_P = 1.  The physical scalar mass enters through
m_tilde = m_phi / H0_eV.

Important conventions
---------------------
* ``phi`` is dimensionless, i.e. the physical field is phi * M_P.
* ``rho_DM`` is dimensionless in units consistent with the supplied
  Friedmann equation.
* The dark-matter initial condition is Omega_DM0 * a_i**(-3).

References
----------
Planck Collaboration, 2018, arXiv:1807.06209.
Eisenstein & Hu, 1998, ApJ 496, 605, arXiv:astro-ph/9709112.
DESI Collaboration, 2025, arXiv:2503.14738.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import simpson, solve_ivp

# ---------------------------------------------------------------------------
# Cosmological constants and unit conversions
# ---------------------------------------------------------------------------
M_P_GEV = 2.435e18
H0_eV = 1.44e-33
H0_KM_S_MPC = 67.4
C_KM_S = 299_792.458
EV_TO_MPC_INV = 1.642e-16

OMEGA_B0 = 0.049
OMEGA_DM0 = 0.265
OMEGA_L0 = 0.685
OMEGA_R0 = 9.2e-5
OMEGA_GAMMA0 = 5.44e-5
NEFF = 3.046

OMEGA_R0_EFF = OMEGA_GAMMA0 * (
    1.0 + NEFF * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
)

# A dimensionless Lambda-CDM reference using the stated density values.
OMEGA_M0 = OMEGA_B0 + OMEGA_DM0


def _validate_parameters(beta: float, m_phi_eV: float, phi_i: float) -> None:
    """Validate user-facing background parameters."""
    vals = (beta, m_phi_eV, phi_i)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("beta, m_phi_eV and phi_i must be finite.")
    if m_phi_eV < 0.0:
        raise ValueError("m_phi_eV must be non-negative.")


def _h2_and_hprime_over_h(
    N: float,
    phi: float,
    y: float,
    rho_dm: float,
    m_tilde: float,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate H^2 and H'/H from the algebraic Friedmann relation.

    Differentiating

        H^2 (1 - y^2/6) = rho_b + rho_DM + rho_L + rho_r
                           + m_tilde^2 phi^2 / 6

    and eliminating y' with the scalar equation gives the compact identity

        H'/H = -y^2/2 - 3/2 (rho_b + rho_DM)/H^2 - 2 rho_r/H^2.

    This avoids an implicit algebraic loop in the ODE right-hand side.
    """
    a = np.exp(N)
    rho_b = OMEGA_B0 * a ** (-3.0)
    rho_r = OMEGA_R0 * a ** (-4.0)
    rho_l = OMEGA_L0
    numerator = rho_b + rho_dm + rho_l + rho_r + (m_tilde**2 * phi**2) / 6.0
    denom = 1.0 - y**2 / 6.0
    if denom <= 1e-12:
        raise FloatingPointError(
            f"Friedmann denominator became non-positive at N={N:.6g}; "
            f"y={y:.6g}."
        )
    h2 = numerator / denom
    if not np.isfinite(h2) or h2 <= 0.0:
        raise FloatingPointError(f"Invalid H^2={h2} at N={N:.6g}.")

    h = -0.5 * y**2 - 1.5 * (rho_b + rho_dm) / h2 - 2.0 * rho_r / h2
    return h2, h, rho_b, rho_r, rho_l, numerator


def _make_rhs(beta: float, m_tilde: float):
    """Build the scalar/background ODE right-hand side.

    The numerical state uses ``q = rho_DM / (Omega_DM0*a^-3)`` instead of
    rho_DM itself.  This exact change of variables keeps the matter variable
    O(1) at early times and substantially improves Radau's conditioning while
    leaving the physical equations unchanged.
    """

    def rhs(N: float, state: np.ndarray) -> np.ndarray:
        phi, y, q_dm = state
        a = np.exp(N)
        rho_dm = OMEGA_DM0 * a ** (-3.0) * q_dm
        h2, hp_over_h, _, _, _, _ = _h2_and_hprime_over_h(
            N, phi, y, rho_dm, m_tilde
        )

        dphi = y
        dy = (
            -hp_over_h * y
            - 3.0 * y
            - (m_tilde**2 / h2) * phi
            + 3.0 * beta * rho_dm / h2
        )
        dq_dm = -beta * y * q_dm
        return np.array([dphi, dy, dq_dm], dtype=np.float64)

    return rhs



def _make_jacobian(beta: float, m_tilde: float):
    """Return the analytic Jacobian of the scaled background system."""
    m2 = m_tilde**2

    def jac(N: float, state: np.ndarray) -> np.ndarray:
        phi, y, q_dm = state
        a = np.exp(N)
        rb = OMEGA_B0 * a ** (-3.0)
        rr = OMEGA_R0 * a ** (-4.0)
        rdm = OMEGA_DM0 * a ** (-3.0) * q_dm
        A, h, _, _, _, _ = _h2_and_hprime_over_h(N, phi, y, rdm, m_tilde)
        D = 1.0 - y**2 / 6.0
        M = rb + rdm
        B = 1.5 * M + 2.0 * rr

        A_phi = (m2 * phi / 3.0) / D
        A_y = A * (y / 3.0) / D
        A_q = (rdm / q_dm) / D if q_dm != 0.0 else 0.0

        h_phi = B * A_phi / A**2
        h_y = -y + B * A_y / A**2
        B_q = 1.5 * (rdm / q_dm) if q_dm != 0.0 else 0.0
        h_q = -B_q / A + B * A_q / A**2

        invA2 = 1.0 / A**2
        J = np.zeros((3, 3), dtype=np.float64)
        J[0, 1] = 1.0
        J[1, 0] = (
            -y * h_phi
            - m2 / A
            + m2 * phi * A_phi * invA2
            - 3.0 * beta * rdm * A_phi * invA2
        )
        J[1, 1] = (
            -(h_y * y + h + 3.0)
            + m2 * phi * A_y * invA2
            - 3.0 * beta * rdm * A_y * invA2
        )
        if q_dm != 0.0:
            J[1, 2] = (
                -y * h_q
                + m2 * phi * A_q * invA2
                + 3.0 * beta * ((rdm / q_dm) / A - rdm * A_q * invA2)
            )
        else:
            J[1, 2] = -y * h_q
        J[2, 1] = -beta * q_dm
        J[2, 2] = -beta * y
        return J

    return jac

def solve_background(
    beta: float,
    m_phi_eV: float,
    phi_i: float = 1e-3,
    N_i: float = np.log(1e-4),
    N_f: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    method: str = "Radau",
    n_eval: int = 3000,
) -> dict:
    """Solve the homogeneous scalar-tensor background.

    Parameters
    ----------
    beta:
        Dimensionless scalar-dark-matter coupling.
    m_phi_eV:
        Scalar mass in eV.
    phi_i:
        Initial scalar field in reduced-Planck units.
    N_i, N_f:
        Initial and final values of N=ln(a).
    rtol, atol:
        Relative and absolute tolerances passed to ``solve_ivp``.
    method:
        ``solve_ivp`` integration method. Radau remains the conservative
        standalone default; optimized MCMC mode may use LSODA.
    n_eval:
        Number of stored output points used by downstream interpolation.

    Returns
    -------
    dict
        Arrays for the full background history plus model metadata.
    """
    _validate_parameters(float(beta), float(m_phi_eV), float(phi_i))
    if N_f <= N_i:
        raise ValueError("N_f must be greater than N_i.")

    m_tilde = float(m_phi_eV) / H0_eV
    a_i = np.exp(N_i)
    rho_dm_i = OMEGA_DM0 * a_i ** (-3.0)
    y_i = 0.0
    # q_DM = rho_DM / (Omega_DM0 a^-3), hence q_DM(a_i)=1 exactly.
    y0 = np.array([phi_i, y_i, 1.0], dtype=np.float64)

    if int(n_eval) < 16:
        raise ValueError("n_eval must be at least 16.")
    n_eval = int(n_eval)
    # A fixed output grid makes downstream interpolation deterministic.
    t_eval = np.linspace(N_i, N_f, n_eval, dtype=np.float64)

    rhs = _make_rhs(beta, m_tilde)
    ivp_kwargs = {
        "method": method,
        "t_eval": t_eval,
        "rtol": rtol,
        "atol": atol,
        "dense_output": False,
    }
    # Explicit methods such as LSODA/DOP853 do not need the analytic Jacobian.
    # Supplying it to DOP853 also causes a scipy warning, so only pass it to
    # methods that consume it.
    if method in {"Radau", "BDF"}:
        ivp_kwargs["jac"] = _make_jacobian(beta, m_tilde)

    try:
        sol = solve_ivp(
            rhs,
            (N_i, N_f),
            y0,
            **ivp_kwargs,
        )
    except Exception as exc:  # pragma: no cover - defensive error reporting
        raise RuntimeError(
            "Background solve_ivp failed. "
            f"beta={beta}, m_phi_eV={m_phi_eV}, phi_i={phi_i}."
        ) from exc

    if not sol.success:
        raise RuntimeError(
            "Background integration did not converge: "
            f"{sol.message} (last N={sol.t[-1]:.6g})."
        )

    N = sol.t.astype(np.float64)
    phi, y, q_dm = sol.y.astype(np.float64)
    a = np.exp(N)
    rho_dm = OMEGA_DM0 * a ** (-3.0) * q_dm
    z = 1.0 / a - 1.0

    H = np.empty_like(N)
    hp = np.empty_like(N)
    rho_b = OMEGA_B0 * a ** (-3.0)
    rho_r = OMEGA_R0 * a ** (-4.0)
    for i in range(N.size):
        h2, hp[i], _, _, _, _ = _h2_and_hprime_over_h(
            N[i], phi[i], y[i], rho_dm[i], m_tilde
        )
        H[i] = np.sqrt(h2)

    # Scalar energy density implied by the Friedmann equation.
    rho_phi = (H**2 * y**2 + m_tilde**2 * phi**2) / 6.0

    # Reference flat LCDM curve.  The supplied densities sum to slightly below
    # unity because they are rounded; retain their values exactly here.
    H_LCDM = np.sqrt(
        OMEGA_B0 * a ** (-3.0)
        + OMEGA_DM0 * a ** (-3.0)
        + OMEGA_L0
        + OMEGA_R0 * a ** (-4.0)
    )

    omega_phi0 = float(rho_phi[-1] / H[-1] ** 2)

    return {
        "N": N,
        "a": a,
        "z": z,
        "phi": phi,
        "y": y,
        "rho_DM": rho_dm,
        "rho_phi": rho_phi,
        "H": H,
        "H_LCDM": H_LCDM,
        "beta": float(beta),
        "m_phi_eV": float(m_phi_eV),
        "phi_i": float(phi_i),
        "m_tilde": float(m_tilde),
        "Omega_phi0": omega_phi0,
        "rho_b": rho_b,
        "rho_r": rho_r,
        "success": True,
        "message": sol.message,
        "hp_over_h": hp,
    }


def z_drag(
    Omega_m_h2: float,
    Omega_b_h2: float,
) -> float:
    """Return the Eisenstein & Hu (1998) baryon drag redshift fit.

    The fit is included exactly as specified by the project brief.  For the
    Planck-like parameters used here it gives z_d ~= 1020.44, which is not the
    Planck 2018 drag-redshift convention associated with r_d=147.09 Mpc.
    """
    if Omega_m_h2 <= 0.0 or Omega_b_h2 <= 0.0:
        raise ValueError("Physical density parameters must be positive.")
    b1 = 0.313 * Omega_m_h2 ** (-0.419) * (
        1.0 + 0.607 * Omega_m_h2**0.674
    )
    b2 = 0.238 * Omega_m_h2**0.223
    return float(
        1291.0
        * Omega_m_h2**0.251
        / (1.0 + 0.659 * Omega_m_h2**0.828)
        * (1.0 + b1 * Omega_b_h2**b2)
    )


def _background_h_dimensionless(background: dict, z_query: np.ndarray) -> np.ndarray:
    """Interpolate dimensionless H(z) from a background dictionary."""
    z_bg = np.asarray(background["z"], dtype=np.float64)[::-1]
    H_bg = np.asarray(background["H"], dtype=np.float64)[::-1]
    return np.interp(z_query, z_bg, H_bg)


def _extend_background_for_rd(background: dict, z_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Extend the background backwards beyond z=9999 for the sound horizon.

    The requested background solver starts at a=1e-4.  The sound-horizon
    integral is conventionally evaluated much earlier.  We therefore continue
    the exact same ODE backwards from the supplied initial point to z_max.
    This is deterministic and preserves the model equations instead of
    silently changing the high-redshift dynamics to LCDM.
    """
    z_bg_max = float(np.max(background["z"]))
    if z_max <= z_bg_max * (1.0 + 1e-12):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    N_start = float(np.min(background["N"]))
    N_end = float(-np.log1p(z_max))
    beta = float(background["beta"])
    m_tilde = float(background["m_tilde"])
    y0 = np.array(
        [
            float(background["phi"][0]),
            float(background["y"][0]),
            1.0,
        ],
        dtype=np.float64,
    )

    # Backward integration, with enough points for smooth quadrature.
    t_eval = np.linspace(N_start, N_end, 2400, dtype=np.float64)
    try:
        sol = solve_ivp(
            _make_rhs(beta, m_tilde),
            (N_start, N_end),
            y0,
            method="Radau",
            t_eval=t_eval,
            rtol=2e-9,
            atol=2e-11,
            jac=_make_jacobian(beta, m_tilde),
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("High-redshift background extension failed.") from exc
    if not sol.success:
        raise RuntimeError(f"High-redshift extension failed: {sol.message}")

    N = sol.t
    phi, y, q_dm = sol.y
    a = np.exp(N)
    rho_dm = OMEGA_DM0 * a ** (-3.0) * q_dm
    H = np.empty_like(N)
    for i in range(N.size):
        h2, *_ = _h2_and_hprime_over_h(N[i], phi[i], y[i], rho_dm[i], m_tilde)
        H[i] = np.sqrt(h2)
    z = np.exp(-N) - 1.0
    # sol.t decreases, so z increases, which is the desired ordering.
    return z.astype(np.float64), H.astype(np.float64)


def compute_rd(background: dict, z_d: float = 1059.94) -> float:
    """Compute the drag-epoch sound horizon in Mpc.

    The default ``z_d=1059.94`` is the convention needed to reproduce the
    Planck-2018 reference ``r_d ~= 147.09 Mpc`` with the supplied cosmological
    numbers.  The Eisenstein-Hu fitting formula is available through
    :func:`z_drag`; with the same density inputs it gives z_d ~= 1020.44 and
    therefore cannot simultaneously reproduce 147.09 Mpc without introducing
    an empirical rescaling.

    The numerical integral is evaluated to z_max=1e7 and the remaining tail is
    radiation dominated.  The high-redshift extension uses the same background
    equations as ``solve_background``.
    """
    if not np.isfinite(z_d) or z_d <= 0.0:
        raise ValueError("z_d must be positive and finite.")

    h = 0.674
    z_d_eh = z_drag(0.314 * h**2, OMEGA_B0 * h**2)
    if abs(z_d - z_d_eh) / z_d > 0.02:
        warnings.warn(
            "compute_rd is using the Planck-style z_d=1059.94 convention; "
            f"the Eisenstein-Hu fit gives {z_d_eh:.2f} for these inputs.",
            RuntimeWarning,
            stacklevel=2,
        )

    z_max = 1.0e7
    z_ext, H_ext = _extend_background_for_rd(background, z_max)

    z_bg = np.asarray(background["z"], dtype=np.float64)[::-1]
    H_bg = np.asarray(background["H"], dtype=np.float64)[::-1]
    if z_ext.size:
        # Drop the duplicated z~9999 endpoint from the extension.
        keep = z_ext > z_bg[-1] * (1.0 + 1e-12)
        z_grid = np.concatenate([z_bg, z_ext[keep]])
        H_grid = np.concatenate([H_bg, H_ext[keep]])
    else:
        z_grid, H_grid = z_bg, H_bg

    # Physical sound speed, using the photon density only in the baryon-loading
    # factor R and the effective radiation density in the high-z H behaviour.
    c_over_H0 = C_KM_S / H0_KM_S_MPC

    def H_dimless(z: float) -> float:
        return float(np.interp(z, z_grid, H_grid))

    def integrand(z: float) -> float:
        a = 1.0 / (1.0 + z)
        R = (3.0 * OMEGA_B0 / (4.0 * OMEGA_GAMMA0)) * a
        c_s_over_c = 1.0 / np.sqrt(3.0 * (1.0 + R))
        return c_over_H0 * c_s_over_c / H_dimless(z)

    z_upper_integral = min(z_max, float(z_grid[-1]))
    if z_d >= z_upper_integral:
        raise ValueError("z_d must lie below the numerical sound-horizon range.")

    # Integrate on a logarithmically spaced grid.  A direct adaptive quadrature
    # over seven decades can trigger harmless roundoff warnings when the
    # integrand approaches its radiation-dominated asymptote.
    n_grid = 30001
    z_samples = np.geomspace(1.0 + float(z_d), 1.0 + z_upper_integral, n_grid) - 1.0
    integrand_values = np.fromiter(
        (integrand(float(zv)) for zv in z_samples),
        dtype=np.float64,
        count=n_grid,
    )
    integral = float(simpson(integrand_values, x=z_samples))

    # Radiation-dominated analytical tail beyond z_upper_integral.
    tail = (
        c_over_H0
        / np.sqrt(3.0 * OMEGA_R0_EFF)
        / (1.0 + z_upper_integral)
    )
    return float(integral + tail)


if __name__ == "__main__":
    bg = solve_background(0.0, 1e-30, 1e-3)
    rd = compute_rd(bg)
    eh_zd = z_drag(0.314 * 0.674**2, OMEGA_B0 * 0.674**2)
    print(f"Eisenstein-Hu z_d = {eh_zd:.3f}")
    print(f"r_d(LCDM) = {rd:.3f} Mpc (target: 147.09)")
    print(f"Omega_phi0 = {bg['Omega_phi0']:.6e}")
    assert abs(rd - 147.09) < 0.5, "r_d calibration failed!"
