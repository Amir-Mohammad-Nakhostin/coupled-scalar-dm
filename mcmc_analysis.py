"""MCMC analysis of the scalar-tensor dark-matter coupling model.

This module extends the deterministic comparison suite with a two-parameter
posterior analysis in (beta, log10(m_phi/eV)).  The BAO likelihood uses the
13-point DESI DR2 data vector with a *block-diagonal approximate covariance*:

* diagonal entries are the quoted sigma^2;
* D_M/r_d and D_H/r_d at the same redshift have rho=-0.4;
* different effective redshifts are assumed independent;
* the z=0.295 D_V/r_d point is independent.

IMPORTANT: this is NOT the official DESI DR2 covariance matrix.  It is the
approximation explicitly requested for this project because the exact
measurement covariance was not supplied with the task.  Set USE_EXACT_COV=True
only after replacing ``build_bao_covariance`` with the official covariance
file and validating its ordering and units.

The likelihood is

    log L = -0.5 * (chi2_BAO_full_cov + chi2_RSD_diagonal).

The MCMC uses emcee's affine-invariant ensemble sampler.  Model evaluations
are parallelised with multiprocessing when RUN_IN_SERIAL=False.  The random
seed is fixed for reproducibility.

The Bayes factor is a Laplace approximation, not an exact evidence integral.
For the two-parameter model the code uses

    ln Z_1 ~= -chi2_min/2 + d/2 ln(2*pi) - 0.5 ln det(H) - ln(V_prior),

with H approximated by the inverse covariance of posterior samples.  LCDM is
treated as the beta=0 benchmark with m_phi=1e-30 eV, matching the previous
suite's reference convention; this is effectively GR/LCDM because the scalar
energy contribution is negligible, but it is not a separately hard-coded
Boltzmann-code LCDM calculation.

The script is deterministic given the same installed package versions,
number of workers, and numerical BLAS/LAPACK libraries.  Multiprocessing can
change the wall-clock order of evaluations but not the master sampler RNG.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

try:
    import emcee
except ImportError:  # pragma: no cover - optional runtime dependency
    emcee = None
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid

from bao_data import ALL_BAO
from background_solver import (
    C_KM_S,
    H0_KM_S_MPC,
    compute_rd,
    solve_background,
)
from growth_solver import solve_growth, compute_fsigma8
from rsd_data import RSD_DATA

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent
FAILURE_LOG = OUT_DIR / "mcmc_failures.log"

RANDOM_SEED = 42
N_DIM = 2
NWALKERS = int(os.environ.get("MCMC_NWALKERS", "16"))
NSTEPS = int(os.environ.get("MCMC_NSTEPS", "1000"))
BURN_IN = int(os.environ.get("MCMC_BURN_IN", "200"))
N_CORES = int(os.environ.get("MCMC_NCORES", str(max(1, min(8, os.cpu_count() or 1)))))
RUN_IN_SERIAL = os.environ.get("MCMC_SERIAL", "0") == "1"
PROGRESS = os.environ.get("MCMC_PROGRESS", "1") != "0"

BETA_MIN, BETA_MAX = 0.0, 0.15
LOGM_MIN, LOGM_MAX = -32.0, -28.0
PRIOR_VOLUME = (BETA_MAX - BETA_MIN) * (LOGM_MAX - LOGM_MIN)
PHI_I = 1.0e-3
K_REF = 0.1
SIGMA8_0 = 0.811

LCDM_BETA = 0.0
LCDM_MASS_Ev = 1.0e-30

# The prompt explicitly requests this approximation.
USE_EXACT_COV = False
RHO_DM_DH = -0.4

# The prior initialisation requested by the prompt.
INIT_CENTER = np.array([0.05, -30.0], dtype=np.float64)
INIT_SIGMA = np.array([0.004, 0.05], dtype=np.float64)

# MCMC-optimized background tolerances requested by the user.
SOLVER_RTOL = 1e-6
SOLVER_ATOL = 1e-8

# Recomputing r_d inside every MCMC evaluation is unnecessarily expensive.
# The default MCMC likelihood therefore uses the Planck-2018 reference value.
# Set USE_FIXED_RD=0 to restore model-by-model compute_rd calls.
USE_FIXED_RD = os.environ.get("USE_FIXED_RD", "1") == "1"
FIXED_RD_MPC = 147.09

# Fast-mode numerical path. LSODA was benchmarked against the conservative
# Radau path at the reference point and is much faster for this workload.
FAST_BG_METHOD = "LSODA"
FAST_GROWTH_METHOD = "LSODA"
FAST_BG_N_EVAL = 3000
FAST_GROWTH_N_EVAL = 800
FAST_GROWTH_RTOL = 1e-8
FAST_GROWTH_ATOL = 1e-10

# Per-process LRU-style cache of the last N background solutions.
CACHE_SIZE = 100
CACHE_BETA_DECIMALS = 5
CACHE_MASS_SIG_DIGITS = 6
FAST_CACHE_BETA_DECIMALS = 3
FAST_CACHE_LOGM_DECIMALS = 1

DESI_OBSERVABLES = {"DV_over_rd", "DM_over_rd", "DH_over_rd"}
N_BAO = 13
N_RSD = len(RSD_DATA)
N_DATA = N_BAO + N_RSD

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("mcmc_analysis")
logger.setLevel(logging.INFO)
logger.propagate = False

# -----------------------------------------------------------------------------
# DESI DR2 BAO data and covariance
# -----------------------------------------------------------------------------

def _desi_dr2_data() -> list[tuple[float, str, float, float, str]]:
    """Return the 13 DESI DR2 BAO rows in their supplied order."""
    rows = [
        row
        for row in ALL_BAO
        if row[4].startswith("DESI") and row[1] in DESI_OBSERVABLES
    ]
    if len(rows) != N_BAO:
        raise RuntimeError(f"Expected {N_BAO} DESI DR2 rows, found {len(rows)}.")
    return rows


DESI_BAO_DATA = _desi_dr2_data()
BAO_VECTOR = np.array([row[2] for row in DESI_BAO_DATA], dtype=np.float64)
BAO_SIGMA = np.array([row[3] for row in DESI_BAO_DATA], dtype=np.float64)
BAO_Z = np.array([row[0] for row in DESI_BAO_DATA], dtype=np.float64)
BAO_KIND = [row[1] for row in DESI_BAO_DATA]


def build_bao_covariance(
    data: Iterable[tuple[float, str, float, float, str]],
    rho: float = RHO_DM_DH,
) -> np.ndarray:
    """Build the requested approximate DESI DR2 covariance matrix.

    Parameters
    ----------
    data:
        The 13-row DESI DR2 data vector.
    rho:
        Correlation coefficient between D_M/r_d and D_H/r_d at the same z.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-definite covariance matrix in data-vector units.

    Notes
    -----
    This is explicitly an approximation, not the official DESI DR2 covariance.
    Different redshifts are assumed independent.
    """
    rows = list(data)
    n = len(rows)
    cov = np.diag(np.array([r[3] ** 2 for r in rows], dtype=np.float64))
    for i in range(n):
        zi, kind_i, _, sigma_i, _ = rows[i]
        if kind_i not in ("DM_over_rd", "DH_over_rd"):
            continue
        for j in range(i + 1, n):
            zj, kind_j, _, sigma_j, _ = rows[j]
            if np.isclose(zi, zj) and {kind_i, kind_j} == {"DM_over_rd", "DH_over_rd"}:
                cov_ij = float(rho * sigma_i * sigma_j)
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij
    # Validate symmetry and positive definiteness now so likelihood calls are cheap.
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-15):
        raise RuntimeError("Constructed BAO covariance is not symmetric.")
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Constructed BAO covariance is not positive definite.") from exc
    return cov


BAO_COV = build_bao_covariance(DESI_BAO_DATA)
BAO_CHO = cho_factor(BAO_COV, lower=True, check_finite=True)

_BACKGROUND_CACHE: OrderedDict[tuple[float, float], dict] = OrderedDict()
_FAST_MODE = False

# -----------------------------------------------------------------------------
# Cosmology helpers
# -----------------------------------------------------------------------------

def _distance_arrays(background: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return z, E(z), and line-of-sight comoving distance arrays."""
    z = np.asarray(background["z"], dtype=np.float64)[::-1]
    E = np.asarray(background["H"], dtype=np.float64)[::-1]
    c_over_h0 = C_KM_S / H0_KM_S_MPC
    inv_h = c_over_h0 / E
    dm = cumulative_trapezoid(inv_h, z, initial=0.0)
    return z, E, dm


def _bao_prediction_vector(background: dict, rd_mpc: float) -> np.ndarray:
    """Compute the 13 DESI BAO observables in the supplied data-vector order."""
    z_grid, H_grid, dm_grid = _distance_arrays(background)
    predictions: list[float] = []
    for z, kind, *_ in DESI_BAO_DATA:
        E = float(np.interp(z, z_grid, H_grid))
        d_h = C_KM_S / (H0_KM_S_MPC * E)
        d_m = float(np.interp(z, z_grid, dm_grid))
        d_v = (z * d_m**2 * d_h) ** (1.0 / 3.0)
        value = {
            "DM_over_rd": d_m / rd_mpc,
            "DH_over_rd": d_h / rd_mpc,
            "DV_over_rd": d_v / rd_mpc,
        }[kind]
        predictions.append(value)
    return np.asarray(predictions, dtype=np.float64)


def chi2_bao_full(background: dict, rd_mpc: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return full-covariance DESI BAO chi2 and residual vectors."""
    prediction = _bao_prediction_vector(background, rd_mpc)
    residual = prediction - BAO_VECTOR
    whitened = cho_solve(BAO_CHO, residual, check_finite=False)
    chi2 = float(np.dot(residual, whitened))
    return chi2, prediction, residual


def chi2_rsd_from_background(background: dict, beta: float, m_phi_eV: float) -> float:
    """Solve growth at k=0.1 Mpc^-1 and return the diagonal RSD chi2."""
    if _FAST_MODE:
        growth = solve_growth(
            background, k=K_REF, beta=beta, m_phi_eV=m_phi_eV,
            N_i=np.log(1e-3), N_f=0.0,
            method=FAST_GROWTH_METHOD, rtol=FAST_GROWTH_RTOL,
            atol=FAST_GROWTH_ATOL, n_eval=FAST_GROWTH_N_EVAL,
        )
    else:
        growth = solve_growth(
            background, k=K_REF, beta=beta, m_phi_eV=m_phi_eV,
            N_i=np.log(1e-3), N_f=0.0,
            method="Radau", rtol=1e-8, atol=1e-10, n_eval=2500,
        )
    fs = compute_fsigma8(growth, sigma8_0=SIGMA8_0)
    z_model = np.asarray(fs["z"], dtype=np.float64)[::-1]
    fs_model = np.asarray(fs["fsigma8"], dtype=np.float64)[::-1]
    chi2 = 0.0
    for z, observed, sigma, _survey in RSD_DATA:
        predicted = float(np.interp(z, z_model, fs_model))
        chi2 += ((predicted - observed) / sigma) ** 2
    return float(chi2)


def _log_failure(beta: float, logm: float, exc: Exception) -> None:
    """Append a concise likelihood failure to the shared log."""
    message = (
        f"Likelihood failure beta={beta:.10g} log10_m={logm:.10g}: "
        f"{type(exc).__name__}: {exc}\n"
    )
    try:
        with FAILURE_LOG.open("a", encoding="utf-8") as f:
            f.write(message)
    except OSError:
        # Logging must never turn a failed model evaluation into a failed run.
        pass


def _cache_key(beta: float, m_phi_eV: float) -> tuple[float, float]:
    """Return the rounded (beta, m_phi_eV) key used by the background cache."""
    if _FAST_MODE:
        beta_key = round(float(beta), FAST_CACHE_BETA_DECIMALS)
        logm_key = round(float(np.log10(m_phi_eV)), FAST_CACHE_LOGM_DECIMALS)
        mass_key = float(10.0 ** logm_key)
    else:
        beta_key = round(float(beta), CACHE_BETA_DECIMALS)
        mass_key = float(f"{float(m_phi_eV):.{CACHE_MASS_SIG_DIGITS}e}")
    return beta_key, mass_key


def _get_background(beta: float, m_phi_eV: float) -> dict:
    """Fetch a cached background or solve it once for this cache cell."""
    key = _cache_key(beta, m_phi_eV)
    cached = _BACKGROUND_CACHE.get(key)
    if cached is not None:
        _BACKGROUND_CACHE.move_to_end(key)
        return cached

    method = FAST_BG_METHOD if _FAST_MODE else "Radau"
    n_eval = FAST_BG_N_EVAL if _FAST_MODE else 3000
    background = solve_background(
        beta, m_phi_eV, PHI_I,
        rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
        method=method, n_eval=n_eval,
    )
    _BACKGROUND_CACHE[key] = background
    _BACKGROUND_CACHE.move_to_end(key)
    while len(_BACKGROUND_CACHE) > CACHE_SIZE:
        _BACKGROUND_CACHE.popitem(last=False)
    return background


def _rd_for_likelihood(background: dict) -> float:
    """Return fixed Planck r_d or compute model-specific r_d when requested."""
    if USE_FIXED_RD:
        return FIXED_RD_MPC
    rd = compute_rd(background)
    if not np.isfinite(rd) or rd <= 0.0:
        raise FloatingPointError(f"Invalid r_d={rd}")
    return float(rd)


def evaluate_model(beta: float, logm: float) -> tuple[float, float, float]:
    """Evaluate (chi2_BAO, chi2_RSD, chi2_total) for one parameter point."""
    if not (BETA_MIN <= beta <= BETA_MAX and LOGM_MIN <= logm <= LOGM_MAX):
        return 1.0e10, 1.0e10, 2.0e10
    m_phi_eV = 10.0 ** float(logm)
    try:
        background = _get_background(beta, m_phi_eV)
        rd = _rd_for_likelihood(background)
        chi2_bao, _, _ = chi2_bao_full(background, rd)
        chi2_rsd = chi2_rsd_from_background(background, beta, m_phi_eV)
        chi2_total = float(chi2_bao + chi2_rsd)
        if not np.isfinite(chi2_total):
            raise FloatingPointError("Non-finite total chi2")
        return float(chi2_bao), float(chi2_rsd), chi2_total
    except Exception as exc:
        _log_failure(beta, logm, exc)
        return 1.0e10, 1.0e10, 2.0e10


def log_probability(theta: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    """Uniform-prior log posterior plus chi-square blobs for emcee."""
    beta = float(theta[0])
    logm = float(theta[1])
    chi2_bao, chi2_rsd, chi2_total = evaluate_model(beta, logm)
    if chi2_total >= 1.0e9:
        return -np.inf, (chi2_bao, chi2_rsd, chi2_total)
    return -0.5 * chi2_total, (chi2_bao, chi2_rsd, chi2_total)


# -----------------------------------------------------------------------------
# Walker initialisation and diagnostics
# -----------------------------------------------------------------------------

def _initial_positions(rng: np.random.Generator, nwalkers: int) -> np.ndarray:
    """Draw the requested number of valid walkers near the starting point."""
    positions = np.empty((nwalkers, N_DIM), dtype=np.float64)
    for i in range(nwalkers):
        for _attempt in range(1000):
            candidate = INIT_CENTER + rng.normal(scale=INIT_SIGMA, size=N_DIM)
            if BETA_MIN < candidate[0] < BETA_MAX and LOGM_MIN < candidate[1] < LOGM_MAX:
                positions[i] = candidate
                break
        else:  # pragma: no cover - unreachable with the wide prior
            raise RuntimeError("Could not initialise valid MCMC walkers.")
    return positions


def _autocorr_time_text(sampler: emcee.EnsembleSampler, nsteps: int) -> tuple[str, np.ndarray | None, bool]:
    """Compute autocorrelation diagnostics, tolerating short chains."""
    try:
        tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=np.float64)
        threshold = nsteps / 50.0
        warn = bool(np.any(tau > threshold))
        status = (
            f"Autocorrelation times: beta={tau[0]:.2f}, log10_m={tau[1]:.2f}; "
            f"criterion nsteps/50={threshold:.2f}."
        )
        if warn:
            status += " WARNING: at least one tau exceeds nsteps/50."
        return status, tau, warn
    except Exception as exc:
        return f"Autocorrelation time unavailable: {exc}", None, True


# -----------------------------------------------------------------------------
# Evidence and plots
# -----------------------------------------------------------------------------

def _laplace_log_evidence(
    chi2_min: float,
    samples: np.ndarray,
) -> tuple[float, float, str]:
    """Approximate model evidence from a Gaussian/Laplace approximation.

    The Hessian H is approximated as the inverse covariance of the posterior
    samples.  This is reliable only when the posterior is reasonably Gaussian
    and the best fit is not pinned to a prior boundary.
    """
    if samples.shape[0] < 20:
        raise ValueError("Too few posterior samples for a Laplace estimate.")
    cov = np.cov(samples.T, ddof=1)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    cov += np.eye(N_DIM, dtype=np.float64) * 1.0e-12
    sign_h, logdet_h = np.linalg.slogdet(np.linalg.pinv(cov))
    if sign_h <= 0 or not np.isfinite(logdet_h):
        raise FloatingPointError("Posterior-derived Hessian is not positive definite.")
    ln_z = (
        -0.5 * chi2_min
        + 0.5 * N_DIM * np.log(2.0 * np.pi)
        - 0.5 * logdet_h
        - np.log(PRIOR_VOLUME)
    )
    eigenvalues = np.linalg.eigvalsh(cov)
    warning = ""
    if np.any(eigenvalues <= 0.0):
        warning = "Posterior covariance was not strictly positive definite."
    return float(ln_z), float(logdet_h), warning


def _jeffreys_label(ln_bayes_factor: float) -> str:
    """Return the requested qualitative label for |ln B|."""
    x = abs(float(ln_bayes_factor))
    if x < 1.0:
        return "inconclusive"
    if x < 3.0:
        return "weak evidence"
    if x < 5.0:
        return "moderate evidence"
    return "strong evidence"


def _save_chain_csv(
    samples: np.ndarray,
    blobs: np.ndarray,
    path: Path,
) -> None:
    """Save flattened post-burn-in chain with chi-square columns."""
    fields = ["beta", "log10_m_phi_eV", "chi2_BAO", "chi2_RSD", "chi2_total"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for theta, blob in zip(samples, blobs):
            writer.writerow([
                f"{theta[0]:.12g}",
                f"{theta[1]:.12g}",
                f"{blob[0]:.12g}",
                f"{blob[1]:.12g}",
                f"{blob[2]:.12g}",
            ])


def _make_trace_plot(chain: np.ndarray, path: Path) -> None:
    """Create a publication-style walker trace plot."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    labels = [r"$\beta$", r"$\log_{10}(m_\phi/\mathrm{eV})$"]
    for j, ax in enumerate(axes):
        ax.plot(chain[:, :, j], alpha=0.18, linewidth=0.35)
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.18)
    axes[-1].set_xlabel("MCMC step")
    fig.suptitle("MCMC trace")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_corner_plot(samples: np.ndarray, path: Path) -> None:
    """Create a dependency-free two-parameter corner plot."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    labels = [r"$\beta$", r"$\log_{10}(m_\phi/\mathrm{eV})$"]
    for i in range(2):
        ax = axes[i, i]
        ax.hist(samples[:, i], bins=50, density=True, histtype="step", linewidth=1.5)
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        ax.axvline(q50, linewidth=1.2)
        ax.axvspan(q16, q84, alpha=0.16)
        ax.set_xlabel(labels[i])
        ax.set_ylabel("density")
        ax.grid(alpha=0.15)

    ax = axes[1, 0]
    ax.hist2d(samples[:, 0], samples[:, 1], bins=60, cmap="Greys", density=True)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.12)
    axes[0, 1].axis("off")
    fig.suptitle("Posterior corner plot")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_beta_posterior(samples: np.ndarray, path: Path) -> None:
    """Create the 1D marginalized posterior for beta."""
    beta = samples[:, 0]
    q16, q50, q84 = np.percentile(beta, [16, 50, 84])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    counts, edges = np.histogram(beta, bins=60, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, linewidth=1.8)
    ax.fill_between(centers, counts, alpha=0.18)
    ax.axvline(q50, linewidth=1.3, label=f"median = {q50:.4f}")
    ax.axvspan(q16, q84, alpha=0.14, label=f"68% = [{q16:.4f}, {q84:.4f}]")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("posterior density")
    ax.set_title(r"Marginalized posterior for $\beta$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_bao_residual_plot(beta: float, logm: float, path: Path) -> dict:
    """Compute and plot standardized DESI BAO residuals for the best fit."""
    m_phi_eV = 10.0 ** logm
    background = solve_background(
        beta, m_phi_eV, PHI_I,
        rtol=SOLVER_RTOL, atol=SOLVER_ATOL,
        method=FAST_BG_METHOD if _FAST_MODE else "Radau",
        n_eval=FAST_BG_N_EVAL if _FAST_MODE else 3000,
    )
    rd = _rd_for_likelihood(background)
    chi2_bao, prediction, residual = chi2_bao_full(background, rd)
    standardized = residual / BAO_SIGMA

    labels = [f"{z:.3g} {kind.replace('_over_rd', '')}" for z, kind in zip(BAO_Z, BAO_KIND)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(1.0, linewidth=0.8, linestyle="--", alpha=0.55)
    ax.axhline(-1.0, linewidth=0.8, linestyle="--", alpha=0.55)
    ax.scatter(x, standardized, s=32)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(r"$(model-data)/\sigma$")
    ax.set_title(
        rf"DESI DR2 BAO residuals — best fit $\beta={beta:.4f}$, "
        rf"$\log_{{10}}m_\phi={logm:.3f}$; $\chi^2_{{BAO}}={chi2_bao:.2f}$"
    )
    ax.grid(alpha=0.18, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "rd_Mpc": float(rd),
        "prediction": prediction,
        "residual": residual,
        "standardized": standardized,
        "chi2_bao": float(chi2_bao),
        "background": background,
    }


def _best_fit_exact(sample_best: np.ndarray) -> tuple[np.ndarray, float, float, float, bool]:
    """Refine the chain minimum with a local bounded optimizer."""
    def objective(theta: np.ndarray) -> float:
        return evaluate_model(float(theta[0]), float(theta[1]))[2]

    try:
        result = minimize(
            objective,
            x0=np.asarray(sample_best, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(BETA_MIN, BETA_MAX), (LOGM_MIN, LOGM_MAX)],
            options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-7},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        chi2_bao, chi2_rsd, chi2_total = evaluate_model(float(theta[0]), float(theta[1]))
        return theta, chi2_bao, chi2_rsd, chi2_total, bool(result.success)
    except Exception as exc:
        logger.error("Best-fit refinement failed: %s: %s", type(exc).__name__, exc)
        return (
            np.asarray(sample_best, dtype=np.float64),
            *evaluate_model(float(sample_best[0]), float(sample_best[1])),
            False,
        )


def _lcdm_chi2() -> tuple[float, float, float, dict]:
    """Evaluate the fixed LCDM benchmark convention used by the previous suite."""
    beta = LCDM_BETA
    m_phi_eV = LCDM_MASS_Ev
    background = solve_background(
        beta,
        m_phi_eV,
        PHI_I,
        rtol=SOLVER_RTOL,
        atol=SOLVER_ATOL,
    )
    rd = _rd_for_likelihood(background)
    chi2_bao, _, _ = chi2_bao_full(background, rd)
    chi2_rsd = chi2_rsd_from_background(background, beta, m_phi_eV)
    return float(chi2_bao), float(chi2_rsd), float(chi2_bao + chi2_rsd), background


def _write_summary(
    path: Path,
    samples: np.ndarray,
    chi2_total_samples: np.ndarray,
    best_theta: np.ndarray,
    best_chi2_bao: float,
    best_chi2_rsd: float,
    best_chi2_total: float,
    lcdm_chi2_bao: float,
    lcdm_chi2_rsd: float,
    lcdm_chi2_total: float,
    sampler: emcee.EnsembleSampler,
    elapsed_s: float,
    nwalkers: int,
    nsteps: int,
    burn_in: int,
    ncores: int,
    run_serial: bool,
    mean_step_s: float,
    estimated_total_s: float,
) -> None:
    """Write the numerical and methodological summary report."""
    q_beta = np.percentile(samples[:, 0], [16, 50, 84])
    q_logm = np.percentile(samples[:, 1], [16, 50, 84])
    acc = float(np.mean(sampler.acceptance_fraction))
    tau_text, tau, tau_warn = _autocorr_time_text(sampler, nsteps)

    dchi2 = float(lcdm_chi2_total - best_chi2_total)
    aic_lcdm = float(lcdm_chi2_total)
    aic_model = float(best_chi2_total + 2 * N_DIM)
    bic_lcdm = float(lcdm_chi2_total)
    bic_model = float(best_chi2_total + N_DIM * np.log(N_DATA))

    try:
        ln_z1, logdet_h, laplace_warning = _laplace_log_evidence(best_chi2_total, samples)
        ln_z0 = -0.5 * lcdm_chi2_total
        ln_b10 = float(ln_z1 - ln_z0)
        bayes_label = _jeffreys_label(ln_b10)
    except Exception as exc:
        ln_z1 = np.nan
        ln_z0 = -0.5 * lcdm_chi2_total
        ln_b10 = np.nan
        logdet_h = np.nan
        bayes_label = "unavailable"
        laplace_warning = f"Laplace evidence failed: {exc}"

    with path.open("w", encoding="utf-8") as f:
        f.write("Scalar-tensor dark-matter coupling MCMC summary\n")
        f.write("=" * 78 + "\n\n")
        f.write("CONFIGURATION\n")
        f.write(f"random_seed = {RANDOM_SEED}\n")
        f.write(f"nwalkers = {nwalkers}\n")
        f.write(f"nsteps = {nsteps}\n")
        f.write(f"burn_in = {burn_in}\n")
        f.write(f"n_cores = {ncores}\n")
        f.write(f"run_in_serial = {run_serial}\n")
        f.write(f"mean_seconds_per_step = {mean_step_s:.6f}\n")
        f.write(f"estimated_sampling_seconds = {estimated_total_s:.6f}\n")
        f.write(f"beta_prior = [{BETA_MIN}, {BETA_MAX}]\n")
        f.write(f"log10_m_phi_prior = [{LOGM_MIN}, {LOGM_MAX}]\n")
        f.write(f"phi_i = {PHI_I}\n")
        f.write(f"k_ref_Mpc_inv = {K_REF}\n")
        f.write(f"sigma8_0 = {SIGMA8_0}\n")
        f.write(f"USE_EXACT_COV = {USE_EXACT_COV}\n")
        f.write(f"approximate_covariance_rho = {RHO_DM_DH}\n")
        f.write(f"solver_rtol = {SOLVER_RTOL}\n")
        f.write(f"solver_atol = {SOLVER_ATOL}\n")
        f.write(f"N_BAO_DESI = {N_BAO}\n")
        f.write(f"N_RSD = {N_RSD}\n")
        f.write(f"N_DATA = {N_DATA}\n")
        f.write(f"wall_time_seconds = {elapsed_s:.3f}\n")
        f.write(f"USE_FIXED_RD = {USE_FIXED_RD}\n")
        f.write(f"FIXED_RD_MPC = {FIXED_RD_MPC:.8f}\n")
        f.write(f"FAST_MODE = {_FAST_MODE}\n")
        f.write(f"FAST_BG_METHOD = {FAST_BG_METHOD}\n")
        f.write(f"FAST_GROWTH_METHOD = {FAST_GROWTH_METHOD}\n")
        f.write(f"FAST_BG_N_EVAL = {FAST_BG_N_EVAL}\n")
        f.write(f"FAST_GROWTH_N_EVAL = {FAST_GROWTH_N_EVAL}\n")
        f.write(f"FAST_CACHE_BETA_DECIMALS = {FAST_CACHE_BETA_DECIMALS}\n")
        f.write(f"FAST_CACHE_LOGM_DECIMALS = {FAST_CACHE_LOGM_DECIMALS}\n\n")

        f.write("POSTERIOR\n")
        f.write("beta = %.8f (+%.8f / -%.8f)\n" % (
            q_beta[1], q_beta[2] - q_beta[1], q_beta[1] - q_beta[0]
        ))
        f.write("log10_m_phi_eV = %.8f (+%.8f / -%.8f)\n" % (
            q_logm[1], q_logm[2] - q_logm[1], q_logm[1] - q_logm[0]
        ))
        f.write(f"best_fit_beta = {best_theta[0]:.10f}\n")
        f.write(f"best_fit_log10_m_phi_eV = {best_theta[1]:.10f}\n")
        f.write(f"best_fit_m_phi_eV = {10.0 ** best_theta[1]:.10e}\n")
        f.write(f"chi2_BAO_best = {best_chi2_bao:.6f}\n")
        f.write(f"chi2_RSD_best = {best_chi2_rsd:.6f}\n")
        f.write(f"chi2_total_best = {best_chi2_total:.6f}\n\n")

        f.write("LCDM COMPARISON\n")
        f.write(f"lcdm_beta = {LCDM_BETA:.8f}\n")
        f.write(f"lcdm_m_phi_eV = {LCDM_MASS_Ev:.10e}\n")
        f.write(f"chi2_BAO_lcdm = {lcdm_chi2_bao:.6f}\n")
        f.write(f"chi2_RSD_lcdm = {lcdm_chi2_rsd:.6f}\n")
        f.write(f"chi2_total_lcdm = {lcdm_chi2_total:.6f}\n")
        f.write(f"Delta_chi2 = chi2_LCDM - chi2_best = {dchi2:.6f}\n")
        f.write(f"AIC_LCDM = {aic_lcdm:.6f}\n")
        f.write(f"AIC_model = {aic_model:.6f}\n")
        f.write(f"Delta_AIC_model-minus-LCDM = {aic_model - aic_lcdm:.6f}\n")
        f.write(f"BIC_LCDM = {bic_lcdm:.6f}\n")
        f.write(f"BIC_model = {bic_model:.6f}\n")
        f.write(f"Delta_BIC_model-minus-LCDM = {bic_model - bic_lcdm:.6f}\n\n")

        f.write("BAYES FACTOR (LAPLACE APPROXIMATION)\n")
        f.write(f"ln_Z0 = {ln_z0:.8f}\n")
        f.write(f"ln_Z1 = {ln_z1:.8f}\n")
        f.write(f"logdet_H = {logdet_h:.8f}\n")
        f.write(f"ln_B10 = ln_Z1 - ln_Z0 = {ln_b10:.8f}\n")
        f.write(f"Jeffreys interpretation = {bayes_label}\n")
        f.write(
            "WARNING: the Bayes factor is a Laplace estimate based on the posterior "
            "sample covariance, not an exact thermodynamic-integration evidence.\n"
        )
        if laplace_warning:
            f.write(f"Laplace warning = {laplace_warning}\n")
        f.write("\n")

        f.write("CONVERGENCE\n")
        f.write(f"mean_acceptance_fraction = {acc:.6f}\n")
        f.write(tau_text + "\n")
        f.write(
            "Convergence warning = "
            + ("YES" if tau_warn else "NO")
            + "\n"
        )
        f.write(
            "Interpretation: nsteps should be comfortably larger than 50*tau; "
            "the nsteps/50 check requested by the project is reported above.\n"
        )

        f.write("\nCOVARIANCE / DATA WARNING\n")
        f.write(
            "The BAO likelihood uses the prompt-specified rho=-0.4 block-diagonal "
            "approximation. It is not the official DESI DR2 covariance matrix.\n"
        )
        f.write(
            "Cross-redshift correlations and any official systematic covariance are "
            "not included. Do not interpret this as a final DESI collaboration likelihood.\n"
        )
        f.write(
            "RSD uses the existing diagonal errors because no RSD covariance matrix "
            "was supplied.\n"
        )
        f.write(
            f"Likelihood r_d is {'fixed at 147.09 Mpc' if USE_FIXED_RD else 'computed for each model'}.\n"
        )
        if _FAST_MODE:
            f.write(
                "FAST_MODE additionally quantises the background cache to "
                f"beta={FAST_CACHE_BETA_DECIMALS} decimal places and "
                f"log10(m_phi)={FAST_CACHE_LOGM_DECIMALS} decimal places; this is "
                "a speed approximation and should be disabled/refined for final publication chains.\n"
            )


# -----------------------------------------------------------------------------
# Runtime configuration / timing
# -----------------------------------------------------------------------------

class _CachedMapPool:
    """Pool-compatible map wrapper with a parent-process fast MCMC cache.

    In fast mode, adjacent proposals are quantised to the same cache cell. The
    wrapper prevents repeated worker evaluations across MCMC steps. This is an
    explicit speed approximation; final publication chains should rerun with a
    finer cache or no quantisation.
    """

    def __init__(self, raw_pool, fast_mode: bool):
        self.raw_pool = raw_pool
        self.fast_mode = bool(fast_mode)
        self.cache: OrderedDict[tuple[float, float], object] = OrderedDict()

    @staticmethod
    def _theta_key(theta: np.ndarray) -> tuple[float, float]:
        beta = round(float(theta[0]), FAST_CACHE_BETA_DECIMALS)
        logm = round(float(theta[1]), FAST_CACHE_LOGM_DECIMALS)
        return beta, logm

    def map(self, func, iterable):
        items = list(iterable)
        if not self.fast_mode:
            if self.raw_pool is None:
                return [func(item) for item in items]
            return self.raw_pool.map(func, items)

        results = [None] * len(items)
        missing_items = []
        missing_positions = []
        for idx, theta in enumerate(items):
            key = self._theta_key(np.asarray(theta))
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                results[idx] = cached
            else:
                missing_items.append(theta)
                missing_positions.append((idx, key))

        if missing_items:
            if self.raw_pool is None:
                missing_results = [func(item) for item in missing_items]
            else:
                missing_results = self.raw_pool.map(func, missing_items)
            for (idx, key), value in zip(missing_positions, missing_results):
                self.cache[key] = value
                self.cache.move_to_end(key)
                while len(self.cache) > CACHE_SIZE:
                    self.cache.popitem(last=False)
                results[idx] = value
        return results

    def clear(self) -> None:
        self.cache.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-mode", action="store_true",
                        help="Enable fixed r_d, rounded background cache, LSODA, and optimized MCMC settings.")
    parser.add_argument("--nwalkers", type=int, default=NWALKERS)
    parser.add_argument("--nsteps", type=int, default=NSTEPS)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--ncores", type=int, default=N_CORES)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark the optimized likelihood-map workload without requiring emcee.")
    parser.add_argument("--benchmark-steps", type=int, default=50)
    return parser.parse_args()


def _worker_init(fast_mode: bool) -> None:
    global _FAST_MODE, _BACKGROUND_CACHE
    _FAST_MODE = bool(fast_mode)
    _BACKGROUND_CACHE = OrderedDict()


def _timing_line(step: int, step_time: float, times: list[float], nsteps: int) -> str:
    recent = times[-min(20, len(times)):]
    mean_step = float(np.mean(recent))
    mean_all = float(np.mean(times))
    eta = max(0, nsteps-step) * mean_step
    return (f"step {step:4d}/{nsteps}: {step_time:6.3f} s | "
            f"mean={mean_all:6.3f} s | ETA={eta/60.0:6.2f} min")


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def _run_speed_benchmark(nwalkers: int, nsteps: int, ncores: int, use_serial: bool) -> float:
    """Benchmark the same 16-walker likelihood-map workload used by emcee.

    This is not an MCMC posterior run. It generates a deterministic, correlated
    random-walk trajectory around the requested initialization point and times
    the exact ``log_probability`` calls through the same cache/pool wrapper.
    It is intentionally runnable without the optional emcee dependency.
    """
    if nsteps <= 0 or nwalkers < 2 * N_DIM:
        raise ValueError("Invalid benchmark dimensions.")
    rng = np.random.default_rng(RANDOM_SEED)
    state = np.column_stack([
        INIT_CENTER[0] + INIT_SIGMA[0] * rng.normal(size=nwalkers),
        INIT_CENTER[1] + INIT_SIGMA[1] * rng.normal(size=nwalkers),
    ])
    raw_pool = None
    if not use_serial:
        ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
        raw_pool = ctx.Pool(processes=ncores, initializer=_worker_init, initargs=(True,))
    else:
        _worker_init(True)
    wrapper = _CachedMapPool(raw_pool, True)
    step_times: list[float] = []
    start = time.perf_counter()
    try:
        for step in range(1, nsteps + 1):
            state = state + rng.normal([0.0, 0.0], [0.0005, 0.01], state.shape)
            state[:, 0] = np.clip(state[:, 0], BETA_MIN + 1e-6, BETA_MAX - 1e-6)
            state[:, 1] = np.clip(state[:, 1], LOGM_MIN + 1e-6, LOGM_MAX - 1e-6)
            batch = [row.copy() for row in state]
            t = time.perf_counter()
            wrapper.map(log_probability, batch)
            dt = time.perf_counter() - t
            step_times.append(dt)
            if step <= 3 or step % 10 == 0 or step == nsteps:
                print(_timing_line(step, dt, step_times, nsteps), flush=True)
    finally:
        if raw_pool is not None:
            raw_pool.close()
            raw_pool.join()
    elapsed = time.perf_counter() - start
    print(f"Benchmark total: {elapsed:.3f} s ({elapsed/60.0:.3f} min)")
    print(f"Benchmark mean step: {np.mean(step_times):.3f} s")
    print(f"Benchmark under 60 s: {elapsed < 60.0}")
    return float(elapsed)


def main() -> None:
    """Run the full MCMC workflow and write all requested artifacts."""
    global _FAST_MODE
    args = _parse_args()
    if args.benchmark:
        _FAST_MODE = True
        _run_speed_benchmark(
            nwalkers=int(args.nwalkers),
            nsteps=int(args.benchmark_steps),
            ncores=max(1, int(args.ncores)),
            use_serial=bool(args.serial or RUN_IN_SERIAL),
        )
        return
    if emcee is None:
        raise RuntimeError(
            "emcee is required to run the MCMC. Install it with: python -m pip install emcee"
        )
    _FAST_MODE = bool(args.fast_mode)

    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    nwalkers = int(args.nwalkers)
    nsteps = int(args.nsteps)
    burn_in = int(args.burn_in)
    ncores = max(1, int(args.ncores))
    run_serial = bool(args.serial or RUN_IN_SERIAL)

    if burn_in >= nsteps:
        raise ValueError("burn-in must be smaller than nsteps.")
    if nwalkers < 2 * N_DIM:
        raise ValueError("emcee requires at least 2*n_dim walkers.")
    if USE_EXACT_COV:
        raise NotImplementedError(
            "USE_EXACT_COV=True requires replacing build_bao_covariance with the official covariance file."
        )

    print("\nScalar-tensor MCMC analysis")
    print("=" * 88)
    print(f"DESI DR2 BAO covariance: APPROXIMATE block diagonal, rho={RHO_DM_DH}")
    print(f"USE_EXACT_COV = {USE_EXACT_COV}")
    print(f"USE_FIXED_RD = {USE_FIXED_RD} (r_d={FIXED_RD_MPC:.2f} Mpc)")
    print(f"FAST_MODE = {_FAST_MODE}")
    print(f"nwalkers={nwalkers}, nsteps={nsteps}, burn-in={burn_in}, cores={ncores}")
    if _FAST_MODE:
        print(f"background/growth methods: {FAST_BG_METHOD}/{FAST_GROWTH_METHOD}")
        print(f"background cache: last {CACHE_SIZE} rounded solutions")
    if nsteps < 1000:
        print("WARNING: chain length is for speed testing/initialization, not final convergence evidence.")

    start = time.perf_counter()
    positions = _initial_positions(rng, nwalkers)

    raw_pool = None
    if not run_serial:
        ctx = mp.get_context("spawn" if os.name == "nt" else "fork")
        raw_pool = ctx.Pool(processes=ncores, initializer=_worker_init, initargs=(_FAST_MODE,))
    else:
        _worker_init(_FAST_MODE)
    pool = _CachedMapPool(raw_pool, _FAST_MODE)

    step_times: list[float] = []
    try:
        sampler = emcee.EnsembleSampler(nwalkers, N_DIM, log_probability, pool=pool)
        print("Starting MCMC ...")
        state = positions
        step_clock = time.perf_counter()
        for step, state in enumerate(sampler.sample(state, iterations=nsteps, progress=False), start=1):
            now = time.perf_counter()
            dt = now - step_clock
            step_clock = now
            step_times.append(dt)
            if step <= 5 or step % 10 == 0 or step == nsteps:
                print(_timing_line(step, dt, step_times, nsteps), flush=True)
    finally:
        if raw_pool is not None:
            raw_pool.close()
            raw_pool.join()

    sampling_elapsed_s = time.perf_counter() - start
    mean_step_s = float(np.mean(step_times)) if step_times else np.nan
    estimated_total_s = mean_step_s * nsteps if np.isfinite(mean_step_s) else np.nan
    print(f"Estimated sampling time: {estimated_total_s/60.0:.2f} min")
    print(f"Actual sampling time:    {sampling_elapsed_s/60.0:.2f} min")

    chain = np.asarray(sampler.get_chain(), dtype=np.float64)
    blobs = np.asarray(sampler.get_blobs(), dtype=np.float64)
    np.save(OUT_DIR / "mcmc_chain.npy", chain)

    if chain.shape != (nsteps, nwalkers, N_DIM):
        raise RuntimeError(f"Unexpected chain shape: {chain.shape}")
    if blobs.shape[:2] != (nsteps, nwalkers) or blobs.shape[-1] != 3:
        raise RuntimeError(f"Unexpected blob shape: {blobs.shape}")

    posterior_chain = chain[burn_in:, :, :].reshape(-1, N_DIM)
    posterior_blobs = blobs[burn_in:, :, :].reshape(-1, 3)
    valid = np.isfinite(posterior_blobs[:, 2]) & (posterior_blobs[:, 2] < 1.0e9)
    posterior_chain = posterior_chain[valid]
    posterior_blobs = posterior_blobs[valid]
    if posterior_chain.shape[0] < 100:
        raise RuntimeError("Too few valid post-burn-in samples for posterior analysis.")

    _save_chain_csv(
        posterior_chain,
        posterior_blobs,
        OUT_DIR / "mcmc_posterior.csv",
    )

    chi2_total_samples = posterior_blobs[:, 2]
    best_idx = int(np.argmin(chi2_total_samples))
    sample_best = posterior_chain[best_idx]
    best_theta, best_chi2_bao, best_chi2_rsd, best_chi2_total, opt_success = _best_fit_exact(sample_best)

    # Fixed LCDM benchmark uses the previous suite's convention.
    lcdm_chi2_bao, lcdm_chi2_rsd, lcdm_chi2_total, _lcdm_bg = _lcdm_chi2()

    # Plots.
    _make_corner_plot(posterior_chain, OUT_DIR / "mcmc_corner.png")
    _make_trace_plot(chain, OUT_DIR / "mcmc_trace.png")
    _make_beta_posterior(posterior_chain, OUT_DIR / "mcmc_1d_posterior.png")
    best_bao = _make_bao_residual_plot(
        float(best_theta[0]),
        float(best_theta[1]),
        OUT_DIR / "mcmc_bao_residuals.png",
    )

    full_elapsed_s = time.perf_counter() - start
    _write_summary(
        OUT_DIR / "mcmc_summary.txt",
        posterior_chain,
        chi2_total_samples,
        best_theta,
        best_chi2_bao,
        best_chi2_rsd,
        best_chi2_total,
        lcdm_chi2_bao,
        lcdm_chi2_rsd,
        lcdm_chi2_total,
        sampler,
        full_elapsed_s,
        nwalkers, nsteps, burn_in, ncores, run_serial,
        mean_step_s, estimated_total_s,
    )

    # stdout summary table
    q_beta = np.percentile(posterior_chain[:, 0], [16, 50, 84])
    q_logm = np.percentile(posterior_chain[:, 1], [16, 50, 84])
    dchi2 = lcdm_chi2_total - best_chi2_total
    aic_lcdm = lcdm_chi2_total
    aic_model = best_chi2_total + 2 * N_DIM
    bic_lcdm = lcdm_chi2_total
    bic_model = best_chi2_total + N_DIM * np.log(N_DATA)
    tau_text, _tau, tau_warn = _autocorr_time_text(sampler, nsteps)
    try:
        ln_z1, _, _ = _laplace_log_evidence(best_chi2_total, posterior_chain)
        ln_z0 = -0.5 * lcdm_chi2_total
        ln_b10 = ln_z1 - ln_z0
    except Exception:
        ln_b10 = np.nan

    print("\nRESULTS")
    print("-" * 88)
    print(f"beta posterior:       {q_beta[1]:.6f} +{q_beta[2]-q_beta[1]:.6f}/-{q_beta[1]-q_beta[0]:.6f}")
    print(f"log10(m_phi/eV):      {q_logm[1]:.6f} +{q_logm[2]-q_logm[1]:.6f}/-{q_logm[1]-q_logm[0]:.6f}")
    print(f"best-fit beta:         {best_theta[0]:.8f}")
    print(f"best-fit log10(m/eV):  {best_theta[1]:.8f}")
    print(f"best-fit chi2 BAO:     {best_chi2_bao:.6f}")
    print(f"best-fit chi2 RSD:     {best_chi2_rsd:.6f}")
    print(f"best-fit chi2 total:   {best_chi2_total:.6f}")
    print(f"LCDM chi2 total:       {lcdm_chi2_total:.6f}")
    print(f"Delta chi2:            {dchi2:.6f}")
    print(f"AIC LCDM/model:        {aic_lcdm:.6f} / {aic_model:.6f}")
    print(f"BIC LCDM/model:        {bic_lcdm:.6f} / {bic_model:.6f}")
    print(f"Laplace ln(B10):       {ln_b10:.6f}")
    print(tau_text)
    if tau_warn:
        print("WARNING: convergence diagnostic is not comfortably satisfied.")
    print(f"Sampling wall time:    {sampling_elapsed_s/60.0:.2f} min")
    print(f"Full workflow time:    {full_elapsed_s/60.0:.2f} min")
    print("\nSaved:")
    for filename in [
        "mcmc_chain.npy",
        "mcmc_posterior.csv",
        "mcmc_summary.txt",
        "mcmc_corner.png",
        "mcmc_trace.png",
        "mcmc_1d_posterior.png",
        "mcmc_bao_residuals.png",
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
