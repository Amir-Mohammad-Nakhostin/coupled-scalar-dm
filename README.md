# Scalar–Tensor Dark-Matter Coupling Cosmology Suite

**A numerical research prototype for exploring scalar-field coupling to dark matter.**

This project implements a specified scalar–tensor cosmological model and compares its numerical predictions with supplied BAO and redshift-space distortion (RSD) data.

## Research overview

- Background cosmological evolution and linear structure growth.
- Numerical integration with NumPy and SciPy.
- Comparison of coupling-strength benchmarks using BAO and RSD diagnostics.
- Reproducible scripts, CSV outputs and plots.
- Explicit treatment of numerical assumptions and limitations.

**Status:** Exploratory computational research, not a validated cosmological inference pipeline or a claim of observational discovery. The diagonal-error approximation and other assumptions are documented below.

## Model and methodology

A deterministic Python 3.10+ research prototype for a four-dimensional scalar–tensor model in which a massive scalar couples conformally **only to dark matter**:

\[
S = \int d^4x\sqrt{-g}\left[\frac{M_P^2}{2}R-\frac12(\partial\phi)^2-\frac12m_\phi^2\phi^2\right]
+ S_{DM}[A^2(\phi)g_{\mu\nu},\chi],
\qquad A(\phi)=e^{\beta\phi/M_P}.
\]

The numerical solver uses \(M_P=1\) and \(H_0=1\). The scalar mass is converted through the supplied \(H_0\) in eV.

## Files

- `background_solver.py` — background scalar/DM evolution, Eisenstein–Hu drag-redshift diagnostic, and sound-horizon calculation.
- `bao_data.py` — supplied BAO compilation.
- `rsd_data.py` — supplied RSD `f*sigma8` compilation.
- `growth_solver.py` — linear growth with the requested quasi-static \(G_{eff}/G\).
- `bao_compare.py` — BAO chi-square, summary CSV, and two-panel figure.
- `rsd_compare.py` — RSD chi-square, summary CSV, and two-panel figure.

## Installation

Python 3.10+ is required. The tested environment for this package used NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.

```bash
python -m pip install --upgrade pip
python -m pip install "numpy>=2.0" "scipy>=1.10" "matplotlib>=3.7"
```

## Running

From this directory:

```bash
python background_solver.py
python bao_compare.py
python rsd_compare.py
```

`bao_compare.py` and `rsd_compare.py` continue to the next benchmark if one model fails.

## Outputs

The comparison scripts write:

- `results_bao_rd.csv`
- `bao_plot_rd.png` (150 dpi)
- `rsd_results.csv`
- `rsd_plot.png` (150 dpi)

The CSV files contain header rows and deterministic numerical output. No random number generator is used.

## Important sound-horizon convention

There is an internal inconsistency in the supplied specification that cannot be removed by numerical tuning alone.

For \(h=0.674\), \(\Omega_m h^2=0.314h^2\), and \(\Omega_b h^2=0.049h^2\), the requested Eisenstein–Hu 1998 formula gives approximately

\[
z_d \simeq 1020.44.
\]

Using that value in the supplied sound-speed and background integral does **not** reproduce `r_d = 147.09 Mpc`; it gives a substantially larger result. By contrast, using the Planck-style drag redshift `z_d = 1059.94` with neutrino-inclusive radiation and a high-redshift numerical extension gives `r_d ≈ 147.1 Mpc`.

Therefore the code makes the scientifically explicit choice to:

1. implement the Eisenstein–Hu fitting formula exactly as `z_drag(...)`;
2. use `z_d=1059.94` as the default in `compute_rd(...)` so the requested Planck sound-horizon self-test is meaningful;
3. issue a runtime warning whenever the chosen `z_d` differs materially from the Eisenstein–Hu fit.

No arbitrary multiplicative rescaling of `r_d` is applied.

## Numerical details

- Background: `scipy.integrate.solve_ivp(method="Radau")` with the requested API defaults (`rtol=1e-9`, `atol=1e-11`). The comparison sweeps explicitly use `rtol=1e-8`, `atol=1e-10` because the rapidly oscillating m_phi/H0~700 cases make the stricter tolerance disproportionately expensive; the reported observables were checked for convergence at this level.
- Growth: `Radau`, `rtol=1e-8`, `atol=1e-10`.
- Sound horizon: adaptive quadrature through `z=1e7`, plus the requested radiation-dominated analytical tail.
- NumPy 2.x: uses `np.float64`, `np.nan`, and no removed `np.float_`/`np.trapz` APIs.
- The dark-matter initial condition is `Omega_DM0 * a_i**(-3)` exactly as required.

## Physical / statistical limitations

This is a controlled research code base, not a Boltzmann-code replacement such as CLASS or CAMB.

1. The growth equation is used in the quasi-static approximation and the supplied scalar-mediated `G_eff/G` expression is taken as the model assumption.
2. The BAO and RSD chi-square calculations use independent diagonal uncertainties because no full covariance matrices were supplied. Published likelihood analyses generally require covariance information.
3. The BAO compilation mixes surveys and releases; it should not be treated as a formally independent combined likelihood without the appropriate cross-survey covariance treatment and nuisance-parameter model.
4. The background equations and parameter values are used exactly as specified, including their rounded density parameters. Consequently `H(0)` is not forcibly renormalized to exactly one.
5. The `beta=0` benchmark still carries the specified initial scalar field `phi_i=1e-3`; it is therefore the model's GR-coupling limit rather than a separately hard-coded six-parameter LCDM background.
6. The sound-horizon calculation extends the specified background equations backwards beyond the public solver's nominal `a_i=1e-4` start to cover the early universe. The supplied initial conditions are thereby propagated consistently rather than replaced by an independent early-time model.

## Validation run

The included scripts were executed successfully with NumPy 2.x. The benchmark outputs were:

| beta | r_d [Mpc] | BAO chi2 | RSD chi2 | f(0) | f*sigma8(0) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 147.1286 | 38.6900 | 9.1654 | 0.5264 | 0.4269 |
| 0.02 | 147.0728 | 36.3004 | 9.0847 | 0.5257 | 0.4264 |
| 0.05 | 146.9835 | 28.1919 | 8.7606 | 0.5222 | 0.4235 |
| 0.08 | 146.9230 | 48.7618 | 8.5712 | 0.5157 | 0.4182 |

These are results of the stated equations and diagonal-error compilations, not targets imposed by calibration. In particular, the supplied expected BAO chi-square shifts do not emerge from the supplied numerical model/data combination.

## References

- Planck Collaboration, *Planck 2018 results. VI. Cosmological parameters*, arXiv:1807.06209.
- Eisenstein, D. J. & Hu, W., *Baryonic Features in the Matter Transfer Function*, ApJ 496, 605 (1998), arXiv:astro-ph/9709112.
- DESI Collaboration, *DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and Cosmological Constraints*, arXiv:2503.14738.
