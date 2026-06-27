# drvarma

**Exact maximum-likelihood estimation, forecasting and diagnostics of
multivariate VARMA (vector ARMA) models** — in pure Python, with an optional
compiled C engine for speed.

```python
import numpy as np
from drvarma import Model, datasets

series = datasets.simulate_varma(phi=[np.diag([0.5, 0.4, 0.3])], sigma=np.eye(3),
                                 n=300, mu=[100., 50., 75.], seed=1,
                                 names=["A", "B", "C"])
m = Model(series, p=2, q=0, include_mean=True).fit()
print(m.phi, m.sigma, m.loglik)
levels, lo, hi = m.forecast(12, bands=True)     # + 95% bands
print(m.diagnostics())                          # Hosking Q, Jarque-Bera
```

drvarma fits a stationary Gaussian VARMA(p, q),
`Φ(B)(wₜ − μ) = Θ(B) aₜ`, `aₜ ~ N(0, Σ)`, by **exact** maximum likelihood (no
conditional/back-forecasting approximation), and gives forecasts (+ error bands),
impulse responses, variance decompositions, residual diagnostics and volatility.

## Install

```sh
pip install drvarma                       # pure-Python (numpy + scipy)
```

Optional extras: `drvarma[plots]` (matplotlib + pyfug charts),
`drvarma[forecast-report]` (HTML forecast reports), `drvarma[c-engine]` (build the
CFFI C engine — needs GSL dev headers, ~10–100× faster but optional).

## Algorithms

drvarma implements the published exact-ML machinery of Mauricio and the standard
multivariate time-series toolkit:

- **Exact Gaussian VARMA likelihood.** The exact log-likelihood is evaluated by
  **Mauricio's algorithm** (Mauricio 1995, *JASA*; published in code form as
  *Algorithm AS 311*, Mauricio 1997) — an innovations-style factorisation that
  works directly on the VARMA form, **not** a Kalman/state-space filter. Cost is
  `O(n)` in the sample size; one of the two most efficient exact methods in the
  literature (with Shea's AS 242). The faithful Python port is `drvarma._as311`.
- **Maximum-likelihood optimisation.** A **factored-BFGS quasi-Newton** method
  with a Dennis–Schnabel line search (Dennis & Schnabel 1983), maximising the
  *concentrated* likelihood (the residual scale σ² profiled out, `Σ = σ²·Q`). The
  parameter covariance / standard errors come from the optimiser's factored
  Hessian. Ported in `drvarma._qnewt`.
- **Initialisation.** OLS VAR(p) seed; optional **Hannan–Rissanen two-step** start
  for VARMA (`-twostep`), which fits a long AR, recovers residuals, then regresses
  on AR and MA lags.
- **Forecasting.** Minimum-MSE forecasts of the modelled (Box-Cox + differenced)
  series with exact forecast-error variances, integrated back to original units
  (level, period and annual variation, each with std and 95 % bands). Includes
  **fixed-parameter recursive forecasting** from multiple origins (`-estwin`) for
  out-of-sample evaluation.
- **Structural analysis.** **Orthogonalised impulse responses** (shocks
  orthogonalised by the Cholesky factor of Σ), accumulated responses, long-run
  gain, and the **forecast-error variance decomposition (FEVD)**.
- **Diagnostics.** **Hosking's multivariate portmanteau** test, the
  **multivariate Jarque–Bera** normality test, and per-series ACF/PACF and
  two-sided cross-correlation (CCF) functions with Ljung–Box / bivariate Q.
- **Volatility.** Conditional covariance of the residuals by **exponential
  weighting** (rational inattention) and by a **moving window**.
- **Transforms.** Box-Cox power + regular/seasonal differencing, and optional
  **harmonic (deseasonalisation)** seasonal adjustment with re-seasonalised
  forecasts.

All of the above run with **no compiled code**. The optional CFFI engine wraps the
validated, Numerical-Recipes-free drvarma C core and is bit-compatible with the
pure-Python path on well-conditioned problems; it is an accelerator only. See
[`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for the complexity discussion
and a pure-Python vs hybrid vs C performance study.

## Command line

```sh
drvarma IPC3 3 0 -mean -deseason auto -forecast 24        # writes IPC3.out, .forecast
drvarma IPC3 3 0 -mean -forecast 24 -html                # + HTML report per series
drvarma IPC3 3 0 -mean -estwin 200 -forecast 12          # recursive (.recursive)
drvarma IPC3 3 0 -mean -volexp 0.05 20 -volmov 20        # volatility (.volexp/.volmov)
```

`<file>.inp` in, text reports out. Flags: `-mean -diagar -diagma -diagcov
-m {1,2} -twostep -deseason [auto|force] -scale S -forecast H -html -estwin N
-volexp [α w] -volmov [w]` (λ, d, D come from the `.inp` header).

## Documentation

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — install, API, CLI, worked examples.
- [`docs/INP_FORMAT.md`](docs/INP_FORMAT.md) — the `.inp` input format (a precise,
  assistant-friendly spec for preparing inputs).
- [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) — internals, algorithmic
  complexity from the literature, and the performance study.

## References

- Mauricio, J. A. (1995). *Exact maximum likelihood estimation of stationary
  vector ARMA models.* **JASA** 90(429), 282–291.
- Mauricio, J. A. (1997). *Algorithm AS 311: the exact likelihood function of a
  vector ARMA process.* **Applied Statistics** 46(1), 157–171.
- Mauricio, J. A. (2002). *An algorithm for the exact likelihood of a stationary
  vector ARMA model.* **J. Time Series Analysis** 23(4), 473–486.
- Shea, B. L. (1989). *Algorithm AS 242: the exact likelihood of a vector ARMA
  model.* **Applied Statistics** 38(1), 161–184.
- Dennis, J. E. & Schnabel, R. B. (1983). *Numerical Methods for Unconstrained
  Optimization and Nonlinear Equations.*
- Hosking, J. R. M. (1980); Jarque, C. M. & Bera, A. K. (1980).

## License

GNU General Public License v2 or later — © A. B. Treadway, J. A. Mauricio,
D. E. Guerrero. See [`COPYING`](COPYING).
