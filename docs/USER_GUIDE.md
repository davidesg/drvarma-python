# drvarma — User Guide

drvarma estimates, diagnoses and forecasts **multivariate VARMA(p, q)** models by
exact maximum likelihood. This guide covers the Python API and the command-line
tool. For internals, performance and the engine-vs-pure-Python trade-offs, see
`docs/DEVELOPER_GUIDE.md`.

## Contents
1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Getting data in](#3-getting-data-in)
4. [Specifying a model](#4-specifying-a-model)
5. [Fitting and reading results](#5-fitting-and-reading-results)
6. [Forecasting](#6-forecasting)
7. [Diagnostics, IRF, FEVD](#7-diagnostics-irf-fevd)
8. [Volatility](#8-volatility)
9. [Reports and the CLI](#9-reports-and-the-cli)
10. [Plots](#10-plots)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Install

```sh
pip install drvarma                 # pure-Python (NumPy/SciPy only)
```

That gives full functionality with no compiled code. For the **optional C engine**
(10–100× faster — see the developer guide) install the GSL headers first:

```sh
sudo apt-get install libgsl-dev     # Debian/Ubuntu
pip install "drvarma[c-engine]"
```

If the engine can't be built the install still succeeds and drvarma runs in
pure-Python mode automatically. Optional plotting: `pip install "drvarma[plots]"`.

---

## 2. Quick start

```python
import numpy as np
import drvarma
from drvarma import Model, datasets

# a 3-variable series (here simulated; normally you'd load yours — see §3)
series = datasets.simulate_varma(
    phi=[np.diag([0.5, 0.4, 0.3])], sigma=np.eye(3),
    n=300, mu=[100., 50., 75.], seed=1, names=["A", "B", "C"])

# specify and fit a VAR(2) with a mean, on the levels (no transform)
model = Model(series, lam=1.0, d=0, D=0, scale=1.0,
              p=2, q=0, include_mean=True).fit()

print(model.ifault)        # 0 = converged OK
print(model.loglik)        # exact log-likelihood
print(model.phi)           # AR matrices, shape (p, m, m)
print(model.sigma)         # innovation covariance, (m, m)

future = model.forecast(12)                 # (12, 3) point forecasts
levels, low, high = model.forecast(12, bands=True)   # + 95% bands
print(model.diagnostics()) # Hosking Q + Jarque-Bera
```

---

## 3. Getting data in

### From a `.inp` file

drvarma reads the same `.inp` format as the C engine:

```python
from drvarma import load, save
series, spec = load("IPC3.inp")     # series: MultiSeries; spec: InpSpec(lam, d, D)
```

The header carries the transform orders, so you usually pass them straight to the
model: `Model(series, lam=spec.lam, d=spec.d, D=spec.D, ...)`. The CLI (`§9`) does
this for you. The full `.inp` token spec (for hand- or assistant-authoring an
input) is in [`docs/INP_FORMAT.md`](INP_FORMAT.md). A `.inp` file looks like:

```
* Trivariate consumer price indices
** Frequency (1=A, 4=Q, 12=M):
 12
** Series, observations, start (subperiod year):
 3 216 1 2002
** Series names:
 IPC_ES IPC_FR IPC_DE
** Box-Cox lambda, regular differences, annual differences:
 0.0 1 0
** Data:
 69.5300 81.9400 77.7000
 ...
```

### From NumPy arrays

```python
from drvarma import MultiSeries
data = np.column_stack([y1, y2, y3])     # shape (nobs, m)
series = MultiSeries(data, freq=12, start=(2002, 1), names=["ES", "FR", "DE"])
save("mydata.inp", series, drvarma.InpSpec(lam=0.0, d=1, D=0))   # optional
```

`MultiSeries` fields: `.data` (nobs×m), `.m`, `.nobs`, `.freq`, `.start=(year, subperiod)`,
`.names`.

---

## 4. Specifying a model

```python
Model(series,
      lam=0.0, d=1, D=0,        # Box-Cox λ; regular & seasonal differencing
      scale=100.0,              # rescale after Box-Cox (engine default 100)
      p=3, q=0,                 # AR, MA orders
      include_mean=False,       # estimate a mean/drift μ
      diag_ar=False,            # restrict Φ_k to diagonal
      diag_ma=False,            # restrict Θ_k to diagonal
      diag_cov=False,           # restrict Σ to diagonal
      method=1,                 # 1 = exact ML (default), 2 = approximate
      twostep=False,            # Hannan–Rissanen start (q>0 only)
      deseason=None)            # None | "auto" | "force"
```

**Transform.** drvarma models the *stationary* series obtained by a Box-Cox power
(`lam`: 0 = log, 1 = identity), then `d` regular differences and `D` seasonal
differences of period `freq`, then a `scale` factor. Forecasts are integrated and
inverse-transformed back to the original units automatically.

**Deseason.** `deseason="auto"` removes a harmonic seasonal component from series
that test as significantly seasonal (`"force"` = all); forecasts are
re-seasonalised. Estimated on the `d=1` differenced basis (see STATUS gotchas).

**Two-step.** For VARMA (`q>0`), `twostep=True` seeds the optimiser from a
per-series Hannan–Rissanen estimate — helps weakly-identified models converge.

---

## 5. Fitting and reading results

`fit()` deseasonalises (if asked), transforms, and estimates by exact ML. After
fitting, read results through accessors:

| Accessor | Meaning |
|----------|---------|
| `model.ifault` | 0 = OK; >0 = problem (see §11) |
| `model.loglik` | exact log-likelihood |
| `model.mu` | mean vector μ, `(m,)` |
| `model.phi` | AR matrices, `(p, m, m)` |
| `model.theta` | MA matrices, `(q, m, m)` |
| `model.sigma` | innovation covariance Σ, `(m, m)` |
| `model.sigma2` | σ̂² scalar (Σ = σ²·Q split, as the C engine) |
| `model.residuals` | residuals, `(nobs, m)` |
| `model.params` / `model.std_errors` | packed estimates and their std errors |

```python
model = Model(series, lam=spec.lam, d=spec.d, D=spec.D,
              p=3, q=0, include_mean=True, deseason="auto").fit()
if model.ifault == 0:
    print(model.phi[0])              # Φ_1
    print(model.std_errors)
```

---

## 6. Forecasting

```python
levels = model.forecast(12)                       # (12, m) in original units
levels, low95, high95 = model.forecast(12, bands=True)
past = model.forecast(12, b=24)                    # forecast from 24 obs back
```

**Recursive / out-of-sample evaluation.** Estimate once on the first `estwin` raw
observations, then forecast `H` steps from every later origin with those *fixed*
parameters (general VARMA(p, q) supported):

```python
rows = model.recursive_forecast(estwin=200, H=12)
# rows: list of (origin_raw_index, series_index, horizon, level)
```

---

## 7. Diagnostics, IRF, FEVD

```python
d = model.diagnostics()          # default lag = freq+2
# {'hosking_Q', 'hosking_df', 'hosking_p', 'hosking_lag', 'JB', 'JB_df', 'JB_p'}

oirf = model.irf(20)             # orthogonalised IRF, (H+1, m, m)
psi  = model.irf(20, orthogonalized=False)   # raw ψ-weights
fevd = model.fevd(20)            # variance decomposition (%), (H, m, m)
```

Per-series and cross-correlation diagnostics live in `drvarma.diagnostics`
(`series_stats`, `acf`, `pacf`, `ljung_box`, `ccf`).

---

## 8. Volatility

Conditional volatility of the residuals (`drvarma.volatility`):

```python
from drvarma import volatility
phi_, thr = volatility.write_volexp("out.volexp", model.residuals, model.sigma,
                                    alpha=0.05, window=20)   # exponential weighting
volatility.write_volmov("out.volmov", model.residuals, window=20)  # moving window
```

---

## 9. Reports and the CLI

### Text reports

```python
from drvarma import report
report.write_out(model, "model.out")            # full .out report
report.write_forecast(model, 12, "model.forecast")
report.write_recursive(model, 200, 12, "model.recursive")
text = report.out_report(model)                 # as a string
```

### HTML forecast report (SPS / fuf)

A self-contained HTML forecast report **per series** (table + two-panel chart),
homologable with the fue *fuf* reports. Needs jinja2
(`pip install "drvarma[forecast-report]"`):

```python
from drvarma import report_forecast
paths = report_forecast.write_forecast_report(model, "model", L=24)
# -> ["model_IPC_ES.html", "model_IPC_FR.html", "model_IPC_DE.html"]
```

### Command line

The `drvarma` console script mirrors the C binary. Input is read from
`<file>.inp`; `<file>.out` is always written.

```sh
drvarma IPC3 3 0 -mean -deseason auto -forecast 12
drvarma IPC3 3 0 -mean -forecast 24 -html            # + HTML report per series
drvarma IPC3 3 0 -mean -estwin 200 -forecast 12      # writes .recursive
drvarma IPC3 3 0 -mean -volexp 0.05 20 -volmov 20    # writes .volexp/.volmov
```

Flags: `-mean -diagar -diagma -diagcov -m {1,2} -twostep -deseason [auto|force]
-scale S -forecast H -html -estwin N -volexp [alpha window] -volmov [window]`.
(λ, d, D come from the `.inp` header.)

---

## 10. Plots

Requires `matplotlib` (`pip install "drvarma[plots]"`); the Jenkins–Treadway
styled diagnostics also use `pyfug`.

```python
from drvarma import plots
plots.plot_forecast(model, 12)          # history + forecast + 95% bands
plots.plot_irf(model, 20)               # m×m OIRF grid
plots.plot_fevd(model, 20)              # stacked FEVD
plots.plot_series(series)
# JT-styled residual diagnostics (need pyfug):
plots.plot_residual_diagnostics(model, j=0)
```

---

## 11. Troubleshooting

**`ifault` codes** (from the estimator):

| code | meaning |
|------|---------|
| 0 | converged OK |
| 1 | Q (covariance) not positive definite at the start |
| 2 | AR has a unit root |
| 3 | AR strictly non-stationary |
| 4 | MA strictly non-invertible |
| 5 | numerical problem |
| 6 | optimiser failed to converge |

**Non-convergence / weak identification.** For VARMA (`q>0`) try `twostep=True`,
or reduce the orders. Near-cancelling Φ/Θ are weakly identified.

**Ill-conditioning.** If series have wildly different variances (e.g. one ~100×
another), the parameter covariance — and therefore the **standard errors** — are
ill-determined; the *point estimates* stay reliable. **Rescale series to
comparable variances.** See `docs/DEVELOPER_GUIDE.md` §4.3 for the quantitative
analysis.

**Speed.** Pure-Python fits of VARMA(q>0) or ill-conditioned models can take
seconds to tens of seconds. Install the C engine (`[c-engine]`, §1) for a
10–100× speed-up; results are identical for well-conditioned problems.
