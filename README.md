# drvarma

**Exact maximum-likelihood estimation, forecasting and diagnostics of
multivariate VARMA (vector ARMA) models** — in Python, with the compiled C
likelihood engine in the binary wheels.

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

> **What this provides.** drvarma is a pure-Python implementation of **Mauricio's
> exact-likelihood algorithm for multivariate VARMA** (1995 JASA / 1997 AS 311):
> an innovations-form factorisation that evaluates the exact Gaussian likelihood
> *directly on the VARMA form, without a state-space / Kalman filter*, maximised
> by a faithful factored-BFGS quasi-Newton. Exact ML for VARMA is *also* available
> in Python through state-space methods (e.g. statsmodels' Kalman-filter `VARMAX`);
> what drvarma adds is a faithful port of **this specific, non-Kalman algorithm** —
> which, as far as we know, is not otherwise available in Python — together with
> the surrounding forecasting/IRF/FEVD/diagnostics toolkit. It runs with no
> compiled code; the optional C engine is only an accelerator.

## Install

```sh
pip install drvarma
```

The binary wheels carry the compiled engine (CPython 3.10–3.13 on Linux
x86_64/aarch64, glibc and musl, macOS arm64 and Windows x86_64);
`drvarma._engine.engine_backend()` says `"c"`. Where no wheel fits, pip builds
the sdist, which needs a C compiler and the GSL headers; if the build or the
load fails, drvarma warns and runs the pure-Python path: the same results,
20–250× slower.

Optional extras: `drvarma[plots]` (matplotlib + pyfug charts),
`drvarma[forecast-report]` (HTML forecast reports). `drvarma[c-engine]` is kept
for old install lines and adds nothing: cffi is a dependency.

drvarma is the engine and has no assistant of its own. The MCP assistant for
VARMA systems is [sima](https://pypi.org/project/sima-tseries/)
(`pip install sima-tseries`), built on drvarma's ladder.

## Features

- **Exact-ML estimation** of stationary VARMA(p, q) for general dimension `m`,
  with mean, diagonal AR/MA/covariance restrictions, and optional
  Hannan–Rissanen two-step start.
- **Forecasting** in original units: level, period and annual variation, each
  with standard errors and 95 % bands; plus fixed-parameter **recursive**
  (out-of-sample) forecasting.
- **Structural analysis**: orthogonalised impulse responses, accumulated
  responses, long-run gain, and the forecast-error variance decomposition.
- **Diagnostics**: Hosking multivariate portmanteau, multivariate Jarque–Bera,
  per-series ACF/PACF and two-sided cross-correlation (CCF).
- **Volatility**: exponential-weight and moving-window residual covariance.
- **Transforms**: Box-Cox + regular/seasonal differencing and harmonic
  deseasonalisation (with re-seasonalised forecasts).
- **I/O & reports**: `.inp` reader/writer, C-format `.out`/`.forecast`/
  `.recursive` text reports, an HTML forecast report per series, and matplotlib
  charts.

Everything also runs with **no compiled code**. The CFFI engine of the binary
wheels wraps the validated drvarma C core and is bit-compatible with the
pure-Python path on well-conditioned problems — an accelerator only. The
numerical methods are tabulated [below](#numerical-methods); see
[`docs/DEVELOPER_GUIDE.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/DEVELOPER_GUIDE.md) for the complexity discussion
and a pure-Python vs hybrid vs C performance study.

## Command line

### The ladder (0.2): fue's models as input

```sh
drvarma ES.pre FR.pre DE.pre 0 0 -diagcov -forecast 24 -estwin 216 -o diag  # the univariates
drvarma ES.pre FR.pre DE.pre 1 0          -forecast 24 -estwin 216 -o var1  # + cross dynamics
drvarma -split IPC3.inp -mean -ar 1 -dir specs/     # the old multivariate .inp -> fue .inp
```

Each series keeps its fue model (Box-Cox, deterministic terms, differencing,
mean, ARMA factors) on the diagonal; `p q` are the cross orders. A diagonal
gate checks that the joint model reproduces the univariate ones and stops the
program if it does not. In Python: `drvarma.ladder.Ladder(files, p, q).fit()`.
See [`docs/LADDER.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/LADDER.md).

### The multivariate `.inp` (deprecated since 0.2 / C 5.0)

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

- [`docs/LADDER.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/LADDER.md) — the ladder: fue's models as input, the gate,
  forecasting, one format, and the parity with the C.
- [`docs/USER_GUIDE.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/USER_GUIDE.md) — install, API, CLI, worked examples.
- [`docs/INP_FORMAT.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/INP_FORMAT.md) — the `.inp` input format (a precise,
  assistant-friendly spec for preparing inputs).
- [`docs/DEVELOPER_GUIDE.md`](https://github.com/davidesg/drvarma-python/blob/master/docs/DEVELOPER_GUIDE.md) — internals, algorithmic
  complexity from the literature, and the performance study.

## Numerical methods

| Algorithm | Reference | Used for |
|-----------|-----------|----------|
| Exact Gaussian VARMA log-likelihood (innovations factorisation, not Kalman) | Mauricio (1995) JASA; Mauricio (1997) AS 311 | Exact likelihood (`drvarma._as311`) |
| Factored-BFGS quasi-Newton + Dennis–Schnabel line search | Dennis & Schnabel (1983) | ML optimisation (`drvarma._qnewt.raxopt`) |
| Concentrated objective `f1ᵐ·f2` (σ² profiled, Σ = σ²·Q) | Mauricio (1995) JASA §3 | Conditioning; covariance / std errors |
| Hannan–Rissanen two-step | Hannan & Rissanen (1982) | VARMA start (`-twostep`) |
| Companion-form Lyapunov / autocovariances | Mauricio (1997) AS 311 | Stationary covariance, ψ-weights |
| Orthogonalised IRF & FEVD (Cholesky of Σ) | Lütkepohl (2005) | Structural analysis |
| Hosking multivariate portmanteau; multivariate Jarque–Bera | Hosking (1980); Jarque & Bera (1980) | Residual diagnostics |
| Exponential / moving-window conditional covariance | — | Volatility (`drvarma.volatility`) |

Cross-references: Mauricio (2002, *JTSA* 23(4)) and Shea (1989, AS 242) are the
other efficient exact-likelihood methods compared in the literature.

## Authors and licence

**drvarma** is developed by **David E. Guerrero** and Arthur B. Treadway, based on
the exact-likelihood algorithms designed and coded by **José Alberto Mauricio**.

Released under the **GNU General Public License v2.0 or later**
(GPL-2.0-or-later) — see [`COPYING`](https://github.com/davidesg/drvarma-python/blob/master/COPYING).
