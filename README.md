# drvarma (Python)

Python port of **drvarma** — maximum-likelihood estimation, diagnostics and
forecasting of multivariate **VARMA** models. Work in progress.

This package mirrors the architecture of the fue Python port: pure-Python layers
(I/O, model API, forecasting, diagnostics) with an optional CFFI-compiled C
engine for speed (the validated, Numerical-Recipes-free drvarma C core). See
[`docs/MIGRATION_PLAN.md`](docs/MIGRATION_PLAN.md).

The C engine lives in the sibling directory `../drvarma_v.04.1` (published at
<https://github.com/davidesg/drvarma>); this folder is the Python port.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| P0 | package skeleton, `.inp` I/O, Box-Cox/differencing transform, synthetic VARMA simulator | **done** |
| P1 | estimation via CFFI over a `drvarma_api.c` | **done** |
| P2 | forecasting (+ bands, recursive `-estwin`), deseason, diagnostics, IRF/FEVD, report writers | **done** |
| P3 | pure-Python exact-ML likelihood — faithful port of Mauricio's AS 311 | **done** (VAR + VARMA); Shea/multshea.c backup pending |
| P4 | synthetic suite & reliability (`datasets.varma_cases`, recovery + agreement + formula tests) | **done** |
| P5 | CLI, packaging, plots, **CI** | **done** |
| PP1–PP5 | **100% pure-Python parity** — estimator (σ²/Q split + std errors via the ported BFGS), `-twostep`, volatility, recursive q>0, engine-free fidelity | **done** |

All numerics are validated against the C engine/binary (parameters, forecasts,
bands, diagnostics, IRF/FEVD, recursive forecasts). **The pure-Python path is now
feature- and fidelity-complete; the CFFI engine is an optional accelerator.**

## Documentation

- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — install, API, CLI, examples.
- [`docs/INP_FORMAT.md`](docs/INP_FORMAT.md) — the `.inp` input format (precise,
  assistant-friendly spec for preparing inputs).
- [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) — internals, complexity from
  the literature, and the pure-Python / hybrid / C performance study.
- **Resuming dev work?** [`docs/STATUS.md`](docs/STATUS.md) (state, architecture,
  gotchas) and [`TODO.md`](TODO.md) (next tasks); 100%-Python plan in
  [`docs/PURE_PYTHON_PLAN.md`](docs/PURE_PYTHON_PLAN.md).

## Install (development)

```sh
python -m pip install -e .
pytest            # or: PYTHONPATH=src python -m pytest tests/
```

Dependencies: `numpy`, `scipy` (runtime); `cffi` only for the optional C engine.

## Quick taste

```python
import drvarma
series, spec = drvarma.load("../drvarma_v.04.1/data/models_group1/IPC3.inp")
w, bc = drvarma.transform.transform(series.data, lam=spec.lam, d=spec.d,
                                    D=spec.D, s=series.freq)

from drvarma.datasets import simulate_varma
import numpy as np
sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.2, 0.4]])], n=300, seed=1)
```

## Command line

Mirrors the C binary (`<file>.inp` in, `.out`/`.forecast`/`.recursive` out):

```sh
drvarma data/models_group1/IPC3 3 0 -mean -deseason auto -forecast 24
# or, without installing: PYTHONPATH=src python -m drvarma.cli <file> p q [flags]
```

Flags: `-mean -diagar -diagma -diagcov -m METHOD -twostep -deseason [auto|force]
-scale FACTOR -forecast H -estwin N`. The `.forecast` output is byte-identical to
the C engine; see [`docs/STATUS.md`](docs/STATUS.md) for `.out` fidelity notes.

## Plots (optional)

With matplotlib (`pip install "drvarma[plots]"`), `drvarma.plots` offers
`plot_series`, `plot_forecast` (history + forecast + 95% bands), `plot_irf`
(m×m OIRF grid), `plot_fevd` and `plot_ccf` (two-sided cross-correlation). Each
returns a matplotlib `Figure`.

The `[plots]` extra also pulls in **pyfug**, whose Jenkins-Treadway diagnostic
plots are reused per series/residual: `plot_series_jt`, `plot_residual_acf_pacf`,
`plot_residual_histogram`, `plot_residual_diagnostics` (series + ACF/PACF) and
`plot_mean_deviation`; `apply_jt_theme()` styles the rest with the JT theme.

## License

GNU General Public License v2 or later (see [`COPYING`](COPYING)).
