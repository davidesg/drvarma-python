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
| P2 | forecasting (+ bands, recursive `-estwin`), deseason, diagnostics, IRF/FEVD | **done** (report writers in P5) |
| P3 | pure-Python general-m likelihood (reference/fallback) | pending |
| P4 | synthetic test suite | started (simulator) |
| P5 | CLI, packaging, report writers, docs | pending |

All P1/P2 numerics are validated against the C engine on the IPC3 reference
case (parameters, forecasts, bands, diagnostics, IRF/FEVD and recursive
forecasts match to rounding precision).

**Resuming work?** Read [`docs/STATUS.md`](docs/STATUS.md) (state, architecture,
build/test/validate, gotchas) and [`TODO.md`](TODO.md) (next tasks).

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

## License

GNU General Public License v2 or later (see [`COPYING`](COPYING)).
