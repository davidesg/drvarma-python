# drvarma Python port — TODO

Status snapshot in `docs/STATUS.md`. P0, P1 and P2 (analytics) are done and
validated against the C engine (29 tests). Tasks below are ordered; each notes
the files to touch and how to validate.

## P2 — remaining (presentation only)
- [ ] **Report writers** (`report.py`): produce the `.out` and `.forecast` text
      files matching the C format (parameters with SE/t/p, Wald tests, IRF/FEVD
      tables, residual diagnostics; forecast Level/Low95/High95 + mon%/ann%).
      Pieces already available: params/cov/sigma from `estimate_w`, forecasts +
      bands from `Model.forecast(bands=True)`, diagnostics/irf/fevd from `Model`.
      Validate by diffing against `../drvarma_v.04.1/*.out` / `*.forecast`.
      (Could fold into P5/CLI.)

## P3 — pure-Python general-m likelihood (reference / fallback)
- [ ] **`elfvarma_py.py`**: exact Gaussian VARMA(p,q) log-likelihood for general
      m in numpy (generalise fue's m=1 `elf_scalar`/`flikam`; the C
      `elfvarma.c`/`multshea.c` are the reference implementation). Start q=0
      (conditional/exact VAR) then add MA.
- [ ] **`estimate_py.py`**: scipy `minimize` (L-BFGS-B) driver over the param
      packing used by `shootx` (mu, phi, theta, chol(Q)); reuse the same packing
      order so results line up with the C.
- [ ] Wire `_engine.estimate` to **fall back** to the pure-Python estimator when
      `_drvarma_engine` is not importable (mirror fue's `_engine.py`).
- [ ] Validate: pure-Python vs C params/loglik to ~1e-6 on IPC3 and pass-through;
      synthetic recovery (VAR and VARMA) via `datasets.simulate_varma`.

## P4 — synthetic test suite & reliability
- [ ] Expand `datasets`: VARMA(1,1), full-Σ, near-unit-root, varying m; seeded
      fixtures with known ground truth.
- [ ] Parameter-recovery tests (CFFI and, once ready, pure-Python) with
      tolerance bands; edge cases (collinearity à la WTI/IPC, small n).
- [ ] Reliability tests mirroring fue's `tests/test_reliability*.py`.

## P5 — CLI, packaging, docs
- [ ] **`cli.py`**: `drvarma file.inp p q [options]` mirroring the C flags
      (`-mean -diagar -diagma -diagcov -deseason -forecast -estwin -scale`),
      writing `.out`/`.forecast`/`.recursive` via `report.py`.
- [ ] Packaging: `cffi_modules` in `pyproject.toml` so `pip install` builds the
      engine; pure-Python wheel fallback; entry point `drvarma = drvarma.cli:main`.
- [ ] Optional `plots.py` (matplotlib): series, forecasts+bands, IRF.
- [ ] Docs: USER_GUIDE / API reference for the Python package; PyPI release.
- [ ] CI workflow building the engine + running pytest (mirror the C repo CI).

## Engine / maintenance
- [ ] Keep `csrc/internal/` in sync with `../drvarma_v.04.1/src` when the C
      engine changes (they are copies).
- [ ] `recursive_forecast` for q>0 (needs full-data residuals at fixed params;
      add a C API entry to filter residuals, or compute in numpy).
- [ ] Consider exposing the C forecast/diagnostics too (currently re-implemented
      in numpy); decide single-source-of-truth per function.

## Decisions / open questions
- [ ] Publish the Python port to its own GitHub repo? (own remote, CI, PyPI.)
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the
      C (via more API). Numpy keeps the port usable without the C engine (P3 goal).
