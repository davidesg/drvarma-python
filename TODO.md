# drvarma Python port — TODO

Status snapshot in `docs/STATUS.md`. P0, P1 and P2 are done and validated against
the C engine (37 tests). Tasks below are ordered; each notes the files to touch
and how to validate.

## P2 — remaining (presentation only)
- [x] **Report writers** (`report.py`): `.forecast` is **byte-exact** vs the C
      (incl. mon%/ann% rates+std); `.recursive` matches the validated engine path
      (<1e-4). `.out` reproduces header, parameters (estimates exact; SE/t/p carry
      the documented <1e-5 engine tolerance), Wald tests, OIRF/accumulated/gain,
      FEVD, multivariate diagnostics, normalized model, inverse roots. Validated
      in `tests/test_report.py` (8 tests). NOT reproduced (by design): the optimizer
      iteration/objective line (engine-internal — log-likelihood shown instead),
      inverse-roots ordering (modulus-sorted, not chekma order), and the per-series
      ASCII residual-plot tail (`diagnose()`).

## P3 — pure-Python general-m likelihood (reference / fallback)
- [x] **`elfvarma_py.py`** (q=0): exact Gaussian **VAR(p)** log-likelihood for
      general m via the companion-form Lyapunov stationary covariance of the first
      p obs + conditional density. Reproduces the C `logelf` to ~1e-7. Vectorised.
- [x] **`estimate_py.py`** (q=0): scipy L-BFGS-B over (mu, phi, chol(Sigma)) from
      OLS start; result dict matches the C `estimate_w` (params in C label order;
      std errors via numerical observed-information Hessian, best-effort). Reports
      `sigma2=1, sigma=Sigma` (the AS-311 sigma2/Q split is not reproduced here).
- [x] Wire `_engine.estimate_w` to **fall back** to `estimate_py.estimate_w_py`
      when `_drvarma_engine` is not importable (mirrors fue's `_engine.py`).
- [x] Validated (`tests/test_estimate_py.py`, 6 tests): elf_var vs C logelf <1e-6;
      pure-Python estimate vs C mu/phi/sigma/logelf <1e-3 on IPC3 (q=0); fallback
      dispatch (incl. via `Model.fit`); synthetic VAR(1) recovery.
- [ ] **MA case (q>0)**: port the multivariate AS 311 (`elfvarma.c`/`multshea.c`)
      — exact VARMA likelihood. `estimate_w_py` currently raises NotImplementedError
      for q>0 (build the C engine for VARMA). Add chol(Q)+theta to the packing.

## P4 — synthetic test suite & reliability
- [ ] Expand `datasets`: VARMA(1,1), full-Σ, near-unit-root, varying m; seeded
      fixtures with known ground truth.
- [ ] Parameter-recovery tests (CFFI and, once ready, pure-Python) with
      tolerance bands; edge cases (collinearity à la WTI/IPC, small n).
- [ ] Reliability tests mirroring fue's `tests/test_reliability*.py`.

## P5 — CLI, packaging, docs
- [x] **`cli.py`**: `drvarma <file> p q [options]` mirroring the C flags
      (`-mean -diagar -diagma -diagcov -m -twostep -deseason -forecast -estwin
      -scale`), reading lambda/d/D from the `.inp` header and writing
      `.out`/`.forecast`/`.recursive` via `report.py`. Entry point
      `drvarma = drvarma.cli:main` is wired in `pyproject.toml`.
      (`-volexp`/`-volmov` not ported — volatility is out of scope for the port.)
- [x] Packaging: `setup.py` builds the engine as an **optional** cffi Extension
      (`pip install` / `build_ext --inplace` compile it into `src/drvarma/`; a
      build failure without GSL degrades to a pure-Python install, tests skip).
      `cffi` added to build-system requires; `MANIFEST.in` ships `csrc/` in the
      sdist. (Entry point `drvarma = drvarma.cli:main` already wired.)
      Remaining: real pure-Python *compute* fallback needs P3; binary wheels/CI.
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
