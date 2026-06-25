# drvarma Python port — TODO

Status snapshot in `docs/STATUS.md`. P0–P4 and most of P5 (CLI, packaging,
per-series diagnostics, base plots) are done and validated against the C engine
(94 tests). The whole Model → forecast → report pipeline also runs on the
pure-Python fallback with no compiled engine. Remaining: the graphics finish
(pyfug JT formats, deferred to last), P5 docs/CI, and the deferred Shea backup.

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
- [x] **`_as311.py`** — faithful Python port of **Mauricio's AS 311**
      (`csrc/internal/elfvarma.c`: `elf`, `cgamma`, `cxi`, `cres`, `chekma`), the
      exact VARMA(p,q) log-likelihood for general m. 1-indexed transcription;
      hot length-n loops vectorised without changing the algorithm. Reproduces the
      C `logelf` to ~1e-11 and the exact residuals to ~1e-12. **No Kalman**
      (that route is Shea's; a faithful `multshea.c` port is the desirable backup).
- [x] **`elfvarma_py.py`**: `elf_varma` (AS 311 wrapper, general p,q) + `elf_var`
      (fast vectorised q=0 specialisation via the companion Lyapunov covariance,
      cross-checked against AS 311).
- [x] **`estimate_py.py`**: scipy L-BFGS-B over (mu, phi, theta, chol(Sigma)) from
      an OLS/θ=0 start; result dict matches the C `estimate_w` (params in C label
      order; std errors via numerical observed-information Hessian, best-effort).
      Reports `sigma2=1, sigma=Sigma` (AS-311 sigma2/Q split not reproduced).
- [x] Wire `_engine.estimate_w` to **fall back** to `estimate_py.estimate_w_py`
      when `_drvarma_engine` is not importable (mirrors fue's `_engine.py`).
- [x] Validated (`tests/test_estimate_py.py`, 11 tests): elf_var/elf_varma vs C
      `logelf` <1e-6 and exact residuals <1e-6 (q=0 and VARMA(1,1)/(2,1)); pure-
      Python estimate vs C mu/phi/theta/sigma/logelf <1e-3; fallback dispatch
      (incl. via `Model.fit`); synthetic VAR(1) recovery and VARMA MLE property;
      full Model→forecast→report pipeline with the engine monkeypatched out.

## P4 — synthetic test suite & reliability
- [x] Expand `datasets`: `varma_cases()` registry (VAR(1)/VAR(2), VARMA(1,1),
      full-Σ, near-unit-root, diagonal; m=2,3) with known ground truth, all
      verified stationary/invertible; `is_stationary`/`is_invertible` helpers.
- [x] Parameter-recovery tests (`tests/test_reliability.py`): C-engine recovery at
      n=4000 within bands (phi<0.07, sigma<0.08; mu excluded — near-unit-root mean
      is noisy); C-vs-pure-Python agreement <3e-3; small-n (n=40) convergence.
- [x] Reliability tests (mirroring fue): Sigma symmetric/PD, std=sqrt(diag cov),
      npar vs diag restrictions, Hosking-Q / Jarque-Bera against their formulas,
      simulation + estimator determinism, near-unit-root convergence. (19 tests.)
- [x] Documented **pass-through** cases (WTI→IPC, `data/passthrough/WTI_IPC_*`):
      ill-conditioned by the ~hundreds-fold WTI/IPC variance disparity. Tests
      (12, in `test_reliability.py`): variance disparity >100×; point estimates
      scale-invariant (C engine, <1e-4); parameter cov ill-conditioned
      (cond>1e4); C-vs-pure-Python point estimates/logelf robust (<2e-3/<1e-4)
      despite the ill-conditioning. Matches the C `MODELS_RESULTS.md` §4 caveat.

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
- [x] **Per-series diagnostics** migrated from `diagnose.c` into `diagnostics.py`
      (drvarma owns these): `series_stats` (mean/var/std/SE/skew/kurt/min/max —
      **exact** vs `IPC3.out`), `acf`, `pacf` (Durbin-Levinson), `ljung_box`
      (ChiTest), `residual_diagnostics`; plus `ccf`/`qccf` (Hosking bivariate).
- [x] `plots.py` (matplotlib, lazy import): `plot_series`, `plot_forecast`
      (history + forecast + 95% bands), `plot_irf` (m×m OIRF grid), `plot_fevd`
      (stacked), `plot_ccf` (two-sided CCF in the drv4.040804/drvus format).
      Smoke-tested with the Agg backend (`tests/test_plots.py`; skip if matplotlib
      absent).
- [x] **`MultiSeries → pyfug.core.Tseries` adapter** (`_pyfug.py`): builds a
      univariate Tseries per series/residual column with the statistics filled
      from drvarma's own `diagnostics.series_stats` (drvarma owns the numbers;
      pyfug only renders). `residual_start` dates residuals from `d+D·s`. pyfug
      added to `[plots]` extras; tests skip if pyfug absent (5 tests).
- [ ] **Residual `.out` section (ASCII)** — compose per residual series: write
      drvarma's `File_StatSer` stats block (own wording, no JB line) + reuse
      `pyfug.ascii` `_write_ascii_plot`/`_write_ascii_histogram`/
      `_write_acf_ascii_bars` via the adapter. See `docs/FUE_REUSE.md`.
- [ ] **Graphics finish (deferred to last)**: reuse `pyfug.graphics` JT formats
      (series, ACF/PACF, histogram, mean-deviation) through the same adapter;
      restyle forecast/IRF/FEVD with the JT theme; add pyfug to `[plots]` extras.
      drvarma keeps stat ownership (`diagnostics.py`); pyfug renders. See
      `docs/FUE_REUSE.md`.
- [ ] Docs: USER_GUIDE / API reference for the Python package; PyPI release.
- [ ] CI workflow building the engine + running pytest (mirror the C repo CI).

## Engine / maintenance
- [ ] Keep `csrc/internal/` in sync with `../drvarma_v.04.1/src` when the C
      engine changes (they are copies).
- [ ] `recursive_forecast` for q>0 (needs full-data residuals at fixed params;
      add a C API entry to filter residuals, or compute in numpy).
- [ ] Consider exposing the C forecast/diagnostics too (currently re-implemented
      in numpy); decide single-source-of-truth per function.

## P6 — Shea (AS 242) backup likelihood  [DEFERRED — do last]

`csrc/internal/multshea.c` (Shea) is present and compiled but **not wired into the
drvarma C estimator**: its entry point `marma(...)` has no callers — only
Mauricio's `elf(...)` (`elfvarma.c`) is used. So Shea has no C reference results to
validate a Python port against yet. Sequence (C first):

- [ ] Wire `marma()` into the C engine (a *new* engine version) as a selectable
      likelihood option (e.g. a method flag), **if feasible**. It already mirrors
      the `elf(...)` interface (`marma(k,n,p,q,mu,phi,theta,...)`), so it should be
      a near drop-in alternative inside `shootx`/`drvmlest`.
- [ ] Compare **C-Shea vs C-Mauricio** on the reference cases (logelf, residuals,
      estimated params) — they should agree to numerical precision.
- [ ] Decide whether to keep Shea as a permanent C option.
- [ ] Only then port Shea to Python (faithful, like `_as311.py`) as an alternative
      to `_as311`, validated against the C-Shea results.

Note: a Kalman/state-space likelihood is essentially Shea's route — do not add one
as a stand-in; port `multshea.c` faithfully instead.

## Decisions / open questions
- [ ] Publish the Python port to its own GitHub repo? (own remote, CI, PyPI.)
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the
      C (via more API). Numpy keeps the port usable without the C engine (P3 goal).
