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
- [x] **Residual `.out` section (ASCII)** — `report.residual_report`: drvarma's
      own File_StatSer stats block + standardized time-series plot reused from
      `pyfug.ascii` (markers normalised ¯/®→`>`) + **drvarma's own** histogram and
      ACF/PACF correlograms (`_ascii.py`: ports of `File_HistSer`, `PlotCor`,
      `PlotCCF`, `Ccf`, `ChiTestC`, `round_local`) + cross-correlation section.
      **Byte-exact vs `IPC3.out`** except the standardized-plot value column
      (residuals differ ~1e-9, engine tolerance). Wired into `out_report`
      (`residuals="auto"`: included when pyfug is importable). `tests/test_residual_report.py`.
- [x] **Graphics finish** (`plots.py`): JT diagnostics delegated to
      `pyfug.graphics` via the adapter — `plot_series_jt`, `plot_residual_acf_pacf`,
      `plot_residual_histogram`, `plot_residual_diagnostics` (combined),
      `plot_mean_deviation`. `apply_jt_theme()` applies pyfug's JT matplotlib
      rcParams globally so drvarma's own forecast/IRF/FEVD/CCF plots adopt the JT
      style too. pyfug in `[plots]` extras; tests skip if pyfug absent (6 tests).
- [ ] Docs: USER_GUIDE / API reference for the Python package; PyPI release.
- [x] CI workflow (done 2026-06-27): `.github/workflows/ci.yml` — a **pure-Python**
      job (matrix py3.10–3.12, no GSL → engine degrades away, asserts it is absent)
      and a **with-engine** job (libgsl-dev → builds the cffi extension, asserts it
      imports), both running `pytest`. Tests that need the C binary / engine / the
      sibling repo's IPC3 skip themselves, so both jobs are green on a standalone
      checkout. (To exercise the C-binary comparisons in CI, also check out
      `../drvarma_v.04.1` and `make drvarma` — left out to keep CI self-contained.)

## PP — 100% pure-Python parity  (full plan: `docs/PURE_PYTHON_PLAN.md`)

Goal: the pure-Python path is feature- and fidelity-complete vs the C engine, so
the CFFI engine is an optional accelerator only. Ordered PP1 → PP5.

- [x] **PP1 (keystone)** — estimator parity (done 2026-06-26). `estimate_py`
      now mirrors the C *exactly*: the `shootx` packing `(μ, φ, θ, raw qq
      lower-tri)`; `init_varma` (OLS AR seed + qq = residual **correlation**
      matrix — the start that pins the σ²/Q split, since the concentrated
      objective `f1^m·f2` is scale-invariant in qq); the concentrated objective
      via AS 311 `elf(σ²=1)`; and a **faithful port of the factored BFGS
      optimiser** (`_qnewt.py` ← `qnewtopt.c`: raxopt/bfgsfac/qrupdate/jacrot/
      cdgrad/lnsrch/umstop). `σ̂²=f1/(n·m)`, `Σ=σ²·Q`; `cov = 2·f·b⁻¹/n` from the
      optimiser's factored Hessian `b` (a plain numerical Hessian can't do this —
      the qq-scale direction is flat). Engine-free IPC3 `.out`: parameter table
      and normalized model **byte-identical** to the C binary except the 6th
      decimal of a few std errors (≤1.4e-4, the documented engine tolerance);
      estimates/logelf/sigma2/Σ/residuals match the C engine to ~1e-10. Closes
      G1 + G2. Tests: `test_estimate_py.py` (`..._split_and_stderrs_match_c`,
      `..._reports_sigma2_q_split`). 109 tests green.
- [x] **PP2** — Hannan-Rissanen two-step init (done 2026-06-26). Ported
      `hannan_rissanen_diag` (per-series HR: AR(L) OLS → residuals → regress on
      AR+MA lags, `theta_d=-coef`, variances scaled to avg 1) and the
      `combine_vectors` merge (diagonal AR/MA/cov from HR into the full start,
      off-diagonal kept from `init_varma`) into `estimate_py`; `-twostep` wired
      through the pure-Python path with the C's exact trigger (q>0 and not
      fully-diagonal). `init_diag_varma` is dead code in the C (no callers) — not
      ported. Validated vs the C engine (twostep=True): params <1e-5, logelf
      <1e-6, Σ <1e-6 on VARMA(1,1)/(2,1). Tests: `test_estimate_py.py`
      (`..._twostep_matches_c`, `..._twostep_runs_without_engine`). 112 green.
- [x] **PP3** — volatility (done 2026-06-26). `volatility.py` ports
      `volatility.c`: exponential weighting (`H_t=Σ φ(1-φ)^k ε_{t-k}ε_{t-k}'`, φ
      from the Mahalanobis-distance exceedance proportion vs the (1-α) percentile)
      and the moving-window unbiased sample covariance. `.volexp`/`.volmov`
      writers (C `%g` format) + the `.out` info line; CLI `-volexp [alpha window]`
      / `-volmov [window]` (defaults 0.05/20/20). **Byte-identical** to the C
      binary with the engine; engine-free only a single last-digit `%g` rounding
      differs (residuals ~1e-10). Tests: `test_volatility.py` (5). 117 green.
      Closes G4.
- [x] **PP4** — recursive forecasting for q>0 (done 2026-06-27). Dropped the
      q=0 restriction in `recursive_forecast`. **Empirically established** (vs the
      C binary on a synthetic well-identified VARMA(1,1)) that the C's
      `forecast_mean` uses the **estimation-window residuals** `varma1.a` (zero
      beyond the window), *not* full-series AS-311 residuals — so the MA term at
      origin e indexes `a[e]` and vanishes past the window. Implemented as
      `a_full[:estwin_eff] = result["residuals"]`. Validated to <1e-6 vs the C
      binary `.recursive` end-to-end (Model→`recursive_report`). `-seasonal` is a
      **vestigial** C flag (it only feeds `forecast_model`'s discarded `v_seas`;
      the `.forecast` annual% comes from `forecast_level_variances` with `s=freq`)
      — deliberately not ported. Test: `test_recursive.py::
      test_recursive_varma_q_positive_matches_c`. 117 green. Closes G5.
- [x] **PP5** — engine-free fidelity locked in (done 2026-06-27).
      `test_pure_python_out.py`: (1) the pure-Python estimator reproduces the C
      engine across the zoo (VAR(3) ±deseason, diag-ar, diag-cov) to logelf <1e-6,
      mu/phi/Σ <1e-5; (2) with the engine monkeypatched **off**, the deterministic
      `.out` sections (OIRF/accumulated, FEVD, multivariate diagnostics, normalized
      model) are **byte-identical** to the C binary for VAR(3) -mean. Findings:
      engine-free == cffi engine to ~1e-9 on raw data; the `.out` differences are
      (a) the **Inverse roots** ordering (modulus vs chekma QR — deliberate) and
      (b) under deseason the σ²/Q *split* drifts ~2.7e-5 (scale-ambiguous flat
      direction; Σ/logelf still ~1e-12) — both documented, neither an estimation
      error. Also established: the C *binary* can be numerically unstable on
      pathological synthetics (VARMA(1,1) n=300 → SIGABRT / garbage Q), where the
      pure-Python path stays stable; and the C/Python Hosking p-values use the same
      `df=m²·s` + upper tail (a 0.0373-vs-0.9627 mismatch was downstream of the
      binary's garbage estimate, not a formula bug).

## Engine / maintenance
- [ ] Keep `csrc/internal/` in sync with `../drvarma_v.04.1/src` when the C
      engine changes (they are copies).
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the C.

## Out of scope for this port — Shea (AS 242)

`csrc/internal/multshea.c` (`marma`) is compiled but **not wired into the C
estimator** (no callers), so there is no C reference to validate a Python port
against. **Deferred — not part of the 100% Python goal.** If ever revived: wire
`marma()` into a new C engine version first, compare C-Shea vs C-Mauricio, then
port faithfully (never a Kalman/state-space stand-in — that *is* Shea's route).

## Decisions / open questions
- [ ] Publish the Python port to its own GitHub repo? (own remote, CI, PyPI.)
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the
      C (via more API). Numpy keeps the port usable without the C engine (P3 goal).
