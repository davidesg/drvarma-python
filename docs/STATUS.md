# drvarma Python port — status & handoff

Last updated: 2026-06-25. Read this first when resuming in a new session.

## What this is

Python port of the **drvarma** multivariate VARMA engine, following the same
dual architecture as the fue port (`/home/david/Dropbox/SRC/atws/fue/fue`):
pure-Python layers (I/O, model API, forecasting, diagnostics) on top of an
**optional CFFI-compiled C engine** that wraps the validated, GPL/NR-free drvarma
C core.

- **C engine repo (sibling):** `../drvarma_v.04.1` — published at
  <https://github.com/davidesg/drvarma> (GPL v2, Numerical-Recipes-free).
- **This folder** `drvarma_source/drvarma/` is the Python port (own git repo,
  branch `master`).
- Plan: `docs/MIGRATION_PLAN.md`. Reference results to validate against:
  `../drvarma_v.04.1/MODELS_RESULTS.md`.
- Reusing the fue/pyfug Python migration (ASCII residual diagnostics + JT plots)
  for the remaining presentation work: see `docs/FUE_REUSE.md`.
- Plan to reach a **100% pure-Python** drvarma (engine optional): see
  `docs/PURE_PYTHON_PLAN.md` (phases PP1–PP5; Shea is out of scope).

## Strategy (decided: A, CFFI-first)

Wrap the validated multivariate C through CFFI to get a correct port fast; the
pure-Python general-m likelihood is a later reference/fallback (P3). **fue's
pure-Python likelihood is m=1 only**, so the multivariate likelihood is *not*
reusable from fue. P3 status: the exact VARMA likelihood is now written in pure
Python as a faithful port of Mauricio's **AS 311** (`_as311.py`, see below);
`estimate_py` fits it without the C engine.

## Done (P0–P5), all validated vs the C binary/engine on IPC3 (107 tests)

| Phase | Module(s) | Validation |
|-------|-----------|------------|
| P0 | `series`, `inp`, `transform`, `datasets` (VARMA simulator) | .inp round-trip; ∇log×100; simulator recovers AR/cov/mean |
| P1 | `csrc/drvarma_api.{c,h}`, `_build_cffi`, `_engine`, `model.fit` | 36 params match C to <1e-5; synthetic recovery |
| P2 | `forecast` (+bands, recursive), `diagnostics`, `irf`, `deseason`, `report` | see below + report section |
| P3 | `_as311` (Mauricio AS 311), `elfvarma_py`, `estimate_py` | logelf ~1e-11, exact residuals ~1e-12; pure-Python estimate vs C <1e-3 |
| P4 | `datasets.varma_cases`, `tests/test_reliability.py` | recovery within bands; pass-through (WTI/IPC) ill-conditioning; formula checks |
| P5 | `cli`, `setup.py` (optional cffi build) | `.forecast` byte-exact; pipeline runs C-free |

Deferred: P5 docs/plots/CI, P6 Shea backup (`marma`, C-first — see TODO).

P2 numeric checks vs C (IPC3, `3 0 -mean [-deseason auto] [-forecast 12] [-estwin 200]`):
- forecast levels <1e-3; **bands** Low95/High95 <1e-3;
- diagnostics **exact** (Hosking Q(126)=461.5874, JB(6)=31.8249);
- IRF OIRF <1e-3, FEVD <0.05;
- deseason F-stats exact (64.580/32.243/21.960), φ[1]_11=0.330424 end-to-end <1e-5;
- recursive (-estwin) all 612 rows <1e-4 (with & without deseason).

`Model` API: `.fit()`, `.forecast(L, b=0, bands=False)`, `.recursive_forecast(estwin, H)`,
`.diagnostics(lag=None)`, `.irf(horizon)`, `.fevd(horizon)`; accessors
`.phi/.theta/.mu/.sigma/.sigma2/.residuals/.params/.std_errors/.loglik/.ifault`.

## Build & test

```sh
cd drvarma_source/drvarma
# build the optional C engine (GSL dev headers required) straight into src/:
python setup.py build_ext --inplace        # or: pip install -e .
PYTHONPATH=src python -m pytest tests/ -q   # 107 tests
```

`pip install` builds the engine via `setup.py` (an *optional* cffi Extension:
if GSL is missing the install still succeeds and the package runs pure-Python,
with the C-engine tests skipping themselves). Running `_build_cffi` standalone
still works but writes the module under `./drvarma/` (move it to `src/drvarma/`);
prefer `build_ext --inplace`, which places it correctly.

The `.so` (and `_drvarma_engine.c`, `*.o`) are gitignored — rebuild as above.
Tests that compare against the C binary auto-skip if `../drvarma_v.04.1/bin/drvarma`
is absent (build it there with `make drvarma`).

## How validation works

Each P1/P2 test runs the C binary on `IPC3.inp` into a tmp dir, parses its
`.out`/`.forecast`/`.recursive`, and asserts the Python results match. To check
manually, run e.g. `../drvarma_v.04.1/bin/drvarma /tmp/x 3 0 -mean -deseason auto
-forecast 12` and compare to `Model(...).fit()`.

## Architecture / file map

```
csrc/
  drvarma_api.{c,h}   # DrvarmaModelSpec -> drvarma_estimate -> DrvarmaResult.
                      # Embeds estimation machinery extracted VERBATIM from
                      # drvarma.c (shootx, init_varma, init_diag_varma,
                      # combine_vectors, hannan_rissanen_diag, calc_nparametrs)
                      # + the file-scope globals.  Estimates on the stationary w.
  internal/           # copies of the shared core: nlatools, elfvarma, multshea,
                      # qnewtopt, drvmlest, main.h.
src/drvarma/
  series.py     MultiSeries
  inp.py        multivariate .inp reader/writer
  transform.py  Box-Cox + differencing + integrate_forecast
  deseason.py   harmonic seasonal adjustment (estimate d=1, recover levels)
  _build_cffi.py / _engine.py   CFFI build + estimate_w() bridge
  model.py      Model (fit/forecast/recursive_forecast/diagnostics/irf/fevd)
  forecast.py   forecast_w, forecast_level_variances, recursive_forecast
  diagnostics.py hosking_q, jarque_bera_mv, series_stats, acf, pacf, ljung_box, ccf
  irf.py        psi_weights, oirf, fevd
  datasets.py   simulate_varma
  report.py     .out/.forecast/.recursive writers (C-format text reports)
  cli.py        `drvarma <file> p q [flags]` entry point (drvarma.cli:main)
  _as311.py     faithful port of Mauricio's AS 311 exact VARMA likelihood
  elfvarma_py.py elf_varma (AS 311 wrapper) + elf_var (fast q=0 specialisation)
  estimate_py.py scipy exact-ML VARMA estimator (pure-Python fallback)
  plots.py      matplotlib plots: drvarma (forecast/IRF/FEVD/CCF) + JT diagnostics via pyfug
  _pyfug.py     MultiSeries/residual -> pyfug.core.Tseries adapter (JT rendering)
  _ascii.py     drvarma ASCII histogram + ACF/PACF/CCF correlograms (diagnose.c)
tests/          one test module per area; compare to the C binary + synthetic
```

## Pure-Python VARMA fallback (P3, done 2026-06-25)

`_as311.py` is a **faithful Python port of Mauricio's AS 311** (the exact VARMA
log-likelihood in `csrc/internal/elfvarma.c`: `elf` + `cgamma` + `cxi` + `cres` +
`chekma`). Mauricio wrote that algorithm, so the port is a step-by-step
transcription (1-indexed arrays; only the length-n inner sums vectorised). It
reproduces the C `logelf` to ~1e-11 and the AS 311 exact residuals to ~1e-12.

**Do not replace this with a Kalman filter** — the state-space/Kalman route is
essentially Shea's approach; a faithful port of `multshea.c` (AS 242) is the
desirable *backup*, tracked in TODO.

`elfvarma_py.elf_varma` wraps AS 311 (general p,q); `elf_var` is a fast vectorised
q=0 specialisation (companion Lyapunov covariance), cross-checked against AS 311.
`estimate_py.estimate_w_py` fits by scipy L-BFGS-B and returns the **same dict
shape** as the C `_engine.estimate_w`, which now **falls back** to it on
ImportError — so model/forecast/report work without the compiled engine.
Validated vs the C engine to <1e-3 on params (q=0 and VARMA). Caveats: std errors
come from a numerical Hessian (best-effort); the fallback reports `sigma2=1,
sigma=Sigma` rather than the C's AS-311 sigma2/Q split; the L-BFGS-B start is
θ=0 (no Hannan-Rissanen two-step), so for weakly-identified VARMA the optimum is
the MLE but may sit far from a poorly-identified "truth" (the C agrees).

## Reports & CLI (P2-presentation + P5-CLI, done 2026-06-25)

`report.py` reproduces the C text outputs; `cli.py` is the `drvarma` entry point
(reads lambda/d/D from the `.inp`, writes `.out` always, `.forecast` with
`-forecast H`, `.recursive` with `-estwin N`). Fidelity vs the C binary
(`tests/test_report.py`, 8 tests):
- **`.forecast` byte-exact** (Level/Low95/High95 + mon%/ann% rates and std).
- **`.recursive`** matches the validated engine path (<1e-4); format ported verbatim.
- **`.out`**: header, OIRF/accumulated/gain, FEVD, multivariate diagnostics,
  normalized model are **byte-exact**; parameter *estimates* exact, but SE/t/p
  carry the documented <1e-5 engine tolerance (Wald chi2 amplify it to ~1e-3,
  p-values/conclusions still agree). NOT reproduced by design: the optimizer
  iteration/objective line (engine-internal — `report` prints the log-likelihood
  instead), inverse-roots *ordering* (modulus-sorted, not chekma's QR order), and
  the per-series ASCII residual-plot tail (`diagnose()`).
- Hosking lag in the report is `floor(sqrt(nobs))` (C `multivariate_diagnostics`),
  *not* the `freq+2` used by `Model.diagnostics()`.

## Gotchas (important when resuming)

- **`result["logelf"]` is the true log-likelihood**, NOT the scalar the C prints
  as "Objective function" (a different normalisation). Validate on PARAMETERS,
  forecasts and diagnostics, not on that printed scalar.
- **deseason is delicate**: harmonic amplitudes are estimated on the **d=1
  differenced basis** (design = d-th difference of cos/sin regressors), then
  mapped to LEVEL dummies (A0 + sum-to-zero) and subtracted from raw *levels*;
  forecasts re-seasonalise with period `(origin + l + sub - 2) % freq`. Don't
  "simplify" this — the differenced-basis estimation is what matches the C.
- **CFFI .so placement**: prefer `python setup.py build_ext --inplace` (or
  `pip install -e .`), which builds straight into `src/drvarma/`. Running
  `_build_cffi` standalone still writes under `./drvarma/` (dotted module name) —
  move it into `src/drvarma/` if you use that path.
- **`csrc/internal/` are copies** of the C core and will drift from
  `../drvarma_v.04.1/src` if that changes — re-sync when the C engine is updated.
- **default scale = 100** (matches the C; estimates are scale-invariant). Keep it
  so cross-validation against the C is exact.
- **recursive_forecast is q=0 only** (the documented `-estwin` use); q>0 needs
  full-data residuals at fixed params (not yet implemented).
- The pass-through (WTI/IPC) cross-term SEs are ill-conditioned by design
  (variance disparity) — see the C `MODELS_RESULTS.md` caveat; expect the same.

## Next

See `TODO.md`.
