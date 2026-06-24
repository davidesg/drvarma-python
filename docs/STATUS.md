# drvarma Python port — status & handoff

Last updated: 2026-06-24. Read this first when resuming in a new session.

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

## Strategy (decided: A, CFFI-first)

Wrap the validated multivariate C through CFFI to get a correct port fast; the
pure-Python general-m likelihood is a later reference/fallback (P3). **fue's
pure-Python likelihood is m=1 only**, so the multivariate likelihood is *not*
reusable from fue — it lives in the shared C (or must be written for P3).

## Done (P0–P2), all validated vs the C binary on IPC3

| Phase | Module(s) | Validation |
|-------|-----------|------------|
| P0 | `series`, `inp`, `transform`, `datasets` (VARMA simulator) | .inp round-trip; ∇log×100; simulator recovers AR/cov/mean |
| P1 | `csrc/drvarma_api.{c,h}`, `_build_cffi`, `_engine`, `model.fit` | 36 params match C to <1e-5; synthetic recovery |
| P2 | `forecast` (+bands, recursive), `diagnostics`, `irf`, `deseason` | see below |

P2 numeric checks vs C (IPC3, `3 0 -mean [-deseason auto] [-forecast 12] [-estwin 200]`):
- forecast levels <1e-3; **bands** Low95/High95 <1e-3;
- diagnostics **exact** (Hosking Q(126)=461.5874, JB(6)=31.8249);
- IRF OIRF <1e-3, FEVD <0.05;
- deseason F-stats exact (64.580/32.243/21.960), φ[1]_11=0.330424 end-to-end <1e-5;
- recursive (-estwin) all 612 rows <1e-4 (with & without deseason).

`Model` API: `.fit()`, `.forecast(L, b=0, bands=False)`, `.recursive_forecast(estwin, H)`,
`.diagnostics(lag=None)`, `.irf(horizon)`, `.fevd(horizon)`; plus `.params/.sigma2/.loglik/.ifault`.

## Build & test

```sh
cd drvarma_source/drvarma
# build the optional C engine (GSL + GLib dev headers required):
python -m drvarma._build_cffi
mv drvarma/_drvarma_engine*.so src/drvarma/ ; rm -rf drvarma   # cffi writes it under ./drvarma/
PYTHONPATH=src python -m pytest tests/ -q                       # 29 tests
```

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
  diagnostics.py hosking_q, jarque_bera_mv
  irf.py        psi_weights, oirf, fevd
  datasets.py   simulate_varma
tests/          one test module per area; compare to the C binary + synthetic
```

## Gotchas (important when resuming)

- **`result["logelf"]` is the true log-likelihood**, NOT the scalar the C prints
  as "Objective function" (a different normalisation). Validate on PARAMETERS,
  forecasts and diagnostics, not on that printed scalar.
- **deseason is delicate**: harmonic amplitudes are estimated on the **d=1
  differenced basis** (design = d-th difference of cos/sin regressors), then
  mapped to LEVEL dummies (A0 + sum-to-zero) and subtracted from raw *levels*;
  forecasts re-seasonalise with period `(origin + l + sub - 2) % freq`. Don't
  "simplify" this — the differenced-basis estimation is what matches the C.
- **CFFI .so placement**: `_build_cffi` writes the module under `./drvarma/`
  (because of the dotted module name); move it into `src/drvarma/`.
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
