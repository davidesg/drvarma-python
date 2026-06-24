# drvarma → Python — Migration Plan

Goal: port drvarma (multivariate VARMA) to Python following the **same approach
used for fue 1.13.1** (`atws/fue/fue`), reusing shared components and validating
incrementally against the reference-model battery and synthetic data.

## 1. Reference: how fue was ported

fue's Python port uses a **dual architecture** (`PERFORMANCE.md` there):

- **Pure-Python core** (numpy + scipy `L-BFGS-B`): `cast_us.estimate_py`,
  `elfvarma.elf_scalar` (Mauricio 1995), `flikam_scalar` (Melard 1984) — the
  reference/fallback estimator.
- **Optional C engine via CFFI** (`_fue_engine.c` = a thin `*_api.c` over the C
  core + GSL): `_engine.estimate()` tries the compiled `.so`, falls back to pure
  Python. cffi is a build-time dep only.
- **Always-Python layers**: `series` (TimeSeries), `model` (Model API,
  `fit()`/`forecast()`), `intervention` (+ FixedFreqFactor), `inp` (.inp parser),
  `report` (.out/.pre), `forecast`, `diagnostics`, `plots`, `cli`.
- Packaged with `pyproject.toml`/`setup.py`; extensive `tests/` (estimation,
  forecast, reliability, real cases) comparing against the C results.

**Key constraint:** fue's pure-Python likelihood is **specialised to m=1**
(`elfvarma.py` says so explicitly). The *general multivariate* likelihood exists
only in the shared **C** code. So drvarma's port cannot simply reuse fue's
Python `elf_scalar`.

## 2. Component map (drvarma C → Python)

| drvarma C | Nature | Plan |
|-----------|--------|------|
| `nlatools.c` (LU, Cholesky, GSL eigen/SVD, memory) | shared / generic | numpy/scipy replaces it (np.linalg, scipy.linalg). No port needed. |
| `qnewtopt.c` (BFGS) | shared | reuse fue's `qnewtopt.py`, or scipy `L-BFGS-B` (fue's pure path already does). |
| `elfvarma.c` / `multshea.c` (exact VARMA log-lik, general m) | **drvarma-critical** | **new**: general-m likelihood in numpy (generalise fue's m=1 `elf_scalar`/`flikam`), AND/OR wrap the C via CFFI. |
| `drvmlest.c` (`est` driver) | shared pattern | Python `estimate()` driver (mirror fue `cast_us`/`_engine`). |
| `drvarma.c` (`main`, `shootx`, init_varma) | orchestration | Python `Model.fit()` + param↔model packing (mirror fue `model.py` + `cast_us`). |
| `transform.c` (Box-Cox + diff + integrate) | shared-ish | port to numpy (check if fue already has Box-Cox/diff; else new). |
| `deseason.c` (harmonic adjustment) | drvarma-specific | port to numpy (GSL HAC F-test → numpy/scipy). |
| `forecast.c` (point + variance, origin offset) | drvarma-specific (multiv.) | port to numpy (generalise fue `forecast.py`). |
| `diagnose.c` (Q, JB, ACF/PACF, IRF, FEVD, Wald) | mixed | reuse fue `diagnostics.py` (ACF/PACF/JB/LB) + new multivariate (Hosking Q, mult-JB), IRF/FEVD, Wald tests. |
| `volatility.c` | drvarma-specific | port later (low priority). |
| `gui/` (GTK, Johansen, VECM) | front-end | out of scope for the core port; later (e.g. a thin CLI/notebook or a separate GUI). |
| `.inp` reader/writer (multivariate) | I/O | adapt fue `inp.py`/`report.py` to the multivariate header/format. |

**Reusable largely as-is from fue's Python port:** `series.py` (extend to
multivariate), `intervention.py`, `qnewtopt.py`, the CFFI scaffolding
(`_build_cffi.py`, `_engine.py` pattern), `cli.py`, `plots.py`, packaging.

## 3. Strategy

Recommended: **CFFI-first on the validated multivariate C, with a pure-Python
general-m reference added incrementally** (mirrors fue, adapted to multivariate).

1. **Expose a C library API** (`drvarma_api.c`): refactor the estimation pipeline
   out of `main()` into a callable `drvarma_estimate(spec) -> result` (params,
   cov, residuals, loglik, …). drvarma's `main` currently does everything inline;
   this extraction is the enabling step (the C is already NR-free/GSL).
2. **CFFI wrapper** (`_drvarma_engine`): build over `drvarma_api.c` + the existing
   `.c` core, exactly like fue's `_fue_engine`. Python `_engine.estimate()` calls
   it.
3. **Python layers** (numpy): `series`/`MultiSeries`, `model.Model` (spec, fit,
   forecast), `inp` (multivariate parser), `transform`, `deseason`, `forecast`,
   `diagnostics` (incl. multivariate), `report`, `cli`. Reuse/adapt fue's.
4. **Pure-Python general-m likelihood** (`elfvarma_py`): generalise fue's m=1
   `elf_scalar`/`flikam` to vector ARMA (numpy + scipy `L-BFGS-B`) as the
   reference/fallback. This is the largest new numerical piece; can land after a
   working CFFI path.

## 4. Validation

### 4.1 Reference-model battery (against the C drvarma)
Reproduce, in order of increasing complexity, comparing Python vs C `.out`
(objective, parameters, SE/Wald, eigenvalues, diagnostics) and `.forecast`:

1. **Univariate / diagonal** — diagonal VAR factorises to univariates; compare to
   both C drvarma and ART (see `MODELS_RESULTS.md`).
2. **Bivariate** — WTI+IPC pass-through (`data/passthrough/`).
3. **Trivariate** — `data/models_group1/IPC3` (diagonal and full VAR(3)).
4. Forecasting and **`-estwin`** recursive output.
Tolerance: match the C objective/parameters to ~1e-6 (same data, options, scale).

### 4.2 Synthetic models (parameter recovery)
Add a **VARMA simulator**: given (m, p, q, Φ, Θ, Σ, μ), generate `w_t`; estimate;
check recovery (coefficients within sampling error, σ̂→Σ). Use it to:
- cover structures absent from the real battery (q>0, full Σ, near-unit roots,
  different m);
- build fast unit tests with known ground truth;
- stress edge cases (collinearity à la WTI/IPC, small n).
Store generators + fixtures under `tests/` (mirror fue's `datasets.py`/tests).

## 5. Package layout (mirror fue)

```
drvarma_py/                 (or src/drvarma/)
  pyproject.toml, setup.py
  src/drvarma/
    __init__.py
    series.py        # TimeSeries / MultiSeries
    model.py         # Model: spec, fit(), forecast()
    intervention.py  # Intervention, FixedFreqFactor
    inp.py           # multivariate .inp reader/writer
    transform.py     # Box-Cox + differencing (+ integrate)
    deseason.py      # harmonic seasonal adjustment
    elfvarma.py      # general-m exact log-likelihood (pure Python)
    estimate.py      # estimation driver (cast/pack + optimiser)
    forecast.py      # multivariate point + variance forecasts, -estwin
    diagnostics.py   # ACF/PACF/JB/LB + Hosking Q + mult-JB
    irf.py           # impulse response + FEVD
    report.py        # .out / .forecast / .recursive writers
    _engine.py, _build_cffi.py, _drvarma_engine.c   # optional C path
    cli.py
  tests/             # battery + synthetic recovery + reliability
```

## 6. Phased roadmap

- **P0 — scaffolding & I/O**: package skeleton; `series`/`MultiSeries`, `inp`
  reader/writer (round-trip the existing `.inp`), `transform` (Box-Cox/diff).
  Reuse fue where possible.
- **P1 — estimation via CFFI**: extract `drvarma_api.c`, build `_drvarma_engine`,
  `Model.fit()` → C; validate objective/params on the battery (univariate →
  trivariate).
- **P2 — forecasting & diagnostics**: `forecast` (+ `-estwin`), `deseason`,
  multivariate diagnostics, IRF/FEVD, `report`; validate `.forecast`/`.recursive`.
- **P3 — pure-Python general-m likelihood**: `elfvarma.py` + scipy optimiser as
  reference/fallback; cross-check vs C and synthetic recovery.
- **P4 — synthetic simulator & test suite**; **P5 — CLI/packaging/docs**, PyPI;
  GUI/VECM later or separate.

## 7. Notes / risks
- Reuse the validated C (NR-free, GSL) through CFFI to get a correct port fast;
  the pure-Python multivariate likelihood is the main novel effort (P3).
- Keep the same conventions as the C (scale default 100; estimates scale-
  invariant) so cross-validation is exact.
- The WTI/IPC variance-disparity caveat (ill-conditioned cross-term SEs) applies
  equally in Python — validate against the documented C results, not idealised
  expectations.
