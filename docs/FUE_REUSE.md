# Reusing the fue → Python migration for drvarma

Evaluation of what the **fue** family already migrated from C to Python, what is
a faithful (exact-algorithm, different-API) copy, and what drvarma's remaining
work can reuse instead of re-porting from C. Written 2026-06-25.

> TL;DR — the univariate **ASCII residual diagnostics** (`diagnose.c`) and the
> **Jenkins-Treadway plots** are already migrated in `pyfug`. drvarma should reuse
> them per residual/series via a small `MultiSeries → Tseries` adapter, rather than
> re-porting `diagnose.c`/`x11plots.c`. The only multivariate gap (CCF) is already
> done in drvarma. drvarma keeps ownership of its own statistics (it computes them);
> pyfug is used for *rendering*.

## Sibling repositories

| Path | What it is |
|------|------------|
| `/home/david/Dropbox/SRC/atws/fue/fue` | **fue** univariate ARIMA Python port (+ its `csrc/`) |
| `/home/david/Dropbox/SRC/atws/fug/pyfug` | **pyfug** v2.0.0 — Python port of the FUG C graphics/ASCII/stats |
| `/home/david/Dropbox/SRC/drv4.040804/drvus` | older drvarma C with the **CCF** graphic (`ccf.c`, `qccf.c`, `x11plots.c`, `ccf*.eps`) |
| `../drvarma_v.04.1` | the current drvarma C engine (what this port wraps/validates against) |

fue and pyfug are **univariate (m=1)**. Everything multivariate is drvarma's own.

## What is already migrated to Python (univariate, faithful copies)

### pyfug — the FUG C migration (graphics + ASCII + statistics)

| Module | Contents | Fidelity vs C |
|--------|----------|---------------|
| `pyfug.ascii` | **`diagnose.c` migrated**: stats block, standardized 55-char time-series plot, histogram, ACF/PACF correlograms, mean-deviation, outlier table (`_write_statistics`, `_write_ascii_plot`, `_write_ascii_histogram`, `_write_acf_ascii_bars`, `_write_meandev`, `generate_ascii_output`) | near byte-exact |
| `pyfug.graphics` | Jenkins-Treadway high-definition matplotlib plots: `plot_series`, `plot_acf_pacf`, `plot_histogram`, `plot_combined`, `plot_mean_deviation`, `JTFigure`, `plot_title`, theme constants (`graphics/base.py`) | high |
| `pyfug.statistics` | `descriptive_stats`, `acf`, `pacf`, `ljung_box`, `chi_test`, `acf_pacf_max`, `series_max`, `series_size`, `compute_all` | same formulas as `diagnose.c` |
| `pyfug.transform` | `boxcox`, `regular_diff`, `seasonal_diff`, `delop`, `apply_diffops` | drvarma already has its own `transform.py` |
| `pyfug.core.Tseries` | univariate series dataclass (`nobs/freq/begyear/begtime/data/mean/var/skew/kurt/jarquebera/...`) | the **adapter target** |
| `pyfug.io`, `pyfug.latex`, `pyfug.jupyter`, `pyfug.cli` | I/O, LaTeX, notebook, CLI | univariate-specific |

### fue Python (univariate ARIMA modelling)

`diagnostics.py` (`acf`, `pacf`, `jarque_bera`, `ljung_box`), `plots.py`
(`plot_series`, `plot_acf_pacf`, `plot_histogram`, `plot_forecast`,
`plot_residual_diagnostics`, `plot_model_diagnostics`), `report.py`
(`write_out`, `write_fuf`, …), `elfvarma.py` (**AS 311 specialised to m=1** —
`elf_scalar`, `flikam_scalar`), `forecast.py`, `model.py`, `_engine.py`,
`cast_us.py`. The dual C-engine-or-pure-Python architecture drvarma copied.

## Empirical check (the decisive evidence)

Running `pyfug.ascii._write_statistics` + `_write_ascii_plot` on the first IPC3
residual series **reproduces the residual section of `IPC3.out` almost byte-for-
byte** — the standardized plot rows, dates, and statistics match. The only
differences are cosmetic:

- pyfug adds a `Jarque-Bera:` line in the stats block; drvarma's `File_StatSer`
  does not.
- pyfug's header reads `Transformed Time Series Data`; drvarma's reads
  `Unconditional residuals`.
- last-digit rounding on a few printed original values.

So `pyfug.ascii` is effectively the migrated `diagnose.c` ASCII renderer.

## Reuse map for drvarma's remaining work

| drvarma task | Reusable from the fue migration | How |
|--------------|--------------------------------|-----|
| **Residual section of `.out`** (ASCII) | `pyfug.ascii` (= `diagnose.c`) | per residual series via a `MultiSeries → Tseries` adapter; wrap with drvarma's `--- Residual series a[i] ---` header and wording |
| **Jenkins-Treadway plots** (series, ACF/PACF, histogram, mean-dev) | `pyfug.graphics` | delegate per series/residual (TODO item: "graphics finish") |
| per-series stats (mean/var/skew/kurt/acf/pacf/Ljung-Box) | `pyfug.statistics` ≡ already ported in drvarma | **done** in `diagnostics.py` (validated exact vs `IPC3.out`) |
| univariate forecast plot | `fue.plots.plot_forecast` | adapt, or keep drvarma's multivariate `plot_forecast` |

## What is drvarma-specific (not in fue — already implemented here)

Multivariate VARMA exact likelihood (AS 311, `_as311.py`), multivariate
forecasting + bands, IRF/FEVD, multivariate Hosking-Q / Jarque-Bera, Wald tests,
the **cross-correlation (CCF)** (`diagnostics.ccf`/`qccf`, `plots.plot_ccf` in the
`drv4.040804/drvus` format — pyfug has no CCF), and all the multivariate `.out`
report sections. These are validated against the C engine in this repo.

## Decision: statistics ownership vs rendering

- **drvarma owns its statistics.** `diagnostics.py` already ports the per-series
  computations from `diagnose.c` (`series_stats`, `acf`, `pacf`, `ljung_box`,
  `residual_diagnostics`) and matches `IPC3.out` exactly. Do **not** depend on
  `pyfug.statistics`/`pyfug.compute_all` for the *numbers*.
- **pyfug is used for rendering only.** Reuse `pyfug.ascii` (text) and
  `pyfug.graphics` (matplotlib) to avoid re-porting the intricate ASCII/JT layout.
  When the renderer recomputes a statistic internally, it agrees with drvarma's
  (same formulas) — acceptable for rendering.

## Recommended plan

1. **`MultiSeries → pyfug.core.Tseries` adapter** (one per residual/series column;
   `begyear/begtime` from the differenced-series start via `report.obs_to_date`).
2. **Residual `.out` section** — compose per series by reusing `pyfug.ascii`
   renderers, wrapped with drvarma's residual headers/wording. Add `pyfug` to the
   `[plots]`/`[report]` extras; the section is optional and skips if pyfug absent.
3. **Graphics finish** (TODO #11) — delegate the JT diagnostic plots to
   `pyfug.graphics` through the same adapter; restyle forecast/IRF/FEVD with the JT
   theme. CCF is already done.

## Caveats / open points

- Reusing `pyfug.ascii` yields fue's wording (`Jarque-Bera:` line, `Transformed
  Time Series Data`). Decide whether to keep drvarma's exact `File_StatSer` wording
  (write that block ourselves, reusing pyfug only for the plot/histogram/
  correlogram) or accept pyfug's wording for the whole section.
- `pyfug` is a sibling dev package (v2.0.0); pin/declare it in `[plots]` extras and
  guard imports so the core package still runs without it.
- The CCF graphic format lives in `drv4.040804/drvus` (not pyfug); drvarma's
  `plot_ccf` already reproduces it.
