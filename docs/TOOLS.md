# `sima` — MCP tool reference

*Generated from the docstrings by `tools/gen_tools_md.py`. Do not edit by hand — edit the docstring.*

**15 tools.** In an MCP server the docstring is what the model reads, so this page and the instruction the model receives are the same text by construction.

---

| tool | what it answers |
|---|---|
| [`characterize_series`](#characterize-series) | SEED step — ART's univariate identification per component, saved to session. |
| [`confirm_and_estimate`](#confirm-and-estimate) | Estimate the final VARMA(p,q) by exact ML and store the fit under `name`. |
| [`cross_correlation_matrices`](#cross-correlation-matrices) | Sample cross-correlation matrices (CCM) of the prepared series. |
| [`diagnose`](#diagnose) | Multivariate residual diagnostics: Hosking Q + Jarque-Bera. A significant Q |
| [`export_fit`](#export-fit) | Residuals and fitted parameters as JSON — the fit in machine-readable form. |
| [`generate_forecast`](#generate-forecast) | Forecast the fitted VARMA `horizon` steps ahead with 95% bands (drvarma's |
| [`identify_varma_order`](#identify-varma-order) | Rank VARMA(p,q) orders by information criteria (AIC/BIC/HQ). |
| [`impulse_response`](#impulse-response) | Orthogonalised impulse responses (OIRF) with 95 % Monte-Carlo bands. |
| [`load_data`](#load-data) | Load the m ORIGINAL (untransformed) series into the session under `name`. |
| [`partial_autoregression_matrices`](#partial-autoregression-matrices) | Tiao-Box partial autoregression matrices — AR-order identification. |
| [`plot_cross_correlation_functions`](#plot-cross-correlation-functions) | PLOT of the TWO-SIDED CCFs for every pair — reads DIRECTION, unlike the CCM. |
| [`plot_cross_correlation_matrices`](#plot-cross-correlation-matrices) | PLOT of the CCM: m×m grid, cell (i,j) = ρ_ij(k) for k=1..K, with ±2/√n bands. |
| [`plot_partial_autoregression_matrices`](#plot-partial-autoregression-matrices) | PLOT of the Tiao-Box partial autoregression matrices (t-ratios). |
| [`series_info`](#series-info) | Per-series descriptive statistics (mean/var/std/skew/kurt/min/max). |
| [`variance_decomposition`](#variance-decomposition) | Forecast-error variance decomposition (FEVD, %) with 95 % Monte-Carlo bands. |

---

## `characterize_series`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |

SEED step — ART's univariate identification per component, saved to session.

    For each series computes λ (Box-Cox), d (unit-root recommended differencing),
    seasonality (HAC F-test) and rough ARMA orders, then derives a JOINT consensus
    (λ, d, deseason) that the VARMA tools use by default. Without this, a VARMA
    spec is blind guessing. Defensive: falls back to log/d=1 per piece if a
    component analysis is unavailable.

---

## `confirm_and_estimate`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `p` | integer | yes | — |
| `q` | integer | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `include_mean` | boolean | no | `True` |
| `diag_ar` | boolean | no | `False` |
| `diag_ma` | boolean | no | `False` |
| `diag_cov` | boolean | no | `False` |

Estimate the final VARMA(p,q) by exact ML and store the fit under `name`.

    Uses the saved characterization (λ/d/deseason) by default. diag_ar/diag_ma/
    diag_cov impose diagonal AR/MA/covariance. Returns loglik, Φ/Σ and lets you
    diagnose/forecast next.

---

## `cross_correlation_matrices`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `n_lags` | integer | no | `6` |

Sample cross-correlation matrices (CCM) of the prepared series.

    Uses the saved characterization (λ/d/deseason) by default. Tiao-Box +/-/.
    (bound 2/√n). CCM that CUT OFF after lag q ⇒ pure MA(q); slow decay ⇒ AR terms.

    The +/- symbols are NOT a significance test, and printing them without
    saying so is how a reader turns them into one. Tiao & Box (1981, p. 806),
    who introduced them, are explicit: the variances of the sample correlations
    "can be considerably greater than n^(-1/2) when the series are highly
    autocorrelated, so that these indicator symbols, if taken literally, can
    lead to OVERPARAMETERIZATION. However, we do not interpret these indicator
    symbols in the sense of a formal significance test, but as a rather crude
    'signal-to-noise' guide."

    Which is exactly the reading an assistant will not arrive at on its own from
    a grid of + and -, so the tool now says it in its own output.

---

## `diagnose`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lag` | integer | no | `0` |

Multivariate residual diagnostics: Hosking Q + Jarque-Bera. A significant Q
    means the order (p or q) is too low.

---

## `export_fit`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `what` | string | no | `all` |
| `max_rows` | integer | no | `0` |

Residuals and fitted parameters as JSON — the fit in machine-readable form.

    `what`: "residuals", "params", "sigma" or "all". `max_rows` truncates the
    residual block (0 = all); the head and tail are kept so the ends stay
    visible.

    This exists because every cross-check has to be possible FROM HERE. In the
    oil pass-through exercise the residual ACF, the OLS arbitration and the
    reproduction of the published table all had to bypass this server and drive
    the library directly — which is exactly the situation in which a wrong
    number survives, because checking it is more work than believing it.

    The text tools state conclusions; this one hands over the evidence.

---

## `generate_forecast`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |

Forecast the fitted VARMA `horizon` steps ahead with 95% bands (drvarma's
    native, level-unit, scale-correct bands — never a hand-rolled relative-std band).

---

## `identify_varma_order`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `p_max` | integer | no | `0` |
| `q_max` | integer | no | `0` |
| `include_mean` | boolean | no | `True` |

Rank VARMA(p,q) orders by information criteria (AIC/BIC/HQ).

    Uses the saved characterization (λ/d/deseason and the p,q ceilings) by default;
    p_max/q_max=0 means "use the seed's ceiling". Fits every (p,q) by exact ML and
    ranks by BIC. Complements the CCM / Tiao-Box evidence.

---

## `impulse_response`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |
| `bands` | boolean | no | `True` |
| `ndraws` | integer | no | `600` |

Orthogonalised impulse responses (OIRF) with 95 % Monte-Carlo bands.

    The band is the point of this tool. A response without one cannot say
    whether it is signal, and the sizes at stake here (a 5 % pass-through
    against a 26 % one) are decided by the interval, not the point.
    `bands=False` skips the draws when only the shape is wanted.

---

## `load_data`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `path` | string | no | `` |
| `values_json` | string | no | `` |
| `freq` | integer | no | `12` |
| `start_year` | integer | no | `2000` |
| `start_period` | integer | no | `1` |
| `series_names` | string | no | `` |

Load the m ORIGINAL (untransformed) series into the session under `name`.

    Provide EITHER `path` (an .xlsx/.xls Excel or a .csv, one column per series,
    optional header row) OR `values_json` (a JSON list of rows). freq = obs/year
    (1/4/12); start = (start_year, start_period). series_names = optional
    comma-separated labels. Do NOT pass pre-transformed or pre-differenced data.

---

## `partial_autoregression_matrices`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `max_order` | integer | no | `6` |

Tiao-Box partial autoregression matrices — AR-order identification.

    Uses the saved characterization by default. Fits VAR(k) by OLS; the partial
    autoregression matrix is the last block Φ_kk. For a VAR(p) it is ≈0 (all '.')
    for k>p, so p = last lag with significant symbols (t-ratios, |t|>1.96).

---

## `plot_cross_correlation_functions`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `n_lags` | integer | no | `0` |
| `path` | string | no | `` |

PLOT of the TWO-SIDED CCFs for every pair — reads DIRECTION, unlike the CCM.

    One panel per pair (i<j) over lags -K..K with ±2/√n bands. Bars on the RIGHT
    (k>0) mean series j leads series i; bars on the LEFT (k<0) mean i leads j. A
    strictly one-sided pattern is the visual signature of an exogenous variable;
    bars on both sides mean feedback. Writes a PNG and returns its path.

    The ±2/√n band is the same crude signal-to-noise guide as the CCM's +/-
    symbols, not a significance test — see `cross_correlation_matrices`. With
    strongly autocorrelated series the true variance of each correlation is
    larger, so reading the band literally over-parameterises. Read the SHAPE
    (cut-off vs decay, one side vs both), not the count of crossings.

---

## `plot_cross_correlation_matrices`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `n_lags` | integer | no | `0` |
| `path` | string | no | `` |

PLOT of the CCM: m×m grid, cell (i,j) = ρ_ij(k) for k=1..K, with ±2/√n bands.

    The visual counterpart of `cross_correlation_matrices` (same statistic, shared
    code). Row i = series i, column j = series j; a bar at lag k in cell (i,j) means
    series j lagged k correlates with series i now. A cell that CUTS OFF after lag q
    points to MA(q); slow decay points to AR terms. Writes a PNG and returns its path.

    The ±2/√n band is the same crude signal-to-noise guide as the CCM's +/-
    symbols, not a significance test — see `cross_correlation_matrices`. With
    strongly autocorrelated series the true variance of each correlation is
    larger, so reading the band literally over-parameterises. Read the SHAPE
    (cut-off vs decay, one side vs both), not the count of crossings.

---

## `plot_partial_autoregression_matrices`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `lam` | number | no | `-99.0` |
| `d` | integer | no | `-1` |
| `D` | integer | no | `-1` |
| `deseason` | string | no | `seed` |
| `max_order` | integer | no | `6` |
| `path` | string | no | `` |

PLOT of the Tiao-Box partial autoregression matrices (t-ratios).

    The visual counterpart of `partial_autoregression_matrices` (same computation,
    shared code). m×m grid; cell (i,j) shows the t-ratio of Φ_kk[i,j] for k=1..K
    with the ±1.96 band. The AR order p is the last lag with bars outside the band.
    Writes a PNG and returns its path.

---

## `series_info`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |

Per-series descriptive statistics (mean/var/std/skew/kurt/min/max).

---

## `variance_decomposition`

**Arguments**

| name | type | required | default |
|---|---|---|---|
| `name` | string | yes | — |
| `horizon` | integer | no | `12` |
| `bands` | boolean | no | `True` |
| `ndraws` | integer | no | `600` |

Forecast-error variance decomposition (FEVD, %) with 95 % Monte-Carlo bands.

    A share reported without an interval cannot be acted on: 5 % and 26 % are
    different claims only if the bands do not overlap. `bands=False` skips the
    draws.

---
