# sima — design of the drvarma MCP (SIMULTANEOUS multivariate VARMA)

Status: **implemented** (14 tools). Renamed from `multiart` on 2026-08-02 — see
§0 and `drtran-python/docs/ARCHITECTURE_MCP.md`, which settles the suite's MCP
architecture.

## 0. The name, and the assistant next door

`multiart` read as "the multivariate ART", claiming a lineage it does not have.
**ART's natural continuation is drtran**, which consumes ART's `.pre` files
directly and inherits Box–Jenkins' prewhitening. drvarma is a *classical
symmetric VARMA* with a different ancestry (Mauricio's exact likelihood, AS 311).

The suite's three assistants:

| | models | engine |
|---|---|---|
| `art` | one series: ARIMA + interventions | fue |
| `mtram` | transfer functions and networks (DAG) | drtran |
| **`sima`** | **simultaneous VARMA** | drvarma |

`sima` — **SI**multaneous **M**ultivariate **A**nalysis — says the thing an
analyst most needs to know before touching it: everything in here is
simultaneous, so the impulse response is **not identified without an ordering**.
`mtram`'s is, because its exogeneity is declared and tested. Two different claims
about the world, which is why they are two servers.

**The handoff.** When `mtram`'s `identify_network` proposes a DAG with a
**cycle**, the system has no topological order and cannot be cast as a triangular
VARMA: it is simultaneous, and that is exactly when the analyst should come here.
It is a contrast, not a preference — it fires on the m6 system.

Section 7's roadmap said "v3 — transfer functions / VARMA networks (converge with
the drtran Python port)". That is superseded: transfer networks are **not** a
later layer of this server, they are `mtram`.

`sima` is the simultaneous-multivariate counterpart of ART's univariate Box-Jenkins MCP: an
MCP server that walks an analyst (and/or an LLM) through building a **VARMA** model
with drvarma, the way ART does it for univariate ARIMA with fue.

---

## 1. Scope of v1

**In:** the *classic* stationary Gaussian VARMA, `Φ(B)(wₜ − μ) = Θ(B) aₜ`,
`aₜ ~ N(0, Σ)` — identification of the order (p, q), estimation (exact ML),
diagnosis, structural analysis (IRF / FEVD) and forecasting.

**Out of v1 (deferred, but the server is *aware* of them):**

- **Cointegration / common trends.** v1 assumes the series are brought to
  stationarity by (per-series) differencing; it does **not** test for common
  trends or model a VECM. sima should, however, *flag* when the data smell
  non-stationary/cointegrated (e.g. individually I(1) series that move together)
  and say it is setting that aside — answering it properly is the headline v2 goal.
- **Transfer functions / VARMA networks** (drtran's territory: exogenous inputs,
  DAG of transfers). drvarma.Model is a symmetric all-endogenous VARMA; transfer
  structure is a constrained VARMA and belongs to a later layer / the drtran port.

---

## 2. The central idea — art-seeded identification

Specifying a VARMA cold is *palos de ciego* (blind guessing). So sima **seeds**
the VARMA spec from a univariate pass before doing anything multivariate:

- **Route A (default) — art-seeded.** Run ART's **automatic** univariate exploration
  on each component series → per-series `λ` (Box-Cox), differencing `d`, seasonal
  `D`, rough ARMA orders, seasonality. This bounds the multivariate search: the
  transformation and differencing to reach stationarity, and a sensible ceiling for
  (p, q).
- **Route B — user-informed.** If the analyst knows the data, they answer a few
  questions (stationary? transform? seasonal?) and **skip** the ART pass.

With the series prepared, sima proceeds to the **canonical VARMA
identification** (below). Seasonality, if present, is handled by **drvarma's own
routines** (`deseason.py`), not necessarily ART.

---

## 3. Architecture — composed in libraries, standalone as a server

- **Standalone server.** The analyst runs `sima` on its own; it does not need
  an ART server running. Uni-vs-multivariate stays cleanly separated (mirrors the
  C world: fue univariate, drtran/drvarma multivariate).
- **Composed internally.** For the art-seeding step sima *imports* `art`
  (art-tseries) as a library and calls its identification functions; drvarma owns
  the joint VARMA part. No cycle — ART does not depend on drvarma.
- **Packaging.** Ships as an optional extra **`drvarma[mcp]`** with entry point
  `sima` (mirrors `art-tseries` ⇒ `art-mcp`). Bundled by `atsw`. `mcp` +
  `art-tseries` + `fue` are pulled by the extra.

---

## 4. The guided workflow (mirrors ART's protocol)

Opening question, like ART: **guided** (step-by-step, plots + confirmation at each
node) or **autonomous** (full pipeline). And a **language directive**: always
respond in the user's language; tool output may be Spanish, translate it.

```
1. load_data            multivariate series (matrix / long / Excel-CSV)
2. seed  ── Route A: seed_from_art   (per-series λ/d/D/orders, automatic ART)
        └─ Route B: user answers a few questions
3. prepare              transform + difference to stationary VARMA form
                        (seasonality via drvarma deseason)
4. identify_varma_order CCM + partial-autoregression matrices + AIC/BIC/HQ grid
                        → propose (p,q); WAIT for confirmation
5. confirm_and_estimate Model(p,q, include_mean).fit()  (+ diag_ar/diag_ma options)
6. diagnose             Hosking Q, residual cross-correlation (CCM of residuals)
7. structure            impulse_response (OIRF), variance_decomposition (FEVD),
                        [granger_causality]
8. generate_forecast    forecast(L, bands=True) + HTML report
9. full_report          out_report / write_out
```

---

## 5. Tool inventory (→ drvarma / art API, and what is new)

| Tool | Backed by | New work |
|---|---|---|
| `load_data`, `preview_data`, `series_info` | `datasets`, `diagnostics.series_stats` | I/O glue |
| `seed_from_art` | **art** guided/auto identification per series | orchestration + summary |
| `prepare_series` (transform/difference/deseason) | `transform`, `deseason.deseasonalize_raw` | glue |
| `identify_varma_order` | `diagnostics.ccf/qccf`, `Model.fit` for the IC grid | **CCM + partial-autoregression (Tiao-Box) matrices + AIC/BIC/HQ grid + recommendation** ← the core new layer |
| `confirm_and_estimate` | `Model(p,q,include_mean).fit()` (`estimate_w_py` diag options) | glue |
| `diagnose` | `Model.diagnostics`, `diagnostics.hosking_q`, residual `qccf` | multivariate residual CCM panel |
| `impulse_response` | `Model.irf` / `irf.oirf` | plot |
| `variance_decomposition` | `Model.fevd` / `irf.fevd` | plot |
| `granger_causality` | VARMA Φ structure | **build** (Wald on Φ blocks) |
| `generate_forecast` | `Model.forecast(bands=True)`, `report_forecast.write_forecast_report` | table + bands (heed BUG-0008-style level-vs-relative std) |
| `update_and_forecast` (SPS) | `forecast.recursive_forecast` | fixed-window re-forecast (carry scale/refactor — cf. ART BUG-0007) |
| `full_report` | `report.out_report/write_out` | glue |

---

## 5b. Rescaling (the ×100 convention) — handle with care

drvarma's C engine (like fue's) **conditions better on rescaled data**; by
convention the factor is **100**, and drvarma already implements it correctly:

- `Model(series, lam, d, D, scale=100.0, …)` — `transform.DEFAULT_SCALE = 100.0`.
  The transform is `w = ∇^d ∇_s^D [ scale · BoxCox_λ(level) ]`; the level is recovered
  by `÷scale`. `scale` is a **single per-model value**, stored on the model — the
  exact analogue of fue's `refactor`, and the **same convention ART already uses**.
- Forecast **bands are already in level units**: drvarma computes them in the scaled
  space and un-scales, `boxcox_inv((cf ± 1.96·sd)/scale, λ)`. So drvarma does **not**
  have ART's BUG-0008 (relative-std band).
- The SPS path `recursive_forecast(…, scale, …)` **carries the scale**, so it does not
  have ART's BUG-0007 (rebuild dropping the factor).

**The risk is re-introducing these bugs at the MCP layer.** sima MUST:

1. **Never hardcode 100.** Read `model.scale` wherever a scale is needed (single
   source of truth — the P1 rule from the ATSW rescaling audit).
2. **Not re-implement forecast bands** from the raw residual std — use drvarma's own
   level-unit bands (`Model.forecast(bands=True)`). A hand-rolled `level ± 1.96·sd`
   table would resurrect BUG-0008.
3. **Carry `scale` through any model rebuild** (e.g. the SPS `update_and_forecast`
   loop) — pass `scale=old.scale`, exactly as `recursive_forecast` already does. A
   rebuild at the default that mismatches the stored scale resurrects BUG-0007.

See the ATSW `rescaling-architecture` notes (fue/ART BUG-0004/0007/0008) for the
failure modes and why the mean/drift term is where a mis-scale explodes the level.

---

## 6. Dependencies

`drvarma[mcp]` → `mcp` (FastMCP) + `art-tseries` (univariate seeding) + `fue`
(pulled by art) + drvarma itself (+ its `[plots]` / `[forecast-report]` extras for
charts and HTML). Pure-Python installable; the C engine remains the optional
accelerator.

---

## 7. Roadmap

- **v1** — this document: classic stationary VARMA, art-seeded, guided/autonomous,
  IRF/FEVD/forecast/diagnostics.
- **v2** — cointegration / common-trends awareness: at least *detect and warn*, then
  a VECM path.
- ~~**v3** — transfer functions / VARMA networks~~ — **not this server's**: that is
  `mtram` (the drtran port). See §0.

---

## 8. Open questions

- ~~Entry-point name~~ — **settled: `sima`** (2026-08-02). See §0.
- Order-identification default: how much to automate the (p,q) pick vs always ask
  the analyst to confirm from the CCM/IC evidence (lean: propose + confirm, like ART).
- How aggressively to reuse ART's `.inp`/`.pre` artifacts vs an in-memory hand-off
  for the seeding step.
