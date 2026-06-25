# Plan: a 100% pure-Python drvarma

Goal: the **pure-Python path is feature- and fidelity-complete** versus the C
engine, so the compiled CFFI engine becomes a *purely optional accelerator* — no
feature and no output fidelity depends on it. Written 2026-06-25.

**Out of scope (this port):** Shea (`multshea.c` / `marma`). It is not wired into
the C estimator (`marma` has no callers), so there is no C reference to validate a
Python port against — deferred, not part of "100% Python". See `docs/FUE_REUSE.md`.

## Where we are

Already pure-Python and validated (no engine needed): `.inp` I/O, transform,
deseason, **exact VARMA likelihood** (Mauricio AS 311, `_as311.py`), VARMA
estimation (`estimate_py`), forecasting (+bands, recursive q=0), IRF/FEVD,
multivariate + per-series diagnostics, and the full `.out`/`.forecast`/`.recursive`
reports (residual section byte-exact). `_engine.estimate_w` already falls back to
pure Python when the extension is absent, and the whole `Model → forecast → report`
pipeline runs engine-free (`tests/test_estimate_py.py::test_full_pipeline_without_engine`).

## Gaps to close for 100% Python

The pure-Python **estimator** differs from the C in three respects, plus two C
features are not ported. These are the only blockers.

| Gap | Impact when running engine-free |
|-----|---------------------------------|
| G1 σ²/Q split | fallback reports `sigma2=1, sigma=Σ`; the C reports `Σ = σ²·Q` with a normalised `Q`, so `.out` cov[] params, "Q matrix" and "Sigma = σ²·Q" differ |
| G2 std errors | numerical Hessian of the full-Σ objective vs the C's `fdhess` of the concentrated objective → SE/t/p and Wald χ² differ more than the engine path |
| G3 HR two-step | `-twostep` not implemented in pure Python (θ=0 start) → weakly-identified VARMA may hit local optima |
| G4 volatility | `-volexp`/`-volmov` (`volatility.c`) not ported at all |
| G5 recursive q>0 | `recursive_forecast` supports q=0 only |

Key realisation: the C's standard errors come from **`fdhess`** (finite-difference
Hessian, `qnewtopt.c`) of its *concentrated* objective in the
`(μ, φ, θ, chol(Q))` parameterisation with `σ̂² = f1/(n·m)` and `Σ = σ²·Q`
(`drvarma_api.c` line ~1031). So matching the C is about **parameterisation +
concentration**, not analytic derivatives — G1 and G2 are one piece of work.

## Phases

### PP1 — Estimator parity: σ²/Q concentration + matching std errors  *(keystone)*
Reparameterise `estimate_py` to the C's packing and concentration:
- optimise over `(μ, φ, θ, chol(Q))` (not full Σ), with `σ̂² = f1/(n·m)`
  concentrated each step; `elf_varma` already accepts `(qq, sigma2)`.
- pin the Q-scale exactly as the C does (verify `Q` reproduces the C, e.g.
  `cov[1,1]=1.237988` and `sigma2=0.05129` on IPC3 `3 0 -mean -deseason auto`).
- compute cov/SE from a finite-difference Hessian of that concentrated objective
  (port of `fdhess`) → SE/t/p and Wald χ² match the C.
- result dict then reports the C's `sigma2`, `Q` (=`sigma`/`sigma2`), `std_errors`.

**Validates:** pure-Python `.out` parameter table (cov[]), normalized model
(Q, Σ=σ²·Q), `sigma2` and SE/t/p match the C `.out` to the documented tolerance,
engine-free. Closes G1 + G2.
Effort: medium-high. Unblocks a complete `.out` with no engine.

### PP2 — Hannan-Rissanen two-step initialisation
Port `init_varma` (OLS AR seed), `hannan_rissanen_diag` and `combine_vectors`
from `drvarma.c`; wire `-twostep` through `estimate_py`.
**Validates:** `-twostep` VARMA fits converge to the C optimum from the HR seed,
engine-free. Closes G3. Effort: medium.

### PP3 — Volatility module
Port `volatility.c` (`compute_exponential_volatility`,
`compute_moving_window_volatility`) → `volatility.py` + a `.volatility` report
writer + CLI `-volexp [alpha window]` / `-volmov [window]`.
**Validates:** `.volatility` output matches the C reference files. Closes G4.
Effort: medium.

### PP4 — recursive forecasting for q>0
Compute exact residuals at fixed parameters over the full series (AS-311 `cres`
at the frozen estimate), then forecast from each origin; drop the q=0 restriction
in `recursive_forecast`. Also expose the minor `-seasonal` flag.
**Validates:** recursive output for a VARMA(p, q>0) model vs the C. Closes G5.
Effort: medium (low priority).

### PP5 — Convergence hardening + engine-free byte-exact `.out`
- Confirm the pure-Python optimum matches the C to ~1e-6 so the `.out` parameter
  table is byte-exact (tighten the scipy tolerances; if needed, port `qnewtopt`'s
  quasi-Newton/line-search to remove optimiser-dependent drift).
- Add a test mode that forces the pure-Python path (engine monkeypatched off) and
  diffs the **full** `.out` (deterministic sections) byte-exact against the C, for
  VAR and VARMA, with and without deseason.
- Document pure-Python as feature/fidelity-complete; the C engine is optional.
Effort: medium. Depends on PP1–PP4.

## Ordering & dependencies

```
PP1 (keystone) ──► PP5 (validation)
PP2 ──────────────►
PP3 (independent) ►
PP4 (independent) ►
```
Recommended order: **PP1 → PP2 → PP3 → PP4 → PP5**. PP1 delivers the biggest jump
(a complete, faithful `.out` with no engine); PP3/PP4 are independent and can be
interleaved. After PP5, drvarma is **100% Python**: the CFFI engine is an optional
speed-up only.

## Definition of done

- `pip install drvarma` (no GSL, no C build) gives full functionality: estimate
  (VAR & VARMA, diag restrictions, deseason, `-twostep`), forecast (+bands,
  recursive incl. q>0), IRF/FEVD, diagnostics, volatility, and byte-exact
  `.out`/`.forecast`/`.recursive` reports.
- The C-engine and pure-Python paths agree on every reported number to the
  documented tolerance; CI runs the suite in a pure-Python-only environment.
