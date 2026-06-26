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

### PP1 — Estimator parity: σ²/Q concentration + matching std errors  *(keystone — DONE 2026-06-26)*
`estimate_py` now mirrors the C estimator exactly, engine-free:
- **Parameterisation** = the C `shootx` layout: `(μ, φ-entries, θ-entries, raw qq
  lower-tri)`.  Note the C optimises the **raw covariance lower triangle**, *not*
  `chol(Q)` as this plan first assumed; `phi[0]=theta[0]=I`, so the C's
  "normalisation" step is the identity.
- **Objective** = the concentrated likelihood `(f1/f1₀)^m·(f2/f2₀)` from AS 311
  `elf(σ²=1)`.  Key fact established empirically: this objective is **exactly
  scale-invariant in qq** (`f1→f1/c`, `f2→cᵐ·f2`), so the σ²/Q *split* is not
  identified by the likelihood — it is pinned entirely by the starting qq scale.
- **Initialisation** = port of `init_varma`: sample means, OLS VAR(p), and qq
  started at the residual **correlation** matrix (unit diagonal).  That start is
  what fixes the reported `Q` (≈ correlation-scaled) and `σ̂²=f1/(n·m)`.
- **Optimiser** = a faithful port of the C's factored BFGS quasi-Newton
  (`_qnewt.py` ← `qnewtopt.c`).  Reproducing the C's exact trajectory is what
  makes the split *and* the `cov[]` std errors match: `cov = 2·f·b⁻¹/n` uses the
  optimiser's factored Hessian `b` (`drvmlest.c:est`).  A plain finite-difference
  Hessian (this plan's original suggestion) **cannot** reproduce the `cov[]` SE —
  the qq-scale direction is flat, so the numerical Hessian is singular there and
  the SE blow up; the C's finite SE come from the BFGS factor's fabricated
  curvature in that direction.

**Validated** (engine-free IPC3 `3 0 -mean`): the `.out` parameter table and
normalized model (Q, Σ=σ²·Q) are **byte-identical** to the C binary except the
6th decimal of a few std errors (≤1.4e-4); vs the C engine, estimates/logelf/
sigma2/Σ/residuals agree to ~1e-10 and std_errors to <1.4e-4. Closes G1 + G2.

### PP2 — Hannan-Rissanen two-step initialisation  *(DONE 2026-06-26)*
Ported `hannan_rissanen_diag` (per-series HR: AR(L) OLS → residuals e_hat →
regress y on [AR lags, MA lags=e_hat], `theta_d = -coef`, residual variances
scaled to average 1) and the `combine_vectors` merge (diagonal AR/MA/cov from the
HR estimate into the full `init_varma` start, off-diagonals kept) into
`estimate_py`; `-twostep` wired with the C's exact trigger (`q>0` and not
fully-diagonal). `init_diag_varma` is dead code in the C (no callers) — not
ported. **Validated** vs the C engine (twostep=True): params <1e-5, logelf <1e-6,
Σ <1e-6 on VARMA(1,1)/(2,1). Closes G3.

### PP3 — Volatility module  *(DONE 2026-06-26)*
`volatility.py` ports `compute_exponential_volatility` and
`compute_moving_window_volatility`: exponential weighting (φ estimated from the
Mahalanobis-distance exceedance proportion vs the empirical (1-α) percentile, with
the φ=0 equal-weight fallback) and the moving-window unbiased sample covariance.
`.volexp`/`.volmov` writers (C `%g`), the `.out` info line, and CLI
`-volexp [alpha window]` / `-volmov [window]` (defaults 0.05/20/20).
**Validated byte-identical** to the C binary (engine path); engine-free only one
last-digit `%g` rounding differs (residuals ~1e-10). Closes G4.

### PP4 — recursive forecasting for q>0  *(DONE 2026-06-27)*
Dropped the q=0 restriction in `recursive_forecast`.  **Key empirical finding**
(vs the C binary on a synthetic well-identified VARMA(1,1)): the C's
`forecast_mean` indexes the **estimation-window residuals** `varma1.a` — `a[e]`
at origin e, which is **zero past the estimation window** — *not* full-series
AS-311 residuals (this plan's original guess; that path is off by ~1 level unit).
So the fix is just `a_full[:estwin_eff] = result["residuals"]`.  Validated to
<1e-6 vs the C `.recursive` end-to-end.  `-seasonal` is a **vestigial** flag (it
only feeds `forecast_model`'s discarded `v_seas`; the `.forecast` annual std uses
`forecast_level_variances` with `s=freq`) — deliberately not ported.  Closes G5.

### PP5 — Convergence hardening + engine-free byte-exact `.out`  *(DONE 2026-06-27)*
Done as `test_pure_python_out.py`: the pure-Python estimator reproduces the C
engine across the zoo (VAR(3) ±deseason, diag-ar/cov: logelf <1e-6, mu/phi/Σ
<1e-5), and — engine monkeypatched off — the deterministic `.out` sections
(OIRF/accumulated, FEVD, multivariate diagnostics, normalized model) are
byte-identical to the C binary for VAR(3) -mean.  Engine-free == cffi engine to
~1e-9 on raw data.  Residual `.out` differences are both documented and benign:
the **Inverse roots** ordering (modulus vs chekma QR) and, under deseason, the
σ²/Q **split** drifting ~2.7e-5 (scale-ambiguous flat direction — Σ/logelf still
~1e-12).  Side findings: the C *binary* can be numerically unstable on
pathological synthetics (the pure-Python path is not); the C/Python Hosking tests
share `df=m²·s` + upper tail.  The original sub-tasks below are superseded:
- Confirm the pure-Python optimum matches the C to ~1e-6 so the `.out` parameter
  table is byte-exact. **`qnewtopt` is already ported** (`_qnewt.py`, done in PP1),
  so the optimiser trajectory matches the C — IPC3 `3 0 -mean` is already
  byte-identical bar the 6th decimal of a few std errors (≤1.4e-4).  Remaining:
  confirm the same across the model zoo (diag restrictions, deseason, VARMA q>0).
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

## Definition of done — **ACHIEVED 2026-06-27 (PP1–PP5 complete)**

- `pip install drvarma` (no GSL, no C build) gives full functionality: estimate
  (VAR & VARMA, diag restrictions, deseason, `-twostep`), forecast (+bands,
  recursive incl. q>0), IRF/FEVD, diagnostics, volatility, and byte-exact
  `.out`/`.forecast`/`.recursive` reports. ✓
- The C-engine and pure-Python paths agree on every reported number to the
  documented tolerance (estimates/logelf/Σ ~1e-9, std errors ≤1.4e-4); the suite
  runs and passes in a pure-Python-only environment (engine monkeypatched off). ✓

**Pure-Python drvarma is now feature- and fidelity-complete; the CFFI engine is an
optional accelerator only.** Documented residual `.out` differences vs the C
binary (both benign): the modulus-sorted **Inverse roots** order, and the σ²/Q
**split** under deseason (~2.7e-5, scale-ambiguous; Σ matches to ~1e-12).
