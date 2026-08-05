# drvarma Python port — TODO

Status snapshot in `docs/STATUS.md`. P0–P4 and most of P5 (CLI, packaging,
per-series diagnostics, base plots) are done and validated against the C engine
(94 tests). The whole Model → forecast → report pipeline also runs on the
pure-Python fallback with no compiled engine. Remaining: the graphics finish
(pyfug JT formats, deferred to last), P5 docs/CI, and the deferred Shea backup.

## BUGS

- [x] **FIXED (deseason, was CRITICAL) — phase off-by-one in `start_sub`:
      `deseason="auto"` makes seasonality WORSE for any series not starting at
      subperiod 1, silently.** Found 2026-07-28 in the oil pass-through exercise.
      Repro: `bench/repro_multiart_passthrough.py` (BUG 4).
      Series start in **February**, and `mcp_server._prepared_w` correctly passes
      `start_sub=ms.start[1]=2` to `deseasonalize_raw`. Lag-12 autocorrelation of
      `dlog(CPI)` after deseasonalising, vs not deseasonalising at all:

      | country | none | start_sub=2 (used) | start_sub=1 | monthly-dummy OLS |
      |---|---|---|---|---|
      | US | +0.328 | +0.245 | **-0.140** | -0.137 |
      | ES | +0.798 | **+0.866** | **+0.098** | +0.056 |
      | FR | +0.728 | **+0.862** | **+0.250** | +0.201 |
      | DE | +0.608 | **+0.808** | **+0.177** | +0.163 |

      Sweeping `start_sub` over 1..12 puts the minimum |ACF(12)| at **1 for all four
      countries**, where the output matches a plain monthly-dummy OLS regression to
      ~0.04. So the harmonic algorithm itself is CORRECT — the phase convention
      between caller and callee is off by one. At the phase actually used the
      adjustment *adds* seasonal variance (it subtracts the pattern in the wrong
      month), which is why ES/FR/DE get worse than doing nothing.
      **Impact: contaminates the entire pipeline** — characterization seed, CCM,
      Tiao-Box, order search, estimation, IRF, FEVD — with no warning. In this
      exercise it produced residual ACF(12) ≈ 0.7 in the CPI equation, an implausible
      AR(1) persistence of -0.707 for DE inflation and a -12.7 coefficient on lagged
      CPI in the US WTI equation, all of which looked like modelling problems rather
      than a transform bug.
      **Root cause:** the amplitudes are estimated by `harmonic_regression_differenced`
      → `_harmonic_design(n, d, s)`, which takes NO `start_sub` and builds harmonics
      from `t = i + d + 1` — i.e. the dummies come out indexed by OFFSET FROM THE
      START of the series. They were then applied in ABSOLUTE subperiod phase,
      `(np.arange(nobs) + start_sub - 1) % s`. The two agree only at `start_sub == 1`.
      **FIX APPLIED** (`deseason.py`): rotate the estimated dummies into absolute
      subperiod indexing before storing/applying them —
      `level[(np.arange(s) + start_sub - 1) % s] = level_rel`. Identity at
      `start_sub == 1`, so C-parity goldens are untouched.
      Verified two ways: (a) synthetic series with a known pattern started at each of
      the 12 subperiods now recovers it to <0.1 and drives seasonal sd to ~0.001;
      (b) on real IPC data, estimating from January vs from February changed the
      pattern by 36–83 % of its amplitude before the fix (the vector came out
      circularly shifted by one month) and by <1 % after.
      Regression tests: `tests/test_regression_bugs.py` (36 phase tests +
      `test_seasonal_pattern_invariant_to_dropping_leading_observations`).
      **NB — the same defect exists in the C** (`drvarma_v.04.1/src/deseason.c:82-90`,
      reached from `src/drvarma.c:356`, so the main executable is affected). Declared
      in `drvarma_v.04.1/BUGS.md` with the equivalent C patch; NOT fixed there yet.
      Historical impact is limited because `start_sub` defaults to 1 and most series
      start in January — the published pass-through note is unaffected.
      Secondary (separate, lower priority): the seasonal component is estimated on the
      **simple difference of levels** and subtracted in levels, *before* the Box-Cox
      log in `transform`. For a multiplicative seasonal pattern on a trending index
      (ES CPI runs 69→98) an additive level adjustment is the wrong scale; consider
      deseasonalising after the Box-Cox transform.
- [x] **FIXED (report + Model, was HIGH) — the convergence banner was driven by
      `ifault`, not by the optimiser's termination code, and dropped the criterion
      and the iteration count.** Found 2026-07-28.
      The C (`qnewtopt.c:148-153`) prints
      `OPTIMIZER CONVERGED|STOPPED after <k> iterations` — CONVERGED **iff
      `termcode in (1,2)`** — plus `Convergence criterion:` chosen from termcode
      1..5. The port's `_convergence_block` instead did
      `"CONVERGED" if r["ifault"] == 0 else "FAILED"`, omitted `nit`, and printed
      the `termcode == 1` criterion **unconditionally**. Its docstring claimed
      `termcode`/`nit` were "not exposed by the CFFI result", but both ARE present
      in `result` on the pure-Python path.
      `ifault` is MODEL adequacy, not convergence (`estimate_py.py:327-332` says so
      explicitly), so a run that stopped short of an optimum but yielded a formally
      adequate model was labelled "OPTIMIZER CONVERGED" with a fabricated reason.
      **Why it matters (per David):** in multivariate VARMA the *reason* for
      stopping is a first-order diagnostic — ill-conditioned likelihoods, near
      non-identification and common factors show up as termination on `steptol`
      rather than on the gradient. Measured here: the four final VAR(1)/VAR(2) fits
      all stop on **termcode=1 (gradtol)** — genuine convergence — while every
      VARMA(3,2) stops on **termcode=3** ("last global step failed to locate a lower
      point") **with ifault=0**, i.e. non-converged fits were silently entering the
      order-search ranking.
      **Fixes applied:** (a) `report._convergence_block` mirrors the C, reports the
      real criterion and the iteration count, warns explicitly on termcode 2
      (steptol → suspect ill-conditioning, distrust the standard errors) and on
      3/4/5 (not a convergence at all); (b) `Model` now exposes `termcode`,
      `nit` and `converged`; (c) `mcp_server.confirm_and_estimate` reports the
      convergence diagnosis, and `identify_varma_order` **rejects non-converged
      fits** from the ranking instead of ranking them.
      Also handled: **termcode 0**, which is not a termination code at all (in the C
      it is the "keep iterating" state). It means the optimiser never ran — no free
      parameters, or the initial evaluation failed — so the reported values are the
      STARTING values, not estimates. Now reported as `OPTIMIZER NOT RUN`.
      Tests: `test_convergence_is_reported_from_termcode_not_ifault`,
      `test_optimizer_not_run_is_reported_as_such`.
      Open follow-up: `estimate_py.py:329-331` argues termcode 3 "means it is AT the
      optimum, so it is not a fault". The C disagrees (it prints STOPPED). Worth
      settling, since termcode 3 also arises from a bad search direction under
      ill-conditioning, not only from sitting at the optimum.
- [x] **FIXED (pure-Python, era HIGH) — `elf` rechazaba VARMA estacionarios cuando
      Φ_p es singular, que es lo normal en un VARMA NO BALANCEADO.**
      **CAUSA RAÍZ: un fallo de PORT.** `_chol_lower` usaba
      `np.linalg.cholesky`, que es estricta, donde el C usa la Cholesky
      **MODIFICADA** de Gill-Murray-Wright (`nlatools.c:choldcp`, Dennis &
      Schnabel A5.5.2). La diferencia es exactamente el caso semidefinido: el C
      falla sólo si un pivote es negativo **y** `|sum1| > sqrt(macheps)·maxoffl`;
      si el pivote es cero o demasiado pequeño lo **sustituye** por `minljj` y
      sigue. numpy aborta con cualquier matriz no estrictamente PD.
      **Arreglo:** `_chol_lower` es ahora un puerto fiel de `choldcp`.
      **Verificación:** el cast empotrado de drtran (que produce Φ_p singular por
      construcción) homologa con el binario C a ~1e-7 en las cuatro
      combinaciones probadas: (0,0,0) −736.774158, (0,1,0) −721.801539,
      (0,0,1) −718.287406, (1,1,1) −756.602851.
      **Efecto colateral: cerró los TRES fallos que arrastraba la suite.**
      `test_deseason_params_and_forecast_match_c`,
      `test_out_deterministic_sections_byte_exact[marker_pair4]` y
      `test_volexp_volmov_byte_exact` eran los tres tests de paridad con el C, y
      los tres fallaban por esta misma desviación numérica. La suite pasa de
      192/3 a **195 passed, 0 failed**.
      Tests: `test_elf_accepts_stationary_varma_with_singular_phi_p`,
      `test_chol_lower_accepts_semidefinite_and_rejects_indefinite`.
      Descripción original del síntoma:
      Encontrado 2026-07-29 portando el cast empotrado de drtran.
      Repro: `bench/repro_phi_p_singular.py`.
      Un VARMA cuyas ecuaciones tienen órdenes distintos tiene ceros en los
      retardos altos de las filas de menor orden ⇒ Φ_p de rango deficiente. Es la
      situación normal de: (a) el cast EMPOTRADO de drtran
      (`build_embedded_varma`), donde el orden de la fila i es deg(φ_i·D_i); y
      (b) cualquier forma ECHELON con índices de Kronecker desiguales.
      `_elf_f1f2` devuelve **ifault=3** ("non-stationary") sobre estos modelos
      aunque todas las raíces de det(Φ(B)) estén fuera del círculo unidad:

      | Φ₂ | raíz mín. | ifault |
      |---|---|---|
      | `[[-0.12,0],[0,0]]` (fila nula) | 2.500 | **3** |
      | `[[-0.12,0],[0,-0.05]]` | 2.500 | 0 |
      | `[[-0.12,0],[0,1e-8]]` | 2.500 | **0** |
      | `[[-0.12,0],[0,0.0]]` | 2.500 | **3** |

      Perturbar el cero con 1e-8 lo arregla: no es degradación numérica, es un
      test que falla exactamente en el caso singular. Origen: `_as311.cgamma`
      («ifault: 1 if the Yule-Walker system is singular»), que `elf` traduce a 3.
      **ES DEL PORT, NO DEL C** (comprobado): el `drtran` compilado estima esos
      mismos modelos con el cast empotrado —
      `-b 0 -r 1 -s 0 -V` → −721.801539, `-b 1 -r 1 -s 1 -V` → −756.602851 —
      mientras la ruta pure-Python devuelve ifault=3 sobre la misma estructura.
      Impacto: bloquea el puerto del cast empotrado de drtran, que es el cast por
      DEFECTO del C, y cierra la puerta a la forma echelon. El cast por resta no
      está afectado (validado contra el C a 1e-7 en 4 combinaciones de b/r/s).
- [ ] **BUG (C engine, HIGH) — `double free or corruption` on the deseason+VARMA
      path.** Surfaced building the multiart MCP (2026-07-28). Repro: 2-variate
      seasonal series → `Model(..., deseason="auto").fit()` via the compiled engine
      crashes the process. The pure-Python estimator runs the same call fine.
      **Workaround shipped:** `_engine.py` now honours a runtime `DRVARMA_NO_ENGINE`
      env var (force pure-Python without rebuilding); the multiart MCP is registered
      with it. Needs a proper fix in the C engine (likely a free of a
      deseason/dummy buffer, or an ownership bug in the cast when levels were
      deseasonalized upstream). Until fixed, the C engine must not be the default
      for the deseason path.
- [x] **FIXED in the order search (pure-Python numerics still open, HIGH) — `inf` log-likelihood for some
      MA-bearing specs with deseason, and it drives the order *recommendation*.**
      Raised from MEDIUM after the oil pass-through exercise (2026-07-28).
      Repro: `bench/repro_multiart_passthrough.py` (BUG 2), real data, 4 independent
      bivariate datasets (WTI + CPI for US/ES/FR/DE, monthly 2002:02–2019:12, n=215).
      New facts vs the original filing:
      * **Systematic, not data-specific.** Exactly (0,1), (0,2) and (1,2) blow up on
        all four datasets — identical set every time.
      * **Root cause located:** `estimate_py.py:337-338`,
        `logelf = ... - 0.5*n*(m*np.log(f1) + np.log(f2))`. When `f1` or `f2` → 0 the
        log gives `-inf` and the leading minus flips it to `+inf` (hence the
        `RuntimeWarning: divide by zero encountered in log`). It is a sign trap, so
        the bad value sorts *best* under every information criterion.
      * **The diagnosis already exists and is correct:** those fits carry
        **`ifault=3`** (non-stationary). Nothing propagates it.
      * **It corrupts the recommendation, not just the ranking.**
        `mcp_server.identify_varma_order` filters with `r[4] is not None`, which only
        drops raised exceptions; non-finite entries survive, sort first, and `ok[0]`
        is emitted as "Recomendación (mín. BIC): **VARMA(0,1)**" — on all four
        countries. A pure MA(1) for monthly inflation is indefensible and contradicts
        the CCM evidence, so in AUTONOMOUS mode this silently yields a wrong model.
      * **Fix (two lines, independent of the numerical fix):** in
        `identify_varma_order`, keep only `np.isfinite(ll)` **and** `ifault == 0`;
        report the discarded specs instead of hiding them. Then guard the
        degenerate factorisation in the pure-Python MA path.
- [x] **RESOLVED — it was a CONSEQUENCE of the deseason phase bug — exact
      log-likelihood not monotone in `p`: a nested model fitted *better* than the
      larger one, with `ifault=0`.**
      **Update after fixing the phase bug:** the same DE dataset (still starting in
      February, i.e. the case that used to fail) now gives a monotone sequence
      VAR(1) −752.74 → VAR(2) −745.76 → VAR(3) −743.30. The mis-phased
      deseasonalisation was injecting spurious structure that made the optimiser
      fail on the larger model; with the correct phase the failure disappears. No
      separate optimiser bug. The nesting sanity check added to
      `identify_varma_order` stays as a guard, and non-converged fits are now
      rejected outright via `termcode` (see the convergence entry above).
      Original observation, kept for the record:
      Found 2026-07-28, same exercise. Repro: `bench/repro_multiart_passthrough.py`
      (BUG 3). For the DE dataset (WTI + IPC_DE, λ=0, d=1, deseason="auto"):
      `VAR(1) = -865.54`, `VAR(2) = -859.29`, **`VAR(3) = -1032.49`**. VAR(1) is
      nested in VAR(3), so the maximised likelihood cannot decrease — this is an
      optimiser convergence failure reported as a valid fit. US/ES/FR are monotone
      on the same call, so it is specific to this fit, not a systematic formula bug.
      Worse than the `inf` case: the bad fit reports **`ifault=0`**, so there is no
      existing flag to propagate, and `termcode`/`nit` (computed in `estimate_py`,
      see the comment at lines 327-332) are **not exposed as attributes of the fitted
      `Model`** — a caller has no way to detect it. It silently corrupts every
      information criterion derived from it: in the order search DE's VAR(3) row
      showed AIC/BIC/HQ ≈ 2099/2156/2122 against ≈1745/1788/1762 for VAR(2).
      Suggested: expose `termcode`/`nit` on the fit; add a cheap nesting sanity check
      (or a multi-start / better initialisation) in the order search.
- [x] **FIXED (MCP) — `load_data` silently dropped observation 1 of a
      header-less numeric CSV, and ignores the header row when there is one.**
      Found 2026-07-28. Repro: `bench/repro_multiart_passthrough.py` (BUG 1).
      `mcp_server.py:172-177` applies `np.genfromtxt(path, delimiter=",",
      skip_header=1)` unconditionally and only retries without the skip when the
      result is **all** NaN. A purely numeric CSV with no header parses fine after
      skipping, so the retry never fires: `US_levels.csv` (215 rows) loads as 214
      obs starting at the *second* row, with no warning. In the pass-through
      exercise that silently changed the estimation sample (the first differenced
      observation is lost) — exactly the class of error that invalidates a
      published result without ever looking wrong.
      Second defect, same function: the CSV branch never reads column names from the
      header (only the Excel branch does, lines 168-169), so a well-formed CSV loads
      as `['y1', 'y2']` unless `series_names` is passed.
      Fix: sniff the first line (try `float()` on its fields) to decide header vs
      no-header, and take `names` from it when it is a header.

- [x] **FIXED (MCP) — the unit-root search was not capped at d=1, so a seasonal
      series could be over-differenced to d=2.** Found 2026-07-28.
      ART's identification order is λ → d → seasonality *by design*: d is chosen
      before seasonality is handled. ADF/KPSS have low power against a seasonal
      series, so they can fail to reject at d=1 and escalate to d=2 spuriously —
      over-differencing an I(1) series and injecting a spurious MA unit root (the
      same failure mode `recommended_d` already documents as BUG-0002 for KPSS).
      `characterize_series` called `unit_root_tests(ts, lam=lam)` with the default
      `max_d=2`. Measured on IPC_ES (seasonal R²=0.79), ADF is right at the margin:

      | sample | d=0 | d=1 | d=2 | recommended_d |
      |---|---|---|---|---|
      | from January (n=216) | no reject | **no reject** | reject | **2** |
      | from February (n=215) | no reject | **reject** | reject | 1 |
      | deseasonalised | reject | reject | reject | 0 |

      One observation flips the order of integration — the signature of a test with
      no power, not of a genuine I(2) series. **Fix applied:** pass `max_d=1`, so the
      search only goes d=0 → d=1 and the seasonality step handles the rest. All five
      series (WTI, CPI_USA, IPC_ES/FR/DE) then give a stable d=1 on both samples.
      Regression test: `test_characterize_d_stable_to_one_extra_observation`.
      NB: this is a cap in the multiart seed only — `recommended_d` itself behaves as
      specified and was not changed.

## multiart MCP — design gaps (not bugs, but they hid the bugs above)

Found while running the oil pass-through exercise end to end (2026-07-28). Each of
these is a reason a wrong result went unnoticed.

- [x] **FIXED — `confirm_and_estimate` reported only Φ₁ and Σ.** No Φ₂..Φ_p, no Θ, and no
      standard errors — although `Model.std_errors` exists. For a VARMA(3,1) the user
      sees 4 of 16 AR/MA coefficients and cannot judge significance at all. The
      published note's central table is a *t*-ratio table (pass-through coefficient
      and its stars), which multiart currently cannot produce. Emit all Φ_k, Θ_k,
      std errors and t-ratios.
- [x] **FIXED — `diagnose` had no power against seasonal residual autocorrelation.**
      It reports only an aggregated Hosking Q(14) with df=56 plus Jarque-Bera. On the
      US fit it returned "sin autocorrelación cruzada ✓, Modelo adecuado" (p=0.447)
      while the CPI residuals had ACF(12)=+0.27 and ACF(24)=+0.26, both well outside
      2/√n. A single spike at the seasonal lag is diluted by 56 degrees of freedom.
      Report the residual ACF lag by lag (at least s and 2s), or add a seasonal-lag
      Ljung-Box, and never print "Modelo adecuado" on the aggregate test alone.
- [ ] **Nothing verifies that deseasonalisation worked.** `characterize_series`
      detects seasonality and sets `deseason=auto`, but no step checks the result. A
      one-line post-condition — ACF(s) of the prepared series must drop in absolute
      value versus not adjusting — would have caught the CRITICAL phase bug above
      immediately, on the first run.
- [ ] **`characterize_series` reports seasonality as a yes/no.** No F statistic, no
      seasonal R², no amplitude. Here the seasonal component was 40–79 % of the
      variance of monthly inflation (ES R²=0.79, amplitude 2.05 pp) — a first-order
      feature of the data that the summary table renders as "sí".
- [ ] **The seed's (p,q) ceiling is too tight to be useful.** The consensus saved for
      these datasets was `(p,q)≤(0,1)`, and `identify_varma_order` treats
      `p_max=0`/`q_max=0` as "use the seed", so the DEFAULT call searches VMA(1) only
      — it cannot even reach the VAR(1)/VAR(2) of the published note. Combined with
      the `inf` bug this makes "VARMA(0,1)" the default answer twice over.
- [ ] **No access to residuals or fitted parameters through the MCP surface.** Every
      cross-check in this exercise (residual ACF, OLS arbitration, reproducing the
      published table) had to bypass multiart and drive `drvarma` directly. Expose
      residuals and the full parameter vector.
- [ ] **IRF/FEVD come without confidence bands**, so there is no way to tell a
      pass-through share of 5 % from one of 26 % in terms of significance.
- [ ] **The cointegration warning promised in the server instructions never fired.**
      The instructions say to warn when series look I(1) and move together; loading
      four I(1) level pairs produced no such notice.

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
- [ ] **Binary (compiled-engine) wheels — cibuildwheel, next release.** Today PyPI
      ships only the `py3-none-any` pure-Python wheel + sdist, so `pip install
      drvarma` installs cleanly everywhere (Windows too) but runs the **slow**
      pure-Python path; the ~10-100× C engine needs GSL + a compiler
      (`drvarma[c-engine]` / sdist build). Replicate **fue-python's `wheels.yml`**
      (cibuildwheel: cp310-313 × Win/macOS-arm64/Linux manylinux+musllinux
      x86_64+aarch64, GSL bundled) so the fast engine ships out-of-the-box, matching
      fue. Not an install *fix* (drvarma already installs) — a **speed** parity item.
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
- [x] Developer guide + performance study (done 2026-06-27): `docs/DEVELOPER_GUIDE.md`
      — three paths (pure-Python / hybrid-CFFI / pure-C), complexity from the
      literature (`../literature`: Mauricio 1995 JASA / 1997 AS 311 / 2002 JTSA),
      and a reproducible benchmark battery (`bench/benchmark.py` → `results.json`):
      hybrid is 13–100× faster than pure-Python; well-conditioned cross-path
      agreement ~1e-9..1e-13; ill-conditioning (var_disparity cond≈1e8, 226 BFGS
      iters) degrades agreement to ~6e-4 and explains the WTI/IPC caveat.
- [x] User guide (done 2026-06-27): `docs/USER_GUIDE.md` — install, quick start,
      data I/O (.inp + arrays), model spec, fitting/accessors, forecasting
      (+bands, recursive), diagnostics/IRF/FEVD, volatility, reports + CLI, plots,
      troubleshooting (ifault codes, ill-conditioning). All snippets run-verified.
- [x] PyPI release (done 2026-06-28): **drvarma 0.1.0 published to PyPI**
      (<https://pypi.org/project/drvarma/0.1.0/>) — sdist + pure-Python wheel,
      twine check PASSED, install-from-PyPI verified. Metadata polished (SPDX
      licence, classifiers, URLs, author David E. Guerrero); README is the PyPI
      landing page (Features + "Numerical methods" table + honest contribution
      note); `RELEASING.md` + `.github/workflows/publish.yml` (tag-triggered,
      Trusted Publishing). Annotated tag `v0.1.0` created locally.
- [ ] **PENDING — git remote + push.** This repo has **no git remote** (and `gh`
      CLI is absent here). Decide own-repo vs the C-engine repo, then
      `git remote add origin <URL>` and `git push -u origin master --tags`.
      Note: pushing the `v0.1.0` tag triggers `publish.yml`, which will re-attempt
      the 0.1.0 upload and fail harmlessly (already on PyPI). For trusted
      publishing on future tags, configure the GitHub publisher on PyPI
      (project → Settings → Publishing; workflow `publish.yml`, environment `pypi`).
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
- [ ] **`_qnewt` hardcodes the typical parameter size to 1 (Dennis & Schnabel's
      `typx`). STUDIED IN DEPTH AND DELIBERATELY NOT CHANGED (2026-08-04/05).**
      **Full study: `drtran/docs/OPTIMIZER_STOPPING_STUDY.md` — read it before
      touching the stopping criteria.** What matters here:
      * `umstop`'s `max1 = |g|*(|x|+1)/(|f|+1)` becomes an absolute gradient
        tolerance once the parameters are far below 1, and `max2 = |dx|/(|x|+1)` an
        absolute step tolerance. Same four lines in the C (`qnewtopt.c:185,208,215`).
        The optimiser still reaches the optimum; it loses the ability to certify it.
      * Making the tests relative was implemented three ways (adaptive floor,
        norm-relative floor, and D&S's fixed vector) and **all three were rejected
        on measurements**. It buys iterations (`var_disparity` 166 -> 110) but stops
        EARLIER, and the extra depth the historical test forces is what makes two
        rescalings of the same ill-conditioned problem agree:
        `max|phi(scale=100) - phi(scale=25)|` on the WTI/IPC pass-through goes from
        ~1e-5 to 3.5e-4 (ES) and 3.7e-3 (FR), against a 1e-4 tolerance. It broke
        `test_passthrough_point_estimates_scale_invariant` plus four byte-exact
        output comparisons. It also ended 2.55e-04 WORSE in log-likelihood on
        `var_disparity` — the saved iterations are not free.
      * **sima does NOT have the runaway** drtran had: nothing reaches maxits in any
        bench regime, because its parameters are seeded from the data at O(0.1-1)
        and its flat directions are common factors where the line search fails
        cleanly (termcode 3). drtran's escape route was a covariance seeded at ZERO
        on an unbounded ridge.
      * Separate negative result: over-parameterised VARMA fits stop on termcode 2/3
        with AND without the change (VARMA(2,1): 2->3; VARMA(3,2): 3 both ways, same
        likelihood). So the "every VARMA(3,2) stops on termcode 3" observation above
        is weak identification, not this defect, and the open question at
        `estimate_py.py:329-331` is untouched by this study.
      * The most promising UNTRIED variant is a **per-parameter-class typx**
        (AR/MA/transfer/covariances -> 1; mu and the deterministic omegas -> their
        own size, since only those scale with the data). See §8 of the study.
      * Bench note found on the way: the `near_cancellation` regime in
        `bench/benchmark.py:72` is NON-STATIONARY for p >= 2 (phi = [0.6I, 0.6I]
        gives a root at 0.884). The battery only uses it with (1,1) so it does not
        bite today, but a new cell with p=2 would silently get termcode 0.
- [ ] Keep `csrc/internal/` in sync with `../drvarma_v.04.1/src` when the C
      engine changes (they are copies).
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the C.

## Out of scope for this port — Shea (AS 242)

`csrc/internal/multshea.c` (`marma`) is compiled but **not wired into the C
estimator** (no callers), so there is no C reference to validate a Python port
against. **Deferred — not part of the 100% Python goal.** If ever revived: wire
`marma()` into a new C engine version first, compare C-Shea vs C-Mauricio, then
port faithfully (never a Kalman/state-space stand-in — that *is* Shea's route).

## Seasonality detection
- [ ] **Use a HAC F-test in the seasonality detection (superior to the plain OLS
      F).** `deseason.harmonic_regression_differenced` computes the standard OLS
      F, `(ssr/(s-1))/(sse/(n-s))`, which assumes i.i.d. homoskedastic errors. But
      the regression is run on the DIFFERENCED series, whose residuals are
      autocorrelated (differencing induces MA structure; the noise itself may be
      serially correlated), so the OLS F has an incorrect size in general. A
      Newey-West HAC F (Wald on the harmonic coefficients with a Bartlett-kernel
      sandwich covariance, divided by s-1) is robust to that autocorrelation and is
      the more principled test. On the WTI / euro-area IPC / CPI_USA / PCE series
      the correctly-scaled HAC F and this OLS F agree (WTI ~0.95 both, IPCs both
      SEAS, PCE ~1 both), so the two coincide on those cases; but the HAC version
      dominates in general. NB: the correct HAC "meat" is a SUM over t
      (`sum_t x_t x_t' u_t^2 + lags`), NOT an average — dividing by n makes the
      covariance n× too small and the F n× too large (this was a bug in ART's
      `seasonal_detection`, now fixed; see art-python). Port ART's fixed
      `_newey_west_hac` here and offer it as the (default) test.

## Decisions / open questions
- [ ] Publish the Python port to its own GitHub repo? (own remote, CI, PyPI.)
- [ ] Single source of truth for forecasting/diagnostics: numpy (current) vs the
      C (via more API). Numpy keeps the port usable without the C engine (P3 goal).
