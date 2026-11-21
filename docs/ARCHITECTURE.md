# drvarma — architecture of the ladder (C and port)

How the ladder (drvarma 5: one fue file per series) is built, in the C (the
oracle, atsw-gui `engines/drvarma`) and in this port, and what is homologated
between them.

- `docs/LADDER.md`: usage and the parity measured.
- atsw-gui `engines/drvarma/docs/DESIGN-v5-ladder.md`: the 5.0 design.
- `ECHELON_PLAN.md` and the `STUDY-*` files: the echelon form.
- `DEVELOPER_GUIDE.md` and `PURE_PYTHON_PLAN.md`: the deprecated multivariate
  `.inp` path.
- sima's `docs/ARCHITECTURE.md`: the assistant on top and its two routes.

## 1. Two paths

| | the ladder (5.0) | the multivariate `.inp` (deprecated) |
|---|---|---|
| input | one fue `.pre`/`.inp` per series | one drvarma `.inp`, m columns, one λ, d, D |
| the series' model | its own, read and cast by fue (`fue.load`, `fue.cast_us`; C: `lib/fuepre`, `lib/prewhiten`, `lib/fuedet`) | drvarma's transform (`transform.c` / `transform.py`) |
| C | `escalera.c` | `drvarma.c` |
| port | `ladder.py` (`Ladder`, `LadderSeries`) | `model.py`, `estimate_py.py` |
| default start | `ols` with q = 0, zero cross terms with an MA (§5) | `init_varma`: OLS of the VAR, MA at zero; `-twostep` adds a diagonal Hannan-Rissanen |

`-split` (`split.c`, `ladder.split`) turns a multivariate `.inp` into fue files
for the ladder.

**The paths differ with λ = 1.** The `.inp` path transforms with
(y^λ − 1)/λ also at λ = 1, that is y − 1. Its docstrings and guides called
this "the identity", and they are now corrected. The ladder uses fue's
refactor·y. With d = 0 and a mean, a model split from an `.inp` and refitted
on the ladder shows μ shifted by the factor, with the same likelihood. The
`.inp` path is pinned byte for byte by `tests/banco` and deprecated, so it is
documented, not changed.

## 2. The model and its parameters

The series i keeps its univariate model on the diagonal: Box-Cox,
deterministic terms, differences, mean, ARMA factors. The cross orders p and q
add only off-diagonal terms:

    Φ_ii(B) = φ_i(B)Φ_i(B^s)        Θ_ii(B) = θ_i(B)Θ_i(B^s)
    Φ_ij(B) = −Σ_k c_ij,k B^k        Θ_ij(B) = −Σ_k e_ij,k B^k      (i ≠ j)

**The signs are Box and Jenkins'**: Φ(B) = I − Σ Φ_k B^k, so a cross AR
coefficient is +Φ_k[i, j], and an MA one is θ = −(the coefficient on the
lagged innovation).

**The free vector**, in this order (C `esc_pack`, port `Ladder.pack`):
1. each series' free own coefficients: its ARMA factors, unless `-fixarma`;
   the deterministic terms the mask frees (`-det`, `det=`: all of them with
   `-redet`; none by default); its mean if the file estimates it;
2. the free cross slots: AR lags, then MA, then A₀ in the echelon form. Links,
   zeros, rows and the index table decide which are free, and they are named
   `AR1[A<-B]`;
3. log(Q_ii/Q₁₁), for i ≥ 2;
4. q_ij for i > j, unless `-diagcov`.

**Q is normalised**: Σ = σ²·Q, Q₁₁ = 1, and σ² is concentrated (Mauricio
1995, eq. 2.1). The `.inp` path estimates Q unnormalised and holds q₁₁ for
the Hessian instead.

**The own models are re-estimated jointly** with the cross terms (`-fixarma`
keeps them). They start at the univariate optimum: the gate's fit.

## 3. The gate

Before any cross term, each series is fitted alone on the common window, and
the diagonal system is evaluated at those optima. The gate makes two claims:
- **the crossing:** the joint log-likelihood equals the sum of the univariate
  ones, to `GATE_TOL` = 1e-6 relative, or nothing is built on it (C exit
  code 5; port `GateError`);
- **the files:** whether each was an optimum. A file whose coefficients move
  more than `PRE_MOVE` = 1e-3 is a specification, and the report says so.
  A series trimmed to the common window is re-estimated on it.

## 4. The objective and the optimiser

- **The objective** is the concentrated exact likelihood, normalised so that
  f(x₀) = 1. A point the cast or the likelihood rejects returns 1.0. This is
  the article's strategy, in `drvmlest.c` and in the port alike.
- **The optimiser** is the factored BFGS of `qnewtopt.c` (`raxopt`; port
  `_qnewt.py`): 500 iterations, gradtol and steptol 1e-7, typical sizes
  `typx = typf = 1`. **Its termcodes:**
  1. the scaled gradient is small;
  2. the last two steps are close;
  3. the last step found no lower point;
  4. the iteration limit;
  5. five maximal steps.
- **The MA wall:** inverse roots within `MA_WALL_TOL` = 5e-5 of the unit
  circle. A fit there is reported as such, not as an optimum. The engine
  states facts and gives no verdict.
- **Scale.** Because `typx = 1`, a cross coefficient whose natural size is
  far from 1 makes the first steps from zero badly scaled. This happens when
  the series' dispersions differ by orders of magnitude, as with a rate in
  basis points beside prices in %. From zero the search can then stall after
  two iterations (BUG-0027; atsw-gui `docs/ESTUDIO-escala.md`). The optimiser
  is not touched; the start is (§5).

## 5. Starting values

| start | seeds | C | port |
|---|---|---|---|
| `zero` (the default with q > 0, or with no cross AR) | nothing beyond the univariate models | yes | yes |
| `ols` (the default with q = 0) | the original drvarma's `init_varma` in the ladder. Least squares of the VAR(p) on the centred stationary series: cross AR = +Φ_k; Q's covariances = r_ij·√(Q_ii·Q_jj), with the gate's Q_ii; cross MA at zero | `-start ols` | `start="ols"` (API only) |
| `preliminary` | Jenkins and Alavi's preliminary estimates of the cross terms (AR by Yule-Walker, MA from the residuals' cross covariances) | — | API |
| `echelon` | Lütkepohl and Poskitt's stage II least squares: own, cross, A₀, Q (the echelon form's default) | — | API |

- **The own models are never seeded by least squares.** The principle of the
  ladder holds on every route: a `.pre`'s optimum, or the diagonal fit of a
  specification. The echelon form is the exception, because its own terms
  belong to its index table.
- **A seeded start that the likelihood rejects falls back to zero**, and
  `start_used` (or the `.out`) says so.
- **Measured** (BUG-0027, 59 + 54 fits):
  - from zero, elf failed on 2 of 4 monetary VAR(2)s; from `ols` or
    `preliminary`, none;
  - a multivariate Hannan-Rissanen start added nothing, and on Jenkins and
    Alavi's VARMA(2,1) it ended at a worse local maximum.

## 6. The likelihoods

| `lik` | what | where |
|---|---|---|
| `elf` (default) | Mauricio's AS 311. `-m 1`: ξ truncated at 1e-3; `-m 2`: untruncated. Both exact. | C `elfvarma.c`; port `_as311.py` and the compiled `elf_c` |
| `shea` | Shea's AS 242, the independent benchmark | C `lib/lik/multshea.c`; port `_as242.py`, which also runs without the compiled engine |
| `both` | elf, checked against Shea at every point the optimiser visits | C and port |

The residuals reported are always elf's.

## 7. Standard errors and the LR

- **Standard errors:** Mauricio's finite-difference Hessian at the optimum
  (fdhess), the default in both, with `cov = 2F·H⁻¹/n`. If that Hessian is
  not positive definite, the search's BFGS Hessian is used, and the report
  says why (the boundary, a singular Hessian). If the search did not move,
  there is no Hessian at all.
- **The LR of the cross dynamics** is against the diagonal system. Its
  degrees of freedom count the free cross coefficients actually in the model
  (links, zeros, rows: BUG-0017), plus m(m − 1)/2 with a full covariance.

## 8. Homologation

**The C is the oracle.** A fix starts there, and both move together. The
port's ladder tests pin numbers produced by the C: `tests/test_ladder.py`,
with the fixtures copied in `tests/data/ladder`. Some tests run an atsw-gui
binary when `DRVARMA_C_BIN` is set (`test_links.py`, `test_shea.py`). The C
checks the port's numbers in `tests/escalera/pruebas.sh`.

**Homologated** (the parity in `LADDER.md`):

| what | measured |
|---|---|
| the gate | to 1e-13 |
| the logL of the full, VAR(1), pass-through and `-fixarma` models | as in `LADDER.md` |
| 3456 recursive forecasts | within 2.3e-6 |
| `links` | logL 81.114461 |
| the `ols` start | the monetary VAR(2) at −1741.275477 in both, 46 parameters within 6e-6; route V, the gas furnace VARMA(1,1), at −2838.983807 |
| fue's Box-Cox with λ = 1 | in both, since atsw-gui lib/prewhiten BUG-0002 |

**Only in one engine:**

| feature | C | port |
|---|---|---|
| `preliminary`, `echelon` starts; the echelon form, `rows`, `factor_test`; `cross="residual"` | — | yes |
| `-start`, `-zero` on the command line | yes | API only |
| `-irfboot` (parametric bootstrap with refits) and delta-method IRF standard errors | yes | — (`irf_fevd_bands`: Monte Carlo from N(θ̂, cov), no refits) |
| `refit`, `forecast_given`, `identification_mv`, the HTML report | — | yes |

**Known divergences:**
- the deseason phase (port BUG-0001, open: live in the C);
- deseason applied in levels (port BUG-0003, the C pending);
- the `csrc` copies (port BUG-0002, open);
- the `.inp` path's Box-Cox at λ = 1 against the ladder's (§1);
- the presentation-only `.out` differences listed in `STATUS.md`.

## 9. Routes U and V

The school's rule holds on both: **the univariate models first**. They are
the yardstick and give the initial conditions, and the gate is that. Past
the gate the two routes differ only in what the own part is and how it is
seeded.

- **route U** (the default): each series keeps its univariate ARMA on the
  diagonal; the cross terms are added.
- **route V** (`route="v"`, the C's `-route v`; the vector first, Tiao and
  Box): each series' regular ARMA is replaced by a free AR(p)/MA(q), so the
  whole VARMA(p,q) matrix is free; its seasonal and fixed-frequency factors
  multiply its whole row. The seed is the whole matrix; the LR is against
  the same model without cross terms; `factor_test()` checks the row factor.

When every univariate's regular part is exactly an AR(p) and none has
seasonal factors, the two routes are the same model (the golden case).

Until 0.3.x "route V" meant specifications without univariate models (sima's
`write_specs`, `split`): the ladder still takes them, and the gate says they
are not optima, but they are no longer what route V is.
