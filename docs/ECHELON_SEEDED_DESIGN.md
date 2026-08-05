# Echelon form seeded from the univariate models — a design note

**Status: a proposal, not a plan.** Nothing here is implemented. It is written to
be argued with, and §8 lists the things that could sink it.

Date: 2026-08-05. Origin: the observation that `drtran` gets something for free
that `drvarma` currently throws away — the univariate structure of each series —
and that the mechanism it uses (a per-equation operator, kept as a *product*) has
a recognised counterpart in the VARMA literature that this package does not yet
exploit.

---

## 1. The problem, stated precisely

An unrestricted VARMA(p, q) is **not identified**. Different (p, q) and different
coefficient matrices generate the same process, so the likelihood has flat
directions by construction — which is not a numerical accident but the shape of
the parametrisation. This session measured the consequence directly: fitting a
VARMA(2,2) to VAR(1) data makes the compiled and the pure-Python engines
disagree in the 1st decimal of the log-likelihood and stop on different
termination codes, because in a flat basin where you stop depends on the path
(`tests/test_engine_consistency.py`).

Two further costs, both visible in what the package does today:

* **A single (p, q) is imposed on every equation.** If one series is an AR(1)
  and another needs an AR(2) with a seasonal factor, the grid search picks one
  order for both — over-fitting one equation to accommodate the other.
* **The univariate work is discarded.** `characterize_series` runs ART on each
  series and obtains λ, d, seasonality, orders **and estimated coefficients**.
  Of all that, only λ, d and a (p, q) ceiling survive into the VARMA. The
  coefficients — exact-ML estimates of the very dynamics the VARMA has to
  re-learn — are thrown away.

## 2. What the echelon form provides

The echelon form parametrises a VARMA by **Kronecker indices** `(n_1, …, n_ν)` —
one row degree per equation — plus a table of zero restrictions determined by
those indices. It is a *unique* parametrisation of the transfer function: exactly
the property the unrestricted form lacks.

The source is in this repository's sister literature folder:
**Lütkepohl, H. & Poskitt, D. S. (1996), "Specification of Echelon-Form VARMA
Models", JBES 14(1), 69–79** (`ART/literature/`). What follows is from §2 of that
paper, not from memory.

For the pair `[A(z) : M(z)]`, left coprime, the (reversed) echelon canonical form
has:

1. **Diagonal AR**: `a_rr(z) = 1 + a_rr,1 z + … + a_rr,n_r z^{n_r}`
2. **Off-diagonal AR**: `a_rc(z) = a_rc,n_r−n_rc+1 z^{n_r−n_rc+1} + … + a_rc,n_r z^{n_r}`
3. **MA**: `m_rc(z) = m_rc,0 + m_rc,1 z + … + m_rc,n_r z^{n_r}`
4. `deg[A(z) : M(z)]_r = n_r` — row `r` has degree **at most** `n_r`
5. **`A_0 = M_0` is LOWER TRIANGULAR with unit diagonal**

with the index table (their eq. 2.3):

```
n_rj = min(n_r + 1, n_j)    for j < r
     = min(n_r,     n_j)    for j >= r
```

Two things here are easy to get wrong and are worth stating flatly:

* **The free off-diagonal AR coefficients sit at the HIGH lags, not the low
  ones** — `a_rc` starts at lag `n_r − n_rc + 1`. A row of order 2 whose partner
  has index 0 has its only free coefficient at lag 2, with lag 1 constrained to
  zero. This is the opposite of the intuition "lower lags first".
* `A_0` is lower triangular *with unit diagonal*, so contemporaneous effects are
  admitted in one direction only, and the ordering of the variables matters.

The paper's own example (ν = 3, indices 1, 2, 1) gives
`[n_rc] = [[1,1,1],[1,2,1],[1,2,1]]` and

```
A_0 = [[1,0,0],[0,1,0],[0,*,1]]   A_1 = [[*,*,*],[0,*,0],[*,*,*]]   A_2 = [[0,0,0],[*,*,*],[0,0,0]]
M_0 = A_0                          M_1 = [[*,*,*],[*,*,*],[*,*,*]]   M_2 = [[0,0,0],[*,*,*],[0,0,0]]
```

Note `A_1[2,·] = [0,*,0]`: row 2 has index 2, its partners have index 1, so
`n_21 = min(2,1) = 1` and its lag-1 off-diagonals are constrained away while its
lag-2 ones are free. Six of the twelve potentially free parameters are zero.

**The engine already accepts these models.** An echelon form with unequal
Kronecker indices has zero rows at the high lags, hence a rank-deficient `A_p`.
Until 2026-07-31 the pure-Python `elf` rejected exactly that with `ifault=3`
("non-stationary") even when every root of `det A(z)` was outside the unit
circle — a port defect in `_chol_lower`, fixed by using the *modified* Cholesky
the C always used. Verified again today:

```
echelon (row 2 null)     rank(A₂)=1   ifault=0   accepted
both rows of order 2     rank(A₂)=2   ifault=0   accepted
```

So the precondition holds: nothing in the likelihood blocks this.

### 2.1 The paper's specification strategy, and where a seed fits

Lütkepohl & Poskitt §3 lay out five stages:

| stage | what it does |
|---|---|
| I | fit a **long VAR(h)** to get residuals `ε̂_t(h)` |
| II | fit echelon VARMAs **by linear least squares** (lagged residuals replaced by Stage I's) over a range of index sets; select by AIC/HQ/SC |
| III | **ML estimation of the chosen model — with the Stage II estimates as initial values** |
| IV | examine t ratios, impose further zeros, re-estimate |
| V | residual analysis |

Stage III is exactly the seeding idea, and the paper endorses it explicitly. The
proposal in §3 below differs in *where the seed comes from*: from the univariate
exact-ML models rather than from a long VAR plus least squares.

Also worth recording, because it is the same phenomenon this repository met from
the other side: on overparameterised models the paper reports that "the estimated
MA operator has a tendency to converge to the boundary of the invertibility
region", and reads that as a sign "that a reduction of the parameter space is
called for". That is the `termcode 3` of
`drtran/docs/OPTIMIZER_STOPPING_STUDY.md` §5, described in 1996.

## 3. The seed: what the univariate models are actually for

This is the substance of the proposal, and it has three distinct uses — the
second and third are the ones that matter most, and neither is about orders.

### 3.1 Orders → initial Kronecker indices

`p_i` seeded from series `i`'s univariate ARMA orders. Because the echelon ties
the AR and MA row degrees together, the natural seed is `p_i ≈ max(p_i^u,
q_i^u)`. This is a *starting point for a search*, not a claim: the marginal
orders of a component of a VARMA are not the Kronecker indices, and §8 says why.

### 3.2 Coefficients → pre-estimates

`estimate_py` already seeds the optimiser with `_hannan_rissanen_diag` — a
**diagonal, per-series, two-step** estimate. That is precisely the right shape,
and it is a strictly worse version of something the pipeline already has:
ART's univariate models are **exact ML**, not two-step, and they carry the
interventions and deterministics the `.pre` files record.

So the change is not to invent a mechanism but to feed the existing one a better
seed:

```
today     w → hannan_rissanen_diag(w, p, q) → diagonal φ, θ, σ²  → raxopt
proposed  the .pre files → diagonal φ, θ, σ² (exact ML)          → raxopt
                          off-diagonals at zero
```

Starting on the diagonal with each series at its own optimum, and letting the
optimiser add only the cross terms, is exactly what `drtran` does — its search
"starts on the diagonal rung and the optimizer is left to add the dynamics"
(`drtran/estimate.py`). It is why the diagonal rung there is a *proof* the bridge
is correct: with ω = 0 the joint model must reproduce the two univariate ones.
The same test becomes available here.

### 3.3 Seasonality → multiplicative row operators

**This is where the univariate models are not a convenience but the only
sensible route.**

A univariate seasonal model is *multiplicative*: an airline model is
`(1−θB)(1−ΘB^s)`, two parameters spanning order `s+1`. An echelon row is
*additive in lags*: to reach lag 13 it needs `p_i = 13`, and every intervening
lag carries free coefficients. The parsimony is not degraded, it is destroyed —
the same arithmetic that made drtran's `AR(2)×AR(2)` expand to order 26.

The proposal is to let row `i` carry its univariate operator **as a factor**:

```
row i:   φ_i(B) Φ_i(B^s) · w_i,t  +  (cross terms)  =  θ_i(B) Θ_i(B^s) · a_i,t
```

so the seasonal structure is represented with the same parameter count as the
univariate model, and the cross terms are parametrised on top of it.

**The machinery for this already exists in the suite.** `drtran`'s slot table
supports `product` and `lincomb` slot kinds precisely so a coefficient can be a
function of others, and `expand` is applied inside the objective so gradients
need no chain rule. A multiplicative row operator is a product slot. This is a
reuse, not an invention.

### 3.4 What the paper says about seasonality — and it cuts both ways

Two passages from Lütkepohl & Poskitt bear directly on §3.3, and neither is
comfortable.

**In favour.** On Poskitt's single-equation procedure they warn that "in small
samples it is possible that the procedure terminates too early **for processes
with isolated nonzero lags which may occur for seasonal models**". An isolated
nonzero lag at `s` is precisely what a multiplicative seasonal factor produces,
and the standard index-search procedures are documented to miss it. That is an
argument FOR carrying the seasonal structure explicitly as a row factor rather
than hoping an index search discovers it — which is your point, made by the
authors against their own method.

**Against.** On Stage I they warn that "if very long significant lags are
observed, it may signal a **noninvertible MA part** in the generation process
that may be due to inadequate preliminary data adjustments such as differencing
or **seasonal adjustment**. The present procedure is not designed for processes
with noninvertible MA parts."

That is aimed squarely at what `drvarma` does today: harmonic deseasonalisation
before the VARMA. It is the standard Maravall objection — adjusting seasonally
can induce MA unit roots — and it means the current pipeline may be handing the
VARMA a process the echelon machinery is explicitly not built for. Which is an
argument for §3.3 too, but for a stronger reason than parsimony: **modelling the
seasonality inside the row operator avoids an adjustment that can break
invertibility**, where deseasonalising first does not.

They also note that for seasonal data the long VAR of Stage I needs a much
higher order — "for quarterly seasonal data, `h` should be at least 10 for
`T` about 100 and substantially larger" — so the Stage I/II route is expensive
exactly where our data lives. A univariate seed sidesteps Stage I entirely.

### 3.5 Scalar component models: the rival, and why the paper prefers echelon

Tiao & Tsay (1989) and Tsay (1989, 1991) propose scalar-component models, which
"amount to **transforming the original variables first and then fitting
univariate ARMA models to the transformed variables**", then combining those into
a multivariate specification. That is the closest published thing to "start from
univariate structure", and it is worth knowing why Lütkepohl & Poskitt do not
take it:

* an appropriately specified echelon form need not have more nonzero parameters
  than a scalar-component model — the parsimony argument is not decisive;
* **asymptotic inference in echelon forms is straightforward, whereas it is
  problematic for scalar-component models when the transformation of the
  variables is data dependent** — "which it usually is in practice".

That second point matters for us more than for them: this package's whole output
is standard errors, *t* ratios and bands. A parametrisation whose inference is
compromised by a data-dependent transformation is a bad fit for it. It is also
the difference from `drtran`, where the "transformation" is not estimated from
the data at all — it is the analyst's declared exogeneity, and it gets tested.

## 4. Why this is not drtran's parametrisation

`drtran` gets its clean separation from **triangularity**. Declaring exogeneity
makes the system a DAG; after topological ordering `Φ(B)` is triangular, the
input's equation is literally its univariate model, and the output's equation
carries its own `φ_Y·δ` beside the `ω` that parametrises the relation.

That is a strong assumption, and it is *tested* there (the exogeneity
portmanteau at `k < 0`), not assumed. Here it is neither assumed nor available:
`sima` exists for systems where everything is endogenous.

So the two must not be conflated:

|  | drtran / mtram | this proposal |
|---|---|---|
| exogeneity | **declared and tested** | not imposed |
| `Φ(B)` | triangular by construction | full, restricted by the echelon table |
| univariate model of the input | preserved exactly | preserved as a **row factor**, not marginally |
| cross relations | `ω(B)/δ(B)·B^b`, a transfer | free off-diagonal echelon terms |

And one theoretical limit must be stated plainly, because it bounds what can be
claimed: in a VARMA with feedback, the marginal ARMA of **every** component has
`det Φ(B)` as its AR polynomial — common to all of them. The univariate models
therefore cannot be *preserved marginally* by any simultaneous VARMA. What §3.3
preserves is the **structural** row operator, which is a different and weaker
claim — but it is the claim that buys the parsimony, and it is the one drtran
relies on too.

## 5. The objection: Tiao & Box argue for the opposite

**Tiao, G. C. & Box, G. E. P. (1981), "Modeling Multiple Time Series With
Applications", JASA 76(376), 802–816** — `drtran/literature/`. This paper is the
source of the partial autoregression matrices and the ± indicator symbols this
package already implements, and its §4 doctrine is the **reverse** of §3 above.
It belongs in this note, not outside it.

### 5.1 What it confirms

The mtram/sima boundary is in the paper, verbatim (p. 803): transfer-function
models "assume that the series, when suitably arranged, possess a **triangular
relationship** … On the other hand, if `z1` depends on the past of `z2`, and also
`z2` depends on the past of `z1`, then we must have a model that allows for this
**feedback**."

And it is made formal on p. 804: if the `φ`'s and `θ`'s can be arranged so the
coefficient matrices are all **lower triangular**, then the VARMA *is* a transfer
function model. More: if they are lower **block** triangular, one gets a
generalisation "in which both the input vector series and the output vector
series are allowed to have feedback relationships".

That last sentence describes an architecture the suite does not have: a **DAG of
blocks**, with simultaneity *inside* each block and transfer *between* blocks —
`sima` within, `mtram` across. Worth recording even though it is out of scope
here.

### 5.2 What it contradicts

Their §4 (p. 805) is a direct objection to §3 of this note:

> "It is natural that attempts have been made to simplify the general form in the
> model building process, for example by Granger and Newbold (1977) and Wallis
> (1977). **While we sympathize with this aspiration, we feel that so far at
> least these attempts have not been successful.** … **We see no alternative but
> to provide for direct initial fitting** of models of the form (3.1)."

Fit the full form, simplify afterwards. This note proposes the opposite: seed
simple, add relations.

And §6 dismantles the Granger–Newbold approach — which is *almost exactly* the
proposal here: fit univariate ARMAs to each series first, then identify the
dynamic structure from the residuals. Four objections, of which the third is the
one with teeth:

1. the parameters of the residual model are "subject to various complicated
   **nonlinear constraints**";
2. "even if the vector series `{Z_t}` follows a low order ARMA model, the
   corresponding model for the residual vector `{C_t}` can be **complex and
   difficult to identify** in practice";
3. **the arithmetic runs against it.** For `k` series the maximum number of
   parameters goes from `k²(p+q)` in the direct form to `kp + [(k−1)p + q]k²`,
   an increase of `pk(k−1)²`. "Thus, assuming the degree of `H(B)` is correctly
   specified, even for `k` as large as 3 or 4, a very large number of additional
   parameters will have to be estimated **merely to identify correctly a low
   order vector AR model**, say `p = 1` or 2";
4. the correspondence between the degrees "is not necessarily one to one, so it
   is not clear how one determines `p` and `q`".

### 5.3 The door they leave open, and it is the right one

Their own five qualifications include two that legitimise a seeded approach:

> "2. that **occasionally knowledge of the system might allow simplification a
> priori**, although even here prudent checking of the adequacy of the
> simplification would be necessary (see Zellner and Palm 1974),"
>
> "4. that 2 and 3 imply that **provision should be made to allow models to be
> fitted in which certain parameters are fixed or constrained in some other
> way**."

Point 4 is literally the slot table of `drtran` and the constraint machinery of
`drvarma`. Point 2 says a priori simplification is legitimate **when it comes
from knowledge of the system and is then checked** — which is exactly what
`mtram` does with exogeneity: declares it, then submits it to a portmanteau.

### 5.4 Reading

The proposal survives in one half and not the other, and the split is worth
being explicit about:

* **§3.2, the seed, stands.** Starting the optimiser from the univariate
  exact-ML estimates imposes nothing: same model, same parameter space, better
  starting point. None of the four objections in §5.2 applies, because nothing is
  being reparametrised. It is Lütkepohl & Poskitt's Stage III with a better
  Stage II.
* **§3.3, the multiplicative row structure, falls squarely under their
  critique** — it *is* an a priori simplification. It is defensible only on their
  qualification 2: knowledge of the system, plus a check of the adequacy of the
  simplification. That check must therefore be part of the design, not an
  afterthought: a likelihood-ratio test of the factorised row against the
  unrestricted row of the same degree.

There is also a caveat this package should have adopted already and has not.
On the ± indicator symbols for the cross-correlation matrices — which
`cross_correlation_matrices` prints today — the authors warn that the variances
of the sample correlations "can be considerably greater than `n^{-1/2}` when the
series are highly autocorrelated, so that these indicator symbols, **if taken
literally, can lead to overparameterization**. However, we do not interpret these
indicator symbols in the sense of a formal significance test, but as a rather
crude **'signal-to-noise' guide**." The tool's own text should say so.

## 6. Where it plugs into what exists

| piece | state |
|---|---|
| per-series λ, d, seasonality, orders | **exists** — `mcp_server.characterize_series`, saved in `_SEED` |
| per-series exact-ML coefficients | **exists** in the `.pre` files; not read by drvarma |
| diagonal seeding of the optimiser | **exists** — `_hannan_rissanen_diag`, to be superseded |
| likelihood with unequal row orders | **exists and verified** — §2 |
| product/lincomb constrained slots | **exists in drtran** (`drtran.slots`); would need lifting |
| echelon restriction table | **does not exist** |
| Kronecker index search | **does not exist** |

Two of the seven are missing, and one of those is a table from a textbook.

## 7. A staged route

1. **Diagonal seeding from the `.pre` models** (§3.2). Self-contained, testable
   on its own, and valuable with or without the rest: the diagonal rung becomes
   reproducible against the univariate fits, which is a correctness test drvarma
   currently cannot perform.
2. **Unequal row orders without the full echelon** — allow `p_i` per equation by
   zeroing rows, which the likelihood already accepts. Not yet identified, but it
   makes the parametrisation expressible and measurable.
3. **The echelon restriction table**, which turns step 2 into an identified form.
4. **Multiplicative row factors** (§3.3) via lifted product slots — **and, in
   the same step, the likelihood-ratio test of the factorised row against the
   unrestricted row of the same degree**. §5.3 is why: this is the half of the
   proposal Tiao & Box argue against, and the only thing that makes it legitimate
   on their own terms is checking the adequacy of the simplification. Shipping
   the factorisation without the test would be taking the part of their
   qualification that suits us and dropping the condition attached to it.
5. **Index search**: start from the univariate seed and test upward, the way
   `identify_varma_order` searches (p, q) today.

## 8. What could sink this — read before starting

* **The Kronecker indices are not the marginal orders.** §3.1 is a heuristic, and
  the relationship between a component's univariate ARMA orders and its Kronecker
  index is not an identity. The seed may be systematically wrong in a direction
  nobody has measured. **This is the assumption to test first**, on simulated
  systems with known indices, before any of §7 is built. Lütkepohl & Poskitt's
  own example is a warning: three flour-price series, all three methods they try
  return indices `(1,1,1)`, and the final models turn out to be a restricted
  VAR(1) and an MA(1) — "valid echelon forms with Kronecker indices (1,1,1) in
  which certain of the freely varying parameters are 0. **They are not
  lower-order models.**" The index is a ceiling on the row, not its content, so
  a seed that gets the index right can still be wrong about everything inside it.
* **The multiplicative row form may not be identified.** §3.3 imposes a
  factorisation on top of the echelon table; whether the result is still a unique
  parametrisation is an open question, not a detail. A form that looks
  parsimonious and is unidentified is worse than the unrestricted VARMA, because
  it hides the flat directions instead of exposing them.
* **Differencing must still be consensual.** The VARMA needs one `d`; the
  univariate models may disagree. Today `characterize_series` takes the max. That
  over-differences the others, and a multiplicative seasonal row operator does
  not fix it.
* **`termcode 3` is still unexplained.** Over-parameterised VARMA fits stop
  without improvement with *and* without the `typx` change
  (`drtran/docs/OPTIMIZER_STOPPING_STUDY.md` §5), which this session established
  is weak identification rather than an optimiser artefact. If the echelon form
  is doing its job, those stops should become rarer — which is a **testable
  prediction** and the best single check of whether any of this is working.

## 9. References

* Hannan, E. J. & Deistler, M. (1988), *The Statistical Theory of Linear
  Systems* — the echelon form and Kronecker indices.
* **Lütkepohl, H. & Poskitt, D. S. (1996), "Specification of Echelon-Form VARMA
  Models", JBES 14(1), 69–79** — `ART/literature/`. THE source for this note: the
  restriction table (eq. 2.3), the five-stage strategy, and the seasonal
  caveats of §3.4. Read §2 and §3 before implementing anything.
* Lütkepohl, H., *New Introduction to Multiple Time Series Analysis*, ch. 12 —
  the same material in textbook form.
* Poskitt, D. S. (1992) — the **single-equation** index procedure, which
  specifies equation by equation using the fact that the restrictions on row `k`
  are determined by the indices `n_i <= n_k`. Consistent, and cheap: "reestimation
  of equations with fixed Kronecker indices is not necessary". The closest
  published procedure to the equation-by-equation spirit of this note.
* Hannan, E. J. & Kavalieris, L. (1984) — the two-step index search (all indices
  equal first, then varied one at a time). Reduces a 59,049-model full search to
  ~33 fits in their example.
* Tiao, G. C. & Tsay, R. S. (1989), JRSS-B — **scalar component models**: linear
  combinations of the vector that follow low-order ARMA. The closest published
  idea to "find the univariate simplicity hidden in the vector process", and
  worth reading before committing to the echelon route, because it answers the
  same question a different way.
* **Tiao, G. C. & Box, G. E. P. (1981), "Modeling Multiple Time Series With
  Applications", JASA 76(376), 802–816** — `drtran/literature/`. Partial
  autoregression matrices and the ± symbols, both implemented here; the
  triangular/feedback boundary (p. 803–804); and the §4/§6 objection to
  simplification-first modelling that §5 of this note answers.
* Hannan, E. J. & Rissanen, J. (1982) — the two-step initialisation, already
  implemented here as `-twostep`.
