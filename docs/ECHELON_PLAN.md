# Echelon form, stages 2–3 — the plan

*2026-10-11. Follows [`ECHELON_SEEDED_DESIGN.md`](ECHELON_SEEDED_DESIGN.md)
§7 and stage 0 ([`STUDY-echelon-seed.md`](STUDY-echelon-seed.md)), which
dropped the univariate seed for the indices. Stage 1 (the diagonal from the
`.pre` files, and the gate) is the Ladder as it is. Stage 4 is reviewed
after this one; §7's stage 5 (the index search) is folded into 3b here.*

## 0. Decisions

1. **The route is Lütkepohl and Poskitt's (1996, §3), whole.** Stage I is a
   long VAR. Stage II is least squares over candidate index vectors with a
   criterion. Stage III is exact ML from the stage II estimates. The univariate
   models give the transformation, the differencing and the mean (the fue
   files), and the yardstick for forecasting. They do not give the indices.
2. **The echelon form is fitted on route V's specs.** Each series' file carries
   only its characterization (`sima.write_specs`). The diagonal of row `i` is a
   FREE AR(n_i)/MA(n_i), as route V's `estimate(p, q)` already writes.
   - That is the form whose uniqueness LP prove.
   - A route-U diagonal (a `.pre`'s seasonal factors, degree 13 and up) is not
     an echelon row of degree `n_i`. Putting the two together is the
     multiplicative row factor of stage 4, and its identification is §8's
     open question.
   - So stage 3 refuses a series whose ARMA degree exceeds its index (`LadderError`).
3. **One engine, one class.** The echelon form is a structure of the `Ladder`
   (`echelon=(n_1, …, n_m)`), not a new estimator. elf, the gate, fdhess, the
   LR tests, the forecasts and the report are reused unchanged. Only the cast
   learns `A_0`.
4. **The order of the variables is part of the model** (`A_0` is lower
   triangular). It is the Ladder's order, i.e. the files' order. `sima.reorder`
   already changes it.

## 1. Stage 2 — free slots per row

The Ladder already removes cross coefficients in two ways:
- `links` removes a whole pair;
- `zeros` removes single coefficients.

Stage 2 makes the general case first-class: an explicit set of free cross slots.

- **`Ladder(..., rows={"A": (p_A, q_A), …})`**: equation A's cross terms run up
  to lag `p_A` (AR) and `q_A` (MA). Missing rows take `(p, q)`.
  - Internally `_slots()` reads one `free` set, `{(kind, k, i, j)}`.
  - `links`, `zeros` and `rows` are three ways of building that set, and they
    compose.
  - `p` and `q` become the maxima.
- Nothing else changes: `pack`/`unpack`/`names`/`npar` already go through
  `_slots()`.
- **Tests:**
  - `rows` equals the same structure written with `zeros`, with the same logL
    to 1e-10;
  - `links` + `zeros` + `rows` compose;
  - the C has no row orders, so the parity is by that equivalence.
- **Measure:** this form is not identified, which is §7's point 2. On the
  stage-0 systems, fit with row orders `n` and count the `termcode 3` stops
  and the MA-boundary optima. Those counts are the baseline for stage 3.

## 2. Stage 3 — the echelon structure

### 2.1 The model

`Ladder(..., echelon=(n_1, …, n_m))`, with the restriction table of LP (2.3):
`n_rc = min(n_r + 1, n_c)` for `c < r`, and `min(n_r, n_c)` for `c ≥ r`.

| slot | free lags | name |
|---|---|---|
| diagonal AR, MA of row r | 1..n_r | the series' own (from its spec) |
| cross AR `a_rc`, r ≠ c | n_r − n_rc + 1 .. n_r (the HIGH lags) | `AR{k}[r<-c]` |
| contemporaneous `a_rc,0`, c < r, only when n_c > n_r | 0 | `AR0[r<-c]` (new) |
| cross MA `m_rc` | 1..n_r | `MA{k}[r<-c]` |

**The cast.** With `M_0 = A_0`, elf gets the standard form:
`Φ_k = −A_0⁻¹ A_k`, `Θ_k = −A_0⁻¹ M_k`, `Θ_0 = I`. The innovations, Q and its
normalisation `Q_11 = 1` do not change. `A_0 = I` when no `AR0` slot is free,
which leaves today's cast as it is.

**Checks at construction:**
- every series' ARMA degree ≤ its index;
- `cross="additive"` (the residual form of Jenkins and Alavi is not an echelon
  form);
- `links` and `zeros` are allowed, but only as further zeros (LP's stage IV).

**Tests:**
- `echelon=(1,…,1)` is the VARMA(1,1) with every slot free;
- LP's (1,2,1) example gives exactly their table (the free positions printed
  in `ECHELON_SEEDED_DESIGN.md` §2);
- a system from `research/echelon_seed/echelon.py` simulated at nobs 2000:
  - the ML estimates fall within 3 SE of the truth;
  - the cast reproduces the system's Φ, Θ to 1e-12 at the true values;
- the gate still holds, because the diagonal system has no `A_0`.

### 2.2 The index search (LP stages I–II)

This goes in **`drvarma.echelon`**, a new module that takes the pieces of
`research/echelon_seed/echelon.py` that belong in the package: the index
table, the free positions, the standard form. It works on the Ladder's
stationary vector `w` (the same one route V's `identify_matrices` uses).

1. **Long VAR(h)** by least squares, `h` by AIC up to `h_max ≈ (log n)^1.5`.
   The residuals `ε̂_t` are the proxies for the innovations.
2. **For each candidate n:** least squares row by row. Row r regresses
   `w_r,t` on the free lags of `w` and of `ε̂` that the table allows, with
   `A_0`'s terms as regressors on contemporaneous `w`. The criterion is
   `log|Σ̃(n)| + c_T · dim(n) / T`, with `c_T = 2 log log T` (HQ) or
   `log T` (SC); LP report both. *Measured in 3b: SC is the default (HQ
   overfits), and each candidate gets one round of Hannan and Kavalieris'
   residual correction (without it the search does not improve with nobs).*
3. **The search:** Hannan and Kavalieris' (1984) two steps, about m·n_max
   fits instead of (n_max+1)^m:
   - (a) all indices equal, 0..n_max;
   - (b) from the best, lower one index at a time while the criterion
     improves.
   - Full enumeration stays available for m ≤ 3, n_max ≤ 3, as the check of
     (b). *Measured in 3b: full enumeration is the default while there are
     at most 256 candidates (1.3 s for m = 3); (b) cannot raise an index,
     and misses (1,2,1) at nobs 100.*
4. **Output:** the chosen n, the criterion table (both criteria, every fit),
   and the stage II estimates. Those become stage III's start (`start="echelon"`,
   which goes into `Ladder.x_start`).

**The measure is stage 0's grid.** The study runs the same 9 systems × 2
designs × 3 sample sizes × 100 replications (`research/echelon_seed/study.py`
with `--method search`), so each cell of `STUDY-echelon-seed-data.md` gets
its counterpart.

| question | bar to call it working |
|---|---|
| all indices right, generic, nobs 200 | well above stage 0's 0–49 %; LP report most of the mass on the true n at T = 100 for m = 3 |
| does it improve with nobs? | it must (stage 0's did not) |
| the weak design | allowed to fall short; it then underfits, which stage IV's checks see |

**§8's testable prediction** is checked in the same run: exact ML from the
stage II start, against stage 2's row-order fit and the unrestricted VARMA,
with fewer `termcode 3` stops and fewer MA-boundary optima.

### 2.3 In sima — the third form

Two tools, on route V's session:

- **`identify_echelon(max_index, criterion="SC")`** prints the criterion
  table and the chosen n, saying that the variables' order is part of it.
  It records a decision in the guion.
- **`estimate(..., form="echelon", indices=…)`** writes each series' spec with
  AR(n_i)/MA(n_i), fits with `echelon=n` and `start="echelon"`, and reports as
  any Ladder fit does: the LR against the diagonal, and `evaluate` against the
  univariate yardstick.

The manual gets an echelon chapter next to Jenkins–Alavi and Tiao–Box.
`TODO.md`'s stage 3 is marked done when these two tools pass on a simulated
system and on one of the book's data sets.

## 3. Order of work, and what each step ships

| step | repo | content | PR |
|---|---|---|---|
| 2 | drvarma-python | `rows=`, the `free` set, tests, termcode baseline | one |
| 3a | drvarma-python | `echelon=`, `A_0` in the cast, `AR0` slots, tests | one |
| 3b | drvarma-python | `drvarma.echelon` (search), `start="echelon"`, study rerun + `STUDY-echelon-search.md` | one |
| 3c | sima-python | `identify_echelon`, `form="echelon"`, manual, TODO | one |

All of it goes on `ladder`, before the v0.2.0 tag: sima 0.1.0 needs 3a–3b
published in drvarma. §7's stage 5, the index search, is now 3b. What is
left to review afterwards is stage 4: multiplicative row factors with their LR
test, which is what would let a route-U diagonal into the echelon form.

## 4. Stage 4 — the row factor (route U)

*2026-10-12. Decided with the user after the probe below.*

### 4.1 The probe that decided it

Lifting stage 3's refusal and putting each `.pre` on the diagonal as it is,
with the cross terms by the table, works on the IPC (three monthly series,
AR(3) × annual AR(1), AR(1), annual AR(1)): equal indices reproduce the
route-U VARMA(1,1), −85.665. It fails on Jenkins and Alavi's muskrat and
mink (AR(6)/MA(1), AR(4)):
- (3,1) reaches −548.44 and (2,1) −543.51: the larger form below the smaller;
- (2,2) ends below (1,1), with termcode 3 and two MA roots on the wall;
- the stage II start overwrites lags 1–3 of the AR(6) and is not admissible.

The cause is the table itself. It puts the cross AR at the high lags
n_r − n_rc + 1..n_r, which assumes the row's degree is n_r. With the
univariate operator on the diagonal the row's degree is that operator's,
the lags lose their meaning and the forms stop nesting.

### 4.2 Decisions

1. **A model's seasonal and fixed-frequency factors are its row's FACTOR.**
   They multiply the whole row, the cross terms included:

       F_r(B) [A(B) w]_r = G_r(B) [M(B) a]_r,

   with F_r, G_r the AR and MA factors, estimated with the rest. This is
   design note §3.3's row operator. The table keeps LP's meaning, because
   the echelon part of each row is still of degree n_r.
2. **Only the regular ARMA is the row's own part**, and its degree is
   bounded by n_r. On route U the univariate orders are not the indices
   (stage 0), so `Ladder(own="free")` replaces each model's regular ARMA by
   a free AR(n_r)/MA(n_r). It starts at the file's polynomials when they
   fit in n_r, at zero otherwise. `own="spec"` (the default) keeps the
   file's and refuses one that does not fit.
3. **Stage II and the search filter each row by its factor.** Divided by
   G_r, row r is an ordinary echelon row on z⁽ʳ⁾ = (F_r/G_r)(B) w, with the
   same innovations: `echelon.stage2`, `residuals` and `search` take
   `filters=`. The start uses the factors as the gate left them (the
   univariate fits).
4. **The factor is checked, in the same step** (design note §5.4, §7.4).
   `Ladder.factor_test()` refits the same rows with every position the
   product F_r·[A]_r (and G_r·[M]_r) can reach free, own and cross: that is
   the unrestricted row of the same support. It starts at the expansion of
   the factored optimum, with the same likelihood, so the LR is ≥ 0 and is a
   lower bound if the alternative stops early. The support is read from the
   factor's structure, so a lag that is zero by construction stays out (a
   fixed-frequency MA at ω = π/2 has support {0, 2}).
5. **sima** opens `identify_echelon` and `estimate(indices=)` to route U
   with `own="free"`, and `estimate` reports the factor's check.

### 4.3 What the examples say

| data | search (SC) | fit | factor check |
|---|---|---|---|
| flour, route U (Tiao and Box's tools) | (1,0,0); in log levels (1,1,1), LP's own result | 11 par, −670.78, termcode 1, white residuals; VAR(1) 14 par, −671.34 | no factor |
| IPC, route U | (0,1,1); HQ (0,3,1) | MA wall; residuals not white (SC underfits) | p 0.16 |
| muskrat and mink | (3,1) | MA wall, termcode 3 | no factor |
| M6 (EA with a fixed-frequency MA) | (0,0,1) | termcode 1 | (1,0,0): p 0.008; (1,1,1): p 0.15 |

Unfiltered, the search on seasonal rows misses: on simulated quarterly
systems it returns (2,3) for (1,0) and (3,3,2) for (2,1,0), and the
filtered search finds both.

### 4.4 Order of work

| step | repo | content |
|---|---|---|
| 4a | drvarma | the row factor in the cast, `own="free"`, `filters=` in stage II and the search, tests |
| 4b | drvarma | `factor_test`, tests |
| 4s | drvarma | `research/echelon_seed/rowfactor_study.py` and `STUDY-echelon-rowfactor.md`: §8's question on simulated systems |
| 4c | sima | route U in `identify_echelon` and `estimate`, the factor check, the manual |
