---
id: BUG-0030
title: The ladder evaluated points the C's ladder refuses (fue's MA flip in the cast), packed the parameters in another order and seeded ols with another arithmetic — on the MA wall it did not end where the C does
status: fixed
severity: high
component: ladder
found_in: 0.3.4
fixed_in: 0.4.0
reported: 2026-11-22
reporter: David / Claude — sima's trial run on the US monetary VARs, route V, against the C's routes battery
tags:
  - ladder
  - homologation
  - ma-wall
  - cast
references:
  - src/drvarma/ladder.py _cast_us, LadderSeries.c_layout, Ladder._layout, _ols_start, _ols_start_v
  - src/drvarma/_cls.py
  - atsw-gui engines/drvarma/src/escalera.c shootx_esc, esc_pack, start_ols, start_ols_v
  - atsw-gui lib/fuepre/fue_pre_reader.c pack_ar_factors, pack_ma_factors
  - fue-python #14 (cast_us_py(flip=False))
  - atsw-gui #221 (the routes battery: GS1, GS2, GS3 on route V)
  - bugs/BUG-0030-repro/repro.py
---

## Summary

On the VARMA(1,1) of the US monetary systems by route V, the port and the C
ended at different points from the same start (GS2 from zero: the C
−1779.745477 in 172 iterations, on the MA wall; the port −1780.083070 in
130). The two engines were traced evaluation by evaluation. Three causes, all
in the port:

1. **The cast flipped the MA.** The port casts each series with fue's
   `cast_us_py`, and fue takes the invertible form of a regular MA(1) with
   |θ| > 1 (and of a fixed-frequency MA with θ₂ < −1) before it builds the
   polynomial: fue.c's rule for the univariate model. The C's ladder does
   not (`shootx_esc`): the polynomial is built from the parameters as they
   are, and `chekma` refuses the point. At the first point where it matters
   both engines hold the same 45 parameters; the C builds θ_WTI = 1.013852
   and refuses, the port built 0.986337 = 1/1.013852 and went on. After the
   search the port also moved the result to the "invertible form"
   (`canonicalize`); the C reports the point it reached.
2. **The order of the parameters.** The C packs the ARMA of every series,
   then the deterministic terms, then the means (`esc_pack`), each series'
   AR side whole before its MA side and each deterministic variable's omegas
   before its deltas. The port packed series by series, in fue's order. The
   same model; the finite-difference gradient in another sequence, and sums
   that differ in the sixteenth digit.
3. **The ols seed.** numpy's normal equations (BLAS products, LAPACK's solve)
   in place of the C's sums in sequence and its own LU (`ludcp`, `lusol`).

2 and 3 are rounding: harmless where the likelihood has an interior maximum
(every VAR matched, with the same iterations), decisive on the wall, where
there is none. 1 is not rounding: the port evaluated a model the C rejects.

## Where the flip came from

Not from the method. Mauricio (1995, sec. 3) sets the scaled objective to
one at a point that implies non-stationarity, non-invertibility or a Q that
is not positive definite, "thus the minimization routine will reject these
points and continue the search"; AS 311 (1997) detects non-invertibility in
`ELF2` so that "an objective function that penalizes these situations can be
constructed". `elf` returns ifault 4 and the estimator's objective returns
1.0: the paper, to the letter. Nothing takes a reciprocal.

The reciprocal is fue's, the univariate program's: `fue.c` takes the
invertible form of an MA(1) inside its cast ("Take invertible for MA(1)s",
with the author's own caution about what it does to the standard errors),
and so does `fuf.c`. drtran's and drvarma's C do not.

The ladder was written in September 2026, in C and in Python a day apart.
The C builds each series' polynomial with the shared reader (`lib/fuepre`),
which does not flip. The port was designed to cast each series "with fue's
own code, the reference, not a copy", and fue's cast carries the rule: it
came into the multivariate with it. The first symptom was seen the next day
(m6: θ_EU = 1.0722 in the port, 0.932651 = 1/1.0722 in the C) and treated
after the optimisation (`canonicalize`, this tracker's BUG-0007); the cause
stayed in the cast. sima later read another symptom as "the fold of an own
MA(1)" (its BUG-0008).

## Impact

Wherever the search touched the MA wall, on either route, the port could
stop at another point than the C, with another likelihood: up to 2.0 points
in the cases measured (GS1 + FF, VARMA(1,1): the C −1740.636683, the port
−1742.598015). The engines are one (the golden rule); they were not there.

## Reproduction

`bugs/BUG-0030-repro/repro.py`: the order of the own parameters, the model
the likelihood gets when an own MA(1) is at 1.25, and GS1 + FF, VARMA(1,1),
route V, from zero, against the C's number and iterations.

## What to expect after the fix

Without the flip an own MA coefficient can end beyond the unit circle when
the system stops on the wall (θ_WTI = 1.04 on GS3; −1.17 and −1.25 on the
flour prices' echelon form). It is what the C gives: what is judged is the
MA operator of the system, whole, not each coefficient.

## Fix

The port moves; the C is the reference.

- `_cast_us`: the ladder asks fue's cast for the polynomial WITHOUT the flip
  (`cast_us_py(flip=False)`, fue ≥ 0.1.23). With an older fue the ladder
  says so and stops. `canonicalize` is gone.
- `LadderSeries.c_layout` and `Ladder._layout`: the vector in the C's order.
  `pack`, `unpack` and `names` follow it; everything else reads by name.
- `_cls.py`: the least squares of both ols seeds in the C's arithmetic, loop
  for loop.

**What changes for a caller:** the order of `result.names`, `result.x`, the
standard errors and the covariance matrix (the means now come after every
ARMA coefficient). Read them by name.

## Validation

- The 64 monetary cases (4 systems × 2 routes × VAR(1), VAR(2), VAR(3),
  VARMA(1,1) × 2 starts): the C's likelihood to the sixth decimal and the
  C's iterations, all of them. Before, the eight VARMA(1,1) by route V
  matched in three.
- 16 pairs with an MA by route V (VARMA(1,1) and (2,1), 2 starts): all equal
  to the C; eight were not.
- Four of those pairs are in `tests/data/homologation.json`, with their
  iterations, and `tests/test_c_oracle.py` checks them on both sides.
- `tests/test_bug_0030_as_the_c.py`.
