---
id: BUG-0020
title: The diagonal gate fails with NaN when a univariate optimum sits on the MA wall; the same polynomial is admissible alone and ifault 3 in the joint elf
status: fixed
severity: low
component: ladder
found_in: 0.1.7
fixed_in: unreleased (after 0.3.4); the C, atsw-gui engines/drvarma BUG-0009
reported: 2026-10-11
reporter: David / Claude — echelon stage 2's fits study
tags: [ladder, gate, elf, invertibility]
references:
  - bugs/BUG-0020-repro/repro.py
  - src/drvarma/ladder.py (Ladder._run_gate)
  - src/drvarma/_as311.py (chekma, elf: ifault 3 at the Cholesky of mtmp0)
  - research/echelon_seed/fits.py
  - BUG-0007
---

## Summary

On route V's specifications, a series' univariate fit can end with an MA
root at 0.99995. That is inside chekma's tolerance (inverse modulus below
1.00005), so it passes. Evaluated alone, elf accepts it (ifault 0). The
diagonal system with that same polynomial on its diagonal, next to two other
series, fails elf's Cholesky (ifault 3). The gate then reports "difference
nan: the joint cast does not reproduce the univariate models". That is true,
but it does not say why.

## Impact

Low: a univariate optimum on the invertibility wall is already a model to
reconsider. But the gate's message points at the cast, not at the MA, and
the Ladder stops. One replication in 1,800 fits of the echelon study.

## Reproduction

`research/echelon_seed/fits.py`: `data((2, 2, 1), "generic", 200, 45)`, the
specs with own ARMA (2,2), (2,2), (1,1), then `Ladder(files, 0, 0).run_gate()`.
Series y2's MA is `theta = [-0.1143, 0.8858]`, with roots of modulus 1.129
and 0.99995.

## Root cause

Not settled. The joint elf's covariance matrix of the MA part (mtmp0) is
numerically not positive definite at the boundary when m = 3, but it is
for m = 1. To be checked against the C (`elfvarma.c`) before deciding
between two fixes: (a) the gate says "series y2's MA is on the invertibility
wall"; (b) the univariate search keeps away from the wall.

## Fix

Fix (a), in both engines, only the report: when the diagonal system has an
ifault, the gate says it could not be evaluated, with the C's words for the
fault, and names every own MA factor of the univariate optima on the wall
(`_own_ma_outside`, the C's `own_ma_outside`):

```
the diagonal gate failed: the diagonal system could not be evaluated at the
univariate optima (AR nonstationary), and an own MA factor is on the
invertibility wall: y2 (1.0000). Reconsider that univariate model.
```

The numerical cause (why the joint Cholesky of mtmp0 fails at m = 3 where it
passes at m = 1) is in `elf`, the same in both engines: the C gives ifault 3
on the same files, and the port in C and in pure Python too. It is not
touched: a proven engine, and an MA on the wall is already a model to
reconsider. Fix (b), keeping the univariate search off the wall, would
change fue's optima and is not taken.

## Validation

- `bugs/BUG-0020-repro/repro.py` (y1.inp, y2.inp, y3.inp): exits 1 on the
  code before (the "difference nan" message), 0 after.
- `tests/test_bug_0020_puerta_ma_en_la_pared.py`, with the C engine and with
  `DRVARMA_NO_ENGINE=1`.
- The C: atsw-gui `engines/drvarma/bugs/BUG-0009-repro/repro.py`, the same
  three files and the same message; before, "joint diagonal 0.000000" and a
  difference of 1.26e+03.
