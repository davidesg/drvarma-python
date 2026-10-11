---
id: BUG-0020
title: The diagonal gate fails with NaN when a univariate optimum sits on the MA wall; the same polynomial is admissible alone and ifault 3 in the joint elf
status: open
severity: low
component: ladder
found_in: 0.1.7
fixed_in:
reported: 2026-10-11
reporter: David / Claude — echelon stage 2's fits study
tags: [ladder, gate, elf, invertibility]
references:
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

—

## Validation

—
