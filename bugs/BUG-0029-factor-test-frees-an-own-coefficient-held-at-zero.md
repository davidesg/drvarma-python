---
id: BUG-0029
title: factor_test frees in its alternative an own coefficient the model holds at zero, and tests that zero with the row factor
status: fixed
severity: medium
component: ladder
found_in: 0.3.4
fixed_in: 0.4.0
reported: 2026-11-21
reporter: David / Claude — homologation of the C's -zero on own coefficients (atsw-gui #219)
tags:
  - ladder
  - echelon
  - factor_test
  - homologation
references:
  - src/drvarma/ladder.py Ladder.factor_test, LadderSeries.without_row_factor
  - atsw-gui engines/drvarma/src/escalera.c v_factor_test (the C: the zero stays)
  - atsw-gui #219
  - bugs/BUG-0029-repro/repro.py
---

## Summary

`factor_test()` builds its alternative, the rows without the row factor, from
the DEGREES of the row's regular part. With `own_zeros`, a lag of that part
is held at zero in the model, and the alternative freed it: the test was of
that zero and of the factor together, with one degree of freedom too many
for each such lag.

## Reproduction

`bugs/BUG-0029-repro/repro.py`: the echelon form (1, 1) of IPI and WTI with
`theta_IPI[B^1]` at zero. The C gives LR 0.4849 with 2 degrees of freedom;
the port gave 1.3909 with 3, and `theta_IPI[B^1]` among the alternative's
free coefficients.

## Root cause

`without_row_factor` took `{l + jj for l in range(ar + 1) for jj in fa}`: every
lag from 0 to the degree as a lag the factor starts from.

## Fix

The factor starts from lag 0 and from each FREE own lag
(`LadderSeries.own_free_lags`). An own lag held at zero stays at zero in the
alternative, as in the C.

## Validation

The repro exits 0: LR 0.4849, 2 degrees of freedom, the C's.
`tests/test_homologation_c_route_v.py`.
