---
id: BUG-0018
title: A line-search step with log(Q_ii/Q_11) > 709 raises OverflowError out of Ladder.fit; the C rejects the point and goes on
status: fixed
severity: medium
component: ladder
found_in: 0.1.7
fixed_in: 0.2.0
reported: 2026-10-11
reporter: David / Claude — echelon stage 2's fits study
tags: [ladder, optimizer, overflow, motor-c]
references:
  - src/drvarma/ladder.py (Ladder.cast)
  - atsw-gui engines/drvarma/src/escalera.c:418
  - tests/test_rows.py
  - research/echelon_seed/fits.py
---

## Summary

`Ladder.cast` built Q with `math.exp(log(Q_ii/Q_11))`. In an
over-parameterised fit the line search can try a step with that log above
709, and `math.exp` raises `OverflowError`, which aborts the whole fit. The C
(`escalera.c:418`) gets `inf` from `exp()`, the point fails in the
likelihood, and the search shortens the step.

## Impact

Any Ladder fit whose search wanders that far: a VARMA(2,2) on three
simulated series (echelon indices (2,2,1), generic, nobs 200, rep 0) died
after 4 s instead of stopping with termcode 3. In sima this surfaced as an
exception instead of a fit to diagnose.

## Reproduction

```python
L = Ladder(PAIR, 1, 1); v = L.pack()
v[L.names().index("log(Q[IPC_FR]/Q[IPC_ES])")] = 1000.0
L.cast(v)                   # OverflowError
```

## Root cause

`math.exp` raises an exception where C's `exp` returns `inf`.

## Fix

At `log(Q_ii/Q_11) ≥ 700` or NaN, `cast` returns ifault 6, so the point is
inadmissible. The objective then gives the search its penalty value, as for
any inadmissible point.

## Validation

`tests/test_rows.py::test_a_log_variance_out_of_range_is_inadmissible`. The
(2,2,1) fit now ends with termcode 3 and logL −799.117.
