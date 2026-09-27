---
id: BUG-0007
title: The ladder reports an MA(1) coefficient in its non-invertible form (1.0722 instead of 0.9327) — the model is right, the report and the gate's "move" are not
status: fixed
severity: medium
component: ladder
found_in: 0.2.0.dev
fixed_in: 0.2.0
reported: 2026-09-27
reporter: David / Claude — the mtram → sima hand-over on m6
tags: [ladder, ma, invertibility, report]
references:
  - src/drvarma/ladder.py (LadderSeries.canonicalize, Ladder._fit)
  - fue.cast_us.cast_us_py (the flip)
  - tests/test_ladder.py::test_m6_reports_the_invertible_ma_as_the_c_does
---

## Summary

On m6 (six quarterly series, from drtran's data), the ladder reported
`theta_EU[B^1] = 1.0722`. The C reported `0.932651`.

## Impact

The likelihood, the residuals and the forecasts were right, because the model
is the same. What was wrong is what the analyst reads:
- a non-invertible MA coefficient in the parameter table, with its standard
  error computed on that side;
- a gate that said EU had moved 0.192 from its `.pre`, when it had moved
  0.0527.

## Reproduction

`Ladder(m6_files, 0, 0, diagcov=True).fit()` and read `theta_EU[B^1]`.

## Root cause

`fue.cast_us.cast_us_py` flips a regular MA(1) factor with |θ| > 1 to 1/θ,
and a fixed-frequency MA with c₂ < −1 to 1/c₂, before it builds the
polynomial. The objective is therefore identical at θ and at 1/θ, and the
optimiser stopped at 1.0722. The C engine checks invertibility with `chekma`
and never leaves the invertible side.

## Fix

`LadderSeries.canonicalize()` applies fue's own rule to the parameter vector
after the optimisation: the value reported is the one the model uses. The
standard errors are computed at that point. The test pins the C's numbers:
θ, its SE, the gate's per-series moves, and the gate's difference.
