---
id: BUG-0025
title: The ladder's forecast table multiplies the variations by 100/refactor whatever lambda — in levels it prints 100 times the change
status: fixed
severity: medium
component: ladder
found_in: 0.3.2
fixed_in: 0.3.3 (unreleased)
reported: 2026-11-15
reporter: David / Claude — review of inconsistencies between the suite's components, before mtram
tags: [forecast, scale, lambda-1, legado-c]
references:
  - src/drvarma/ladder.py:1886 (_forecast_lines, sc = 100.0 / refactor)
  - src/drvarma/report_forecast.py:226, :404 and src/drvarma/report.py:101 (the drvarma CLI: the same rule)
  - atsw-gui engines/drvarma/src/escalera.c (the forecast table, sc = 100.0 / refactor) and drvarma.c (-forecast)
  - atsw-gui engines/drvarma/src/escalera.c:1279 (its accuracy table: vs = (lambda = 0 ? 100 : 1)/refactor)
  - fue BUG-0027, atsw-gui engines/fuf BUG-0001 (the same defect, fixed 2026-11-15)
  - bugs/BUG-0025-repro/repro.py
---

## Summary

`ladder._forecast_lines` — the `.forecast` file and the forecast block of the
`.out`, which is what sima's `forecast` shows — writes mon%/ann% and their
stds as `100 * delta / refactor`. With lambda = 0 that is a percentage. With
lambda = 1 it is the change of the level times 100:

```
Y: last observation 103.8379, first forecast 102.3802: change -1.4577
the table's mon%: -145.7706
```

fuf (its BUG-0001), fue (BUG-0027), drtran's C and CLI, the C ladder's own
accuracy table and sima's `forecast_uncertainty` all use
`(100 if lambda = 0 else 1) / refactor`. The ladder's forecast table, the C's
and this port's, does not; the header even says "(100*delta/refactor; % for
lambda=0)", i.e. it knows the number is not a percentage otherwise and prints
it anyway.

The drvarma CLI's forecast table and chart (`report.py`, `report_forecast.py`)
follow the same rule with drvarma's `scale`.

## Impact

Medium. In levels the variations and their stds come out 100 times larger,
under a "mon%" heading, and disagree with fue's and drtran's tables for the
same series. Logs (the usual case) are unaffected.

## Reproduction

`bugs/BUG-0025-repro/repro.py`: two synthetic monthly series about 100 in
levels, each an AR(1) with mu fitted by fue with lambda = 1 and refactor = 100;
the ladder VARMA(0,0) and its forecast. Exits 1 while the first forecast's mon%
is 100 times the change of the level.

## Root cause

Ported faithfully from the C (`escalera.c`, `drvarma.c`), which has the defect
fuf had until 2026-11-15.

## Fix

With the C, on the same day (atsw-gui `engines/drvarma` BUG-0005): the engines
are homologated. `(100 if lambda = 0 else 1) / refactor` in `_forecast_lines`
(the ladder) and in `report.write_forecast` (the `.inp` path). With every
series in logs nothing changes, header included; otherwise the header states
both rules, in the C's words. The deprecated `.inp` path's HTML page
(`report_forecast.py`) keeps the old rule: it has no counterpart in the C.

## Validation

The repro exits 0: -1.4577 where it said -145.7706, the number the C prints
for the same files. `tests/test_bug_0025_tabla_en_niveles.py`. The suite: 311
pass.
