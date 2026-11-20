---
id: BUG-0028
title: The ols start falls back to zero in the port but not in the C ladder, which starts from the inadmissible least-squares point and fails
status: open
severity: medium
component: ladder
found_in: 0.3.5
fixed_in:
reported: 2026-11-20
reporter: David / Claude — review of -start ols on route V (gas furnace)
tags:
  - ladder
  - start
  - c-engine
  - homologation
references:
  - src/drvarma/ladder.py Ladder._take_start (the port: restores x0, then zeroes the cross terms)
  - atsw-gui engines/drvarma/src/escalera.c start_ols() (the C)
  - atsw-gui engines/drvarma/BUGS.md BUG-0006, PR #203 (the fix in the C)
  - bugs/BUG-0028-repro/repro.py
---

## Summary

With `start="ols"`, when elf does not accept the least-squares point, the port
goes back to zero (`_take_start`) and says so in `start_used`. The C's ladder
(`-start ols`) prints the same "zero (the ols start was not admissible)" but
starts from the least-squares point and fails with "AR nonstationary" (rc 3).
The port is right; the two engines disagree on this case.

## Impact

The port and sima are not affected. The C (sim_gui, atsw-cli) does not pass
`-start` yet, but route V will, and route V is where the least-squares start
is most likely inadmissible (the VAR(p) estimate together with the own AR of
the diagonal fit).

## Reproduction

`bugs/BUG-0028-repro/repro.py`: the gas furnace by route V, VAR(6). The port
reaches -2638.762693 from both starts; the C with `-start ols` fails (rc 3).
It needs the C binary (`DRVARMA_C`). Exits 1 while open.

## Root cause

In the C, not here: `start_ols()` zeroes the cross terms and then calls
`fit_free(&chk)`, whose last `shootx_esc(lastx = 1)` unpacks `chk.x` and
puts the least-squares point back.

## Fix

In the C (atsw-gui BUG-0006, PR #203): free the check fit before zeroing.
Nothing changes in the port.

## Validation

With the C of #203 the repro exits 0: -2638.762693 from both engines.
