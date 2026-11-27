---
id: BUG-0032
title: DRVARMA_NO_ENGINE makes engine_backend() answer "python" but elf_c and marma_c ignore it and compute in C
status: fixed
severity: low
component: engine
found_in: 0.3.4
fixed_in: unreleased (after 0.3.4)
reported: 2026-11-27
reporter: David / Claude — timing the port for the speed study (TODO.md)
tags: [engine, environment, diagnostics]
references:
  - src/drvarma/_engine.py:42 (engine_backend), :73 (estimate_w), :166 (marma_c), :205 (elf_c)
  - bugs/BUG-0032-repro/repro.py
---

## Summary

The variable is documented as "asks for the port". `estimate_w` honours it, and
so does `engine_backend()`, which then answers "python". `marma_c` and `elf_c`
call `_load_c()` directly and keep computing in C. Those two functions are what
the ladder, and so sima, use.

## Impact

The diagnosis is misleading. Timing sima "on the port" with the variable
measures the C: 11-12 s in both cases on the monetary system, while the real
port does not finish in 36 minutes. That is how the first measurement of the
port's speed (2026-11-27) went wrong. And a provenance that records
`engine_backend()` would say "python" for a computation done in C.

## Reproduction

`python3 bugs/BUG-0032-repro/repro.py`

```
engine_backend() with DRVARMA_NO_ENGINE=1: 'python'
elf_c: 0.15 ms without the variable, 0.13 ms with it, 15.7 ms on the port
OPEN: the variable says python and elf_c computes in C
```

## Root cause

The check `if not os.environ.get("DRVARMA_NO_ENGINE")` only appears in
`estimate_w`. The other entry points were added later (the ladder's scored
likelihood) and call `_load_c()`.

## Fix

Move the check into `_load_c()`, or into one helper used by every entry point,
so that the variable means the same thing everywhere `engine_backend()` speaks.

## Validation

The repro: with the variable set, `elf_c` costs what the port costs.

## Resolution (2026-11-27)

The check moved into `_load_c()`, which every entry point uses: with the
variable set it returns None and `engine_load_error()` says
"DRVARMA_NO_ENGINE is set". `estimate_w` and `engine_backend()` lost their own
copies. `tests/test_bug_0032_no_engine_en_todas.py`.
