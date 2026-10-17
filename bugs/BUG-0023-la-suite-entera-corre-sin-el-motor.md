---
id: BUG-0023
title: A full pytest run computes with the pure-Python port — collecting test_regression_bugs.py sets DRVARMA_NO_ENGINE for the whole session
status: fixed
severity: medium
component: tests
found_in: 0.2.1
fixed_in: 0.3.0
reported: 2026-10-17
reporter: David / Claude — running the full suite for drvarma 0.3.0 (GSL's eigensolvers)
tags: [tests, engine, isolation]
references:
  - tests/test_regression_bugs.py (module level)
  - bugs/BUG-0023-repro/repro.py
  - tests/test_stderr.py test_review_case_against_ols
---

## Summary

`tests/test_regression_bugs.py` runs `os.environ.setdefault("DRVARMA_NO_ENGINE",
"1")` at module level. pytest imports every test module during collection,
before running any test, so in a full run (`pytest -q`, the CI's command)
every test after collection runs with the variable set: `estimate_w`,
`elf_c` and `marma_c` use the pure-Python port, and whatever a test meant to
check on the compiled engine is checked on the port. Run file by file, the
same tests use the engine.

## Impact

Medium, on the tests only (no user-visible code is involved):
- the compiled engine is barely exercised by the CI's full run, which is
  what the suite is for; a defect of the C path would go unseen;
- the full run is very slow: `test_bug_0015_hosking_gl.py::
  test_model_diagnostics_has_the_size_it_says` (150 VARMA(1,1) fits) takes
  ~9 s alone and did not finish in 20 minutes inside the full run;
- tests that need the engine and check for it with `importorskip` (the
  extension imports fine) run on the port anyway; tests that ask
  `engine_backend()` or `gsl_available()` at collection time and call the
  engine at run time fail (the new `tests/test_gsl_eigen.py` would raise
  EngineUnavailableError).

## Reproduction

    python3 bugs/BUG-0023-repro/repro.py

imports the module as collection does and prints the variable afterwards;
exits 1 while it leaks.

## Root cause

A process-wide side effect at import time, meant for that module's tests
only (they compare against the pure-Python path).

## Fix

An autouse fixture in that module that sets the variable with `monkeypatch`
for each of its tests and restores it afterwards; nothing at module level.

What the leak hid: `test_stderr.py::test_review_case_against_ols` fails on
the compiled engine. Its anchor, phi21's s.e. 2.40 at 2%, was measured on
the port; the engine gives 2.4496 and the port 2.4434 (the finite-difference
Hessian differs in the last digits between the two likelihoods; the logL and
the estimates agree to 1e-9, and both s.e. are 0.99 of the OLS ones, which the
test also checks). It fails the same with 0.2.1's engine: not a regression.
The anchor is now 2.45 at the same 2%, which both engines meet.

## Validation

The repro exits 0. The full suite runs with the engine except in that
module: 4 min 23 s instead of more than 50 (stopped there), and
test_review_case_against_ols passes on both engines.
