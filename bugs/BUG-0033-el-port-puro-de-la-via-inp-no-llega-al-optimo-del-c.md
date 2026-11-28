---
id: BUG-0033
title: The pure-Python .inp path is not the C's — on PSW VARMA(1,1) it stops 0.22 below the C's optimum, and on the boundary its BFGS standard errors are another path's
status: open
severity: medium
component: engine
found_in: 0.3.4
fixed_in:
reported: 2026-11-28
reporter: David / Claude — the resync of csrc with the C (BUG-0002)
tags: [inp-path, estimate_py, homologation, pure-python]
references:
  - bugs/BUG-0033-repro/repro.py
  - tests/test_inp_path_as_the_c.py (the four cases marked xfail)
  - tests/data/inp_c/cases.json (the C's numbers)
  - src/drvarma/estimate_py.py
  - BUG-0002
---

## Summary

The deprecated `.inp` path (`estimate_w`, `Model`) was homologated against
the frozen 0.4.1 program, not the current C. With `csrc/` resynced to the C
of record (BUG-0002), the port with the compiled engine gives the C's
parameter table line for line on six cases of the C's battery. The
pure-Python estimator (`estimate_py`) does not, on four of them:

| case | ΔlogL (pure − C) | what |
|---|---|---|
| c1 IPC AR(1) | 0 | fdhess SEs to 1.6e-4: fine |
| c2 IPC ARMA(2,1) diag Σ | −0.00005 | on the boundary: BFGS SEs up to 2× |
| c3 PSW ARMA(1,1) | **−0.218** | stops elsewhere |
| c5 EU diag AR/MA | 0 | same logL, estimates 1e-2 apart (a flat direction?) |
| c6 IPC AR(1), method 2 | 0 | fine |
| c7 IPC ARMA(1,1) twostep | +0.0016 | the C stops lower |

## Impact

Medium. Only the deprecated `.inp` path and only without the compiled engine
(no binary wheel: free-threaded builds, 32-bit Windows, DRVARMA_NO_ENGINE).
The ladder, what sima uses, is homologated separately and is not affected.

## Reproduction

`bugs/BUG-0033-repro/repro.py`: PSW VARMA(1,1) with DRVARMA_NO_ENGINE=1
gives logL −871.542051; the C, −871.32401308. Exits 1 while open.

## Root cause

Not studied. The C's `.inp` path moved after 0.4.1 (the line-search
objective, the boundary sentinel in `objcfunc`, `varma_lik`, the start
`init_varma`/`twostep`), and `estimate_py` followed 0.4.1. To be found by
following the two paths iteration by iteration on c3, as BUG-0030 did for
the ladder.

## Fix

—

## Validation

—
