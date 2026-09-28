---
id: BUG-0010
title: The meaning of termcode 3 contradicts itself inside the package
status: open
severity: medium
component: optimizer
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [termcode, convergence, report]
references: [BUG-0007]
---

## Summary

`estimate_py.py` says a line search that cannot find a lower point
(termcode 3) "means it is AT the optimum, so it is not a fault". The MCP
server says the opposite: "NO es convergencia: el optimizador se rindió".
The C prints STOPPED. The review's VARMA(1,1) on IPC + WTI (λ = 0, d = 1, no
seasonal adjustment) ends with termcode 3, ifault 0, Θ₁ eigenvalues at
modulus 1.00005 and Φ₁ ≈ Θ₁ (near-common factors).

## Impact

The same fit is read as converged or not depending on which layer reports
it. `ifault` does not flag the MA on the invertibility wall.

## Reproduction

Review §2.4 and §4.5. Still so on 2026-09-28 (`estimate_py.py`, the
termcode-3 comment; `mcp_server.py`, the "se rindió" warning). TODO.md
already lists this as an open follow-up.

## Root cause

termcode 3 arises both when the search sits at the optimum and when a bad
direction under ill-conditioning stalls it. Each layer picked one reading.

## Fix

One reading, stated where it is decided, and facts instead of verdicts: the
ladder already reports a stop on the MA wall (one tolerance, 5e-5, since
2026-09-28, as the C). Do the same in `Model`/`estimate_py`, and add a
near-common-factor note when Φ and Θ nearly cancel (sima's `roots_text`
already computes it).

## Validation

The review's §4.5 case reports the stop on the wall; one test per layer
with the same wording.
