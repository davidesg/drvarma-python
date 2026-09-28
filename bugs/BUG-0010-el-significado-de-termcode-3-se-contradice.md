---
id: BUG-0010
title: The meaning of termcode 3 contradicts itself inside the package
status: fixed
severity: medium
component: optimizer
found_in: 0.1.7
fixed_in: 0.2.0
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

## Resolution (2026-09-28)

**In the engine, one reading of termcode 3, and the wall as a fact.**

- `estimate_py`: the comment no longer calls termcode 3 "AT the optimum";
  it says the code is read neither way there.
- `report.py`: termcode 3 is no longer "the estimates are not a maximum". It
  says what happened (the last line search found no lower point), that this
  happens both at an optimum reached to rounding and in a stall, and how to
  tell them apart (re-estimate from the values: an optimum stays put).
- `Model.ma_boundary` / `Model.ma_nroots`: MA inverse roots within 5e-5 of
  the unit circle, the rule of the ladder and the C (atsw-gui MA_WALL_TOL).
  `Model.converged` is False on the wall whatever the termcode, and the
  report writes "OPTIMIZER STOPPED at the MA invertibility boundary" and "MA
  boundary: k of n …", as the C.
- The near-common-factor note stays with the assistant: sima-tseries'
  `study_estimation` (`roots_text`) already reports near-common AR/MA pairs.

**Validation:** `tests/test_bug_0010_termcode_3_y_la_pared_ma.py`, the
review's case (data in `tests/data/bug_0010/`): Θ₁ at 1.000049, termcode 3,
ifault 0 → ma_boundary 2 of 2, converged False, and the report as above.

**The old server's "NO es convergencia: el optimizador se rindió" is not
touched.** The old sima (`drvarma.mcp_server`) is what atsw 1.6.1 installs today, as
the `sima` command of drvarma 0.1.7. From drvarma 0.2.0 that command belongs
to sima-tseries, built on the ladder, and the old server stays only as
`sima-legacy`, deprecated. Decided on 2026-09-28 not to patch it.
