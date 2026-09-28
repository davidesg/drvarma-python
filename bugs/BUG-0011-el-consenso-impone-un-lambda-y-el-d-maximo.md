---
id: BUG-0011
title: The consensus imposes one lambda and d = max(d_i), which over-differences an I(0) series
status: open
severity: low
component: mcp
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [methodology, box-cox, differencing]
references: []
---

## Summary

`characterize_series` derives one λ for all series (0 if every |λᵢ| < 0.25,
else the median) and `d = max(dᵢ)`. `Model` only takes a scalar λ. An I(0)
series is then differenced, which introduces an MA unit root, exactly what
the code elsewhere tries to avoid by capping `max_d = 1`.

## Impact

A design decision of v1, to be discussed rather than a coding error: in
mixed systems it produces over-differenced components and a model on the MA
wall.

## Reproduction

Review §2.5. Still so on 2026-09-28 (`mcp_server.py`: `d_c = max(ds)`).

## Root cause

v1 has no per-series transformation.

## Fix

To decide: per-series λ and d in `Model`, or document the limitation and
warn when the dᵢ differ.

## Validation

Pending the decision.
