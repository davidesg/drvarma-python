---
id: BUG-0013
title: The cointegration warning uses ADF critical values on a residual, and always normalises on the first series
status: wontfix
severity: low
component: mcp
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [cointegration, engle-granger]
references: []
---

## Summary

The Engle-Granger warning tests the static regression's residual with
`unit_root_tests`, which uses the standard ADF critical values instead of
MacKinnon's for residuals, so it warns too often. The result also depends on
which series is the dependent one: it always normalises on the first.

## Impact

Spurious "possible cointegration" warnings, and a warning that changes with
the order of the series.

## Reproduction

Review §2.5. Still so on 2026-09-28 (`mcp_server.py`, the COINTEGRATION
block).

## Root cause

The cheap version of Engle-Granger, as its own comment says.

## Fix

MacKinnon's residual-based critical values (they depend on m), and the test
with each series as the dependent one (warn only if the evidence agrees), or
Johansen.

## Validation

A simulated pair of independent random walks: no warning at the nominal
rate.

## Decision (2026-10-04): documented, not rebuilt

sima supersedes this path and sends cointegration to drvec, where
Johansen's test decides. The Engle-Granger notice in `characterize_series`
stays as a hint and now says so: it uses ADF critical values, not
MacKinnon's for residuals, so it warns too often, and it normalises on the
first series, so it can change with the order. Johansen, via sima → drvec,
is what decides.

Test: `tests/test_bug_0011_0013_limitaciones.py` (a cointegrated pair
raises the notice with that text).
