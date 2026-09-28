---
id: BUG-0009
title: load_data takes a date column as a series, and the sample start is not read from it
status: open
severity: medium
component: mcp
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [load_data, dates, calendar]
references: []
---

## Summary

With a CSV `date,IPC,WTI`, the format of the package's own example files,
`load_data` loads `date` as a third series (mean = nan). `characterize_series`
then characterises it (λ = 0, d = 1). The failure only appears at estimation,
with a misleading message ("the transformed series is not finite for ['date']
(lam=0.0) … needs strictly positive data"). The start of the sample is not
inferred from the dates either: it defaults to (2000, 1) while the data start
in 2002-01.

## Impact

The default flow with the example data breaks late, with a message that
points at the wrong cause. If `start_year` is not passed, the calendar is
shifted, and with it the phase of the seasonal adjustment and the forecast
dates.

## Reproduction

Review §2.3 and §4.4. Still so on 2026-09-28: `load_data` has no date or
non-numeric column handling.

## Root cause

`load_data` converts every column to numeric and keeps all of them.

## Fix

Detect non-numeric or date columns and leave them out of the series; use
them to infer `freq` and `start` when not given (and say so); refuse any
column that ends up entirely NaN.

## Validation

Tests with `date,IPC,WTI`: two series, start (2002, 1), freq 12.
