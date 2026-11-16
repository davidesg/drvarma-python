---
id: BUG-0026
title: The HTML forecast page multiplies the variations by 100/scale whatever lambda — in levels it prints 100 times the change, labelled "(%)"
status: fixed
severity: medium
component: report_forecast
found_in: 0.3.3
fixed_in: 0.3.4
reported: 2026-11-16
reporter: David / Claude — what was left of the scale round (atsw-gui docs/ESTUDIO-escala.md)
tags: [forecast, scale, lambda-1, report]
references:
  - src/drvarma/report_forecast.py (_forecast_arrays sc = 100.0 / scale; _table_data; _make_charts_svg)
  - BUG-0025 (the .forecast, fixed in 0.3.3; its text said the page followed it, and it did not)
  - fue BUG-0028 (fue's page, of which this one is the multivariate port; fixed in fue 0.1.22)
  - atsw-gui gui/sim/src/prevision.c (the C forecast chart of the VARMA, by the rule since atsw-gui #177)
  - bugs/BUG-0026-repro/repro.py
---

## Summary

`write_forecast_report` — the SPS page per series — wrote the period and
annual variations, their standard deviations, the ERR column, the annual chart
and the ERR bands as `100 * delta / scale`, under "(%)" headings, whatever
lambda was. With lambda = 0 that is a percentage. With lambda = 1 it is the
change of the level times 100:

```
level 100.4529 -> 100.2198: change -0.2331; page shows -23.3112; '(%)' labels on the page: 9
```

The page says its numbers are the `.forecast`'s, and since 0.3.3 (BUG-0025)
the `.forecast` uses `(100 if lambda = 0 else 1)/scale`. BUG-0025 listed this
page as following the same rule; it had not been changed.

## Impact

Medium. In levels the page shows variations 100 times too large, labelled as
percentages, and disagrees with the `.forecast`, with fue's page and with
atsw-gui's gui/sim for the same model. Logs (the usual case) are unaffected:
the page is the same, byte for byte apart from the chart's timestamp.

## Reproduction

`bugs/BUG-0026-repro/repro.py`: two synthetic monthly AR(1) series about 100,
lambda = 1, scale = 100. Exits 1 while the first forecast's monthly change on
the page is 100 times the change of the level, or the page says "(%)".

## Root cause

The page was ported from fue's before fue BUG-0028 was fixed, with fue's
`100.0 / refactor`.

## Fix

`_pct(lam)`: 100 with logs, 1 otherwise, for the table, the chart and the ERR
bands; the headings say "(%)" only with logs ("Monthly change", "Annual
change", "Value" otherwise), as fue's page. A history row with no datum a year
before shows "—" instead of a level passed off as a variation.
