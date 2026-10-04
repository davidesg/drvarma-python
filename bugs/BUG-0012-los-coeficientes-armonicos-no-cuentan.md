---
id: BUG-0012
title: The harmonic seasonal coefficients are not counted in degrees of freedom, k or forecast uncertainty
status: wontfix
severity: low
component: deseason
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [methodology, seasonality, df]
references: [BUG-0003]
---

## Summary

The s − 1 harmonic coefficients of the seasonal adjustment (11 for monthly
data) enter neither the degrees of freedom of Hosking's test, nor k in the
information criteria, nor the uncertainty of the forecast bands.

## Impact

Hosking's p-values and the information criteria are slightly optimistic, and
the forecast bands too narrow, whenever the series is seasonally adjusted.

## Reproduction

Review §2.5. Still so on 2026-09-28 (`diagnostics.hosking_q`: df = m²·s).

## Root cause

The adjustment is a preprocessing step outside the model.

## Fix

To decide: count the harmonics where the adjustment was estimated, or say in
the report that they are not counted.

## Validation

Pending the decision.

## Decision (2026-10-04): documented; the Hosking part is BUG-0015

The adjustment stays a preprocessing step; the output says what it does not
count:
- `diagnose`, when the harmonic adjustment was applied: the s−1 coefficients
  per series are not in k, so AIC/BIC leave them out. That does not change
  the ranking of orders, because every candidate carries them. And the
  forecast bands leave out their uncertainty.
- `generate_forecast`: the bands are somewhat narrow for that reason.

Hosking's degrees of freedom: checking this showed a larger defect, now
BUG-0015. `hosking_q` subtracts no estimated parameter at all, not even the
ARMA ones, and the test almost never rejects. The harmonics themselves, to
first order, do not change the asymptotic distribution of the residual
autocorrelations (they are coefficients of deterministic regressors), so
they belong in k for the criteria and the bands, not in Hosking's df.
