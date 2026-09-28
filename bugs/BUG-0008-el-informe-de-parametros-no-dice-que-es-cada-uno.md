---
id: BUG-0008
title: confirm_and_estimate labels every parameter theta[i] and gives t-ratios to the covariance parameters
status: open
severity: medium
component: mcp
found_in: 0.1.7
fixed_in:
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [report, labels, t-ratios]
references: []
---

## Summary

`confirm_and_estimate` prints the estimates as `θ[0] … θ[k]`, the packing
order of the optimizer. The name clashes with the MA's Θ and does not say
which element of which matrix each one is (μ, Φₖ[i,j], Θₖ[i,j], qq). It also
prints a t-ratio and a significance star for the covariance elements
(`θ[8] = 1.9035`, t = 914.4), which have no inferential meaning in the
concentrated, scale-invariant parametrisation.

## Impact

An analyst, or an LLM, cannot tell that `θ[4] = −2.65` is Φ₁[WTI,IPC]
without knowing the internal packing. The covariance t-ratios invite a
reading that has no meaning.

## Reproduction

Review §2.2 and §4.2: `load_data`, `characterize_series`,
`confirm_and_estimate("x", 1, 0)` on IPC + WTI. Still so on 2026-09-28:
`mcp_server.py`, the `θ[{i}]` loop in `confirm_and_estimate`.

## Root cause

The tool formats `mod.params` and `mod.std_errors` directly, without the
labels `report.py` already builds (`mu[i]`, …).

## Fix

Label as `mu[IPC]`, `Phi1[WTI,IPC]`, `Theta1[…]`, from the same code as
`report.py`. Move the covariance parameters to a separate block with no
t-ratios or stars.

## Validation

A test on the tool's text: names, not indices; no star in the covariance
block.
