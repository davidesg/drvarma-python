---
id: BUG-0019
title: An exactly singular fdhess Hessian passes Cholesky and then raises LinAlgError out of Ladder.fit, instead of falling back to BFGS and saying so
status: fixed
severity: medium
component: stderr
found_in: 0.1.7
fixed_in: 0.2.0
reported: 2026-10-11
reporter: David / Claude — echelon stage 2's fits study
tags: [stderr, fdhess, identification, ladder]
references:
  - src/drvarma/stderr.py (fd_covariance)
  - src/drvarma/ladder.py (Ladder._std_errors)
  - tests/test_stderr.py
  - research/echelon_seed/fits.py
---

## Summary

`fd_covariance` checks that the Hessian is positive definite with a
Cholesky factorisation, then inverts it. A Hessian whose smallest eigenvalue
is about 2e-16 passes Cholesky, `np.linalg.inv` raises `LinAlgError`, and
the whole fit is lost. The intended path when the Hessian cannot be used,
falling back to BFGS and saying why, was never reached.

## Impact

Any model with an exactly unidentified direction. In the echelon study this
was the `rows=` fit of an echelon (1,0) system, weak design, nobs 200, rep 38:
equation 1 has a cross AR1 and a cross MA1 on series 2. Series 2 has index 0,
so it is white noise, and the two coefficients are the same regressor. The
echelon restriction table removes exactly that AR term (n_12 = 0). Through
sima this is the kind of over-parameterised model an analyst tries, and it
showed up as an exception instead of a fit with a warning.

## Reproduction

`research/echelon_seed/fits.py`: `fit_form("rows", (1, 0), data((1, 0),
"weak", 200, 38), tmp)`. The Hessian has eigenvalues
[2.0e-16, 0.187, 0.474, …], and two of its diagonal entries are identical
(1.8793).

## Root cause

Cholesky is taken as the only test of positive definiteness, but a
numerically zero eigenvalue can pass it.

## Fix

If `inv` refuses, `fd_covariance` returns no covariance with
`singular=True`. The Ladder then falls back to BFGS, and `se_method` says
"the Hessian is singular: some coefficients are not identified".

## Validation

`tests/test_stderr.py::test_a_singular_hessian_is_refused_not_raised`. The
reproduction case now fits and reports BFGS standard errors with that reason.
