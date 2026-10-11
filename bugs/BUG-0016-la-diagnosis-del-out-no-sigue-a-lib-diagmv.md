---
id: BUG-0016
title: The .out's residual diagnosis and the line search lag behind the C (drvec BUG-41 and BUG-42)
status: fixed
severity: medium
component: diagnostics
found_in: 0.1.7
fixed_in: 0.2.0
reported: 2026-10-11
reporter: David / Claude — following atsw-gui PR #22
tags: [hosking, jarque-bera, ljung-box, lnsrch, motor-c]
references:
  - BUG-0015
  - atsw-gui lib/diagmv/diagmv.c
  - atsw-gui lib/optim/lnsrch.c
  - atsw-gui engines/drvarma/BUGS.md
  - src/drvarma/diagnostics.py
  - src/drvarma/report.py
  - src/drvarma/_qnewt.py
  - csrc/internal/qnewtopt.c
---

## Summary

On 2026-10-11 the C, in atsw-gui, gathered the multivariate residual
diagnosis of drvarma, drtran and drvec in a single `lib/diagmv` and fixed it
(drvec BUG-42). It also fixed the line search's give-up branch (BUG-41). The
port lagged behind in four places:

1. **Hosking's lags in the `.out`**: `report.py` used `floor(sqrt(n))`, and
   `Model.diagnostics` used freq + 2. The C now takes freq + 2 (10 for
   non-seasonal data), at most n - 2, everywhere.
2. **Multivariate Jarque-Bera**: a sum of univariate JBs, which is chi2(2m)
   only when the residuals are uncorrelated. With correlated residuals it
   over-rejected (7-10 % at a nominal 5 %). It is now Lütkepohl's (2005, 4.5):
   the residuals are standardised by the SYMMETRIC square root of their
   covariance, and the skewness and kurtosis parts are each chi2(m).
3. **Per-series Ljung-Box** in the residual panel: it subtracted 0. It now
   subtracts the equation's coefficients on its own lags, p + q (the diagonal
   of every phi_k and theta_j).
4. **`lnsrch`**: when it gave up on x-convergence it put the point back to
   xk but returned the rejected trial's objective. This was in `_qnewt.py`
   and in the wheel's copy of the C (`csrc/internal/qnewtopt.c`). The value
   at the optimum came out wrong in the 9th or 10th significant digit (in the
   bench: `c3_psw_arma11`, `c7_ipc_twostep`).

The histogram cell also follows the C. A count wider than its cell prints as
`**` or `****` (the C overflowed the row, BUG-42.1).

## Fix

- `diagnostics.hosking_lags(freq, nobs)`.
- `jarque_bera_mv(res, parts=False)`, Lütkepohl.
- `report._diagnostics_block` writes `multivariate_diagnostics` from
  lib/diagmv line for line, including the `df = m^2 s - k` line and the
  skewness and kurtosis line.
- `residual_report` passes `p + q` to `plot_cor`.
- `_qnewt.lnsrch` and `csrc/internal/qnewtopt.c` set `fkp1 = fk`.

The Python ladder writes no multivariate block, so nothing changes there.

## Validation

The C comparison tests now check the diagnosis against the drvarma built in
atsw-gui (`tests/_cdiag.py`, `DRVARMA_C_BIN`), and `tests/_bug0015.py` is
gone: the C now subtracts k too. The other sections are still compared with
`drvarma_v.04.1`.

- On IPC3, VAR(3) and VARMA(1,1), with both pure Python and CFFI, the
  `MULTIVARIATE RESIDUAL DIAGNOSTICS` block and the 288 lines of the ACF and
  PACF correlograms are byte-identical to the C.
- `test_diagnostics.py::test_hosking_jb_match_c` fixes lag 14, the same Q and
  df, and the same JB.
