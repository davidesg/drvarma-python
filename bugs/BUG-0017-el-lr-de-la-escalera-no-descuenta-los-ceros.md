---
id: BUG-0017
title: Ladder.lr_test counts (p + q) cross coefficients per linked pair, whatever `zeros` fixed, so the LR's df is too large
status: fixed
severity: medium
component: ladder
found_in: 0.1.7
fixed_in: 0.2.0
reported: 2026-10-11
reporter: David / Claude — while adding rows= (echelon stage 2)
tags: [ladder, lr-test, grados-de-libertad, sima]
references:
  - src/drvarma/ladder.py (Ladder.lr_test)
  - tests/test_rows.py
  - sima-python/src/sima/evidence.py (estimation_text)
  - sima-python/src/sima/mcp_server.py (estimate)
---

## Summary

`lr_test()` computed its df as `(p + q) · (linked pairs) + covariances`.
A cross coefficient fixed by `zeros` is not a parameter, but it was counted.
The LR against the diagonal then had one df too many per zero, and its
p-value came out too high.

## Impact

Every LR of a model with `zeros`, i.e. after Tiao and Box's simplification by
coefficient in sima: `estimate` prints `L.lr_test()`, and so does
`evidence.estimation_text`. sima's `simplify` is not affected: its LR takes
the df from the difference of `npar`. The C has no `zeros`, so this was in
the port only.

## Reproduction

```python
L = Ladder(["IPC_ES_m10.pre", "IPC_FR_msar.pre"], 1, 1, zeros="MA1[IPC_ES<-IPC_FR]")
L.fit(); L.lr_test()[1]     # 5; the model has 3 cross coefficients + Q12 = 4
```

## Root cause

`lr_test` counted lags per pair instead of the free cross slots.

## Fix

The df counts each linked pair's row orders (`rows=`, stage 2), minus
`zeros`, plus the covariances. For any model this equals the difference
of `npar` against the diagonal system.

## Validation

`tests/test_rows.py::test_the_lr_counts_rows_and_zeros` checks df = 4 for the
case above, and that it equals the `npar` difference.
