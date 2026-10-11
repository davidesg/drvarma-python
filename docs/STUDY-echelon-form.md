# Echelon stage 3a: the form, fitted — the study

*2026-10-11. Stage 3a of [`ECHELON_PLAN.md`](ECHELON_PLAN.md) §2.1. Data:
[`STUDY-echelon-form-data.md`](STUDY-echelon-form-data.md). The design is
[`STUDY-echelon-rows.md`](STUDY-echelon-rows.md)'s: the same systems, seeds
and specs; nobs 200, 50 replications.*

## The question

Two questions about `Ladder(echelon=n)`, fitted on systems whose indices are
known:
- does it contain the truth?
- does exact ML find its maximum?

## The yardstick, corrected

Stage 2 used the LR against the unrestricted VARMA(N, N) (`full`). **That is
not a valid test.** `full` contains the truth, but at a point where some of
its parameters are not identified: it has McMillan degree to spare, so
common factors cancel. At such a point the LR does not follow a χ²
distribution (the Davies problem), and it rejects more than its size.

The yardstick here is the **truth**: the echelon form evaluated at the true
parameters (`truth` in `fits.py`; σ² concentrated, mean 10). For a fit that
reaches its maximum, LR = 2 (logL_fit − logL_truth) is ≥ 0 and ~ χ²(npar).

Stage 2's comparison between `rows` and `full` is still useful as a relative
measure: `rows` is far worse. But its rejection rates are not test sizes.

## Results

Two starts are measured: zero (`echelon`, the default) and Jenkins and
Alavi's preliminary estimates (`echelon_pre`).

1. **The form contains the truth, and the search finds its maximum, in every
   system and design but (2,2,1) generic.** LR against the truth:
   - the median is close to npar in every system;
   - the fit ends below the truth in 0–2 % of the replications;
   - the share above χ²(npar)'s 95 % point is 4–14 % in the generic
     design, and 4–26 % in the weak one. The tail is heavier there, which is
     the weak identification of small cross terms in a finite sample.
2. **Compared with the forms of stage 2**, on the systems with unequal
   indices except (2,2,1), i.e. (1,0), (2,1), (1,2), (1,2,1) and (2,1,0).
   With equal indices the three forms are the same model.

   | generic design | `full` | `rows` | `echelon` |
   |---|---|---|---|
   | MA wall | 8–92 % | 0–4 % | 0–4 % |
   | termcode 3 | 26–92 % | 2–82 % | 2–10 % |

   In the weak design `echelon` stops with termcode 3 in 2–36 % of the fits.

   `echelon` has no `rows` misspecification: it contains the truth. Its time
   is close to `rows`'.
3. **(2,2,1), generic, N = 5, 38 parameters: the search fails.**
   - 10 of 49 fits end below the truth, some far below: LR = 2 ΔlogL of
     −516, −609 and −583;
   - started at the truth, the search finds a higher maximum, even in
     replications where the fit from zero was already above the truth
     (rep 1: −797.4 against −805.0);
   - the stops are termcode 3, often with an MA root on the wall.

   This is a start problem: the likelihood has local maxima the search
   from zero does not get past.
4. **The preliminary start does not help.** `echelon_pre` gives the same
   numbers as zero, within a few replications, including (2,2,1)'s 18 %
   below the truth. Jenkins and Alavi's estimates are a start for the cross
   terms of a standard VARMA, and they know nothing of A₀.

## Decision

- **The default start stays zero.** The measurement does not support
  changing it to the preliminary start.
- **Stage 3b's start is the decisive piece for large systems.** Those are
  Lütkepohl and Poskitt's stage II least-squares estimates, which include A₀.
  3b is judged on two counts:
  - does it close (2,2,1)'s 20 %?
  - does it keep the other systems at ≤ 2 %?
- **The yardstick for 3b and later is the LR against the truth**, not
  against `full`.
- The (2,2,1) gate failure (BUG-0020, rep 45) recurs in every form, as
  expected: it is in the univariate step.

## Reproduce

```
cd research/echelon_seed
python3 fits.py --forms full,rows --out fits.csv                # stage 2
python3 fits.py --forms echelon --out fits_echelon.csv          # ~35 min
python3 fits.py --forms truth,echelon_pre --out fits_more.csv   # ~35 min
python3 fits_analyze.py fits.csv fits_echelon.csv fits_more.csv
```
(with `--reps 50 --nobs 200 --workers 4`).
