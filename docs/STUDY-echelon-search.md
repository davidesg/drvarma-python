# Echelon stage 3b: the index search and the least-squares start — the study

*2026-10-12. Stage 3b of [`ECHELON_PLAN.md`](ECHELON_PLAN.md) §2.2. Data:
[`STUDY-echelon-search-data.md`](STUDY-echelon-search-data.md). Code:
`drvarma.echelon`; `research/echelon_seed/search_study.py`,
`search_analyze.py`, `fits.py --forms echelon_ls`.*

## The questions

1. Does Lütkepohl and Poskitt's stage I–II search find the Kronecker
   indices, where stage 0's univariate seed did not (0–49 %)?
2. Started at its stage II estimates, does exact ML reach its maximum in
   (2,2,1) generic, where from zero it ended below the true parameters in
   20 % of the fits (stage 3a)?

## Design

- The index search runs on stage 0's grid: 9 systems × 2 designs × nobs 100,
  200, 500 × 100 replications, the same seeds, n_max 3. Four variants:
  - Hannan and Kavalieris' two steps (`hk`) or full enumeration (`full`);
  - HQ or SC.
  
  Each runs twice: plain stage II, and with one round of HK's correction
  (each candidate's residuals recomputed from its own stage II model, then
  stage II again).
- Exact ML runs on stage 3a's design (nobs 200, 50 replications, the same
  data), with `start="echelon"`. The yardstick is stage 3a's: the LR
  against the truth, 2 (logL_fit − logL_truth).

## Results: the search

1. **The search works where the univariate seed did not.** Plain stage II
   with SC finds the true indices in 65–86 % of the generic replications on
   average (stage 0: 0–49 %).
2. **HQ overfits.** With full enumeration it finds the true indices less
   often than SC in every generic cell, and the misses checked by hand are
   larger indices. LP report both; SC is the one to use here.
3. **Plain stage II does not improve with the sample, and the plan requires
   that it must.** With full enumeration and SC its mean hit rate stays at
   74 → 84 → 84 % (nobs 100, 200, 500). The cause is two systems:
   - (2,1,0) is found in 40–57 %, and loses to (2,1,1);
   - (1,2,1) is found in 50–78 %.
   
   A longer stage I VAR makes this worse: with h at its maximum, (2,1,0) at
   nobs 500 is found in 0 % of 40 replications. So the culprit is not the
   proxies' lag length but the proxies themselves.
4. **One round of HK's correction fixes it.** With SC, the generic mean
   rises with the sample, as it should:

   | generic, SC | nobs 100 | 200 | 500 |
   |---|---|---|---|
   | plain, full | 74 | 84 | 84 |
   | refined, full | 80 | 91 | 94 |
   | refined, hk | 68 | 88 | 97 |

   - (2,1,0) goes from 40–57 % to 88–97 % (full);
   - (1,2,1) from 50–78 % to 79–99 %;
   - (2,2,1) is the price: 67–82 % → 35–68 % with full enumeration,
     overfitting to (2,2,2) and (2,3,1). The two-step search keeps it at
     44–89 %.

   A second round does not help (an extra experiment, 30 replications:
   (2,2,1) at nobs 200 drops to 33 %). Nor does giving every candidate the
   same refined residuals: those of the best equal-index model. That leaves
   (2,1,0) at its plain rate.
5. **Full enumeration against the two steps.** Over the three sample sizes,
   full is ahead (88 against 84 % on average) and its worst cell is better
   (35 against 7 %). The two steps miss (1,2,1) at nobs 100, because the
   step that lowers indices cannot raise one: if the equal-index step stops
   at 1, the 2 is out of reach. Full enumeration costs 1.3 s for m = 3 and
   n_max = 3 (64 candidates).
6. **The weak design underfits whatever the variant**: 0–6 % for (1,1),
   (1,1,1) and (1,2,1) at nobs 100–200. The weak cross terms are below
   what 100–200 observations detect, and the search drops them. The plan
   allows this: an underfitted form shows in the residual checks (stage
   IV). Refinement makes the weak (1,2) and (2,1,0) worse at nobs 100–200
   with the two steps, and changes little with full enumeration.

## Results: exact ML from the stage II start

7. **The 20 % is closed.** From the stage II estimates, no fit in any system
   or design ends below the true parameters. That includes (2,2,1) generic:
   from 20 % with zero (18 % with the preliminary start) to 0 %. Its median
   LR is 38.7 on 38 parameters, as χ²(38) says.
   - Every other system keeps its numbers or improves: 0 % below the truth,
     the same median LR, the same share above χ²'s 95 % point.
   - Of the 5 replications where zero ended below the truth (2, 7, 10, 11,
     13), all now end above it.
   - Replication 1 now reaches the higher maximum that only the refit from
     the truth had found (−797.38).
8. **The search is easier as well.** In the generic design:
   - iterations fall in every system, by 18–69 %, and time in all but (1,1),
     by up to half;
   - in (2,2,1) the MA wall falls from 28 % to 2 %, termcode 3 from 38 % to
     8 %, and time from 25 s to 13 s.
   
   In the weak design the stops are mixed. Some improve ((2,2): the wall
   from 36 % to 16 %) and some get worse ((2,1,0): termcode 2 from 18 % to
   48 %). The flat directions there are the data's, whatever the start.
9. **Least squares can leave the MA outside the unit circle.** Seen in 1 of
   the 8 replications checked by hand. The roots are then pulled inside,
   scaling M_k by λᵏ, which scales every inverse root by λ: the largest
   lands at 0.95 (`echelon.pull_inside`). `start_used` says so.

## Decisions

- **`echelon.search`'s defaults**:
  - SC;
  - one round of HK's correction (`iterate=1`);
  - full enumeration while (n_max+1)^m ≤ 256, the two steps beyond.
  
  The plan had HQ. The data say SC.
- **`Ladder(echelon=n)` starts at the stage II estimates by default.**
  Stage 3a kept zero because the preliminary start did not help; this
  start does (point 7). The plain Ladder keeps zero, as the C.
- **The start uses plain stage II, not the refined one.** That is the
  configuration measured in point 7, and it already ends at 0 %.
- **The weak design's underfitting is left to stage IV's checks**, as the
  plan said.

## Reproduce

```
cd research/echelon_seed
python3 search_study.py --iterate 0 --out search.csv        # ~15 min, 3 workers
python3 search_study.py --iterate 1 --out search_it1.csv    # ~40 min, 2 workers
python3 search_analyze.py search.csv:plain search_it1.csv:refined
python3 fits.py --forms echelon_ls --out fits_ls.csv        # ~70 min, 2 workers
python3 fits_analyze.py fits.csv fits_echelon.csv fits_more.csv fits_ls.csv
```
(`fits.py` with `--reps 50 --nobs 200`; `search_study.py` with `--reps 100`.)
