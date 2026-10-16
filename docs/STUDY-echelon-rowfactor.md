# Echelon stage 4: the row factor — the study

*2026-10-13. Stage 4 of [`ECHELON_PLAN.md`](ECHELON_PLAN.md) §4. Data:
[`STUDY-echelon-rowfactor-data.md`](STUDY-echelon-rowfactor-data.md). Code:
`Ladder(echelon=, own=)`, `Ladder.factor_test`, `ladder.search_row_factors`,
`echelon.stage2(filters=)`; `research/echelon_seed/rowfactor_study.py`,
`rowfactor_twopass.py`, `rowfactor_analyze.py`.*

## The questions

1. Design note §8 warned that "the multiplicative row form may not be
   identified". When a seasonal factor multiplies a whole echelon row, does
   exact ML still reach its maximum?
2. Does the index search find the indices when each row is filtered by its
   factor? And with which factors: the true ones, or the univariate models'
   that route U has?

## Design

- **The systems** are stage 0's nine, generic design, made quarterly.
  Each row is multiplied by an annual factor:
  - **sar**: every row has F = 1 − 0.5 B⁴, an annual AR;
  - **mixed**: row 1 has F = 1 − 0.5 B⁴, row 2 has G = 1 − 0.4 B⁴ (an
    annual MA), row 3 has none.
- **Sample:** nobs 200, 50 replications per cell, 900 in all.
- **The specs:** each series is a fue `.inp` with a free regular
  AR(n_i)/MA(n_i) and its annual factor, everything starting at zero.
  The gate's univariate fits give the factors that the start and the
  search see.
- **The fits:** `Ladder(echelon=n)` from the stage II start (ls) and from
  zero. The yardstick is stage 3a's, the LR against the truth,
  2(logL_fit − logL_truth).
- **The search** (SC, n_max 3) runs three ways: rows filtered by the true
  factors, by the gate's, and not filtered. Separately,
  `search_row_factors` runs the search in passes.

## Results

1. **The factored form behaves as identified.** From the stage II start, no
   fit in any of the 18 cells ends below the true parameters. The median LR
   is close to npar, as χ²(npar) says. The share above χ²'s 95 % point is
   2–12 %, with one cell at 18 %, (2,1,0) mixed, the same from both starts.
   Termcode 3 stops are 0–14 %, MA-wall stops 0–4 %. §8's fear is not
   borne out on these systems.
2. **The stage II start matters, as in stage 3b.**
   - In (2,2,1), from zero, 22 % (sar) and 42 % (mixed) of the fits end
     below the truth, with the MA wall in 28–38 % and termcode 3 in 30–44 %.
     From the stage II start: 0 % below the truth, the wall in 2–4 %.
   - Elsewhere the two starts reach the same optimum in 80–100 % of the
     replications. In the three-series systems the stage II start needs
     33–59 % of zero's iterations and 41–62 % of its time.
3. **Unfiltered, the search fails.** It finds the indices in 0–6 % of the
   replications, rising to 14–42 % only in (2,2) and (2,2,1): an annual
   factor looks like a long row. Filtered by the true factors, it finds them
   in 82–100 %, except (2,2,1) sar at 50 %.
4. **The univariate factors mislead the search when the rows differ.** A
   series' univariate model mixes the other rows' factors. With the gate's
   factors the search finds:
   - in the mixed pattern: (2,1) 28 %, (1,1,1) 38 %, (2,2,1) 48 %, (1,2) 52 %;
   - in the sar pattern, where every row has the same factor: 64–100 %,
     except (2,2,1) at 30 %.
5. **The search in passes fixes it.** `search_row_factors` searches with
   the univariate factors. Then it fits the choice by exact ML, which
   estimates the factors jointly, and searches again with those. It stops
   when the choice repeats, after at most three passes, 2.0–2.6 on average.
   The final choice is right in:
   - (2,1) mixed 88 %, (1,1,1) mixed 88 %, (2,2,1) mixed 84 %, (1,2) mixed
     96 %;
   - every other cell 84–100 %, except (2,2,1) sar at 36 %, where even
     the true factors give only 50 %.

## On the data sets

Results are in `ECHELON_PLAN.md` §4.3:
- **Flour prices (route U):** (1,0,0), white residuals.
- **IPC:** the factor holds, p 0.16.
- **M6:** the fixed-frequency factor is rejected on (1,0,0), p 0.008, and
  holds on (1,1,1), p 0.15.
- **Muskrat and mink:** (3,1), on the MA wall.

## Decisions

- **The row factor stands as designed.** It shows no sign of
  non-identification. The stage II start, which filters each row by its
  factor, is the default.
- **`factor_test` ships with it**, as design note §5.4 requires. It is the
  only guard against a factor the data do not support: in M6's (1,0,0) the
  data reject it.
- **The search on route U is `search_row_factors`**, in passes, not a single
  search with the univariate factors. sima's `identify_echelon` uses it and
  prints the passes.
- **(2,2,1) sar stays hard at nobs 200**: 36 % with the passes, 50 % with
  the true factors. It is a large system with a common annual factor at
  this sample size. The residual checks and HQ's alternative are the guard.

## Reproduce

```
cd research/echelon_seed
python3 rowfactor_study.py --reps 50 --nobs 200 --workers 3      # ~2 h
python3 rowfactor_twopass.py --reps 50 --workers 3               # ~1 h
python3 rowfactor_analyze.py rowfactor.csv rowfactor_twopass.csv
```
