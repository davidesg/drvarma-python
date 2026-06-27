# The drvarma `.inp` input format

This is a precise, self-contained specification for generating a drvarma `.inp`
file. It is written to be unambiguous for an automated assistant: follow the
token order exactly and any `.inp` you produce will load with
`drvarma.load("file.inp")`.

## How the parser reads the file

drvarma's reader is **token-based**, not line-based. It:

1. **drops every line whose first non-space character is `*`** (so `*` title
   lines and `**` field-label lines are *comments* — purely for humans), and
2. reads the remaining content as a flat stream of **whitespace-separated
   tokens** (spaces, tabs and newlines are all just separators).

Therefore the *only* thing that matters is the **order of the data tokens**. The
`*`/`**` comment lines are conventional and recommended for readability, but the
parser ignores them.

## Token order (this is the contract)

Emit these tokens in this exact order:

| # | Token(s) | Type | Meaning |
|---|----------|------|---------|
| 1 | `freq` | int | seasonal frequency: `1`=annual, `4`=quarterly, `12`=monthly |
| 2 | `m` | int | number of series (variables), `m ≥ 1` |
| 3 | `nobs` | int | number of observations (time points) |
| 4 | `start_sub` | int | starting subperiod: month `1..12` (monthly), quarter `1..4`, or `1` (annual) |
| 5 | `start_year` | int | starting calendar year, e.g. `2002` |
| 6 | `name_1 … name_m` | str ×`m` | one **whitespace-free** name per series (e.g. `IPC_ES`) |
| 7 | `lambda` | float | Box-Cox λ: `0` = log transform, `1` = identity (no transform) |
| 8 | `d` | int | number of regular differences (often `1` for price/level series) |
| 9 | `D` | int | number of seasonal differences (lag = `freq`); often `0` |
| 10 | `data` | float ×`(nobs·m)` | the observations, **row-major**: for each time `t` (oldest→newest) the `m` values, series order matching the names |

**Data layout (critical):** the `nobs·m` numbers are read as `nobs` rows of `m`
values. Row `t` holds all series at time `t`, in the same order as the names.
Lay one observation per line for readability.

## Canonical template

```
* <free-text title / description>
** Frequency (1=A, 4=Q, 12=M):
 <freq>
** Series, observations, start (subperiod year):
 <m> <nobs> <start_sub> <start_year>
** Series names:
 <name_1> <name_2> ... <name_m>
** Box-Cox lambda, regular differences, annual differences:
 <lambda> <d> <D>
** Data:
 <y1_t1> <y2_t1> ... <ym_t1>
 <y1_t2> <y2_t2> ... <ym_t2>
 ...
 <y1_tN> <y2_tN> ... <ym_tN>
```

## Minimal worked example (2 series, 4 monthly obs)

```
* Example: two monthly series
** Frequency (1=A, 4=Q, 12=M):
 12
** Series, observations, start (subperiod year):
 2 4 1 2020
** Series names:
 SALES PRICE
** Box-Cox lambda, regular differences, annual differences:
 0.0 1 0
** Data:
 100.0 10.0
 101.5 10.1
 103.2 10.0
 102.8 10.2
```

This declares a monthly (`freq=12`), 2-series, 4-observation dataset starting in
January 2020, modelled in logs (`lambda=0`) with one regular difference (`d=1`)
and no seasonal differencing (`D=0`).

## Rules and pitfalls

- **`m`, `nobs` must match the data.** Exactly `nobs·m` data tokens must follow;
  too few raises a parse error, extras are ignored.
- **Series names cannot contain spaces** (each is one token). Use `_` (e.g.
  `IPC_ES`, not `IPC ES`).
- **One row = one time point**, not one series. Do *not* write all of series 1
  then all of series 2.
- **`start_sub` is the subperiod, `start_year` the year**, in that order
  (`1 2002` = January 2002 for monthly data).
- **Transform belongs in the header, not the data.** Put raw levels in `Data:`;
  `lambda`, `d`, `D` tell drvarma how to make them stationary. Don't pre-difference.
- **Decimal point** (`.`), not comma. Any whitespace separates tokens.
- **Encoding** is latin-1-tolerant; plain ASCII is safest.

## Programmatic generation

You don't have to format text by hand — build a `MultiSeries` and write it:

```python
import numpy as np
from drvarma import MultiSeries, InpSpec, save

data = np.column_stack([sales, price])          # shape (nobs, m)
series = MultiSeries(data, freq=12, start=(2020, 1), names=["SALES", "PRICE"])
save("example.inp", series, InpSpec(lam=0.0, d=1, D=0))
```

`MultiSeries(data, freq, start=(year, subperiod), names)` and
`InpSpec(lam, d, D)` mirror the header fields above; `save` writes the canonical
template. Read it back with `series, spec = drvarma.load("example.inp")`.
