# The ladder: fue's models as the input of a VARMA

drvarma is the third rung of the ladder fue → drtran → drvarma. drtran casts a
transfer network as a triangular VARMA, which needs a DAG. **A cycle is where
drtran ends**, and the way on is back to the ladder: drvarma takes the same
fue files as a general VARMA. It is not the best way to parametrise a VARMA,
but it is a good place to seed one. Each series arrives with its
autocorrelation already modelled, and **the univariate model is the yardstick
for forecasting**: a VARMA that cannot beat it has no reason to exist.

This module is the Python port of the C's ladder mode
(`engines/drvarma/src/escalera.c` in the atsw-gui monorepo, design in
`engines/drvarma/docs/DESIGN-v5-ladder.md`). The C is the oracle.

## Use

```python
from drvarma.ladder import Ladder, split

L = Ladder(["ES.pre", "FR.pre", "DE.pre"], p=1, q=0)      # cross orders
r = L.fit()                    # the gate, the diagonal system, the model
L.gate                         # per-series logL, sigma2, how far each file moved
L.lr_test()                    # cross dynamics against the univariates
L.forecast(24)                 # levels with 95% bands, per series

L = Ladder(files, 0, 0, diagcov=True, estwin=216)        # the univariates
L.fit(); rows, summary = L.recursive(24)                  # fixed-parameter, every origin
```

Options: `diagcov`, `redet` (re-estimate the deterministic terms, fixed at the
file by default), `fixarma`, `method` (1 exact, 2 approximate), `estwin`.

## The model

Each series `i` brings its univariate model from its file: Box-Cox with fue's
rescaling factor, deterministic terms, the non-stationary operator, the mean,
and the ARMA factors (seasonal and fixed-frequency ones included). On the
stationary series `w`:

    Phi(B) (w_t - mu) = Theta(B) a_t,        a_t ~ N(0, sigma2 Q)
    Phi_ii = phi_i(B) Phi_i(B^s)      Theta_ii = theta_i(B) Theta_i(B^s)
    Phi_ij = -SUM_k c_ij,k B^k        Theta_ij = -SUM_k e_ij,k B^k   (i != j)

Q is normalised with Q11 = 1 (the likelihood concentrates sigma2). Standard
errors come from a finite-difference Hessian at the optimum.

**Nothing about a univariate model is re-implemented.** `fue.load` reads the
file: it is the reference parser, and it decides what a file is by its content,
not its name. `fue.cast_us.cast_us_py` gives each series its stationary series
and its expanded ARMA polynomials: the code fue estimates with, and the one
drtran-python uses too. A forecast goes back to the level with the pieces of
`fue.forecast`: deterministic terms, the operator, the inverse Box-Cox.

## The gate

Each series is fitted alone, on the common window of all the series. The
diagonal system is then **evaluated** at those optima, with Q_ii/Q_11 =
σ²_i/σ²_1. The identity is exact, and if it fails `GateError` is raised: a
gate that only prints a warning is not a gate. It also reports how far each
file moved: a genuine `.pre` is a fixed point (~1e-5, the rounding of its six
decimals).

## Alignment (BUG-2)

The same frequency and the same last date, or `LadderError`. The series are
aligned at the end and trimmed to the shortest stationary series. It is the
same rule as the C's `fuepre_check_alignment` and `drtran.cast.check_alignment`.

## One format

The multivariate drvarma `.inp` (one λ, d, D for m columns) is deprecated.
`split` writes one fue `.inp` per series. What was a run option of the old
path becomes part of each specification. With `mean=True, ar=P, ma=Q` and
cross orders P, Q, the ladder is the old full VARMA(P,Q): `tests/test_ladder.py`
checks that it reaches the same μ and φ as the old path.

## Parity with the C (`tests/test_ladder.py`)

| check | against |
|---|---|
| the gate, 1e-13 | the identity |
| σ² of each series; the AR(3)×SAR(1) of DE | fue's `.out` |
| logL: full covariance 143.245159, VAR(1) 160.828677, pass-through −724.199406, `-fixarma` 57.837680 | the C |
| the trimmed case (different lengths, same end) | the C |
| 3456 fixed-parameter forecasts, ≤ 2.3e-6 | fue (`fue_recursive_reference.csv`) |
| bands and recursive MAE/MAPE | the C's `.out` |
| ladder on `split` files = old VAR(1) with mean | the old path |

## Speed

The system's likelihood is `elf_c`, the compiled `elf` shipped in the binary
wheels (~250× the pure-Python port). fue's cast is pure Python, so the
stationary series of each model is computed once and cached. At each
evaluation only its ARMA polynomials are asked for, from `cast_us_py` on a
copy of the model without deterministic terms. On the CPI trio a diagonal fit
takes ~1 s, a VARMA(1,1) ~4 s, and 48 origins of recursive forecasts ~3 s.
`redet` is the slow one (~6 s), because the stationary series is rebuilt at
every evaluation.
