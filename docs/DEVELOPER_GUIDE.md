# drvarma — Developer Guide & Performance Study

Audience: developers of **exact-maximum-likelihood VARMA software**. This guide
documents drvarma's three estimation paths, their algorithmic and numerical
foundations (with references to the original literature in `../literature`), and
an empirical **efficiency / accuracy / robustness study** comparing them across a
battery of conditions — including deliberately **ill-conditioned** estimation
problems. The benchmark is reproducible: `bench/benchmark.py` → `bench/results.json`.

---

## 1. The three implementations

drvarma exposes one model API (`Model.fit`) over a single result contract
(`_engine.estimate_w`), backed by three interchangeable estimation paths:

| Path | Entry point | What it is |
|------|-------------|-----------|
| **pure-Python** | `estimate_py.estimate_w_py` | AS 311 likelihood (`_as311.py`) + a faithful port of the C factored-BFGS optimiser (`_qnewt.py`). No compiled code; pure NumPy/SciPy. |
| **hybrid (CFFI)** | `_engine.estimate_w` → `_drvarma_engine` | The validated drvarma **C core** (`csrc/`) compiled as a CFFI extension and called from Python. Same numbers as the binary, returned as a Python dict. |
| **pure C** | the `drvarma` binary | The standalone C program: reads `.inp`, estimates, writes `.out`/`.forecast`/… End-to-end (so its wall-time also includes I/O and report generation). |

The hybrid and pure-C paths share the **same C estimation core** (`shootx`,
`init_varma`, `est`, `elf`, `raxopt`), so they agree to the last bit on the
estimates; they differ only in surrounding I/O. The pure-Python path is an
independent re-implementation of that same algorithm — which is exactly why the
cross-checks below are meaningful.

`_engine.estimate_w` prefers the compiled engine and **falls back to pure Python**
when it is absent, so the package is fully functional with no C build (`pip
install` without GSL).

---

## 2. Algorithmic & numerical background

### 2.1 The estimation problem

For an `m`-dimensional stationary Gaussian VARMA(p, q),
`Φ(B) (w_t − μ) = Θ(B) a_t`, `a_t ~ N(0, Σ)`, exact maximum likelihood maximises
the exact Gaussian log-likelihood of `w_1…w_n`. drvarma evaluates and maximises that
likelihood by **Mauricio's exact-ML method** (Mauricio 1995, JASA), as published
in code form as **Algorithm AS 311** (Mauricio 1997) — `elf`/`cgamma`/`cxi`/`cres`
in `elfvarma.c`, ported verbatim to `_as311.py`. The optimiser is a factored-BFGS
quasi-Newton (Dennis & Schnabel 1983) with a Dennis–Schnabel line search
(`qnewtopt.c` → `_qnewt.py`), over the concentrated objective `f1^m · f2`
(σ² profiled out; see `docs/PURE_PYTHON_PLAN.md` §PP1).

### 2.2 Computational complexity (from the literature)

The cost of one exact-likelihood **evaluation** is **O(n)** in the series length
plus a fixed *preliminary* cost (theoretical autocovariances). For scalar models
the per-step operation counts of the main methods are (Mauricio 2002, §1, citing
the original sources):

| Method | time-consuming ops (mult/div) |
|--------|-------------------------------|
| Ansley (1979) | `n[p + (q+1)(q+4)/2]` — quadratic in q, linear in p |
| Pearlman (1980)/Mélard (1984, AS 197) | `n(2p+3q+2)` (p<q+1), `n(p+4q+6)` (p≥q+1) |
| Kohn & Ansley (1985) | `n(p+3q+2)` |

Kalman-filter methods (Pearlman/Mélard/Kohn–Ansley) also pay a **preliminary**
cost to start the recursions (autocovariances of orders 0…g, g=max(p,q)), which
is quadratic in p, q, g (Mélard 1984) and *cannot* use the scalar fast
autocovariance algorithms in the multivariate case. Outside the state-space
framework (Ansley 1979; Ljung–Box 1979), only orders 0…p−1 are needed — and none
when p=0. So for `q > p` (especially q ≫ p) the non-Kalman methods recover ground
despite the `O(q²)` term.

For the **multivariate** case, Mauricio (1995, 1997 = AS 311) and Shea (1989 =
AS 242) are established as the two most efficient exact-likelihood methods; the
later innovations-form algorithm (Mauricio 2002) is faster than Shea by a factor
"close to three in many cases" (and its edge grows with the dimension m), except
for seasonal models with q large and ≫ p. **drvarma implements AS 311**, not the
2002 method.

### 2.3 Numerical conditioning

The literature flags that methods relying on **explicit matrix inversion** incur
"computational inefficiency and a loss of numerical precision" (Mauricio 1993,
§1). This matters here: under ill-conditioning (near-unit roots, near-singular Σ,
or large variance disparities across series) the parameter information matrix
becomes ill-conditioned, the optimiser needs many more iterations, and the two
independent implementations (pure-Python vs C) are expected to **diverge more** —
which §4.3 confirms empirically.

---

## 3. Benchmark methodology

`bench/benchmark.py` times each path on synthetic series simulated from known
VARMA parameters (`datasets.simulate_varma`), across:

- **scaling in n** — well-conditioned VAR(2), m=3, n ∈ {200, 500, 1000, 2000};
- **scaling in m** — well-conditioned VAR(1), n=500, m ∈ {2, 3, 4};
- **orders** — m=3, n=500, (p,q) ∈ {(1,0),(3,0),(1,1),(2,1)};
- **conditioning regimes** — m=3, n=500: `near_unit_root` (dominant root ≈0.97),
  `var_disparity` (innovation variances ~100× apart, WTI/IPC-like), `high_corr`
  (Σ correlation ≈0.97), and `near_cancellation` (φ ≈ −θ, weakly-identified VARMA).

Metrics per cell: wall-time (pure-Python ×1, hybrid ×5, C-binary ×3, medians);
pure-Python BFGS iteration count (`nit`); cross-implementation accuracy
(`|Δlogelf|`, `max|Δparams|`, `max|ΔΣ|` between pure-Python and hybrid); and the
condition number of the C engine's parameter covariance (`cond_cov`) as a
conditioning proxy. Environment for the numbers below: Python 3.12, NumPy 1.26,
single core; the C engine and binary built from `../drvarma_v.04.1`.

---

## 4. Results

### 4.1 Speed

```
regime            m/n/p/q   BFGS  py_s   hybrid_s  C_s   speedup(py/hybrid)
well              3/ 200/2/0   26  2.34   0.037   0.098     64×
well              3/ 500/2/0   27  2.67   0.076   0.203     35×
well              3/1000/2/0   25  2.79   0.134   0.432     21×
well              3/2000/2/0   25  3.24   0.245   1.018     13×
well              2/ 500/1/0   11  0.15   0.006   0.034     26×
well              3/ 500/1/0   23  1.00   0.032   0.106     32×
well              4/ 500/1/0   22  2.30   0.076   0.314     30×
well              3/ 500/3/0   38  9.98   0.194   0.422     51×
well              3/ 500/1/1   50 12.99   0.130   0.277    100×
well              3/ 500/2/1   52 18.08   0.245   0.442     74×
```

- **Hybrid vs pure-Python: ~13–100×.** The compiled core wins everywhere.
- **The relative gap shrinks as n grows** (64× → 13× from n=200 to 2000) because
  pure-Python is *nearly flat in n*: its cost is dominated by a fixed
  per-iteration overhead (Python-level BFGS loop + central-difference gradient =
  2·npar likelihood calls per gradient), while the compiled per-evaluation cost
  grows O(n). For long series the constant Python overhead amortises.
- **q>0 is the worst case for pure-Python** (VARMA(2,1): 18 s, 74×) — the AS 311
  MA recursion (`cxi`/`cres`) is inherently sequential and resists vectorisation.
- **Pure C binary** is ~2–4× the hybrid *estimation* time, but that gap is I/O +
  full `.out` report generation, not compute — the binary and the hybrid run the
  identical estimation core. It remains far faster than pure-Python.

### 4.2 Accuracy (pure-Python vs the C core)

Where the problem is well-conditioned the two independent implementations are
**numerically identical**:

```
well-conditioned cells:  |Δlogelf| ~ 1e-13 … 1e-12,  max|Δparams| ~ 1e-9
```

This is the strongest possible validation of the pure-Python port: a from-scratch
NumPy re-implementation of AS 311 + factored BFGS reproduces the GSL/C engine to
the last several digits.

### 4.3 Ill-conditioning

The interesting cases. As conditioning worsens (rising `cond_cov`), pure-Python
needs **many more BFGS iterations**, runs **much slower**, and the pure-Python /
C agreement **degrades** — exactly the precision-loss behaviour the literature
attributes to explicit matrix inversion:

```
regime            cond_cov   BFGS  py_s    max|Δparams|(py vs C)
well (3/500/2/0)     5e+02     27   2.67    5e-09          (baseline)
near_unit_root       1e+05     46   4.55    3e-07
high_corr            3e+05     51   5.38    2e-07
well (3/500/3/0)     2e+06     38   9.98    3e-07
var_disparity        1e+08    226  21.56    6e-04   ← worst
```

- **`var_disparity`** (variances ~100× apart) is the pathological case: the
  parameter covariance is ill-conditioned (cond ≈ 1e8), BFGS takes **226**
  iterations (vs ~25 well-conditioned), pure-Python takes **21.6 s**, and the
  two implementations diverge to `6e-4` in the parameters. This mirrors the C
  engine's own documented WTI→IPC pass-through caveat (`MODELS_RESULTS.md` §4):
  the point estimates remain scale-robust, but the *standard errors* are
  ill-determined. **Pre-scaling series to comparable variances is the practical
  fix.**
- Even in the worst case the discrepancy (`6e-4`) is small relative to the
  parameters; the *likelihood* still agrees to `~4e-7`. The point estimates are
  trustworthy; what degrades is the precision of the second-order quantities
  (covariance / std errors) — for both implementations.
- **Robustness asymmetry.** Separately (see `docs/STATUS.md` §PP5) the *C binary*
  has been observed to abort (SIGABRT) or emit a garbage covariance on a
  pathological weakly-identified VARMA(1,1), where the pure-Python path converges
  cleanly. So "C = always better" does not hold: the compiled path is much
  faster, but the pure-Python path can be more numerically forgiving.

---

## 5. Guidance — which path to use

- **Default to the hybrid (CFFI) engine** for anything interactive or batched:
  10–100× faster, and bit-identical to the reference binary. Build it with GSL
  (`pip install` with `libgsl-dev`, or `setup.py build_ext --inplace`).
- **Pure-Python is the right default when** there is no C toolchain/GSL, for
  reproducibility/portability (wheels with no native deps), for teaching/reading
  the algorithm, or as an independent oracle in tests. Expect 10–100× slower;
  budget for it on VARMA(q>0) and ill-conditioned fits.
- **The standalone binary** is for the file-in/file-out CLI workflow; its extra
  wall-time over the hybrid is report I/O, not estimation.
- **Under ill-conditioning** (near-unit roots, near-singular Σ, variance
  disparities): expect more iterations, slower fits, and larger cross-path
  disagreement in std errors. Mitigate by **rescaling series to comparable
  variances**, and treat std errors / Wald tests as approximate (both paths).

---

## 6. Reproducing

```sh
PYTHONPATH=src python bench/benchmark.py            # full battery → bench/results.json
PYTHONPATH=src python bench/benchmark.py --quick    # smaller/faster
```

The harness auto-detects the compiled engine and the C binary
(`../drvarma_v.04.1/bin/drvarma`); paths it cannot find are simply skipped.
`bench/results.json` holds the raw per-cell timings, accuracies and conditioning
numbers.

---

## 7. References

Original articles in `../literature` (drvarma's algorithmic basis), plus the
methods they benchmark against:

- **Mauricio, J. A. (1995).** *Exact maximum likelihood estimation of stationary
  vector ARMA models.* Journal of the American Statistical Association 90(429),
  282–291. — the **published** estimation method (exact likelihood evaluation +
  maximisation) that drvarma implements.
- **Mauricio, J. A. (1997).** *Algorithm AS 311: The exact likelihood function of
  a vector autoregressive moving average process.* Applied Statistics (JRSS C)
  46(1), 157–171. — the algorithm/code companion to the 1995 paper; the routines
  ported verbatim in `_as311.py`. [`518-2013-11-11-JAM197.pdf`]
- **Mauricio, J. A. (2002).** *An algorithm for the exact likelihood of a
  stationary vector autoregressive-moving average model.* Journal of Time Series
  Analysis 23(4), 473–486. — innovations form; the operation-count comparisons in
  §2.2/§2.3. [`518-2013-11-11-JAM102.pdf`]
- **Mauricio, J. A. (1993).** *Vector ARMA Models* (ICAE working paper,
  Universidad Complutense de Madrid) — the working-paper precursor of the 1995
  JASA article; the matrix-inversion precision discussion. [`9316.pdf`]
- **Mélard, G. (1984).** *Algorithm AS 197: A fast algorithm for the exact
  likelihood of ARMA models.* Applied Statistics 33(1), 104–114. [`as197.pdf`]
- **Shea, B. L. (1989).** *Algorithm AS 242: The exact likelihood of a vector
  ARMA model.* Applied Statistics 38(1), 161–184. — the AS 242 backup route
  (`multshea.c`), out of scope for the Python port.
- Ansley (1979); Ljung & Box (1979); Pearlman (1980); Kohn & Ansley (1985);
  Hillmer & Tiao (1979); Hall & Nicholls (1980) — methods compared in the above.
- **Dennis, J. E. & Schnabel, R. B. (1983).** *Numerical Methods for Unconstrained
  Optimization and Nonlinear Equations.* — the factored-BFGS optimiser
  (`qnewtopt.c` / `_qnewt.py`).

## Releasing: do NOT publish by hand

This package compiles a C engine, so its wheels are platform-specific and are
built **only** by `.github/workflows/publish.yml`, which runs `cibuildwheel`
across macOS, manylinux, musllinux and Windows for cp310–cp313. Release 0.1.3
shipped 26 files that way.

`python -m build` on a developer machine produces a `linux_x86_64` wheel, and
**PyPI rejects that tag** — only `manylinux`/`musllinux` are accepted. So a
manual `twine upload` publishes the sdist alone, and a user without a compiler
who installs that version gets a build failure where the previous version
worked.

The release is therefore: bump the version, commit, **push a `v*` tag**, and let
the workflow do the rest.
