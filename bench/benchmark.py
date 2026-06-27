"""Performance / efficiency benchmark: pure-Python vs hybrid (CFFI) vs pure C.

Compares the three estimation paths of drvarma across a battery of conditions
(dimension m, length n, orders p/q, and *conditioning* of the estimation
problem, including deliberately ill-conditioned cases).  Writes the raw results
to ``bench/results.json``; ``docs/DEVELOPER_GUIDE.md`` summarises them.

Paths timed
-----------
* **pure-Python** — ``estimate_py.estimate_w_py`` (AS 311 + the ported BFGS).
* **hybrid**      — ``_engine.estimate_w`` via the CFFI-compiled C core.
* **pure C**      — the standalone ``drvarma`` binary (subprocess, full CLI:
  read .inp → estimate → write .out); end-to-end, so it also carries I/O and
  report overhead (noted in the guide).

Run:  PYTHONPATH=src python bench/benchmark.py [--quick]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.datasets import simulate_varma          # noqa: E402
from drvarma import transform as T                    # noqa: E402
from drvarma.estimate_py import estimate_w_py         # noqa: E402
from drvarma.inp import save, InpSpec                 # noqa: E402

try:
    from drvarma._engine import estimate_w
    import drvarma._drvarma_engine                     # noqa: F401
    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

_C_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "drvarma_v.04.1")
C_BIN = os.path.join(_C_DIR, "bin", "drvarma")
HAS_CBIN = os.path.exists(C_BIN)

SCRATCH = os.path.join(os.path.dirname(__file__), "_scratch")
os.makedirs(SCRATCH, exist_ok=True)


# --------------------------------------------------------------------------- #
#  parameter generators (well- and ill-conditioned regimes)                   #
# --------------------------------------------------------------------------- #

def _diag(m, val):
    return np.eye(m) * val


def make_params(regime, m, p, q, rng):
    """Return (phi_list, theta_list, sigma, mu, label) for a conditioning regime."""
    mu = np.zeros(m)
    # baseline well-behaved Σ (unit-ish, moderate correlation 0.3)
    sigma = np.full((m, m), 0.3) + np.diag(np.full(m, 0.7))

    if regime == "well":
        phi = [_diag(m, 0.5 / (k + 1)) + 0.05 * np.tril(np.ones((m, m)), -1)
               for k in range(p)]
        theta = [_diag(m, 0.3 / (k + 1)) for k in range(q)]
    elif regime == "near_unit_root":
        # dominant root near the unit circle (persistent)
        phi = [_diag(m, 0.0) for _ in range(p)]
        if p >= 1:
            phi[0] = _diag(m, 0.97)
        theta = [_diag(m, 0.2) for _ in range(q)]
    elif regime == "near_cancellation":
        # φ ≈ θ → AR and MA nearly cancel (weakly identified VARMA)
        base = _diag(m, 0.6)
        phi = [base.copy() for _ in range(p)]
        theta = [(-(base) + 0.02 * np.eye(m)) for _ in range(q)]   # ≈ -φ
    elif regime == "var_disparity":
        # innovation variances differ ~100× across series (WTI/IPC-like)
        d = np.array([100.0 if i == 0 else 1.0 for i in range(m)])
        sigma = np.outer(np.sqrt(d), np.sqrt(d)) * (np.full((m, m), 0.3)
                                                    + np.diag(np.full(m, 0.7)))
        phi = [_diag(m, 0.5 / (k + 1)) for k in range(p)]
        theta = [_diag(m, 0.3 / (k + 1)) for k in range(q)]
    elif regime == "high_corr":
        # near-singular Σ (correlation ≈ 0.97)
        sigma = np.full((m, m), 0.97) + np.diag(np.full(m, 0.03))
        phi = [_diag(m, 0.5 / (k + 1)) for k in range(p)]
        theta = [_diag(m, 0.3 / (k + 1)) for k in range(q)]
    else:
        raise ValueError(regime)
    return phi, theta, 0.5 * (sigma + sigma.T), mu, regime


# --------------------------------------------------------------------------- #
#  timing helpers                                                             #
# --------------------------------------------------------------------------- #

def _time(fn, repeats):
    """Median wall-time (s) of `fn` over `repeats`; returns (median, last_result)."""
    ts, res = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


def time_c_binary(w, p, q, repeats):
    """Time the standalone C binary end-to-end on a .inp built from w."""
    from drvarma.series import MultiSeries
    n, m = w.shape
    ser = MultiSeries(w.copy(), freq=12, start=(2000, 1),
                      names=["s%d" % i for i in range(m)])
    base = os.path.join(SCRATCH, "bench")
    save(base + ".inp", ser, InpSpec(lam=1.0, d=0, D=0))
    args = [os.path.abspath(C_BIN), base, str(p), str(q), "-mean"]
    ts, rc = [], 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = subprocess.run(args, capture_output=True)
        ts.append(time.perf_counter() - t0)
        rc = r.returncode
    return float(np.median(ts)), rc


# --------------------------------------------------------------------------- #
#  one cell                                                                   #
# --------------------------------------------------------------------------- #

def run_cell(regime, m, n, p, q, rng, quick):
    phi, theta, sigma, mu, _ = make_params(regime, m, p, q, rng)
    sim = simulate_varma(phi=phi, theta=(theta or None), sigma=sigma, n=n,
                         mu=mu, seed=int(rng.integers(1 << 30)))
    w, _ = T.transform(np.asarray(sim.data).reshape(n, m),
                       lam=1.0, d=0, D=0, s=12, scale=1.0)

    rep_py = 1
    rep_hy = 2 if quick else 5
    out = {"regime": regime, "m": m, "n": n, "p": p, "q": q,
           "npar": None, "py": {}, "hybrid": {}, "c": {}}

    # pure-Python
    try:
        t_py, r_py = _time(lambda: estimate_w_py(w, p, q, include_mean=True), rep_py)
        out["npar"] = int(r_py["npar"])
        out["py"] = {"time": t_py, "ifault": int(r_py["ifault"]),
                     "logelf": float(r_py["logelf"]), "nit": int(r_py.get("nit", -1))}
    except Exception as e:
        out["py"] = {"error": repr(e)}
        r_py = None

    # hybrid (CFFI)
    if HAS_ENGINE:
        try:
            t_hy, r_hy = _time(lambda: estimate_w(w, p, q, include_mean=True), rep_hy)
            cov = r_hy.get("cov")
            condc = float(np.linalg.cond(cov)) if cov is not None and cov.size else float("nan")
            out["hybrid"] = {"time": t_hy, "ifault": int(r_hy["ifault"]),
                             "logelf": float(r_hy["logelf"]), "cond_cov": condc}
            if r_py is not None and "error" not in out["py"]:
                out["acc"] = {
                    "d_logelf": abs(r_py["logelf"] - r_hy["logelf"]),
                    "d_params": float(np.max(np.abs(r_py["params"] - r_hy["params"]))),
                    "d_sigma": float(np.max(np.abs(r_py["sigma"] - r_hy["sigma"]))),
                    "speedup_py_over_hybrid": (out["py"]["time"] / t_hy
                                               if t_hy else float("nan")),
                }
        except Exception as e:
            out["hybrid"] = {"error": repr(e)}

    # pure C binary (end-to-end); skip large n in quick mode
    if HAS_CBIN and not (quick and n > 1000):
        try:
            t_c, rc = time_c_binary(w, p, q, 1 if quick else 3)
            out["c"] = {"time": t_c, "returncode": rc,
                        "crashed": rc != 0}
        except Exception as e:
            out["c"] = {"error": repr(e)}

    tag = f"{regime:16s} m={m} n={n:<4} ({p},{q})"
    py = out["py"].get("time")
    hy = out["hybrid"].get("time")
    print(f"  {tag}  py={py and round(py,3)}s  hybrid={hy and round(hy,4)}s  "
          f"c={out['c'].get('time') and round(out['c']['time'],3)}s  "
          f"crashed_C={out['c'].get('crashed')}  npar={out['npar']}")
    return out


# --------------------------------------------------------------------------- #
#  battery                                                                     #
# --------------------------------------------------------------------------- #

def battery(quick):
    rng = np.random.default_rng(20260627)
    cells = []

    # A. scaling in n (well-conditioned VAR(2), m=3)
    for n in ([200, 500] if quick else [200, 500, 1000, 2000]):
        cells.append(("well", 3, n, 2, 0))
    # B. scaling in m (well-conditioned VAR(1), n=500)
    for m in [2, 3, 4]:
        cells.append(("well", m, 500, 1, 0))
    # C. orders (m=3, n=500)
    for (p, q) in [(1, 0), (3, 0), (1, 1), (2, 1)]:
        cells.append(("well", 3, 500, p, q))
    # D. ill-conditioned regimes (m=3, n=500)
    for regime in ["near_unit_root", "var_disparity", "high_corr"]:
        cells.append((regime, 3, 500, 2, 0))
    cells.append(("near_cancellation", 2, 500, 1, 1))
    cells.append(("near_cancellation", 3, 300, 1, 1))

    print(f"engine={HAS_ENGINE}  c_binary={HAS_CBIN}  cells={len(cells)}  quick={quick}\n")
    results = []
    for (regime, m, n, p, q) in cells:
        results.append(run_cell(regime, m, n, p, q, rng, quick))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer/smaller cells")
    args = ap.parse_args()
    t0 = time.perf_counter()
    results = battery(args.quick)
    meta = {"engine": HAS_ENGINE, "c_binary": HAS_CBIN, "quick": args.quick,
            "numpy": np.__version__, "python": sys.version.split()[0],
            "total_time_s": round(time.perf_counter() - t0, 1)}
    out = {"meta": meta, "results": results}
    dst = os.path.join(os.path.dirname(__file__), "results.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}  (total {meta['total_time_s']}s)")


if __name__ == "__main__":
    main()
