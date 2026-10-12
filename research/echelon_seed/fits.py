"""Stage 2's measure (docs/ECHELON_PLAN.md §1): on the stage-0 systems, how
does exact ML behave on each parametrisation of the same true system?

  full : VARMA(N, N) with N = max n_i, every series and every pair
  rows : each series' own AR(n_i)/MA(n_i), cross orders (n_i, n_i) per row
         (Ladder rows=): the row degrees, not yet the echelon table
  echelon : own AR(n_i)/MA(n_i), Ladder(echelon=n): the index table and A_0
  echelon_pre : the same, from Jenkins and Alavi's preliminary start
            (Ladder start="preliminary") instead of zero
  truth   : the echelon form EVALUATED at the true system (mean 10, Sigma's
            ratios; sigma2 concentrated): 2 (logL_echelon - logL_truth) is
            chi2(npar) if the fit reaches its maximum

Both go through the Ladder on route V's specifications (each series a fue
.inp with its mean and free own ARMA), as sima's route V does. For each fit:
termcode, MA inverse roots on the wall, logL, parameters, iterations,
seconds.

    python3 fits.py [--reps 50] [--nobs 200] [--workers 4] [--forms full,rows]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
import warnings
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["indices", "design", "nobs", "rep", "form", "npar", "logL", "termcode",
          "ma_boundary", "ifault", "nit", "seconds", "error"]


def _operator(title, order):
    if order <= 0:
        return [f"** {title}:", "0"]
    return [f"** {title}:", f"1 {order}", "**"] + ["0.000000  1"] * order


def write_spec(path, name, y, p, q):
    """A route-V fue .inp (as sima.raw.write_inp, annual, lambda 1, d = 0):
    the mean free at the sample mean, a free own AR(p)/MA(q) at zero."""
    L = ["* route V specification for the echelon study", "",
         "** Frequency of time series: either 1(A), 4(Q) or 12(M):", " 1",
         "** Number of observations and starting date of time series:",
         f" {len(y)}  1 1 {name}",
         "** Number of deterministic variables (including seasonal components):", "0"]
    L += _operator("Number and orders of regular AR operators", p)
    L += _operator("Number and orders of annual AR operators", 0)
    L += _operator("Number and orders of regular MA operators", q)
    L += _operator("Number and orders of anual MA operators", 0)
    L += ["** Number and frequencies of regular AR(2) operators with fixed frequency:", "0",
          "** Number and frequencies of regular MA(2) operators with fixed frequency:", "0",
          "** Mean parameter (mu):", f"{float(np.mean(y)):.6f} 1",
          "** Box-Cox lambda, regular differences and complete annual differences:",
          " 1  0  0",
          "** Individual factors of the annual difference (starting at freq 0.0):", " 0",
          "** ACF/PACF bands (0 Automatic) and reescaling factor:", " 0 1",
          "** Time series (stochastic and non-standard deterministic variables):"]
    L += [f"{v:.15g}" for v in y]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def data(n, design, nobs, rep):
    from drvarma.datasets import simulate_varma
    import echelon as E
    A, M, sigma = S.system(n, design)
    phi, theta = E.standard(A, M)
    seed = zlib.crc32(repr((n, design, nobs, rep)).encode())   # = stage 0's
    # a level far from zero: fue's mean starts at the sample mean anyway
    return simulate_varma(phi=phi, theta=theta, sigma=sigma, n=nobs, burnin=300,
                          seed=seed, freq=1).data + 10.0


def fit_form(form, n, y, tmp):
    from drvarma.ladder import Ladder
    m, N = len(n), max(n)
    names = [f"y{i + 1}" for i in range(m)]
    if form == "full":
        own = [(N, N)] * m
        kw = dict(p=N, q=N)
    elif form == "rows":
        own = [(k, k) for k in n]
        kw = dict(p=N, q=N, rows={nm: (k, k) for nm, k in zip(names, n)})
    elif form in ("echelon", "echelon_pre", "truth"):
        own = [(k, k) for k in n]
        kw = dict(echelon=n)
        if form == "echelon_pre":
            kw["start"] = "preliminary"
    else:
        raise ValueError(form)
    files = [write_spec(os.path.join(tmp, f"{form}_{nm}.inp"), nm, y[:, i], *own[i])
             for i, nm in enumerate(names)]
    L = Ladder(files, **kw)
    if form != "truth":
        return L.fit()
    L._set_structure(list(range(m)), L.p, L.q, L.diagcov)
    v = L.pack()
    val = true_values(n, design_of[0], names)
    nm = L.names()
    missing = [x for x in nm if x not in val]
    if missing:
        raise RuntimeError(f"no true value for {missing}")
    L.unpack(np.array([val[x] for x in nm]))
    return L._fit(optimize=False)


design_of = [None]          # set by one(): the design of the task in hand


def true_values(n, design, names):
    """The Ladder's parameters at the true system: own phi/theta and cross
    AR/MA/AR0 from the LP-convention A, M (signs flipped), the mean 10, Q's
    log-ratios and covariances relative to Sigma_11."""
    A, M, sigma = S.system(n, design)
    m = len(n)
    v = {}
    for i, a in enumerate(names):
        v[f"mu_{a}"] = 10.0
        for k in range(1, n[i] + 1):
            v[f"phi_{a}[B^{k}]"] = -A[k, i, i]
            v[f"theta_{a}[B^{k}]"] = -M[k, i, i]
        for j, b in enumerate(names):
            if i == j:
                continue
            v[f"AR0[{a}<-{b}]"] = -A[0, i, j]
            for k in range(1, A.shape[0]):
                v[f"AR{k}[{a}<-{b}]"] = -A[k, i, j]
                v[f"MA{k}[{a}<-{b}]"] = -M[k, i, j]
    for i in range(1, m):
        v[f"log(Q[{names[i]}]/Q[{names[0]}])"] = float(np.log(sigma[i, i] / sigma[0, 0]))
        for j in range(i):
            v[f"Q[{names[i]},{names[j]}]"] = sigma[i, j] / sigma[0, 0]
    return v


def one(task):
    n, design, nobs, rep, forms = task
    warnings.simplefilter("ignore")
    y = data(n, design, nobs, rep)
    design_of[0] = design
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for form in forms:
            row = dict(indices="".join(map(str, n)), design=design, nobs=nobs, rep=rep,
                       form=form, error="")
            t0 = time.time()
            try:
                r = fit_form(form, n, y, tmp)
                row.update(npar=r.npar, logL=f"{r.logL:.6f}", termcode=r.termcode,
                           ma_boundary=r.ma_boundary, ifault=r.ifault, nit=r.nit)
            except Exception as e:                       # noqa: BLE001
                row.update(npar="", logL="", termcode="", ma_boundary="", ifault="",
                           nit="", error=f"{type(e).__name__}: {e}"[:200])
            row["seconds"] = f"{time.time() - t0:.1f}"
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--nobs", default="200")
    ap.add_argument("--designs", default=",".join(S.DESIGNS))
    ap.add_argument("--indices", default="")
    ap.add_argument("--forms", default="full,rows")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "fits.csv"))
    a = ap.parse_args()
    forms = a.forms.split(",")
    idx = ([tuple(int(c) for c in t) for t in a.indices.split(",")] if a.indices
           else S.INDICES)
    tasks = [(n, d, int(nobs), r, forms) for n in idx for d in a.designs.split(",")
             for nobs in a.nobs.split(",") for r in range(a.reps)]
    with open(a.out, "w", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, rows in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            w.writerows(rows)
            fh.flush()
            if k % 20 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
    print("rc=0", flush=True)


if __name__ == "__main__":
    main()
