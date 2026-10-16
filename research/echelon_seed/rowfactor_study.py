"""Stage 4's measure (docs/ECHELON_PLAN.md, stage 4; ECHELON_SEEDED_DESIGN.md
§8, "the multiplicative row form may not be identified"): on stage 0's
generic systems, quarterly, each row times a seasonal factor,

    F_r(B) [A(B) w]_r = G_r(B) [M(B) a]_r,

does exact ML reach its maximum, and does the index search, each row
filtered by its factor, find the indices?

Factor patterns (s = 4):
  sar   : every row F = 1 - .5 B^4 (an annual AR, the same for all)
  mixed : row 1 F = 1 - .5 B^4, row 2 G = 1 - .4 B^4 (annual MA), row 3 none

Fits (the Ladder on fue specs: a free regular AR(n_i)/MA(n_i) and the
row's annual factor, all starting at zero; the gate's univariate fits give
the factors the start sees):
  ls    : Ladder(echelon=n), the stage II start on the filtered rows
  zero  : the same from zero
  truth : the form evaluated at the true system: 2 (logL_fit - logL_truth)
          is chi2(npar) if the fit reaches its maximum
Search: SC, n_max 3, the rows filtered by the true factors (true), by the
gate's estimates (gate), or not at all (none).

    python3 rowfactor_study.py [--reps 50] [--nobs 200] [--workers 3]
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
import fits as FT
import study as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["indices", "pattern", "nobs", "rep", "form", "npar", "logL", "termcode",
          "ma_boundary", "ifault", "nit", "seconds", "found", "error"]
SAR, SMA = 0.5, 0.4


def factors(pattern, m):
    """Per row (f, g): annual AR and MA coefficients by lag (s = 4)."""
    a = [0.0, 0.0, 0.0, SAR]
    b = [0.0, 0.0, 0.0, SMA]
    if pattern == "sar":
        return [(a, [])] * m
    return [[(a, []), ([], b), ([], [])][r % 3] for r in range(m)]


def data(n, pattern, nobs, rep):
    from drvarma import echelon as E
    from drvarma.datasets import simulate_varma
    A, M, sigma = S.system(n, "generic")
    fac = factors(pattern, len(n))
    A = E.times_row_factors(A, [f for f, _g in fac])
    M = E.times_row_factors(M, [g for _f, g in fac])
    K, m = max(len(A), len(M)), len(n)
    A = np.concatenate([A, np.zeros((K - len(A), m, m))])
    M = np.concatenate([M, np.zeros((K - len(M), m, m))])
    phi, theta = E.standard(A, M)
    seed = zlib.crc32(repr((n, pattern, nobs, rep)).encode())
    return simulate_varma(phi=phi, theta=theta, sigma=sigma, n=nobs, burnin=300,
                          seed=seed, freq=4).data + 10.0


def write_spec(path, name, y, p, q, P, Q):
    """fits.write_spec, quarterly, with an annual AR(P)/MA(Q)."""
    FT.write_spec(path, name, y, p, q)
    L = open(path).read().split("\n")
    L[L.index("** Frequency of time series: either 1(A), 4(Q) or 12(M):") + 1] = " 4"
    for title, order in (("Number and orders of annual AR operators", P),
                         ("Number and orders of anual MA operators", Q)):
        i = L.index(f"** {title}:")
        if order:
            L[i + 1:i + 2] = [f"1 {order}", "**"] + ["0.000000  1"] * order
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def files(n, pattern, y, tmp):
    fac = factors(pattern, len(n))
    return [write_spec(os.path.join(tmp, f"y{i + 1}.inp"), f"y{i + 1}", y[:, i], k, k,
                       int(bool(f)), int(bool(g)))
            for i, (k, (f, g)) in enumerate(zip(n, fac))]


def fit(form, n, pattern, y, tmp):
    from drvarma.ladder import Ladder
    fs = files(n, pattern, y, tmp)
    if form == "zero":
        return Ladder(fs, echelon=n, start="zero").fit()
    if form == "ls":
        return Ladder(fs, echelon=n).fit()
    L = Ladder(fs, echelon=n, start="zero")
    m = len(n)
    L._set_structure(list(range(m)), L.p, L.q, L.diagcov)
    names = L.names()
    val = FT.true_values(n, "generic", [f"y{i + 1}" for i in range(m)])
    for i, (f, g) in enumerate(factors(pattern, m)):
        if f:
            val[f"Phi_y{i + 1}[B^4]"] = SAR
        if g:
            val[f"Theta_y{i + 1}[B^4]"] = SMA
    missing = [x for x in names if x not in val]
    if missing:
        raise RuntimeError(f"no true value for {missing}")
    L.unpack(np.array([val[x] for x in names]))
    return L._fit(optimize=False)


def search(how, n, pattern, y, tmp):
    from drvarma import echelon as E
    from drvarma.ladder import Ladder
    m = len(n)
    L = Ladder(files(n, pattern, y, tmp), echelon=n, start="zero")
    L.run_gate()
    L._set_structure(list(range(m)), 0, 0, True)
    _mu, _p, _t, _q, W, _if = L.cast(L.pack())
    if how == "true":
        filt = [(np.array(f), np.array(g)) for f, g in factors(pattern, m)]
    elif how == "gate":
        filt = [tuple(s.split_polynomials()[2:4]) for s in L.series]
    else:
        filt = None
    return E.search(W, n_max=3, criterion="SC", filters=filt).indices


def one(task):
    n, pattern, nobs, rep, forms = task
    warnings.simplefilter("ignore")
    y = data(n, pattern, nobs, rep)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for form in forms:
            row = dict(indices="".join(map(str, n)), pattern=pattern, nobs=nobs, rep=rep,
                       form=form, error="", found="")
            t0 = time.time()
            try:
                if form.startswith("search_"):
                    got = search(form[7:], n, pattern, y, tmp)
                    row["found"] = "".join(map(str, got))
                else:
                    r = fit(form, n, pattern, y, tmp)
                    row.update(npar=r.npar, logL=f"{r.logL:.6f}", termcode=r.termcode,
                               ma_boundary=r.ma_boundary, ifault=r.ifault, nit=r.nit)
            except Exception as e:                       # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"[:200]
            row["seconds"] = f"{time.time() - t0:.1f}"
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--nobs", default="200")
    ap.add_argument("--patterns", default="sar,mixed")
    ap.add_argument("--indices", default="")
    ap.add_argument("--forms", default="truth,ls,zero,search_true,search_gate,search_none")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "rowfactor.csv"))
    a = ap.parse_args()
    forms = a.forms.split(",")
    idx = ([tuple(int(c) for c in t) for t in a.indices.split(",")] if a.indices
           else S.INDICES)
    tasks = [(n, p, int(nobs), r, forms) for n in idx for p in a.patterns.split(",")
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
