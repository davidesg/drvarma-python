"""Stage 0 of docs/ECHELON_SEEDED_DESIGN.md: do the univariate orders an
analyst of the school would identify predict the Kronecker indices?

For each system (echelon.py: known indices, two designs), each sample size
and each replication: simulate, identify each series with art's
`suggest_orders` (route U's call: d = D = 0, lambda = 1, art's defaults),
and record the first-ranked orders and the tied ones. The seed of §3.1 is
max(p, q). Results: results.csv, one row per series and replication.

    python3 study.py [--reps 100] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelon as E

INDICES = [(1, 0), (1, 1), (2, 1), (1, 2), (2, 2),
           (1, 1, 1), (1, 2, 1), (2, 1, 0), (2, 2, 1)]
DESIGNS = {"generic": 1.0, "weak": 0.25}     # cross coefficients x this
NS = (100, 200, 500)
SIGMA_RHO = 0.3                               # innovation correlation

HERE = os.path.dirname(os.path.abspath(__file__))


def system(n, design):
    seed = 1000 * len(n) + 100 * sum(n) + 10 * n[0] + (1 if design == "weak" else 0)
    A, M = E.draw(n, seed, cross=DESIGNS[design])
    m = len(n)
    sigma = np.full((m, m), SIGMA_RHO) + (1 - SIGMA_RHO) * np.eye(m)
    return A, M, sigma


def _degree(sp):
    p = sp.sparse_ar_lag or sp.p
    q = sp.sparse_ma_lag or sp.q
    return int(p), int(q)


def one(task):
    n, design, nobs, rep = task
    warnings.simplefilter("ignore")
    import art
    import fue
    from drvarma.datasets import simulate_varma
    A, M, sigma = system(n, design)
    phi, theta = E.standard(A, M)
    seed = zlib.crc32(repr((n, design, nobs, rep)).encode())   # reproducible
    y = simulate_varma(phi=phi, theta=theta, sigma=sigma, n=nobs, burnin=300,
                       seed=seed, freq=1).data
    rows = []
    for i in range(len(n)):
        ts = fue.TimeSeries(list(map(float, y[:, i])), freq=1, start=(1, 1),
                            name=f"y{i + 1}")
        try:
            specs = art.suggest_orders(ts, d=0, D=0, lam=1.0, top_n=5)
        except Exception as e:                       # noqa: BLE001
            rows.append(dict(indices="".join(map(str, n)), design=design, nobs=nobs,
                             rep=rep, series=i + 1, n_i=n[i], p=-1, q=-1,
                             label=f"error: {e}", tied=""))
            continue
        p, q = _degree(specs[0])
        tied = [_degree(s) for s in specs if getattr(s, "tied", False)]
        rows.append(dict(indices="".join(map(str, n)), design=design, nobs=nobs,
                         rep=rep, series=i + 1, n_i=n[i], p=p, q=q,
                         label=specs[0].label().strip(),
                         tied=";".join(f"{a},{b}" for a, b in tied)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
    a = ap.parse_args()
    tasks = [(n, d, nobs, r) for n in INDICES for d in DESIGNS for nobs in NS
             for r in range(a.reps)]
    fields = ["indices", "design", "nobs", "rep", "series", "n_i", "p", "q",
              "label", "tied"]
    with open(a.out, "w", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k, rows in enumerate(pool.imap_unordered(one, tasks, chunksize=4)):
            w.writerows(rows)
            if k % 200 == 0:
                fh.flush()
                print(f"{k}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
