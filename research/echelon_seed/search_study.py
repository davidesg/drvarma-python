"""Stage 3b's first measure (docs/ECHELON_PLAN.md §2.2): does Lutkepohl and
Poskitt's stage I-II search find the Kronecker indices? On stage 0's grid
(the same systems, designs, sample sizes and seeds), four variants: Hannan
and Kavalieris' two steps or full enumeration, with HQ or SC. n_max 3.

    python3 search_study.py [--reps 100] [--workers 4] [--iterate 0]
        [--out search.csv]

--iterate: the rounds of Hannan and Kavalieris' residual correction per
candidate (echelon.stage2_refined); 0 is plain stage II.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S
from fits import data

VARIANTS = [("hk", "HQ"), ("hk", "SC"), ("full", "HQ"), ("full", "SC")]
FIELDS = ["indices", "design", "nobs", "rep", "h"] + [f"{a}_{c}" for a, c in VARIANTS] \
    + ["hk_fits"]


def one(task):
    from drvarma import echelon as X
    n, design, nobs, rep, it = task
    warnings.simplefilter("ignore")
    y = data(n, design, nobs, rep)
    row = dict(indices="".join(map(str, n)), design=design, nobs=nobs, rep=rep)
    for crit in ("HQ", "SC"):
        s = X.search(y, n_max=3, criterion=crit, iterate=it)
        row[f"hk_{crit}"] = "".join(map(str, s.indices))
    row["hk_fits"] = len(s.fits)
    f = X.search(y, n_max=3, full=True, iterate=it)
    row["h"] = f.h
    for crit in ("HQ", "SC"):
        best = min(f.fits, key=lambda k: f.fits[k].criterion(crit))
        row[f"full_{crit}"] = "".join(map(str, best))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iterate", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "search.csv"))
    a = ap.parse_args()
    tasks = [(n, d, nobs, r, a.iterate) for n in S.INDICES for d in S.DESIGNS for nobs in S.NS
             for r in range(a.reps)]
    with open(a.out, "w", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k, row in enumerate(pool.imap_unordered(one, tasks, chunksize=4)):
            w.writerow(row)
            if k % 500 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
    print("rc=0", flush=True)


if __name__ == "__main__":
    main()
