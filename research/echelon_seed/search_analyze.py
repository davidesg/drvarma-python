"""Summary of search_study.py's CSVs: the share of replications where each
variant finds the true indices, and where it over- or underfits (every
index >= or <= the truth).

    python3 search_analyze.py search.csv[:label] search_it1.csv[:label] ...
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S

VARIANTS = ["hk_HQ", "hk_SC", "full_HQ", "full_SC"]


def main(args):
    runs = []
    for a in args:
        path, _, label = a.partition(":")
        runs.append((label or os.path.splitext(os.path.basename(path))[0],
                     list(csv.DictReader(open(path)))))
    cols = [f"{lab} {v}" for lab, _r in runs for v in VARIANTS]
    print("% of replications with the true indices.\n")
    print("| design | indices | nobs | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for d in S.DESIGNS:
        for n in S.INDICES:
            k = "".join(map(str, n))
            for T in S.NS:
                cells = []
                for _lab, R in runs:
                    rs = [r for r in R if r["design"] == d and r["indices"] == k
                          and int(r["nobs"]) == T]
                    cells += [f"{100 * sum(r[v] == k for r in rs) / len(rs):.0f}"
                              if rs else "-" for v in VARIANTS]
                print(f"| {d} | {n} | {T} | " + " | ".join(cells) + " |")
    print()
    print("Over- and underfitting of hk_SC (% of replications).\n")
    print("| design | indices | nobs | " + " | ".join(f"{lab} over | {lab} under"
                                                     for lab, _r in runs) + " |")
    print("|---|---|---|" + "---|---|" * len(runs))
    for d in S.DESIGNS:
        for n in S.INDICES:
            k = "".join(map(str, n))
            for T in S.NS:
                cells = []
                for _lab, R in runs:
                    rs = [r for r in R if r["design"] == d and r["indices"] == k
                          and int(r["nobs"]) == T]
                    x = [r["hk_SC"] for r in rs if r["hk_SC"] != k]
                    over = sum(all(a >= b for a, b in zip(s, k)) for s in x)
                    under = sum(all(a <= b for a, b in zip(s, k)) for s in x)
                    cells += [f"{100 * over / max(len(rs), 1):.0f}",
                              f"{100 * under / max(len(rs), 1):.0f}"]
                print(f"| {d} | {n} | {T} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
