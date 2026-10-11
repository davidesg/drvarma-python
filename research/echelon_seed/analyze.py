"""Summary of results.csv (study.py): how often the seed max(p, q) of the
first-ranked univariate model hits, falls short of or exceeds the Kronecker
index n_i; and whether any tied candidate's seed hits it.

    python3 analyze.py [results.csv]  > summary.md
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echelon as E
import study as S


def load(path):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if int(r["p"]) >= 0]


def seed(r):
    return max(int(r["p"]), int(r["q"]))


def tied_seeds(r):
    return {max(map(int, t.split(","))) for t in r["tied"].split(";") if t}


def pct(k, n):
    return f"{100.0 * k / n:5.1f}" if n else "  -  "


def per_series(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["design"], r["indices"], int(r["series"]), int(r["nobs"]))].append(r)
    print("## Per series: the seed max(p, q) of art's first-ranked model against n_i\n")
    print("N is the McMillan degree, sum n_i: the order of every component's")
    print("marginal ARMA, exactly (echelon.marginal_orders).\n")
    for design in S.DESIGNS:
        print(f"### Design `{design}` (cross coefficients x {S.DESIGNS[design]})\n")
        print("| indices | N | series | n_i | nobs | = n_i % | < n_i % | > n_i % "
              "| mean seed | tied hit % |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for n in S.INDICES:
            key = "".join(map(str, n))
            for i in range(1, len(n) + 1):
                for nobs in S.NS:
                    rs = g.get((design, key, i, nobs), [])
                    if not rs:
                        continue
                    ni = n[i - 1]
                    sd = [seed(r) for r in rs]
                    eq = sum(s == ni for s in sd)
                    lo = sum(s < ni for s in sd)
                    hi = sum(s > ni for s in sd)
                    th = sum(ni in tied_seeds(r) for r in rs)
                    print(f"| {n} | {sum(n)} | {i} | {ni} | {nobs} | {pct(eq, len(rs))} "
                          f"| {pct(lo, len(rs))} | {pct(hi, len(rs))} "
                          f"| {sum(sd) / len(sd):.2f} | {pct(th, len(rs))} |")
        print()


def per_system(rows):
    """Does the vector of seeds equal the vector of indices? And the
    weaker readings: the same ordering, and the sum against N."""
    g = defaultdict(dict)
    for r in rows:
        g[(r["design"], r["indices"], int(r["nobs"]), int(r["rep"]))][int(r["series"])] = seed(r)
    agg = defaultdict(lambda: [0, 0, 0, 0, 0])
    for (design, key, nobs, rep), sd in g.items():
        n = tuple(int(c) for c in key)
        if len(sd) != len(n):
            continue
        v = [sd[i + 1] for i in range(len(n))]
        a = agg[(design, key, nobs)]
        a[0] += 1
        a[1] += v == list(n)
        a[2] += all((v[i] - v[j]) * (n[i] - n[j]) >= 0 and
                    ((v[i] == v[j]) == (n[i] == n[j]))
                    for i in range(len(n)) for j in range(i + 1, len(n)))
        a[3] += sum(v) == sum(n)
        a[4] += sum(v)
    print("## Per system: all the indices at once\n")
    print("| design | indices | N | nobs | all equal % | same ordering % "
          "| sum = N % | mean sum |")
    print("|---|---|---|---|---|---|---|---|")
    for design in S.DESIGNS:
        for n in S.INDICES:
            key = "".join(map(str, n))
            for nobs in S.NS:
                a = agg.get((design, key, nobs))
                if not a or not a[0]:
                    continue
                print(f"| {design} | {n} | {sum(n)} | {nobs} | {pct(a[1], a[0])} "
                      f"| {pct(a[2], a[0])} | {pct(a[3], a[0])} | {a[4] / a[0]:.2f} |")
    print()


def overall(rows):
    print("## Overall, by design and sample size\n")
    print("| design | nobs | series | = n_i % | < n_i % | > n_i % |")
    print("|---|---|---|---|---|---|")
    for design in S.DESIGNS:
        for nobs in S.NS:
            rs = [r for r in rows if r["design"] == design and int(r["nobs"]) == nobs]
            if not rs:
                continue
            d = [seed(r) - int(r["n_i"]) for r in rs]
            print(f"| {design} | {nobs} | {len(rs)} | {pct(sum(x == 0 for x in d), len(d))} "
                  f"| {pct(sum(x < 0 for x in d), len(d))} | {pct(sum(x > 0 for x in d), len(d))} |")
    print()


def systems():
    print("## The systems\n")
    print("Free values uniform in ±[0.2, 0.7], the cross ones times the design's")
    print(f"factor; inverse roots ≤ 0.85; Σ with correlation {S.SIGMA_RHO}.\n")
    print("| indices | design | AR / MA max inverse root | marginal (p*, q*) |")
    print("|---|---|---|---|")
    for n in S.INDICES:
        for d in S.DESIGNS:
            A, M, _ = S.system(n, d)
            print(f"| {n} | {d} | {E.inverse_root_max(A):.2f} / {E.inverse_root_max(M):.2f} "
                  f"| {E.marginal_orders(A, M)} |")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(S.HERE, "results.csv")
    rows = load(path)
    systems()
    overall(rows)
    per_system(rows)
    per_series(rows)
