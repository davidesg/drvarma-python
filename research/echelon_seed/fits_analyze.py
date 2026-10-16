"""Summary of fits.csv (fits.py): per form, how the exact-ML search stops,
and, since every form is nested in `full`, the LR of each against it.

    python3 fits_analyze.py [fits.csv [fits_echelon.csv ...]] > fits_summary.md
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(k, n):
    return f"{100.0 * k / n:5.1f}" if n else "  -  "


def main(paths):
    rows = [r for p in paths for r in csv.DictReader(open(p))]
    forms = list(dict.fromkeys(r["form"] for r in rows))
    by = defaultdict(dict)
    for r in rows:
        by[(r["design"], r["indices"], r["nobs"], r["rep"])][r["form"]] = r
    print("## How the search stops\n")
    print("termcode 1: gradient; 2: step; 3: no improvement (BUG-0010: the flat")
    print("directions of a weakly identified model); wall: an MA inverse root on")
    print("the invertibility boundary at the stop.\n")
    print("| design | indices | form | fits | npar | tc 1 % | tc 2 % | tc 3 % "
          "| wall % | error % | mean nit | mean s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for design in S.DESIGNS:
        for n in S.INDICES:
            key = "".join(map(str, n))
            for form in forms:
                rs = [g[form] for (d, k, _o, _r), g in by.items()
                      if d == design and k == key and form in g]
                if not rs:
                    continue
                ok = [r for r in rs if not r["error"]]
                tc = [int(r["termcode"]) for r in ok]
                print(f"| {design} | {n} | {form} | {len(rs)} "
                      f"| {ok[0]['npar'] if ok else '-'} "
                      f"| {pct(tc.count(1), len(rs))} | {pct(tc.count(2), len(rs))} "
                      f"| {pct(tc.count(3), len(rs))} "
                      f"| {pct(sum(int(r['ma_boundary']) > 0 for r in ok), len(rs))} "
                      f"| {pct(len(rs) - len(ok), len(rs))} "
                      f"| {sum(int(r['nit']) for r in ok) / max(len(ok), 1):.0f} "
                      f"| {sum(float(r['seconds']) for r in rs) / len(rs):.1f} |")
    print()
    print("## Each form against `full` (nested): LR = 2 (logL_full - logL_form)\n")
    print("A negative LR means the search for `full` stopped below the smaller")
    print("model's optimum, i.e. it did not find its own maximum.\n")
    print("| design | indices | form | pairs | df | median LR | reject 5 % | LR < 0 % |")
    print("|---|---|---|---|---|---|---|---|")
    for design in S.DESIGNS:
        for n in S.INDICES:
            key = "".join(map(str, n))
            for form in forms:
                if form == "full":
                    continue
                lr, dfs, rej, neg = [], set(), 0, 0
                for (d, k, _o, _r), g in by.items():
                    if d != design or k != key or "full" not in g or form not in g:
                        continue
                    a, b = g["full"], g[form]
                    if a["error"] or b["error"]:
                        continue
                    x = 2.0 * (float(a["logL"]) - float(b["logL"]))
                    df = int(a["npar"]) - int(b["npar"])
                    lr.append(x); dfs.add(df)
                    rej += x > chi2.ppf(0.95, df)
                    neg += x < -1e-3
                if not lr:
                    continue
                lr.sort()
                print(f"| {design} | {n} | {form} | {len(lr)} | {'/'.join(map(str, sorted(dfs)))} "
                      f"| {lr[len(lr) // 2]:.2f} | {pct(rej, len(lr))} | {pct(neg, len(lr))} |")
    print()
    if "truth" not in forms:
        return
    print("## Each fit against the truth: LR = 2 (logL_form - logL_truth)\n")
    print("The echelon form evaluated at the true system (sigma2 concentrated).")
    print("For a fit that reaches its maximum, LR >= 0 and ~ chi2(npar); LR < 0")
    print("means the search stopped below the TRUE parameters.\n")
    print("| design | indices | form | pairs | npar | median LR | LR < 0 % | above chi2 95 % |")
    print("|---|---|---|---|---|---|---|---|")
    for design in S.DESIGNS:
        for n in S.INDICES:
            key = "".join(map(str, n))
            for form in forms:
                if not form.startswith("echelon"):
                    continue
                lr, k, neg, hi = [], None, 0, 0
                for (d, kk, _o, _r), g in by.items():
                    if d != design or kk != key or "truth" not in g or form not in g:
                        continue
                    a, b = g[form], g["truth"]
                    if a["error"] or b["error"]:
                        continue
                    x = 2.0 * (float(a["logL"]) - float(b["logL"]))
                    k = int(a["npar"])
                    lr.append(x)
                    neg += x < -1e-3
                    hi += x > chi2.ppf(0.95, k)
                if not lr:
                    continue
                lr.sort()
                print(f"| {design} | {n} | {form} | {len(lr)} | {k} | {lr[len(lr) // 2]:.2f} "
                      f"| {pct(neg, len(lr))} | {pct(hi, len(lr))} |")
    print()


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(HERE, "fits.csv")])
