"""Tables of rowfactor_study.py's csv: per system and factor pattern,

  fits   : how often each start ends below the true parameters (LR < 0),
           the median LR against chi2(npar), termcode 3 and MA-wall stops,
           iterations and seconds; and ls against zero
  search : the share of replications where the search finds the indices,
           with the true factors, the gate's, and none

  passes : with rowfactor_twopass.py's csv, ladder.search_row_factors --
           the first pass (the univariate factors) and the last

    python3 rowfactor_analyze.py rowfactor.csv [rowfactor_twopass.csv]
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import chi2


def main(path):
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(dict)
    for r in rows:
        by[(r["indices"], r["pattern"], r["nobs"], r["rep"])][r["form"]] = r
    cells = sorted({k[:3] for k in by}, key=lambda c: (len(c[0]), c[0], c[1], c[2]))
    print("## Fits (LR = 2 (logL_fit - logL_truth))\n")
    print("| system | pattern | nobs | form | n | below truth | median LR / npar | "
          "> chi2 95% | tc 3 | MA wall | median nit | median s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cmp_rows = []
    for c in cells:
        reps = [v for k, v in by.items() if k[:3] == c]
        for form in ("ls", "zero"):
            lr, npar, tc3, wall, nit, sec, err = [], [], 0, 0, [], [], 0
            for v in reps:
                f, t = v.get(form), v.get("truth")
                if not f or not t or f["error"] or t["error"] or not f["logL"]:
                    err += 1
                    continue
                lr.append(2 * (float(f["logL"]) - float(t["logL"])))
                npar.append(int(f["npar"]))
                tc3 += f["termcode"] == "3"
                wall += f["ma_boundary"] not in ("", "0")
                nit.append(int(f["nit"]))
                sec.append(float(f["seconds"]))
            if not lr:
                continue
            lr = np.array(lr)
            k = len(lr)
            q95 = chi2.ppf(0.95, npar[0])
            print(f"| ({','.join(c[0])}) | {c[1]} | {c[2]} | {form} | {k}"
                  + (f" ({err} err)" if err else "")
                  + f" | {100 * np.mean(lr < -1e-3):.0f} % | {np.median(lr):.1f} / {npar[0]} "
                  f"| {100 * np.mean(lr > q95):.0f} % | {100 * tc3 / k:.0f} % "
                  f"| {100 * wall / k:.0f} % | {np.median(nit):.0f} | {np.median(sec):.1f} |")
        d = []
        for v in reps:
            a, b = v.get("ls"), v.get("zero")
            if a and b and a["logL"] and b["logL"]:
                d.append(float(a["logL"]) - float(b["logL"]))
        if d:
            d = np.array(d)
            cmp_rows.append((c, np.mean(d > 1e-3), np.mean(np.abs(d) <= 1e-3),
                             np.mean(d < -1e-3)))
    print("\n## The least-squares start against zero (logL)\n")
    print("| system | pattern | nobs | ls higher | same | ls lower |")
    print("|---|---|---|---|---|---|")
    for c, hi, eq, lo in cmp_rows:
        print(f"| ({','.join(c[0])}) | {c[1]} | {c[2]} | {100 * hi:.0f} % | "
              f"{100 * eq:.0f} % | {100 * lo:.0f} % |")
    print("\n## The index search (SC, n_max 3): found the true indices\n")
    print("| system | pattern | nobs | true factors | gate's factors | unfiltered |")
    print("|---|---|---|---|---|---|")
    for c in cells:
        reps = [v for k, v in by.items() if k[:3] == c]
        out = []
        for how in ("true", "gate", "none"):
            got = [v[f"search_{how}"]["found"] for v in reps
                   if f"search_{how}" in v and not v[f"search_{how}"]["error"]]
            out.append(f"{100 * np.mean([g == c[0] for g in got]):.0f} %" if got else "–")
        print(f"| ({','.join(c[0])}) | {c[1]} | {c[2]} | " + " | ".join(out) + " |")


def passes(path):
    rows = list(csv.DictReader(open(path)))
    cells = sorted({(r["indices"], r["pattern"]) for r in rows},
                   key=lambda c: (len(c[0]), c[0], c[1]))
    print("\n## The search in passes (ladder.search_row_factors, SC, n_max 3)\n")
    print("| system | pattern | n | pass 1 (univariate factors) | final | mean passes |")
    print("|---|---|---|---|---|---|")
    for c in cells:
        rs = [r for r in rows if (r["indices"], r["pattern"]) == c and not r["error"]]
        err = sum(1 for r in rows if (r["indices"], r["pattern"]) == c and r["error"])
        print(f"| ({','.join(c[0])}) | {c[1]} | {len(rs)}" + (f" ({err} err)" if err else "")
              + f" | {100 * np.mean([r['pass1'] == c[0] for r in rs]):.0f} % "
              f"| {100 * np.mean([r['final'] == c[0] for r in rs]):.0f} % "
              f"| {np.mean([int(r['passes']) for r in rs]):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
    if len(sys.argv) > 2:
        passes(sys.argv[2])
