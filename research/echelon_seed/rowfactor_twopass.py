"""Stage 4: ladder.search_row_factors -- the search in passes, the first
with the univariate (gate) factors, the next with those of the exact-ML
fit of the previous choice -- on rowfactor_study.py's systems and data.

    python3 rowfactor_twopass.py [--reps 50] [--workers 3]
"""
import argparse, csv, tempfile, warnings
from multiprocessing import Pool
import rowfactor_study as R
import study as S


def one(task):
    n, pattern, rep = task
    warnings.simplefilter("ignore")
    from drvarma.ladder import search_row_factors
    y = R.data(n, pattern, 200, rep)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            sr, hist = search_row_factors(R.files(n, pattern, y, tmp), n_max=3)
            ps = [("".join(map(str, h[0]))) for h in hist]
            err = ""
        except Exception as e:                                # noqa: BLE001
            ps, err = [], f"{type(e).__name__}: {e}"[:200]
    return dict(indices="".join(map(str, n)), pattern=pattern, rep=rep,
                pass1=ps[0] if ps else "", final=ps[-1] if ps else "",
                passes=len(ps), error=err)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="rowfactor_twopass.csv")
    a = ap.parse_args()
    tasks = [(n, p, r) for n in S.INDICES for p in ("sar", "mixed") for r in range(a.reps)]
    with Pool(a.workers) as pool, open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["indices", "pattern", "rep", "pass1", "final",
                                           "passes", "error"])
        w.writeheader()
        for k, row in enumerate(pool.imap_unordered(one, tasks)):
            w.writerow(row); fh.flush()
            if k % 50 == 0:
                print(f"{k}/{len(tasks)}", flush=True)
    print("rc=0", flush=True)
