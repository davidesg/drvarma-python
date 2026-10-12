"""Stage 4: the search with the gate's factors, then again with the factors
of the exact-ML fit of its choice (two passes). On the cells where the
gate's factors misled it."""
import sys, tempfile, warnings, csv
from multiprocessing import Pool
import numpy as np
import rowfactor_study as R

def one(task):
    n, pattern, rep = task
    warnings.simplefilter("ignore")
    from drvarma import echelon as E
    from drvarma.ladder import Ladder
    y = R.data(n, pattern, 200, rep)
    with tempfile.TemporaryDirectory() as tmp:
        fs = R.files(n, pattern, y, tmp)
        L = Ladder(fs, echelon=n, start="zero"); L.run_gate()
        m = len(n); L._set_structure(list(range(m)), 0, 0, True)
        W = L.cast(L.pack())[4]
        f1 = [tuple(s.split_polynomials()[2:4]) for s in L.series]
        n1 = E.search(W, n_max=3, filters=f1).indices
        try:
            L2 = Ladder(fs, echelon=n1, own="free"); L2.fit()
            f2 = [tuple(s.split_polynomials()[2:4]) for s in L2.series]
            n2 = E.search(W, n_max=3, filters=f2).indices
        except Exception as e:
            n2 = ("err",)
    return dict(indices="".join(map(str,n)), pattern=pattern, rep=rep,
                pass1="".join(map(str,n1)), pass2="".join(map(str,n2)))

if __name__ == "__main__":
    cells = [((2,1),"mixed"), ((1,2),"sar"), ((1,0),"mixed")]
    tasks = [(n,p,r) for n,p in cells for r in range(50)]
    with Pool(1) as pool, open("rowfactor_twopass.csv","w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["indices","pattern","rep","pass1","pass2"]); w.writeheader()
        for row in pool.imap_unordered(one, tasks):
            w.writerow(row); fh.flush()
    print("rc=0")
