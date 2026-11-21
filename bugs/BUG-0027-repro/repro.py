"""BUG-0027: Ladder.fit, VAR(2) of route U from the zero start, stops after 2
iterations (termcode 3, "last global step failed to locate a lower point")
with every cross term at ~0 and Sigma diagonal: logL -1793.05, BELOW the
nested VAR(1) (-1744.88). Shea's likelihood from the same start reaches
-1741.28. Only this system: the same p = 2 with FF, GS2 or GS3 in place of
GS1 converges.

Files: art's .pre of the US monetary VAR 1990:01-2007:12 (IPI m07, IPC m07,
WTI m03, GS1 m01), in this order.

    python3 bugs/BUG-0027-repro/repro.py      # exits 1 while open (~5 min)

Fixed in 0.4.0: the default start with q = 0 is the least squares ("ols"),
as in the C, and the VAR(2) reaches -1741.2755. Asking for start="zero"
still stalls: a known limit, not the default's path.
"""
import os, sys, time
from drvarma.ladder import Ladder
HERE = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(HERE, f) for f in
         ("IPI_m07.pre", "IPC_m07.pre", "WTI_m03.pre", "GS1_m01.pre")]


def run(p, lik="elf", start=None):
    t = time.time()
    L = Ladder(files, p, 0, lik=lik, start=start)
    L.fit()
    r = L.result
    print(f"p={p} lik={lik:<5} start={L.start_used:<12} logL {r.logL:.4f}  "
          f"nit {r.nit:>3}  termcode {r.termcode}  se {r.se_method}  "
          f"({time.time() - t:.0f}s)", flush=True)
    return r


r1 = run(1)
r2 = run(2)
r2s = run(2, lik="shea")
nested_violated = r2.logL < r1.logL - 1e-6
print("VAR(2) elf below the nested VAR(1):", nested_violated)
print("Shea reaches above elf by %.2f" % (r2s.logL - r2.logL))
sys.exit(1 if nested_violated else 0)
