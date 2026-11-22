"""BUG-0029: factor_test freed, in its alternative, an own coefficient the
model holds at zero (own_zeros), so it tested that zero and the row factor
together.

The echelon form (1, 1) of IPI and WTI with theta_IPI[B^1] at zero. IPI has
an AR(1)_12 that multiplies its row. Without the factor the product reaches
two positions more (the own lag 13 and the cross lag 13 of IPI's row): 2
degrees of freedom, and LR 0.4849 (the C's, atsw-gui #219). The defect gave
3, with theta_IPI[B^1] free in the alternative.

    python3 bugs/BUG-0029-repro/repro.py      # exits 1 while open
"""
import os, sys, warnings
warnings.simplefilter("ignore")
from drvarma.ladder import Ladder
HERE = os.path.dirname(os.path.abspath(__file__))
files = [os.path.join(HERE, f) for f in ("IPI_m07.pre", "WTI_m03.pre")]
L = Ladder(files, echelon=(1, 1), own="free", own_zeros={"IPI": {"ma": {1}}}, start="zero")
L.fit()
r = L.factor_test()
extra = sorted(set(r["free"].names()) - set(L.names()))
print(f"LR {r['lr']:.4f}  df {r['df']}  free in the alternative only: {extra}")
sys.exit(1 if "theta_IPI[B^1]" in extra or r["df"] != 2 else 0)
