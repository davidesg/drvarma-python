"""BUG-0020: the diagonal gate fails with NaN when a univariate optimum sits
on the MA wall, and its message blames the cast.

The echelon study's system (2,2,1), rep 45, on route V's specifications
(y1.inp, y2.inp, y3.inp, written by research/echelon_seed/fits.py with a
fue header). y2's univariate MA(2) ends with an inverse root at modulus
~1.00005, inside chekma's tolerance: alone elf accepts it; the diagonal
system of three fails elf (ifault 3, the Cholesky of mtmp0). The gate said
"difference nan: the joint cast does not reproduce the univariate models".
The C said the same with a 0 for the joint logL (its BUG-0009).

Exits 1 while the GateError does not say that the diagonal system could not
be evaluated and name y2's MA on the wall, 0 when it does, 2 if the case
passes the gate (then there is nothing to say).

    python3 bugs/BUG-0020-repro/repro.py   # exits 1 while open
"""
import glob
import os
import sys
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from drvarma.ladder import GateError, Ladder  # noqa: E402

L = Ladder(sorted(glob.glob(os.path.join(HERE, "y*.inp"))), 0, 0)
try:
    L.run_gate()
except GateError as e:
    msg = str(e)
else:
    print("the case passes the gate: nothing to show")
    sys.exit(2)
print(msg)
if "could not be evaluated" in msg and "invertibility wall: y2 (" in msg:
    print("closed: the gate says the joint diagonal failed and names y2's MA")
    sys.exit(0)
print("OPEN: the gate blames the cast and does not name the MA on the wall")
sys.exit(1)
