"""BUG-0030: the ladder evaluated points the C's ladder refuses, and walked
the parameters in another order.

Three things, all in the port, kept it from following the C where the search
runs along the MA invertibility wall:

1. THE CAST FLIPPED THE MA. The port casts each series with fue's
   cast_us_py, and fue takes the invertible form of a regular MA(1) with
   |theta| > 1 (1/theta) before it builds the polynomial. The C's ladder
   (escalera.c shootx_esc) does not: it builds the polynomial from the
   parameters as they are, and chekma refuses the point. Same x in both
   engines, theta_WTI = 1.013852 in the C (refused), 0.986337 in the port
   (evaluated).
2. THE ORDER OF THE PARAMETERS. The C packs the ARMA of every series, then
   the deterministic terms, then the means (esc_pack); the port packed series
   by series. The same model, the gradient in another sequence.
3. THE OLS SEED by numpy's normal equations instead of the C's sequential
   sums and LU: the same numbers to the sixteenth digit.

The repro checks 1 and 2 on a pair, and the wall case the C gives
(-1740.636683 in 62 iterations; the port gave -1742.598015 in 38).

    python3 bugs/BUG-0030-repro/repro.py   # exits 1 while open
"""
import os
import sys
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from drvarma.ladder import Ladder   # noqa: E402

D = os.path.join(HERE, "..", "..", "tests", "data", "ladder")
files = [os.path.join(D, "GS1_m01.pre"), os.path.join(D, "FF_m03.pre")]
bad = 0

# 2. the order: every ARMA coefficient before every mean (IPC and WTI have one)
ipc = [os.path.join(D, "IPC_m07.pre"), os.path.join(D, "WTI_m03.pre")]
L = Ladder(ipc, 1, 0)
names = L.names()
own = [n for n in names if n.split("_")[0] in ("phi", "Phi", "theta", "Theta", "mu")]
last_arma = max(i for i, n in enumerate(own) if not n.startswith("mu_"))
first_mu = min(i for i, n in enumerate(own) if n.startswith("mu_"))
print("own parameters:", " ".join(own))
if first_mu < last_arma:
    print("OPEN: a mean is packed before another series' ARMA (the C packs all the ARMA first)")
    bad = 1

# 1. the cast: an own MA(1) beyond the unit circle stays where it is
L = Ladder(files, 1, 1, route="v", start="zero")
r = L.fit()
print(f"GS1 + FF, VARMA(1,1), route V, from zero: logL {r.logL:.6f} in {r.nit} "
      "iterations (the C: -1740.636683 in 62)")
if abs(r.logL - (-1740.636683)) > 1e-5 or r.nit != 62:
    print("OPEN: the port does not follow the C along the MA wall")
    bad = 1
v = L.pack()
k = L.names().index("theta_GS1[B^1]")
v[k] = 1.25
_mu, _phi, theta, _qq, _w, ifa = L.cast(v)
print(f"theta_GS1 = 1.25 in x -> {theta[0][0][0]:.6f} in the model the likelihood gets")
if ifa == 0 and abs(theta[0][0][0] - 1.25) > 1e-12:
    print("OPEN: the cast took the reciprocal (fue's rule); the C does not")
    bad = 1
print("open" if bad else "closed: the same order, the same cast, the same path as the C")
sys.exit(bad)
