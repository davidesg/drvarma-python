"""BUG-0028: the ols start's fall-back to zero. The port's Ladder (start="ols")
falls back to zero when elf does not accept the least-squares point; the C's
ladder (atsw-gui engines/drvarma, -start ols, BUG-0006 there) says it does and
starts from the least-squares point instead, failing with "AR nonstationary"
(rc 3). The homologation breaks on exactly the case the fall-back is for.

Files: Box and Jenkins' gas furnace (series J), route V of Tiao and Box: two
specifications with only their transformation and a free own AR(6)
(GAS_v.inp, CO2_v.inp), VAR(6). The least squares of the VAR(6) with the own
AR of the diagonal fit is not stationary.

    DRVARMA_C=/path/to/engines/drvarma/bin/drvarma python3 bugs/BUG-0028-repro/repro.py

(DRVARMA_C is required: a `drvarma` on the PATH may be another program). Exits 1 while the C with -start ols does not
reach the port's optimum, 0 when it does, 2 if the C is missing or the port's
own fit is not the reference (-2638.762693, both starts), or the C has no
-start ols (before atsw-gui #199).
"""
import os, re, shutil, subprocess, sys, tempfile
from drvarma.ladder import Ladder

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("GAS_v.inp", "CO2_v.inp")
OPTIMUM = -2638.762693

for start in ("zero", "ols"):
    L = Ladder([os.path.join(HERE, f) for f in FILES], 6, 0, start=start)
    L.fit()
    print(f"port  start={start:<4} used: {L.start_used}  logL {L.result.logL:.6f}")
    if abs(L.result.logL - OPTIMUM) > 1e-5:
        print("the port is not at the reference optimum")
        sys.exit(2)

c = os.environ.get("DRVARMA_C")
if not c or not os.path.exists(c):
    print("no drvarma C: set DRVARMA_C")
    sys.exit(2)
with tempfile.TemporaryDirectory() as t:
    for f in FILES:
        shutil.copy(os.path.join(HERE, f), t)
    r = subprocess.run([c, *FILES, "6", "0", "-start", "ols", "-o", "ols"], cwd=t,
                       capture_output=True, text=True, timeout=900)
m = re.search(r"^Exact log-likelihood\s*:?\s*(-?[\d.]+)", r.stdout, re.M)
used = re.search(r"^Start of the cross terms: (.*)$", r.stdout, re.M)
ll = float(m.group(1)) if m else None
if not used:
    print("this C has no -start ols")
    sys.exit(2)
print(f"C     start=ols  used: {used.group(1)}  rc {r.returncode}  logL {ll}")
if r.returncode != 0 or ll is None or abs(ll - OPTIMUM) > 1e-5:
    print("the C does not fall back to zero: it starts from the least-squares point")
    sys.exit(1)
print("the C and the port agree")
sys.exit(0)
