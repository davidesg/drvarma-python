"""BUG-0033: the pure-Python .inp path does not reach the C's optimum.

PSW, VARMA(1,1) (the C battery's case c3_psw_arma11). The C (atsw-gui
engines/drvarma, tests/banco/ref) and the port with the compiled engine end
at logL -871.32401308; the pure-Python estimator (DRVARMA_NO_ENGINE=1) at
-871.542051, 0.22 lower.

Exits 1 while the pure-Python logL is more than 1e-5 from the C's, 0 when
it is not.

    python3 bugs/BUG-0033-repro/repro.py   # exits 1 while open
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
C_LOGL = -871.32401308

with tempfile.TemporaryDirectory() as t:
    shutil.copy(os.path.join(HERE, "PSW.inp"), t)
    env = dict(os.environ, DRVARMA_NO_ENGINE="1",
               PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run([sys.executable, "-m", "drvarma.cli", "PSW", "1", "1"], cwd=t,
                   env=env, capture_output=True, text=True, timeout=1800)
    out = open(os.path.join(t, "PSW.out"), errors="replace").read()
logl = float(re.findall(r"Log-likelihood = (-?[\d.]+)", out)[-1])
print(f"pure-Python logL {logl:.6f}, the C's {C_LOGL:.6f}")
if abs(logl - C_LOGL) > 1e-5:
    print("OPEN: the pure-Python .inp path stops elsewhere")
    sys.exit(1)
print("closed")
sys.exit(0)
