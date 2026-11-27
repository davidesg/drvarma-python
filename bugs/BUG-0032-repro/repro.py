"""BUG-0032: DRVARMA_NO_ENGINE is documented as "asks for the port", and
engine_backend() then answers "python"; but elf_c and marma_c do not read it
and keep computing in C. Anyone timing the port with the variable (as the
2026-11-27 investigation did) measures the C and concludes both engines cost
the same.

    python3 bugs/BUG-0032-repro/repro.py   # exits 1 while open
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import drvarma._engine as EN   # noqa: E402

if EN._load_c() is None:
    sys.exit("needs the compiled engine")
m, n, p = 3, 200, 2
rng = np.random.default_rng(33)
w = rng.normal(size=(n, m))
args = (m, n, p, 0, np.zeros(m), np.zeros((p, m, m)) + 0.05, np.zeros((0, m, m)),
        np.eye(m), w)


def t(reps=5):
    s = time.perf_counter()
    for _ in range(reps):
        EN.elf_c(*args)
    return (time.perf_counter() - s) / reps


c = t()
os.environ["DRVARMA_NO_ENGINE"] = "1"
said, asked = EN.engine_backend(), t()
real = EN._load_c
EN._load_c = lambda: None
port = t(1)
EN._load_c = real
print(f"engine_backend() with DRVARMA_NO_ENGINE=1: {said!r}")
print(f"elf_c: {c*1e3:.2f} ms without the variable, {asked*1e3:.2f} ms with it, "
      f"{port*1e3:.1f} ms on the port")
bad = said == "python" and asked < port / 5
print("OPEN: the variable says python and elf_c computes in C" if bad else "fixed")
sys.exit(1 if bad else 0)
