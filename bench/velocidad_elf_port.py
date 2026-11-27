"""Speed of the pure-Python elf (_as311.elf) against the compiled one, at the
orders sima sends it. Starting measurement of the study in TODO.md
(«ESTUDIO — La velocidad del port en Python»); not a bug: the port is a
faithful, loop-by-loop port of elfvarma.c.

sima's fits carry each series' univariate AR polynomial expanded inside
Phi(B): on the US monetary system (IPI, IPC, WTI, FF, 1990-2007) elf receives
m = 4, n = 215, p = 15, and a sima estimate calls it 3,293 times with p = 0
and 8,795 with p = 1. The documented ratio (DEVELOPER_GUIDE 4.1, 13-100x;
README, 20-250x) was measured with estimate_w and p <= 3.

The script scores the same synthetic stationary VAR(p), m = 4, n = 215, with
the engine and with the port at p = 2 and p = 15, and prints the time per
evaluation and the ratio. Measured 2026-11-27, Linux x86_64: 0.12 / 46 ms
(379x) at p = 2, 14.9 / 550 ms (37x) at p = 15; on the real system 4.9 / 704
ms (143x), so about 40 min (p = 0) and 1.7 h (p = 1) for a sima estimate.

    python3 bench/velocidad_elf_port.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import drvarma._engine as EN   # noqa: E402

if EN._load_c() is None:
    sys.exit("the compiled engine is needed to compare (build it first)")

m, n = 4, 215
rng = np.random.default_rng(31)


def case(p):
    phi = np.zeros((p, m, m))
    for k in range(p):                       # small, decaying: stationary
        phi[k] = rng.normal(0, 0.04, (m, m)) * 0.8 ** k
    phi[0] += np.diag([0.3, 0.2, 0.25, 0.1])
    qq = np.eye(m)
    w = np.zeros((n + p, m))
    e = rng.normal(size=(n + p, m))
    for t in range(p, n + p):
        w[t] = e[t] + sum(phi[k] @ w[t - k - 1] for k in range(p))
    return (m, n, p, 0, np.zeros(m), phi, np.zeros((0, m, m)), qq, w[p:])


def per_eval(args, reps):
    t = time.perf_counter()
    for _ in range(reps):
        out = EN.elf_c(*args)
    return (time.perf_counter() - t) / reps, out[0]


real = EN._load_c
for p in (2, 15):
    args = case(p)
    tc, lc = per_eval(args, 20)
    EN._load_c = lambda: None                # what the py3-none-any wheel gives
    tp, lp = per_eval(args, 1)
    EN._load_c = real
    r = tp / tc
    print(f"m={m} n={n} p={p:2d}: C {tc*1e3:7.2f} ms   port {tp*1e3:8.1f} ms   "
          f"{r:5.0f}x   (logL {lc:.6f} / {lp:.6f})")
print(f"a sima estimate of that size: ~3,300 (p = 0) to ~8,800 (p = 1) evaluations")
