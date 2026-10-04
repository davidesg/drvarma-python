"""BUG-0003 — deseasonalize in levels (today, as the C) or after the log?

On IPC3 (IPC_ES, IPC_FR, IPC_DE; 2002-2019, monthly) it measures:
  1. the seasonal amplitude on each half of the sample, estimated in levels
     and in logs: a multiplicative pattern keeps its amplitude in logs and
     grows with the level in levels;
  2. the seasonal ACF of the residuals of the VAR(3) with mean fitted after
     each order, which is the comparison the report asks for.

    python tools/deseason_order.py
"""
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import test_report as T                                    # noqa: E402
from drvarma import load                                   # noqa: E402
from drvarma._engine import estimate_w                     # noqa: E402
from drvarma.deseason import (deseasonalize_raw,           # noqa: E402
                              harmonic_regression_differenced)

s, spec = load(T.IPC3)
Y, F, sub = s.data, s.freq, s.start[1]
n = len(Y)
h = n // 2


def amp(y):
    """The seasonal pattern's amplitude: the norm of the harmonic coefficients
    (on the d=1 basis, as deseason estimates it)."""
    c, _f, _r2 = harmonic_regression_differenced(y, 1, F)
    return float(np.linalg.norm(c))


print("1. seasonal amplitude, second half / first half (1 = constant)")
print(f"   {'series':8}{'level ratio':>13}{'in levels':>11}{'in logs':>9}")
for j, nm in enumerate(s.names):
    y = Y[:, j]
    lv = y[h:].mean() / y[:h].mean()
    print(f"   {nm:8}{lv:13.3f}{amp(y[h:]) / amp(y[:h]):11.3f}"
          f"{amp(np.log(y[h:])) / amp(np.log(y[:h])):9.3f}")


def acf(x, k):
    x = x - x.mean()
    return float((x[:-k] * x[k:]).sum() / (x * x).sum())


def fit(w):
    r = estimate_w(w, 3, 0, include_mean=True)
    res = r["residuals"]
    return r, [[acf(res[:, j], k) for k in (12, 24)] for j in range(res.shape[1])]


# today: deseason the levels, then log, then difference (x100 as the Model)
lv, _d, _i = deseasonalize_raw(Y, s=F, start_sub=sub, mode="force")
w_lv = np.diff(100 * np.log(lv), axis=0)
# the fix: log first, deseason the logs, then difference
lg, _d, _i = deseasonalize_raw(np.log(Y), s=F, start_sub=sub, mode="force")
w_lg = np.diff(100 * lg, axis=0)
r_lv, a_lv = fit(w_lv)
r_lg, a_lg = fit(w_lg)
band = 2 / np.sqrt(len(w_lv))
print(f"\n2. VAR(3) with mean, residual ACF at 12 and 24 (band ±{band:.3f})")
print(f"   {'series':8}{'levels':>18}{'logs':>18}")
for j, nm in enumerate(s.names):
    print(f"   {nm:8}{a_lv[j][0]:+9.3f}{a_lv[j][1]:+9.3f}{a_lg[j][0]:+9.3f}{a_lg[j][1]:+9.3f}")
print(f"   logelf   {r_lv['logelf']:18.3f}{r_lg['logelf']:18.3f}")
