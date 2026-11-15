#!/usr/bin/env python3
"""BUG-0025: the ladder's forecast table multiplies the variations by
100/refactor whatever lambda: in levels it prints 100 times the change.

`ladder._forecast_lines` (the .forecast and the forecast block of the .out,
what sima's forecast shows) writes mon%/ann% and their stds as
100 * delta / refactor. With lambda = 0 that is a percentage; with lambda = 1 it
is the change of the level times 100. The C does the same (escalera.c, the
forecast table; drvarma.c, -forecast); its own accuracy table, and fuf, drtran
and sima's forecast_uncertainty, use (100 if lambda = 0 else 1)/refactor.

Two synthetic monthly series about 100 in levels, each an AR(1) with mu fitted
by fue with lambda = 1 and refactor = 100; the ladder VARMA(0,0) on them and its
forecast. Exits 1 while the first forecast's mon% is 100 times the change of
the level; 0 when it is the change.
"""
import os
import sys
import tempfile
import warnings

import numpy as np

import fue
from drvarma.ladder import Ladder, write_forecast

warnings.filterwarnings("ignore")


def main():
    rng = np.random.default_rng(11)
    n = 160
    paths = []
    with tempfile.TemporaryDirectory() as d:
        ys = []
        for nm in ("Y", "X"):
            y = np.zeros(n)
            y[0] = 100.0
            for t in range(1, n):
                y[t] = 100.0 + 0.6 * (y[t - 1] - 100.0) + rng.normal(0, 1.0)
            ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name=nm)
            m = fue.Model(ts, d=0, D=0, boxlam=1.0, ar=[[0.1]], ar_free=[[True]],
                          mu=100.0 * float(y.mean()), estimate_mu=True, refactor=100.0)
            m.fit()
            p = os.path.join(d, f"{nm}.pre")
            m.write_pre(p)
            paths.append(p)
            ys.append(y)
        L = Ladder(paths, 0, 0)
        L.fit()
        fc = L.forecast(12)
        out = os.path.join(d, "t.forecast")
        write_forecast(L, fc, out)
        text = open(out).read().splitlines()
    i = next(k for k, l in enumerate(text) if l.startswith("Series Y"))
    row = text[i + 2].split()
    level, mon = float(row[1]), float(row[4])
    change = level - ys[0][-1]
    print(f"Y: last observation {ys[0][-1]:.4f}, first forecast {level:.4f}: change {change:+.4f}")
    print(f"the table's mon%: {mon:+.4f}")
    abierto = abs(mon) > 10 * abs(change) + 1e-3
    print("x100 in levels" if abierto else "the change of the level")
    return 1 if abierto else 0


if __name__ == "__main__":
    sys.exit(main())
