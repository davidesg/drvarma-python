#!/usr/bin/env python3
"""BUG-0026: the HTML forecast page multiplies the variations by 100/scale
whatever lambda: in levels it prints 100 times the change.

`report_forecast` (write_forecast_report, the SPS page per series) writes the
period and annual variations, their standard deviations, the ERR column and
the chart as 100 * delta / scale, and labels them "(%)". With lambda = 0 that is
a percentage; with lambda = 1 it is the change of the level times 100. The
page says it matches the .forecast, which since 0.3.3 (BUG-0025) uses
(100 if lambda = 0 else 1)/scale, as fue's page (fue BUG-0028) and atsw-gui's
gui/sim forecast do.

Two synthetic monthly AR(1) series about 100, in levels, scale = 100. Exits 1
while the first forecast's monthly change on the page is 100 times the change
of the level; 0 when it is the change and the labels drop the "(%)".
"""
import os
import re
import sys
import tempfile
import warnings

import numpy as np

from drvarma import Model
from drvarma import report_forecast as rf
from drvarma.datasets import simulate_varma

warnings.filterwarnings("ignore")


def main():
    sim = simulate_varma(phi=[np.diag([0.5, 0.4])], sigma=np.eye(2), n=120,
                         mu=[100.0, 50.0], seed=1, freq=12, start=(2010, 1),
                         names=["A", "B"])
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=100.0, p=1, q=0,
                include_mean=True).fit()
    arr = rf._forecast_arrays(mdl, 12)
    level = arr["level"][:, 0]
    last = float(np.asarray(sim.data)[-1, 0])
    change = level[0] - last                       # the change, in the series' units
    shown = arr["diff1"][0, 0]
    with tempfile.TemporaryDirectory() as d:
        html = open(rf.write_forecast_report(mdl, os.path.join(d, "fc"), L=12)[0],
                    encoding="utf-8").read()
    pct = len(re.findall(r"\(%\)", html))
    print(f"level {last:.4f} -> {level[0]:.4f}: change {change:.4f}; page shows "
          f"{shown:.4f}; '(%)' labels on the page: {pct}")
    ok = abs(shown - change) < 1e-6 * max(1.0, abs(change)) and pct == 0
    print("FIXED" if ok else "OPEN: in levels the page shows 100 x the change, "
          "labelled as a percentage")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
