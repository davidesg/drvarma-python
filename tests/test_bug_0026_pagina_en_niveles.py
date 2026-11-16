"""BUG-0026: the HTML forecast page follows the .forecast's rule — a
percentage with logs, the change itself otherwise."""
import os
import re

import numpy as np
import pytest

pytest.importorskip("jinja2", reason="jinja2 not installed")
pytest.importorskip("matplotlib", reason="matplotlib not installed")

from drvarma import Model
from drvarma import report_forecast as rf
from drvarma.datasets import simulate_varma


def _model(lam, scale):
    sim = simulate_varma(phi=[np.diag([0.5, 0.4])], sigma=np.eye(2), n=120,
                         mu=[100.0, 50.0], seed=1, freq=12, start=(2010, 1),
                         names=["A", "B"])
    if lam == 0.0:
        sim.data = np.exp(np.asarray(sim.data) / 100.0)
    return sim, Model(sim, lam=lam, d=0, D=0, scale=scale, p=1, q=0,
                      include_mean=True).fit()


def test_in_levels_the_page_shows_the_change(tmp_path):
    sim, mdl = _model(1.0, 100.0)
    arr = rf._forecast_arrays(mdl, 12)
    change = arr["level"][0, 0] - float(np.asarray(sim.data)[-1, 0])
    assert arr["diff1"][0, 0] == pytest.approx(change, rel=1e-9)
    html = open(rf.write_forecast_report(mdl, str(tmp_path / "fc"), L=12)[0],
                encoding="utf-8").read()
    assert "(%)" not in html
    assert "Monthly change" in html and "Annual change" in html


def test_with_logs_it_is_a_percentage(tmp_path):
    sim, mdl = _model(0.0, 100.0)
    arr = rf._forecast_arrays(mdl, 12)
    last = float(np.asarray(sim.data)[-1, 0])
    pct = 100.0 * (np.log(arr["level"][0, 0]) - np.log(last))
    assert arr["diff1"][0, 0] == pytest.approx(pct, rel=1e-9)
    html = open(rf.write_forecast_report(mdl, str(tmp_path / "fc"), L=12)[0],
                encoding="utf-8").read()
    assert "Monthly (%)" in html and "Annual (%)" in html
    assert re.search(r"ERR<br>\(%\)", html)
