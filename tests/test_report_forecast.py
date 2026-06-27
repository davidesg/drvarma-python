"""Tests for the HTML SPS / fuf forecast report (one per series)."""
import numpy as np
import pytest

pytest.importorskip("jinja2", reason="jinja2 not installed")
pytest.importorskip("matplotlib", reason="matplotlib not installed")

from drvarma import Model
from drvarma import report_forecast as rf
from drvarma.datasets import simulate_varma


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_varma(phi=[np.diag([0.5, 0.4])], sigma=np.eye(2),
                         n=120, mu=[100.0, 50.0], seed=1, freq=12,
                         start=(2010, 1), names=["A", "B"])
    return Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0,
                 include_mean=True).fit()


def test_write_forecast_report_one_html_per_series(fitted, tmp_path):
    paths = rf.write_forecast_report(fitted, str(tmp_path / "fc"), L=12)
    assert len(paths) == 2                                  # one per series
    for name, path in zip(("A", "B"), paths):
        assert path.endswith("_%s.html" % name)
        html = open(path, encoding="utf-8").read()
        assert "<!DOCTYPE html>" in html
        assert "Forecast — %s" % name in html               # table caption
        assert "<svg" in html                                # embedded chart
        assert "Annual rate of change" in html and "ERR" in html
        assert "drvarma" in html


def test_report_horizon_beyond_freq_has_end_row(fitted, tmp_path):
    # L>freq -> the table includes the separated H=L horizon row
    paths = rf.write_forecast_report(fitted, str(tmp_path / "fc24"), L=24)
    html = open(paths[0], encoding="utf-8").read()
    assert 'class="fore blank"' in html                      # the gap before H=L


def test_requires_fit():
    sim = simulate_varma(phi=[np.diag([0.5, 0.4])], n=60, seed=2)
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0, include_mean=True)
    with pytest.raises(RuntimeError):
        rf.write_forecast_report(mdl, "x", L=6)
