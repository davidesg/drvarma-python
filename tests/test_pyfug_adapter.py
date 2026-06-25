"""Tests for the MultiSeries -> pyfug.core.Tseries adapter."""
import io

import numpy as np
import pytest

pytest.importorskip("pyfug", reason="pyfug not installed")

from drvarma import Model, diagnostics
from drvarma import _pyfug
from drvarma.datasets import simulate_varma


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_varma(phi=[np.array([[0.5, 0.1], [0.0, 0.4]])],
                         n=180, seed=1, freq=12, start=(2000, 3),
                         names=["A", "B"])
    return Model(sim, lam=1.0, d=1, D=0, scale=1.0, p=1, q=0,
                 include_mean=True).fit(), sim


def test_to_tseries_uses_drvarma_stats(fitted):
    _, sim = fitted
    x = sim.column(0)
    ts = _pyfug.series_to_tseries(sim, 0)
    st = diagnostics.series_stats(x)
    assert ts.name == "A" and ts.nobs == sim.nobs and ts.freq == 12
    assert ts.begyear == 2000 and ts.begtime == 3
    assert ts.mean == st["mean"]
    assert ts.var == st["variance"]
    assert ts.skew == st["skew"] and ts.kurt == st["kurt"]
    assert ts.max_idx == st["max_idx"] and ts.min_idx == st["min_idx"]


def test_residual_start_advances_by_differencing(fitted):
    mdl, sim = fitted
    # d=1: residual series starts one period after the raw start (3/2000 -> 4/2000)
    yr, sub = _pyfug.residual_start(mdl)
    assert (yr, sub) == (2000, 4)


def test_residual_to_tseries_matches_residuals(fitted):
    mdl, _ = fitted
    ts = _pyfug.residual_to_tseries(mdl, 1)
    assert ts.nobs == mdl.residuals.shape[0]
    assert np.allclose(ts.data, mdl.residuals[:, 1])
    assert ts.name == "B" and ts.boxlam == 1.0 and ts.d == 1


def test_pyfug_renders_from_adapter(fitted):
    # the adapted Tseries drives pyfug's ASCII renderer without error
    import pyfug.ascii as A
    mdl, _ = fitted
    ts = _pyfug.residual_to_tseries(mdl, 0)
    f = io.StringIO()
    A._write_ascii_plot(f, ts)
    out = f.getvalue()
    assert "Standardized time series plot" in out
    assert out.count("\n") > ts.nobs        # one row per observation + header


def test_require_pyfug_returns_module():
    assert _pyfug.have_pyfug() is True
    assert _pyfug.require_pyfug() is not None
