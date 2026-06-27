"""Smoke tests for the optional matplotlib plots (Agg backend)."""
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib", reason="matplotlib not installed")
mpl.use("Agg")

from drvarma import Model, plots
from drvarma.datasets import simulate_varma


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_varma(phi=[np.array([[0.5, 0.1], [0.0, 0.4]])],
                         n=150, seed=1, names=["A", "B"])
    return Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0,
                 include_mean=True).fit(), sim


def _is_figure(obj):
    from matplotlib.figure import Figure
    return isinstance(obj, Figure)


def test_plot_series(fitted):
    _, sim = fitted
    fig = plots.plot_series(sim, title="series")
    assert _is_figure(fig) and len(fig.axes) == 2


def test_plot_forecast(fitted):
    mdl, _ = fitted
    fig = plots.plot_forecast(mdl, 12, history=40, bands=True)
    assert _is_figure(fig) and len(fig.axes) == 2


def test_plot_forecast_no_bands(fitted):
    mdl, _ = fitted
    fig = plots.plot_forecast(mdl, 6, bands=False)
    assert _is_figure(fig)


def test_plot_irf(fitted):
    mdl, _ = fitted
    fig = plots.plot_irf(mdl, 10)
    assert _is_figure(fig) and len(fig.axes) == 4    # 2x2 grid


def test_plot_fevd(fitted):
    mdl, _ = fitted
    fig = plots.plot_fevd(mdl, 10)
    assert _is_figure(fig) and len(fig.axes) == 2


def test_plot_ccf(fitted):
    mdl, _ = fitted
    res = mdl.residuals
    fig = plots.plot_ccf(res[:, 0], res[:, 1], lags=24, freq=12,
                         names=("A", "B"))
    assert _is_figure(fig) and len(fig.axes) == 1


def test_plot_residual_ccf(fitted):
    mdl, _ = fitted
    figs = plots.plot_residual_ccf(mdl)          # one figure per pair (m=2 -> 1)
    assert len(figs) == 1 and all(_is_figure(f) and len(f.axes) == 1 for f in figs)


def test_plot_before_fit_raises():
    sim = simulate_varma(phi=[np.array([[0.5, 0.0], [0.0, 0.4]])], n=50, seed=2)
    mdl = Model(sim, lam=1.0, d=0, D=0, scale=1.0, p=1, q=0, include_mean=True)
    with pytest.raises(RuntimeError):
        plots.plot_forecast(mdl, 6)
    with pytest.raises(RuntimeError):
        plots.plot_irf(mdl, 6)


def test_plot_series_into_given_axes(fitted):
    import matplotlib.pyplot as plt
    _, sim = fitted
    fig, axes = plt.subplots(2, 1)
    out = plots.plot_series(sim, axes=axes)
    assert out is fig
