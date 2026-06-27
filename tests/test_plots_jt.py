"""Tests for the Jenkins-Treadway diagnostic plots (delegated to pyfug.graphics)."""
import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib", reason="matplotlib not installed")
mpl.use("Agg")
pytest.importorskip("pyfug", reason="pyfug not installed")

from matplotlib.figure import Figure

from drvarma import Model, plots
from drvarma.datasets import simulate_varma


@pytest.fixture(scope="module")
def fitted():
    sim = simulate_varma(phi=[np.array([[0.5, 0.1], [0.0, 0.4]])],
                         n=180, seed=1, freq=12, start=(2000, 1), names=["A", "B"])
    return Model(sim, lam=1.0, d=1, D=0, scale=1.0, p=1, q=0,
                 include_mean=True).fit(), sim


def test_plot_series_jt(fitted):
    _, sim = fitted
    assert isinstance(plots.plot_series_jt(sim, 0), Figure)


def test_plot_residual_acf_pacf(fitted):
    mdl, _ = fitted
    fig = plots.plot_residual_acf_pacf(mdl, 0)
    assert isinstance(fig, Figure) and len(fig.axes) == 2


def test_plot_residual_histogram(fitted):
    mdl, _ = fitted
    assert isinstance(plots.plot_residual_histogram(mdl, 1), Figure)


def test_plot_residual_diagnostics_combined(fitted):
    mdl, _ = fitted
    fig = plots.plot_residual_diagnostics(mdl, 0)
    assert isinstance(fig, Figure) and len(fig.axes) >= 2


def test_plot_residual_diagnostics_all(fitted):
    mdl, _ = fitted
    figs = plots.plot_residual_diagnostics_all(mdl)    # one per series (m=2)
    assert len(figs) == 2 and all(isinstance(f, Figure) for f in figs)
    assert all(tuple(f.get_size_inches()) == (15.0, 5.5) for f in figs)


def test_plot_mean_deviation(fitted):
    _, sim = fitted
    assert isinstance(plots.plot_mean_deviation(sim, 0), Figure)


def test_apply_jt_theme_then_forecast(fitted):
    import matplotlib
    mdl, _ = fitted
    plots.apply_jt_theme()                       # global JT rcParams
    assert "DejaVu Sans" in matplotlib.rcParams["font.sans-serif"]
    fig = plots.plot_forecast(mdl, 6)            # drvarma plot picks up the theme
    assert isinstance(fig, Figure)
