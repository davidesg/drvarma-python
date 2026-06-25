"""Optional matplotlib plots for drvarma (series, forecasts, IRF, FEVD).

matplotlib is an optional dependency (``pip install "drvarma[plots]"``); it is
imported lazily inside each function, so importing this module never requires it.
Every function accepts an existing axis/axes (or creates a figure) and returns the
matplotlib ``Figure``.
"""

import numpy as np


def _need_mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError('matplotlib is required for drvarma.plots; install it '
                          'with: pip install "drvarma[plots]"') from exc
    return plt


def _dates(start, freq, n, offset=0):
    """Fractional-year x-axis for `n` observations from `start=(year, sub)`."""
    year, sub = start
    idx = np.arange(offset, offset + n)
    return year + (sub - 1 + idx) / float(freq)


def plot_series(series, axes=None, title=None):
    """Plot each column of a MultiSeries on its own stacked subplot."""
    plt = _need_mpl()
    m = series.m
    x = _dates(series.start, series.freq, series.nobs)
    if axes is None:
        fig, axes = plt.subplots(m, 1, figsize=(10, 2.2 * m), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()
    for j in range(m):
        axes[j].plot(x, series.data[:, j], color="k", lw=0.9)
        axes[j].set_ylabel(series.names[j], fontsize=9)
        axes[j].tick_params(direction="out", labelsize=8)
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_forecast(model, L, axes=None, history=None, bands=True, b=0):
    """Plot history + L-step forecast (with 95% bands) per series.

    `history` limits how many in-sample points are drawn (default: all).
    """
    plt = _need_mpl()
    if model.result is None:
        raise RuntimeError("call fit() before plot_forecast()")
    s = model.series
    m = s.m
    if bands:
        fc, lo, hi = model.forecast(L, b=b, bands=True)
    else:
        fc = model.forecast(L, b=b); lo = hi = None

    nobs = s.nobs
    h = nobs if history is None else min(history, nobs)
    x_hist = _dates(s.start, s.freq, h, offset=nobs - h)
    x_fc = _dates(s.start, s.freq, L, offset=nobs - b)

    if axes is None:
        fig, axes = plt.subplots(m, 1, figsize=(10, 2.4 * m), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()
    for j in range(m):
        ax = axes[j]
        ax.plot(x_hist, s.data[nobs - h:, j], color="k", lw=0.9, label="observed")
        ax.plot(x_fc, fc[:, j], color="C0", lw=1.3, label="forecast")
        if bands:
            ax.fill_between(x_fc, lo[:, j], hi[:, j], color="C0", alpha=0.2,
                            label="95%")
        ax.axvline(x_fc[0] - 0.5 / s.freq, color="0.6", lw=0.8, ls="--")
        ax.set_ylabel(s.names[j], fontsize=9)
        ax.tick_params(direction="out", labelsize=8)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("VARMA(%d,%d) forecast" % (model.p, model.q),
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_irf(model, horizon, orthogonalized=True, axes=None):
    """Grid (m x m) of impulse responses: response of variable i to shock j."""
    plt = _need_mpl()
    if model.result is None:
        raise RuntimeError("call fit() before plot_irf()")
    names = model.series.names
    m = model.series.m
    irf = model.irf(horizon, orthogonalized=orthogonalized)   # (H+1, m, m)
    hx = np.arange(horizon + 1)
    if axes is None:
        fig, axes = plt.subplots(m, m, figsize=(2.6 * m, 2.2 * m),
                                 sharex=True, squeeze=False)
    else:
        axes = np.atleast_2d(axes)
        fig = axes[0, 0].get_figure()
    for i in range(m):
        for j in range(m):
            ax = axes[i, j]
            ax.axhline(0, color="0.7", lw=0.7)
            ax.plot(hx, irf[:, i, j], color="C0", lw=1.2)
            ax.tick_params(direction="out", labelsize=7)
            if i == 0:
                ax.set_title("shock %s" % names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)
    kind = "Orthogonalized IRF" if orthogonalized else "IRF (psi weights)"
    fig.suptitle(kind, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_fevd(model, horizon, axes=None):
    """Stacked-area forecast-error variance decomposition, one panel per variable."""
    plt = _need_mpl()
    if model.result is None:
        raise RuntimeError("call fit() before plot_fevd()")
    names = model.series.names
    m = model.series.m
    fevd = model.fevd(horizon)                       # (H, m, m) percentages
    hx = np.arange(1, horizon + 1)
    if axes is None:
        fig, axes = plt.subplots(m, 1, figsize=(9, 2.2 * m), sharex=True,
                                 squeeze=False)
        axes = axes[:, 0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].get_figure()
    for i in range(m):
        ax = axes[i]
        shares = [fevd[:, i, j] for j in range(m)]
        ax.stackplot(hx, *shares, labels=["shock %s" % n for n in names],
                     alpha=0.85)
        ax.set_ylim(0, 100)
        ax.set_ylabel(names[i], fontsize=9)
        ax.tick_params(direction="out", labelsize=8)
    axes[0].legend(fontsize=7, loc="upper right", ncol=m)
    fig.suptitle("Forecast error variance decomposition (%)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig
