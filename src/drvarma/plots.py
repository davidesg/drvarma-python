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


def _snap_cmax(value):
    """Snap a CCF y-limit up to a tidy 0.1 step, with a 0.3 floor (drvus style)."""
    import math
    c = math.ceil(max(value, 0.25) * 10.0) / 10.0
    return max(0.3, min(c, 1.0))


# The CCF's drawing is pyfug's (pyfug.graphics.ccf, pyfug >= 2.1.0): the suite's
# one, in the format of GraphMaker and drvus that atsw-gui's lib/ccfplot draws.
# It was here; these names stay so that nothing that used them breaks.

def ccf_default_lags(freq):
    """GraphMaker's lags for the CCF (``pyfug.graphics.ccf.ccf_default_lags``)."""
    from pyfug.graphics.ccf import ccf_default_lags as _f
    return _f(freq)


def _ccf_scale(value):
    """GraphMaker's vertical scale (``pyfug.graphics.ccf.ccf_scale``)."""
    from pyfug.graphics.ccf import ccf_scale
    return ccf_scale(value)


def plot_ccf(w1, w2, lags=None, freq=12, names=("1", "2"), ax=None):
    """The two-sided CCF of (w1, w2) in GraphMaker's format (`_draw_ccf_panel`).

    At lag k > 0, w2 leads (``diagnostics.ccf``), so the title names w2 FIRST:
    ``names`` are those of (w1, w2) and the title is "name2 - name1" —
    GraphMaker's "second series - first series", drtran's "input - output".
    Under the panel, Hosking's bivariate portmanteau as GraphMaker labels it,
    ``P ( m^2 K ) = ...`` (P, not to be confused with Ljung-Box's Q).
    """
    plt = _need_mpl()
    from .diagnostics import ccf as _ccf, qccf as _qccf
    w1 = np.asarray(w1, float).ravel()
    w2 = np.asarray(w2, float).ravel()
    n = w1.shape[0]
    if lags is None:
        lags = min(ccf_default_lags(freq), n // 4)
    rho = _ccf(w1, w2, lags)
    Q, df, _ = _qccf(w1, w2, lags)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 3.2))
    else:
        fig = ax.get_figure()
    _draw_ccf_panel(ax, rho, lags, n, freq, "%s - %s" % (names[1], names[0]),
                    "P ( %d ) = %.1f" % (df, Q))
    fig.tight_layout()
    return fig


def _draw_ccf_panel(ax, rho, lags, n, freq, title, q_label, band=None, cmax=None):
    """One two-sided CCF panel: ``pyfug.graphics.ccf.draw_ccf_panel``."""
    from pyfug.graphics.ccf import draw_ccf_panel
    draw_ccf_panel(ax, rho, lags, n, freq, title, q_label, band=band, cmax=cmax)


def plot_residual_ccf(model, lags=None, save_prefix=None, dpi=150):
    """Residual cross-correlation functions, in GraphMaker's CCF panel.

    Produces **one figure per residual pair** (i>j) — as drvus writes a separate
    ``ccf<i>_<j>.eps`` for each — with Hosking's P below, GraphMaker's d.f.
    ``4 (K - (p + q))``.
    Lags default to the drvus graphic window ``3·(freq+1)``.  ``k>0`` pairs series
    *i* leading *j* (as in the ``.out`` report).

    If ``save_prefix`` is given, writes ``<save_prefix>_ccf_<i>_<j>_<ni>_<nj>.png``
    per pair.  Returns the list of figures.
    """
    plt = _need_mpl()
    from .diagnostics import ccf as _ccf, qccf as _qccf
    res = model.result["residuals"]
    n, m = res.shape
    names = getattr(model.series, "names", None) or ["a[%d]" % (k + 1)
                                                     for k in range(m)]
    freq = getattr(model.series, "freq", 1)
    if lags is None:                               # drvus graphic window = 3·(f+1)
        lags = 3 * (freq + 1) if freq > 1 else min(3 * 3, n // 4)
        # at least 2 lags beyond p + q, so that P keeps degrees of freedom
        lags = min(max(lags, model.p + model.q + 2), n - 2)

    pairs = [(i, j) for i in range(1, m) for j in range(i)]      # (1,0),(2,0),(2,1)
    figs = []
    for (i, j) in pairs:
        fig, ax = plt.subplots(figsize=(11.0, 3.0), layout="constrained")
        # orient k>0 as i→j (i leading), matching the .out report's convention
        rho = _ccf(res[:, j], res[:, i], lags)
        Q, _df, _ = _qccf(res[:, i], res[:, j], lags)
        df = 4 * (lags - (model.p + model.q))     # GraphMaker: 4 (K - (p + q))
        _draw_ccf_panel(ax, rho, lags, n, freq, "%s - %s" % (names[i], names[j]),
                        "P ( %d ) = %.1f" % (df, Q) if df > 0 else
                        "P = %.1f  (K \u2264 p + q)" % Q)
        if save_prefix is not None:
            # the PNG, and in vectors (SVG, PDF): pyfug.graphics.ccf.save_ccf
            from pyfug.graphics.ccf import save_ccf
            save_ccf(fig, "%s_ccf_%d_%d_%s_%s.png" % (save_prefix, i + 1, j + 1,
                                                       names[i], names[j]), dpi=dpi)
        figs.append(fig)
    return figs


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


# --------------------------------------------------------------------------- #
#  Jenkins-Treadway diagnostics — reuse pyfug.graphics per series/residual     #
# --------------------------------------------------------------------------- #

def _jt_graphics():
    """Import pyfug.graphics (which also applies the JT matplotlib style)."""
    from . import _pyfug
    _pyfug.require_pyfug()
    import pyfug.graphics as G
    return G


def apply_jt_theme():
    """Apply pyfug's Jenkins-Treadway matplotlib rcParams globally.

    Call once so drvarma's own plots (forecast/IRF/FEVD/CCF) also pick up the JT
    fonts and line weights.  Requires pyfug.
    """
    _jt_graphics()
    from pyfug.graphics import base as _b
    _b._setup_matplotlib_rc()


def plot_series_jt(series, j=0, **kw):
    """Jenkins-Treadway standardized plot of column `j` (via pyfug.graphics)."""
    from . import _pyfug
    G = _jt_graphics()
    return G.plot_series(_pyfug.series_to_tseries(series, j), **kw)


def plot_residual_acf_pacf(model, j=0, npar=None, **kw):
    """JT ACF/PACF correlogram of residual series `j` (via pyfug.graphics)."""
    from . import _pyfug
    G = _jt_graphics()
    if npar is None:
        npar = model.result["npar"]
    return G.plot_acf_pacf(_pyfug.residual_to_tseries(model, j), npar=npar, **kw)


def plot_residual_histogram(model, j=0, **kw):
    """JT standardized histogram of residual series `j` (via pyfug.graphics)."""
    from . import _pyfug
    G = _jt_graphics()
    return G.plot_histogram(_pyfug.residual_to_tseries(model, j), **kw)


def plot_residual_diagnostics(model, j=0, npar=None, **kw):
    """JT combined plot (series + ACF/PACF) of residual series `j`."""
    from . import _pyfug
    G = _jt_graphics()
    if npar is None:
        npar = model.result["npar"]
    return G.plot_combined(_pyfug.residual_to_tseries(model, j), npar=npar, **kw)


def plot_residual_diagnostics_all(model, npar=None, save_prefix=None, dpi=150,
                                  **kw):
    """JT residual diagnostics (standardized series + ACF/PACF) for *every* series.

    The multivariate analogue of fue's single-series ``plot_model_diagnostics``:
    one Jenkins-Treadway combined panel per residual column, each at pyfug's
    native landscape proportions (do **not** resize the returned figures — that
    squashes the time-series panel; save them directly).

    If ``save_prefix`` is given, writes ``<save_prefix>_resid_<j>_<name>.png`` for
    each series at ``dpi`` with ``bbox_inches='tight'``.  Returns the list of
    figures (one per series).
    """
    from . import _pyfug
    G = _jt_graphics()
    if npar is None:
        npar = model.result["npar"]
    m = model.result["sigma"].shape[0]
    names = getattr(model.series, "names", None) or ["a[%d]" % (j + 1)
                                                     for j in range(m)]
    figs = []
    for j in range(m):
        fig = G.plot_combined(_pyfug.residual_to_tseries(model, j), npar=npar,
                              title="A.%s (residuals)" % names[j], **kw)
        if save_prefix is not None:
            fig.savefig("%s_resid_%d_%s.png" % (save_prefix, j + 1, names[j]),
                        dpi=dpi, bbox_inches="tight")
        figs.append(fig)
    return figs


def plot_mean_deviation(series, j=0, **kw):
    """JT mean-standard-deviation chart of column `j` (via pyfug.graphics)."""
    from . import _pyfug
    G = _jt_graphics()
    return G.plot_mean_deviation(_pyfug.series_to_tseries(series, j), **kw)
