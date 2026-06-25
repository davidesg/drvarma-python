"""Adapter from drvarma's multivariate objects to pyfug's univariate Tseries.

pyfug (the FUG Python migration) renders the Jenkins-Treadway ASCII diagnostics
and plots for a single series.  drvarma reuses those renderers per residual/series
column through this adapter, **populating the statistics from drvarma's own**
``diagnostics`` (drvarma owns the numbers; pyfug only renders).  See
``docs/FUE_REUSE.md``.

pyfug is an optional dependency (``pip install "drvarma[plots]"``); import it
lazily and guard with :func:`require_pyfug`.
"""

import numpy as np

from . import diagnostics


def have_pyfug():
    """True if pyfug is importable."""
    try:
        import pyfug.core  # noqa: F401
        return True
    except ImportError:
        return False


def require_pyfug():
    """Import and return the pyfug package, or raise an informative error."""
    try:
        import pyfug
        return pyfug
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "pyfug is required for the Jenkins-Treadway diagnostics/plots; "
            'install it with: pip install "drvarma[plots]"') from exc


def _jarque_bera(stats):
    """Univariate Jarque-Bera from drvarma stats: n(skew^2/6 + kurt^2/24)."""
    n = stats["n"]
    return float(n * (stats["skew"] ** 2 / 6.0 + stats["kurt"] ** 2 / 24.0))


def to_tseries(data, name, freq, start, *, d=0, ds=0, boxlam=1.0, lags=0):
    """Build a ``pyfug.core.Tseries`` from a 1-D series.

    Statistics (mean/var/skew/kurt/jarquebera/min_idx/max_idx) are filled from
    drvarma's :func:`diagnostics.series_stats`, so pyfug's renderers use drvarma's
    numbers rather than recomputing.

    Parameters
    ----------
    data : 1-D array
    name : series label
    freq : observations per year
    start : (year, period) of the first observation of THIS series
    d, ds, boxlam : differencing / Box-Cox metadata (for plot titles)
    lags : ACF/PACF lags (0 = let the renderer auto-detect)
    """
    require_pyfug()
    from pyfug.core import Tseries

    data = np.ascontiguousarray(np.asarray(data, float).ravel())
    st = diagnostics.series_stats(data)
    return Tseries(
        name=str(name), nobs=data.shape[0], freq=int(freq),
        begyear=int(start[0]), begtime=int(start[1]),
        data=data, d=int(d), ds=int(ds), boxlam=float(boxlam), lags=int(lags),
        mean=st["mean"], var=st["variance"], skew=st["skew"], kurt=st["kurt"],
        jarquebera=_jarque_bera(st), max_idx=st["max_idx"], min_idx=st["min_idx"],
    )


def residual_start(model):
    """(year, period) of the residual series = original start advanced by d+D·s.

    The residuals belong to the differenced/stationary series, whose first value
    is raw observation ``d + D*freq + 1``.
    """
    from .report import obs_to_date
    s = model.series
    off = model.d + model.D * s.freq
    year, sub = obs_to_date(s.start[0], s.start[1], off + 1, s.freq)
    return (year, sub)


def series_to_tseries(series, j, *, lags=0):
    """Adapt column j of a MultiSeries to a pyfug Tseries (raw level series)."""
    return to_tseries(series.column(j), series.names[j], series.freq,
                      series.start, lags=lags)


def residual_to_tseries(model, j, *, lags=0):
    """Adapt residual series j of a fitted Model to a pyfug Tseries."""
    if model.result is None:
        raise RuntimeError("call fit() before residual_to_tseries()")
    res = model.residuals[:, j]
    name = model.series.names[j]
    return to_tseries(res, name, model.series.freq, residual_start(model),
                      d=model.d, ds=model.D, boxlam=model.lam, lags=lags)
