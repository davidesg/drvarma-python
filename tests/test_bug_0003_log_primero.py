"""BUG-0003 — the seasonal adjustment is log -> d=1 -> deseasonalization, as
art: the pattern is estimated and removed on the Box-Cox scale, and put back
there."""
import os
import warnings

import numpy as np
import pytest

from drvarma import Model, MultiSeries, transform
from drvarma.deseason import deseasonalize, reseasonalize


def _multiplicative(n=216, seed=0):
    """A trending series with a MULTIPLICATIVE seasonal pattern."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    pat = 0.03 * np.cos(2 * np.pi * t / 12) + 0.01 * np.sin(2 * np.pi * t / 6)
    lg = np.log(50) + 0.004 * t + np.cumsum(0.003 * rng.standard_normal(n)) + pat
    return np.exp(lg), pat


def test_the_pattern_is_estimated_in_logs():
    y, pat = _multiplicative()
    des, dm, _info = deseasonalize(y[:, None], 0.0, s=12, mode="force")
    # the dummies are the log pattern, month by month (sum-to-zero)
    true = np.array([pat[k::12].mean() for k in range(12)])
    assert np.max(np.abs(dm[0] - (true - true.mean()))) < 2e-3
    # and the series comes back in levels, without it
    assert np.allclose(np.log(des[:, 0]) + dm[0][np.arange(len(y)) % 12],
                       np.log(y), atol=1e-12)


def test_reseasonalize_undoes_it_on_the_same_scale():
    y, _ = _multiplicative()
    des, dm, _ = deseasonalize(y[:, None], 0.0, s=12, mode="force")
    back = reseasonalize(des[:, 0], dm[0][np.arange(len(y)) % 12], 0.0)
    assert np.allclose(back, y, rtol=1e-12)


def test_the_dummies_are_arts():
    pytest.importorskip("art")
    import fue
    from art import seasonal_detection as sd
    y, _ = _multiplicative()
    _, dm, _ = deseasonalize(y[:, None], 0.0, s=12, mode="force")
    ts = fue.TimeSeries.from_array(y.tolist(), freq=12, start=[2000, 1], name="Y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sd.detect_seasonality(ts, d=1, lam=0.0)
    # art works on 100*log
    assert np.max(np.abs(np.asarray(r.dummies, float) - 100 * dm[0])) < 1e-9


def test_the_annual_rate_is_free_of_the_dummies():
    """In logs the dummy of a month cancels against the same month a year
    before: the forecast's annual rate (TLVA) is exactly the one of the
    re-seasonalized levels."""
    pytest.importorskip("drvarma._drvarma_engine")
    from drvarma import report_forecast as RF
    y1, _ = _multiplicative(seed=1)
    y2, _ = _multiplicative(seed=2)
    ms = MultiSeries(np.column_stack([y1, y2]), freq=12, start=(2000, 1),
                     names=["A", "B"])
    mdl = Model(ms, lam=0.0, d=1, D=0, p=1, q=0, include_mean=True,
                deseason="force").fit()
    L = 18
    q = RF._forecast_arrays(mdl, L)
    lev = q["level"]
    full = np.vstack([ms.data, lev])
    n = ms.nobs
    tlva = 100 * np.log(full[n:] / full[n - 12:n - 12 + L])
    assert np.allclose(q["annual"], tlva, atol=1e-8)


def test_auto_decides_with_arts_identification_test():
    """drvarma's "auto" and art's identification share the test: the HAC F
    (art BUG-0206)."""
    pytest.importorskip("art")
    import fue
    from art import seasonal_detection as sd
    for seed in (4, 5):
        y, _ = _multiplicative(seed=seed)
        _, _, info = deseasonalize(y[:, None], 0.0, s=12, mode="auto")
        ts = fue.TimeSeries.from_array(y.tolist(), freq=12, start=[2000, 1], name="Y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sd.detect_seasonality(ts, d=1, lam=0.0)
        assert info[0]["f_stat"] == pytest.approx(r.f_stat, rel=1e-9)
        assert info[0]["p_value"] == pytest.approx(r.p_value, abs=1e-12)
        assert info[0]["seasonal"] == r.seasonal_detected

