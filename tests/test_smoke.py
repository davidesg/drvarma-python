"""Per-wheel smoke test: does this build import and estimate at all?

Run by cibuildwheel inside every built wheel, on every platform. Deliberately
NOT the golden numerical battery: exact log-likelihoods are platform- and
BLAS-sensitive for ill-conditioned models, so pinning them across 16 build
targets would produce failures that say nothing about the wheel. What this
asks is the question a wheel can actually answer — does the extension load,
and does a fit come out finite and sane.
"""
import numpy as np

import drvarma
from drvarma import MultiSeries, Model


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.4, 0.9]]).T
    w = np.zeros((n, 2))
    P = np.array([[0.5, 0.2], [-0.1, 0.4]])
    for t in range(1, n):
        w[t] = P @ w[t - 1] + e[t]
    return MultiSeries(data=100.0 * np.exp(np.cumsum(0.005 * w, 0)),
                       names=["A", "B"], freq=12, start=(2000, 1))


def test_imports():
    assert hasattr(drvarma, "Model") and hasattr(drvarma, "MultiSeries")


def test_estimates_a_var1():
    m = Model(_series(), p=1, q=0, d=1).fit()
    r = m.result
    assert r["ifault"] == 0
    assert np.isfinite(r["logelf"])
    assert np.all(np.isfinite(np.asarray(r["phi"])))
    assert np.all(np.isfinite(np.asarray(r["sigma"])))


def test_reports_why_the_optimiser_stopped():
    """Whichever engine this wheel carries, it must say how the fit ended.

    The C engine used to return no termination code at all — `report()` writes
    the announcement to `outputv`, which the binding points at /dev/null — so
    every convergence diagnosis was silently inert on the default path.
    """
    m = Model(_series(), p=1, q=0, d=1).fit()
    assert m.termcode is not None and m.termcode >= 1
    assert m.nit is not None and m.nit > 0


def test_forecasts():
    m = Model(_series(), p=1, q=0, d=1).fit()
    lev, lo, hi = m.forecast(6, bands=True)
    assert lev.shape == (6, 2)
    assert np.all(np.isfinite(lev)) and np.all(lo <= hi)


def test_says_which_engine_it_is_using(capsys):
    """Informational: prints the engine so the CI log records what was tested."""
    try:
        from drvarma._drvarma_engine import lib  # noqa: F401
        print("drvarma smoke: COMPILED C engine")
    except ImportError:
        print("drvarma smoke: pure-Python fallback (no C extension in this wheel)")
