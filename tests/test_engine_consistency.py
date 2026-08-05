"""The two engines must say the SAME thing about the same fit.

drvarma ships two estimators — the compiled C engine (the default) and the pure
Python one (the fallback). They were not equally talkative: `raxopt` announces
the termination code through `report()`, which writes to `outputv`, and
`quiet_mode` exists to silence that trace for batch simulations. The binding
points `outputv` at /dev/null, so the announcement was computed and discarded:
the C engine returned no `termcode` and no `nit` at all.

The consequence was not cosmetic. `Model.termcode` was `None` on the DEFAULT
path, so `report._convergence_block` — which exists precisely because "WHY it
stopped matters as much as WHETHER it stopped" — and the MCP's convergence
diagnosis were both inert in production, while working perfectly in the
fallback nobody runs. A diagnosis that silently does not fire is worse than no
diagnosis, because its silence reads as an all-clear.

These tests pin the contract: same keys, same verdict, either engine.
"""
import importlib
import os

import numpy as np
import pytest

from drvarma import _engine
from drvarma.estimate_py import estimate_w_py


@pytest.fixture(autouse=True)
def _force_the_c_engine(monkeypatch):
    """Make sure this file really compares the TWO engines.

    `tests/test_regression_bugs.py` does `os.environ.setdefault(
    "DRVARMA_NO_ENGINE", "1")` at MODULE level, and pytest imports every test
    module during collection — before running anything. So in a full-battery run
    the variable is set for the whole session and `_engine.estimate_w` quietly
    returns the pure-Python estimator. Without this fixture these tests then
    compare pure-Python against pure-Python, agree trivially, and pass while
    testing nothing: measured, the divergence cases went from `xfail` to `pass`
    in the full run and stayed `xfail` in isolation. A test that silently turns
    vacuous is worse than one that fails.
    """
    monkeypatch.delenv("DRVARMA_NO_ENGINE", raising=False)
    importlib.reload(_engine)
    yield
    importlib.reload(_engine)


def _really_the_c_engine():
    """The C engine is importable AND `_engine` will actually reach for it."""
    if os.environ.get("DRVARMA_NO_ENGINE"):
        return False
    try:
        from drvarma._drvarma_engine import lib  # noqa: F401
        return True
    except ImportError:
        return False


needs_engine = pytest.mark.skipif(
    not _really_the_c_engine(),
    reason="the compiled C engine is not built")


@pytest.fixture(scope="module")
def w():
    """A stationary VAR(1) with correlated innovations."""
    rng = np.random.default_rng(1)
    n = 200
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
    out = np.zeros((n, 2))
    P = np.array([[0.6, 0.25], [-0.1, 0.4]])
    for t in range(1, n):
        out[t] = P @ out[t - 1] + e[t]
    return out


@needs_engine
@pytest.mark.parametrize("p,q", [(1, 0), (2, 0), (1, 1)])
def test_both_engines_walk_the_same_path_when_the_model_is_identified(w, p, q):
    """Same termcode, same ITERATION COUNT, same likelihood.

    The iteration count agreeing is the strong part: it means the two are
    walking the identical quasi-Newton trajectory, which is the premise of the
    port — it is what makes the pure-Python standard errors comparable with the
    C's. Measured on this data the parameters agree to 1e-11..1e-8.
    """
    c = _engine.estimate_w(w, p, q)
    py = estimate_w_py(w, p, q)
    assert c["termcode"] == py["termcode"]
    assert c["nit"] == py["nit"]
    assert c["logelf"] == pytest.approx(py["logelf"], rel=1e-9)


@needs_engine
@pytest.mark.parametrize("p,q", [(2, 1), (2, 2)])
def test_the_engines_may_diverge_when_the_model_is_NOT_identified(w, p, q):
    """A KNOWN and understood divergence — recorded, not asserted away.

    The data is a VAR(1). Fitting VARMA(2,1) or VARMA(2,2) puts AR and MA
    factors in that cancel, so the likelihood is flat in those directions, and
    in a flat basin WHERE YOU STOP DEPENDS ON THE PATH. A rounding difference
    between the compiled and the interpreted arithmetic is then enough to send
    the two searches to different points. Measured here:

        (2,1)  termcode 3 both, 78 vs 88 iterations, logL differs 2.5e-03
        (2,2)  termcode 1 vs 3,  187 vs 48 iterations, logL differs 1.84

    This is the same phenomenon as the `refactor=1` study
    (`drtran/docs/OPTIMIZER_STOPPING_STUDY.md`) seen from another angle, and it
    is a reason to distrust the FIT, not the engines: two implementations of the
    same algorithm disagreeing is a symptom that the model is not identified.

    So this test asserts only what is actually guaranteed — that both agree the
    data has been fitted at all — and documents the rest.
    """
    assert not os.environ.get("DRVARMA_NO_ENGINE"), \
        "the fixture did not clear the opt-out; this would compare py with py"
    c = _engine.estimate_w(w, p, q)
    py = estimate_w_py(w, p, q)
    assert c["ifault"] == 0 and py["ifault"] == 0
    assert c["nit"] > 0 and py["nit"] > 0
    if c["termcode"] != py["termcode"] or c["nit"] != py["nit"]:
        pytest.xfail(f"engines diverge on the unidentified ({p},{q}): "
                     f"termcode {c['termcode']}/{py['termcode']}, "
                     f"nit {c['nit']}/{py['nit']} — see the docstring")


@needs_engine
def test_the_c_engine_reports_a_termination_code_at_all(w):
    """The regression itself: the C engine used to omit these keys entirely."""
    c = _engine.estimate_w(w, 1, 0)
    assert "termcode" in c and "nit" in c
    assert c["termcode"] in (1, 2, 3, 4, 5), \
        "0 means the optimiser never ran; on a real fit that is a bug"
    assert c["nit"] > 0


@needs_engine
def test_model_exposes_convergence_on_the_default_engine():
    """`Model.termcode` was None on the C path — the diagnosis was inert."""
    from drvarma import Model, MultiSeries
    rng = np.random.default_rng(7)
    n = 250
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
    x = np.zeros((n, 2))
    P = np.array([[0.6, 0.25], [-0.1, 0.4]])
    for t in range(1, n):
        x[t] = P @ x[t - 1] + e[t]
    ms = MultiSeries(data=100 * np.exp(np.cumsum(0.004 * x, 0)),
                     names=["A", "B"], freq=12, start=(2000, 1))
    m = Model(ms, p=1, q=0, d=1).fit()
    assert m.termcode is not None and m.termcode >= 1
    assert m.nit is not None and m.nit > 0
    assert m.converged is True


def test_prepare_refuses_a_non_finite_transform():
    """A log of non-positive data gave Sigma = NaN with ifault = 0 — a fit that
    declares itself healthy while every number in it is NaN. The guard lived
    only in the MCP helper, so `Model.fit()` did not have it."""
    from drvarma import Model, MultiSeries
    rng = np.random.default_rng(3)
    lev = np.column_stack([np.cumsum(rng.normal(0, 1, 150)),      # crosses zero
                           100 + np.arange(150) * 0.1])
    ms = MultiSeries(data=lev, names=["A", "B"], freq=12, start=(2000, 1))
    with pytest.raises(ValueError, match="not finite"):
        Model(ms, p=1, q=0, d=1, lam=0.0).fit()
