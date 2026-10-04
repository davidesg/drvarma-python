"""BUG-0011 and BUG-0013 — drvarma's MCP v1 says its limitations (sima
supersedes this path; the decision was to document, not to rebuild)."""
import json
import warnings

import numpy as np
import pytest

S = pytest.importorskip("drvarma.mcp_server")
pytest.importorskip("art")


def _load(name, cols, names):
    S.load_data(name, values_json=json.dumps(np.column_stack(cols).tolist()),
                series_names=names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.characterize_series(name)


def test_mixed_orders_warn_of_over_differencing():
    rng = np.random.default_rng(1)
    n = 200
    rw = 100 + np.cumsum(rng.standard_normal(n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + rng.standard_normal()
    out = _load("bug11", [rw, ar + 50], "RW,AR")
    assert "BUG-0011" in out and "SOBREDIFERENCIAN AR" in out and "sima" in out


def test_same_orders_do_not_warn():
    # A draw where ART gives both walks d=1 (on some draws it calls one I(0),
    # and then the notice is right to appear).
    rng = np.random.default_rng(4)
    n = 200
    a = 100 + np.cumsum(rng.standard_normal(n))
    b = 80 + np.cumsum(rng.standard_normal(n))
    assert "BUG-0011" not in _load("bug11b", [a, b], "A,B")


def test_the_cointegration_notice_says_it_is_a_hint():
    rng = np.random.default_rng(3)
    n = 240
    x = 100 + np.cumsum(rng.standard_normal(n))
    y = 0.8 * x + 20 + rng.standard_normal(n)
    out = _load("bug13", [y, x], "Y,X")
    if "COINTEGRACIÓN" not in out:
        pytest.skip("this draw did not raise the Engle-Granger notice")
    assert "INDICIO" in out and "MacKinnon" in out and "Johansen" in out
