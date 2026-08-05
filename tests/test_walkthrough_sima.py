"""sima end to end, through the MCP surface and with no shortcuts.

This is the condition for publishing: an analyst must be able to run the whole
cycle — load, characterise, identify, estimate, diagnose, IRF/FEVD, forecast and
export — without leaving the tools. Every step checks something specific about
its output; "it did not raise" is not a pass.

WHY ONE TEST PER STEP. The sequence is stateful — each tool builds on the
session the previous one left — so it runs ONCE in a module-scoped fixture and
the results are then asserted step by step. That way a failure names the step
that broke instead of the first one, and the ones after it still report.

The data starts in subperiod 2 deliberately: that is the phase that broke
deseasonalisation, so the walk goes through the hard case rather than around it.
"""
import json

import numpy as np
import pytest

pytest.importorskip("mcp", reason="needs the MCP extra: pip install 'drvarma[mcp]'")

from drvarma import mcp_server as S  # noqa: E402


def _data(n=240, s=12, seed=17):
    """A bivariate VAR(1) with seasonality, in POSITIVE levels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = rng.normal(size=(n, 2)) @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
    w = np.zeros((n, 2))
    P = np.array([[0.55, 0.30], [-0.10, 0.45]])
    for k in range(1, n):
        w[k] = P @ w[k - 1] + e[k]
    seas = np.column_stack([0.8 * np.sin(2 * np.pi * t / s),
                            0.3 * np.cos(2 * np.pi * t / s)])
    return 100.0 * np.exp(np.cumsum(0.004 * w, 0) + 0.02 * seas)


def _cointegrated(n=240):
    """Two I(1) series sharing a common trend — the warning must fire on these."""
    tr = np.cumsum(np.random.default_rng(5).normal(0, 1, n))
    return np.column_stack([100 + tr + np.random.default_rng(6).normal(0, .4, n),
                            50 + 0.6 * tr + np.random.default_rng(7).normal(0, .4, n)])


def _check_export(o):
    d = json.loads(o)
    if "residuals" not in d or "params" not in d:
        return "residuals or params missing"
    if np.asarray(d["residuals"]).ndim != 2:
        return "the residuals are not a matrix"
    if d["termcode"] < 1:
        return "termcode=%s: the convergence diagnosis is inert again" % d["termcode"]
    if d["params"]["std_errors"] is None:
        return "no standard errors"
    return None


# (label, call, check) — check returns None to pass, or the reason it failed.
STEPS = [
    ("load_data",
     lambda: S.load_data("PT", values_json=repr(_data().tolist()), freq=12,
                         start_year=2005, start_period=2, series_names="IPC,WTI"),
     lambda o: None if "240" in o else "does not confirm the 240 observations"),

    ("series_info", lambda: S.series_info("PT"),
     lambda o: None if "IPC" in o else "does not name the series"),

    ("characterize_series", lambda: S.characterize_series("PT"),
     lambda o: None if ("Consenso" in o
                        and "Comprobación de la desestacionalización" in o)
     else "no consensus, or no deseasonalisation post-condition"),

    ("deseason post-condition does NOT report a failure",
     lambda: S.characterize_series("PT"),
     lambda o: "deseasonalisation reports itself broken" if "🛑" in o else None),

    ("cross_correlation_matrices", lambda: S.cross_correlation_matrices("PT"),
     lambda o: None if "lag 1" in o else "no lag table"),

    ("CCM states that the symbols are not a test",
     lambda: S.cross_correlation_matrices("PT"),
     lambda o: None if "NO un contraste" in o else "the Tiao-Box caveat is missing"),

    ("partial_autoregression_matrices",
     lambda: S.partial_autoregression_matrices("PT"),
     lambda o: None if "lag" in o.lower() else "no Tiao-Box output"),

    ("identify_varma_order reaches a pure VAR",
     lambda: S.identify_varma_order("PT"),
     lambda o: None if any(f"({p},0)" in o or f"({p}, 0)" in o for p in (1, 2))
     else "the ceiling still cannot reach a pure VAR"),

    ("confirm_and_estimate", lambda: S.confirm_and_estimate("PT", p=1, q=0),
     lambda o: None if "log" in o.lower() else "no likelihood"),

    ("the estimate reports its convergence",
     lambda: S.confirm_and_estimate("PT", p=1, q=0),
     lambda o: None if ("termcode" in o.lower() or "converg" in o.lower())
     else "does not say why the optimiser stopped"),

    ("diagnose", lambda: S.diagnose("PT"),
     lambda o: None if "Hosking" in o else "no Hosking"),

    ("impulse_response WITH bands", lambda: S.impulse_response("PT", horizon=8),
     lambda o: None if ("[" in o and "Bandas 95" in o) else "no bands"),

    ("variance_decomposition WITH bands",
     lambda: S.variance_decomposition("PT", horizon=12),
     lambda o: None if ("[" in o and "Bandas 95" in o) else "no bands"),

    ("generate_forecast", lambda: S.generate_forecast("PT", horizon=6),
     lambda o: None if "IC 95" in o else "no interval"),

    ("export_fit (residuals + params + termcode)",
     lambda: S.export_fit("PT"), _check_export),

    ("the cointegration warning fires on cointegrated data",
     lambda: (S.load_data("CO", values_json=repr(_cointegrated().tolist()),
                          freq=12, start_year=2000, start_period=1,
                          series_names="A,B"),
              S.characterize_series("CO"))[1],
     lambda o: None if "COINTEGRACIÓN" in o else "does not warn"),
]

LABELS = [lbl for lbl, _, _ in STEPS]


@pytest.fixture(scope="module")
def walk():
    """Run the whole sequence once; return {label: None | reason}."""
    out = {}
    for label, call, check in STEPS:
        try:
            res = call()
        except Exception as exc:                           # noqa: BLE001
            out[label] = "raised %s: %s" % (type(exc).__name__, str(exc)[:160])
            continue
        try:
            out[label] = check(res)
        except Exception as exc:                           # noqa: BLE001
            out[label] = "the check itself raised: %s" % str(exc)[:120]
    return out


@pytest.mark.parametrize("label", LABELS)
def test_step(walk, label):
    reason = walk[label]
    assert reason is None, reason
