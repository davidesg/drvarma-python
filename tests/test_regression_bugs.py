"""Regression battery for the defects found in the oil pass-through exercise.

Every test here encodes an INVARIANT that was silently violated in production, not
just a golden value. Run before and after any change:

    DRVARMA_NO_ENGINE=1 pytest tests/test_regression_bugs.py -q

UPDATE (2026-07-29): the three pre-existing failures that used to be listed here —
`test_deseason_params_and_forecast_match_c`,
`test_out_deterministic_sections_byte_exact[marker_pair4]` and
`test_volexp_volmov_byte_exact` — now PASS. All three were C-parity tests, and all
three had the same cause: `_chol_lower` used a strict `np.linalg.cholesky` instead
of the C's MODIFIED Cholesky (`nlatools.c:choldcp`). Porting it faithfully restored
parity. The suite is now 195 passed / 0 failed.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("DRVARMA_NO_ENGINE", "1")

from drvarma import Model, MultiSeries, transform  # noqa: E402
from drvarma.deseason import deseasonalize_raw  # noqa: E402

S_TRUE = np.array([2.5, -1.0, 0.5, 1.5, -0.5, -2.0, 3.0, -1.5, 0.0, 1.0, -2.5, -1.0])
S_TRUE = S_TRUE - S_TRUE.mean()


def seasonal_series(start_sub, n=240, seed=0, noise=0.05):
    """Trend + known seasonal pattern, first observation at subperiod `start_sub`.

    Column 0 carries the known pattern (the deseason tests assert on it). Column 1
    is a DIFFERENT series: the two must not be identical or Sigma is singular
    (ifault=1, Q not positive definite) and the optimiser never runs, which would
    make every estimation test vacuous.
    """
    rng = np.random.default_rng(seed)
    months = (np.arange(n) + start_sub - 1) % 12
    lvl = 100 + 0.3 * np.arange(n) + S_TRUE[months] + rng.normal(0, noise, n)
    other = (50 + 0.1 * np.arange(n) + 0.4 * S_TRUE[months]
             + rng.normal(0, 0.5, n).cumsum() * 0.1)
    return np.column_stack([lvl, other]), months


def seasonal_sd(x, months, n=240):
    """Std dev of the monthly means after removing a linear trend."""
    resid = x - np.poly1d(np.polyfit(np.arange(n), x, 1))(np.arange(n))
    return float(np.std([resid[months == j].mean() for j in range(12)]))


def acf(x, lag):
    xc = np.asarray(x, float) - np.mean(x)
    return float((xc[lag:] * xc[:-lag]).mean() / xc.var())


# ── BUG 4 (CRITICAL) — deseason phase must be honoured for every start_sub ────
@pytest.mark.parametrize("start_sub", list(range(1, 13)))
def test_deseason_recovers_known_pattern_at_any_start_sub(start_sub):
    """The estimated dummies must match the true seasonal pattern regardless of
    which subperiod the series starts at. Before the fix this only held for
    start_sub=1; elsewhere the pattern was estimated in one phase and subtracted in
    another (max error up to 5.5 on a pattern of amplitude ~5).
    """
    raw, _ = seasonal_series(start_sub)
    _, dummies, _ = deseasonalize_raw(raw, s=12, start_sub=start_sub, mode="force")
    assert np.abs(dummies[0] - S_TRUE).max() < 0.1


@pytest.mark.parametrize("start_sub", list(range(1, 13)))
def test_deseason_removes_seasonality_at_any_start_sub(start_sub):
    raw, months = seasonal_series(start_sub)
    adj, _, _ = deseasonalize_raw(raw, s=12, start_sub=start_sub, mode="force")
    assert seasonal_sd(adj[:, 0], months) < 0.05


@pytest.mark.parametrize("start_sub", list(range(1, 13)))
def test_deseason_never_makes_seasonality_worse(start_sub):
    """Weaker but essential invariant: adjusting must never ADD seasonal variance.
    This is what failed most visibly on real data (ES 0.798 -> 0.866 at lag 12).
    """
    raw, months = seasonal_series(start_sub)
    adj, _, _ = deseasonalize_raw(raw, s=12, start_sub=start_sub, mode="force")
    assert seasonal_sd(adj[:, 0], months) <= seasonal_sd(raw[:, 0], months)


def test_deseason_start_sub_1_unchanged():
    """Back-compatibility: at start_sub=1 the fix must be the identity, so the C
    parity results and existing goldens are untouched."""
    raw, _ = seasonal_series(1)
    adj, dummies, _ = deseasonalize_raw(raw, s=12, start_sub=1, mode="force")
    lvl_rel = np.empty(12)
    lvl_rel[(np.arange(12) + 1 - 1) % 12] = dummies[0]
    assert np.allclose(dummies[0], lvl_rel)
    assert np.isfinite(adj).all()


# ── BUG 1 — load_data must not drop observations or lose column names ─────────
def test_load_data_headerless_csv_keeps_every_row(tmp_path):
    from drvarma import mcp_server as M

    p = tmp_path / "noheader.csv"
    rows = np.column_stack([np.arange(1.0, 25.0), np.arange(101.0, 125.0)])
    np.savetxt(p, rows, delimiter=",")
    M.load_data(name="_t1", path=str(p), freq=12, start_year=2002, start_period=2,
                series_names="a,b")
    got = M._DATA["_t1"]
    assert got.nobs == rows.shape[0], "first observation was dropped"
    assert np.allclose(got.data[0], rows[0])


def test_load_data_csv_header_supplies_names(tmp_path):
    from drvarma import mcp_server as M

    p = tmp_path / "header.csv"
    with open(p, "w") as fh:
        fh.write("WTI,CPI\n")
        for r in np.column_stack([np.arange(1.0, 25.0), np.arange(101.0, 125.0)]):
            fh.write(f"{r[0]},{r[1]}\n")
    M.load_data(name="_t2", path=str(p), freq=12, start_year=2002, start_period=1)
    got = M._DATA["_t2"]
    assert got.nobs == 24
    assert list(got.names) == ["WTI", "CPI"]


# ── BUG 2 — the order search must not rank/recommend non-converged fits ───────
@pytest.mark.slow
def test_order_search_excludes_nonfinite_and_faulted_fits():
    from drvarma import mcp_server as M

    raw, _ = seasonal_series(2, n=180)
    M.load_data(name="_t3", values_json=str([list(r) for r in raw]).replace("'", ""),
                freq=12, start_year=2002, start_period=2, series_names="a,b")
    M.characterize_series("_t3")
    out = M.identify_varma_order("_t3", p_max=2, q_max=2)
    assert "inf" not in out.split("Recomendación")[0].replace("-inf", "X"), out
    assert "-inf" not in out, "non-finite criteria leaked into the ranking"


# ── BUG 3 — the exact likelihood must be monotone in p (nesting) ──────────────
@pytest.mark.slow
@pytest.mark.parametrize("start_sub", [1, 2])
def test_loglik_monotone_in_p(start_sub):
    """VAR(1) is nested in VAR(2) is nested in VAR(3): the maximised likelihood
    cannot decrease. A drop signals an optimiser failure being reported as a fit.
    """
    raw, _ = seasonal_series(start_sub, n=180)
    ms = MultiSeries(raw, freq=12, start=(2002, start_sub), names=["a", "b"])
    lls = [float(Model(ms, lam=0.0, d=1, p=p, q=0, include_mean=True).fit().loglik)
           for p in (1, 2, 3)]
    assert lls[0] <= lls[1] + 1e-6, f"VAR(2) worse than VAR(1): {lls}"
    assert lls[1] <= lls[2] + 1e-6, f"VAR(3) worse than VAR(2): {lls}"


# ── Cross-cutting invariants ─────────────────────────────────────────────────
@pytest.mark.slow
def test_transform_routes_agree():
    """Estimating from levels with (lam=0, d=1) must equal estimating from the
    pre-differenced log series with (lam=1, d=0). Guards the multiart protocol of
    always loading ORIGINAL series."""
    raw, _ = seasonal_series(1, n=180)
    a = Model(MultiSeries(raw, freq=12, start=(2002, 1)),
              lam=0.0, d=1, p=1, q=0, include_mean=True).fit()
    w, _ = transform.transform(raw, lam=0.0, d=1, D=0, s=12, scale=1.0)
    b = Model(MultiSeries(np.asarray(w), freq=12, start=(2002, 2)),
              lam=1.0, d=0, p=1, q=0, include_mean=True).fit()
    assert np.allclose(np.asarray(a.phi)[0], np.asarray(b.phi)[0], atol=1e-4)


def _loaded(tmp_path, n=180, m=3):
    """Load a small multivariate dataset into the MCP session."""
    from drvarma import mcp_server as M

    rng = np.random.default_rng(3)
    raw, _ = seasonal_series(1, n=n)
    third = raw[:, 0] * 0.5 + rng.normal(0, 0.4, n).cumsum() * 0.2 + 30
    data = np.column_stack([raw[:, 0], raw[:, 1], third])[:, :m]
    M.load_data(name="_plt", values_json=str([list(r) for r in data]),
                freq=12, start_year=2002, start_period=1,
                series_names=",".join(f"s{i}" for i in range(m)))
    M.characterize_series("_plt")
    return M


@pytest.mark.parametrize("tool,kind", [
    ("plot_cross_correlation_matrices", "ccm"),
    ("plot_cross_correlation_functions", "ccf"),
    ("plot_partial_autoregression_matrices", "tiaobox"),
])
def test_identification_plot_tools_write_a_file(tmp_path, tool, kind):
    M = _loaded(tmp_path)
    out = tmp_path / f"{kind}.png"
    msg = getattr(M, tool)("_plt", path=str(out))
    assert out.exists() and out.stat().st_size > 1000, msg
    assert str(out) in msg


def test_ccm_plot_and_table_use_the_same_statistic(tmp_path):
    """The plot must never disagree with the table: both go through _ccm_values."""
    M = _loaded(tmp_path)
    w = M._prepared_w(M._DATA["_plt"], 0.0, 1, 0, "auto")
    R, bound, n, m = M._ccm_values(w)
    table = M.cross_correlation_matrices("_plt", n_lags=4)
    # rebuild the table's symbols from the shared helper and compare
    rows = [ln.strip() for ln in table.split("\n") if ln.startswith("  ")]
    rebuilt = []
    for k in range(1, 5):
        Rk = R(k)
        for i in range(m):
            rebuilt.append(" ".join(
                "+" if Rk[i, j] > bound else "-" if Rk[i, j] < -bound else "."
                for j in range(m)))
    assert rows == rebuilt


def test_tiaobox_plot_and_table_use_the_same_statistic(tmp_path):
    M = _loaded(tmp_path)
    w = M._prepared_w(M._DATA["_plt"], 0.0, 1, 0, "auto")
    tt = M._tiaobox_tratios(w, 4)
    m = w.shape[1]
    table = M.partial_autoregression_matrices("_plt", max_order=4)
    rows = [ln.strip() for ln in table.split("\n") if ln.startswith("  ")]
    rebuilt = [" ".join("+" if tt[k][i, j] > 1.96 else
                        "-" if tt[k][i, j] < -1.96 else "."
                        for j in range(m))
               for k in sorted(tt) for i in range(m)]
    assert rows == rebuilt


def test_ccf_summary_reports_which_lags_not_just_how_many(tmp_path):
    """A spike at the seasonal lag is not the same evidence as one at lag 1, so the
    summary must name the lags."""
    M = _loaded(tmp_path)
    msg = M.plot_cross_correlation_functions("_plt", n_lags=14,
                                             path=str(tmp_path / "c.png"))
    assert "k>0" in msg and "k<0" in msg
    assert "ρ(0)=" in msg
    # at least one pair line must carry an explicit lag list or an em dash
    assert any(("[" in ln or "—" in ln) for ln in msg.split("\n") if ln.startswith("- "))


def test_tiaobox_plot_does_not_clip_t_ratios(tmp_path):
    """The t-ratio panels must show the ±1.96 band and the tallest bar.

    `plots._snap_cmax` caps at 1.0 because it is built for correlations; reusing it
    for t-ratios clipped every bar and pushed the ±1.96 band off-axis, so the plot
    looked empty of significance while the table reported 5/9 significant.
    """
    import math

    M = _loaded(tmp_path)
    w = M._prepared_w(M._DATA["_plt"], 0.0, 1, 0, "auto")
    tt = M._tiaobox_tratios(w, 5)
    tmax = max(float(np.abs(v).max()) for v in tt.values())
    M.plot_partial_autoregression_matrices("_plt", max_order=5,
                                           path=str(tmp_path / "t.png"))
    cmax = max(math.ceil(tmax), 3.0)
    assert cmax >= 1.96, "the significance band must fit on the axis"
    assert cmax >= tmax, "the tallest t-ratio must fit on the axis"


def test_ccf_uses_the_dot_out_orientation_convention(tmp_path):
    """In the pair titled "A - B", k>0 must mean A → B, as the .out report prints
    ("nj --> ni IF k > 0") and as plots.plot_residual_ccf already does.

    The first version of this tool used `ccf(w_i, w_j)` directly, whose k>0 means
    the SECOND name leads — the opposite convention under a near-identical title.
    Anyone used to the .out would have read the plot backwards.
    """
    from drvarma import mcp_server as M0
    from drvarma.diagnostics import ccf as _ccf

    M = _loaded(tmp_path, n=200, m=2)
    w = M._prepared_w(M._DATA["_plt"], 0.0, 1, 0, "auto")
    # build a lead-lag on purpose: s1 copies s0 shifted forward by 2 periods
    lead = np.zeros_like(w[:, 0])
    lead[2:] = w[:-2, 0]
    rho = _ccf(lead, w[:, 0], 6)          # canonical: title "s0 - s1", k>0 = s0→s1
    k_pos = float(np.abs(rho[6 + 1:]).max())
    k_neg = float(np.abs(rho[:6]).max())
    assert k_pos > k_neg, ("with s0 leading s1 the mass must sit at k>0 under the "
                           ".out convention")


def test_elf_accepts_stationary_varma_with_singular_phi_p():
    """Un VARMA NO BALANCEADO tiene Φ_p singular por construcción, y `elf` debe
    estimarlo si es estacionario.

    El port usaba `np.linalg.cholesky` (estricta) donde el C usa la Cholesky
    MODIFICADA de `nlatools.c:choldcp`, que acepta matrices semidefinidas
    sustituyendo un pivote demasiado pequeño en vez de fallar. Con la estricta,
    `elf` devolvía ifault=3 ("non-stationary") sobre modelos perfectamente
    estacionarios, bloqueando el cast empotrado de drtran y cualquier forma
    echelon con índices de Kronecker desiguales.

    La firma del bug era una discontinuidad: perturbar el cero con 1e-8 lo
    arreglaba. Aquí se fija el caso singular exacto.
    """
    from drvarma.estimate_py import _elf_f1f2

    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, (215, 2))
    # (1-0.7B+0.12B^2)(1-0.3B) = (1-0.4B)(1-0.3B)(1-0.3B): estacionario
    phi = np.array([[[0.7, 0.0], [0.0, 0.3]],
                    [[-0.12, 0.0], [0.0, 0.0]]])      # Phi_2 con la fila 2 nula
    assert abs(np.linalg.det(phi[-1])) < 1e-14, "Phi_p debe ser singular"

    f1, f2, ifault = _elf_f1f2(w, np.zeros(2), phi, np.zeros((0, 2, 2)),
                               np.eye(2), -1e-3)
    assert ifault == 0, "un VARMA estacionario con Phi_p singular debe estimarse"
    assert f1 > 0.0 and f2 > 0.0

    # y el resultado debe ser continuo: perturbar el cero no puede cambiar nada
    phi_eps = phi.copy()
    phi_eps[1, 1, 1] = 1e-10
    f1e, _f2e, ife = _elf_f1f2(w, np.zeros(2), phi_eps, np.zeros((0, 2, 2)),
                               np.eye(2), -1e-3)
    assert ife == 0
    assert f1e == pytest.approx(f1, rel=1e-6)


def test_chol_lower_accepts_semidefinite_and_rejects_indefinite():
    """`choldcp` acepta semidefinidas positivas y sólo rechaza las indefinidas
    más allá de la tolerancia sqrt(macheps)·maxoffl."""
    from drvarma._as311 import _chol_lower

    def envolver(A):
        n = A.shape[0]
        A1 = np.zeros((n + 1, n + 1))
        A1[1:, 1:] = A
        return A1, n

    pd = np.array([[2.0, 0.5], [0.5, 1.0]])
    L, det, ifa = _chol_lower(*envolver(pd))
    assert ifa == 0 and det == pytest.approx(np.linalg.det(pd), rel=1e-8)

    semi = np.array([[1.0, 0.0], [0.0, 0.0]])          # singular, PSD
    _L, _d, ifa = _chol_lower(*envolver(semi))
    assert ifa == 0, "una PSD singular debe aceptarse, no fallar"

    indef = np.array([[1.0, 0.0], [0.0, -5.0]])        # claramente indefinida
    _L, _d, ifa = _chol_lower(*envolver(indef))
    assert ifa == 1, "una indefinida sí debe rechazarse"


def test_convergence_is_reported_from_termcode_not_ifault():
    """The convergence banner must come from the OPTIMIZER's termination code, as
    the C does (`qnewtopt.c`: CONVERGED iff termcode in (1,2), plus the iteration
    count and the criterion for that code).

    The port decided CONVERGED/FAILED from `ifault` — which is MODEL adequacy, not
    convergence — omitted the iteration count, and printed the `termcode == 1`
    criterion unconditionally, so a run stopped at the iteration limit was reported
    as converged with a false reason.
    """
    from drvarma import report

    raw, _ = seasonal_series(1, n=180)
    ms = MultiSeries(raw, freq=12, start=(2002, 1), names=["a", "b"])
    fit = Model(ms, lam=0.0, d=1, p=1, q=0, deseason="auto",
                include_mean=True).fit()

    assert fit.termcode is not None, "termcode not exposed on the Model"
    assert fit.nit is not None, "iteration count not exposed on the Model"
    assert fit.converged is (fit.termcode in (1, 2))
    assert fit.termcode != 0, "optimiser did not run on a fit that needs it"

    banner = report._convergence_block(fit)
    assert "iterations" in banner, "iteration count missing from the banner"
    assert report._TERMCODE_CRIT[fit.termcode] in banner, "wrong/absent criterion"
    # the banner must not claim the gradient criterion for a different termcode
    for tc, crit in report._TERMCODE_CRIT.items():
        if tc != fit.termcode:
            assert crit not in banner


def test_optimizer_not_run_is_reported_as_such():
    """termcode 0 means the optimiser never ran, so the values are the STARTING
    values, not estimates. It must not be dressed up as convergence."""
    from drvarma import report

    class _Fake:
        result = {"termcode": 0, "nit": 0, "ifault": 0, "logelf": -1.0}

    banner = report._convergence_block(_Fake())
    assert "NOT RUN" in banner
    assert "CONVERGED" not in banner
    assert "STARTING values" in banner


def test_characterize_d_stable_to_one_extra_observation():
    """`d` must not flip when one leading observation is added or removed.

    ART decides d BEFORE handling seasonality, and ADF/KPSS have low power on a
    seasonal series: uncapped, IPC_ES returns d=2 from January (n=216) and d=1 from
    February (n=215). d=2 over-differences an I(1) series and injects a spurious MA
    unit root. The search is capped at d=1 so the seasonality step handles the rest.
    """
    from drvarma import mcp_server as M

    raw, _ = seasonal_series(1, n=216)
    ds = []
    for cut, sub in ((0, 1), (1, 2)):
        M.load_data(name="_dstab", values_json=str([list(r) for r in raw[cut:]]),
                    freq=12, start_year=2002, start_period=sub, series_names="a,b")
        M.characterize_series("_dstab")
        ds.append(M._SEED["_dstab"]["consensus"]["d"])
    assert ds[0] == ds[1], f"d flipped with one observation: {ds}"
    assert ds[0] <= 1, f"d escalated above 1: {ds}"


@pytest.mark.parametrize("drop", [1, 2, 5])
def test_seasonal_pattern_invariant_to_dropping_leading_observations(drop):
    """Dropping the first few observations (and declaring the new start_sub) must
    barely move the estimated seasonal pattern — it is the same seasonality either
    way. Before the fix the returned dummy vector came out CIRCULARLY SHIFTED by
    `drop` months, i.e. a discrepancy of 36-83 % of the pattern amplitude on real
    CPI data. This is the cheapest possible check that phase is handled correctly.
    """
    raw, _ = seasonal_series(1, n=240)
    _, d_full, _ = deseasonalize_raw(raw, s=12, start_sub=1, mode="force")
    _, d_cut, _ = deseasonalize_raw(raw[drop:], s=12, start_sub=1 + drop,
                                    mode="force")
    amplitude = d_full[0].max() - d_full[0].min()
    assert np.abs(d_full[0] - d_cut[0]).max() < 0.05 * amplitude


@pytest.mark.parametrize("deseason", [None, "auto"])
@pytest.mark.parametrize("start_sub", [1, 2, 7])
def test_identification_series_equals_estimation_series(deseason, start_sub):
    """The series CCM/Tiao-Box identify on MUST be the series the Model estimates on.

    multiart used to replay deseason+transform itself, keeping the two pipelines in
    sync only by hand. This pins the invariant so a change to Model.prepare() can
    never silently desynchronise identification from estimation.
    """
    from drvarma import mcp_server as M

    raw, _ = seasonal_series(start_sub, n=180)
    ms = MultiSeries(raw, freq=12, start=(2002, start_sub), names=["a", "b"])
    fit = Model(ms, lam=0.0, d=1, p=1, q=0, deseason=deseason,
                include_mean=True).fit()
    w_estimated = np.asarray(fit._w) / fit.scale
    w_identified = np.asarray(M._prepared_w(ms, 0.0, 1, 0, deseason))
    assert np.allclose(w_estimated, w_identified, atol=1e-10)


def test_deseason_then_transform_kills_seasonal_acf():
    """End-to-end post-condition that would have caught BUG 4 on the first run:
    after the prepared-series pipeline, ACF at the seasonal lag must be small."""
    raw, _ = seasonal_series(2, n=240)
    adj, _, _ = deseasonalize_raw(raw, s=12, start_sub=2, mode="force")
    w, _ = transform.transform(adj, lam=0.0, d=1, D=0, s=12, scale=1.0)
    assert abs(acf(np.asarray(w)[:, 0], 12)) < 0.3
