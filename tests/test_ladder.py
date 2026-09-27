"""The ladder (fue's files as input): parity with drvarma 5.0's C ladder mode.

The C is the oracle. The numbers pinned here are the C's (atsw-gui,
``engines/drvarma/tests/escalera``) and fue's own (its ``.out`` files and its
fixed-parameter forecasts, ``fue_recursive_reference.csv``). The fixtures are
copies of the C bench, in ``tests/data/ladder``.
"""
import csv
import os
import re

import numpy as np
import pytest

fue = pytest.importorskip("fue")

from drvarma.ladder import GateError, Ladder, LadderError, load, split  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
ES, FR, DE = (os.path.join(D, f) for f in
              ("IPC_ES_m10.pre", "IPC_FR_msar.pre", "IPC_DE_mar3sar.pre"))
WTI = os.path.join(D, "WTI_ar1.pre")
EXT = [os.path.join(D, f"IPC_{c}_ext.pre") for c in ("ES", "FR", "DE")]


def _fue_sigma2(pre):
    txt = open(pre[:-4] + ".out", encoding="latin-1").read()
    return float(re.search(r"^sigma2:\s*([0-9.eE+-]+)", txt, re.M).group(1))


@pytest.fixture(scope="module")
def trio_diag():
    L = Ladder([ES, FR, DE], 0, 0, diagcov=True)
    L.fit()
    return L


# -- the gate ---------------------------------------------------------------- #

def test_the_gate_is_an_identity(trio_diag):
    g = trio_diag.gate
    assert g["passed"]
    assert abs(g["difference"]) < 1e-9


def test_each_univariate_fit_is_fues(trio_diag):
    for row, pre in zip(trio_diag.gate["rows"], (ES, FR, DE)):
        assert row["sigma2"] == pytest.approx(_fue_sigma2(pre), rel=1e-8)


def test_a_genuine_pre_is_a_fixed_point(trio_diag):
    for row in trio_diag.gate["rows"]:
        assert row["move"] < 1e-4 and row["trimmed"] == 0


def test_coefficients_are_fues(trio_diag):
    r = trio_diag.result
    got = dict(zip(r.names, r.x))
    for name, v in (("phi_IPC_DE[B^1]", 0.041309), ("phi_IPC_DE[B^3]", -0.147537),
                    ("Phi_IPC_DE[B^12]", 0.165910), ("mu_IPC_DE", 0.114851)):
        assert got[name] == pytest.approx(v, abs=2e-6)


def test_the_gate_stops_the_program(monkeypatch):
    L = Ladder([ES, FR], 0, 0, diagcov=True)
    real = L._fit

    def broken(optimize=True, **kw):          # the joint evaluation drifts
        f = real(optimize=optimize, **kw)
        if not optimize:
            f.logL += 1.0
        return f
    monkeypatch.setattr(L, "_fit", broken)
    with pytest.raises(GateError):
        L.fit()


# -- the models, against the C ----------------------------------------------- #

@pytest.mark.parametrize("files,p,q,kw,logL", [
    ((ES, FR, DE), 0, 0, {}, 143.245159),              # full covariance
    ((ES, FR, DE), 1, 0, {}, 160.828677),              # VAR(1) cross
    ((WTI, ES), 1, 0, {}, -724.199406),                # the pass-through
    ((ES, FR, DE), 0, 0, dict(diagcov=True, fixarma=True), 57.837680),
])
def test_loglik_is_the_cs(files, p, q, kw, logL):
    L = Ladder(list(files), p, q, **kw)
    assert L.fit().logL == pytest.approx(logL, abs=2e-6)


def test_the_pass_through_coefficient():
    L = Ladder([WTI, ES], 1, 0)
    r = L.fit()
    got = dict(zip(r.names, r.x))
    assert got["AR1[IPC_ES<-WTI]"] == pytest.approx(0.009932, abs=2e-6)
    lr, df, pv = L.lr_test()
    assert df == 3 and lr == pytest.approx(86.4717, abs=1e-3)


# -- alignment (BUG-2) ------------------------------------------------------- #

def test_a_series_fourteen_years_late_is_refused(tmp_path):
    lines = open(WTI).read().split("\n")
    lines[8] = lines[8].replace(" 2002 ", " 2016 ")
    moved = tmp_path / "WTI_late.pre"
    moved.write_text("\n".join(lines))
    with pytest.raises(LadderError, match="do NOT end on the same date"):
        Ladder([str(moved), ES], 1, 0)


def test_different_lengths_same_end_align_at_the_end(tmp_path):
    lines = open(ES).read().split("\n")
    lines[8] = lines[8].replace("216  1 2002", "204  1 2003")
    i = next(k for k, l in enumerate(lines) if l.startswith("** Time series"))
    del lines[i + 1:i + 13]
    short = tmp_path / "ES_2003.pre"
    short.write_text("\n".join(lines))
    L = Ladder([str(short), FR], 0, 0, diagcov=True)
    L.fit()
    rows = {r["series"]: r for r in L.gate["rows"]}
    assert rows["IPC_FR"]["trimmed"] == 12 and rows["IPC_ES"]["trimmed"] == 0
    # the C's numbers for the same case (tests/escalera, "corto")
    assert rows["IPC_ES"]["logL"] == pytest.approx(-8.820102, abs=2e-6)
    assert rows["IPC_FR"]["logL"] == pytest.approx(52.128376, abs=2e-6)
    assert abs(L.gate["difference"]) < 1e-9


# -- forecasting ------------------------------------------------------------- #

def test_the_diagonal_forecasts_as_fue():
    """3456 fixed-parameter forecasts: 48 origins x 24 horizons x 3 series."""
    L = Ladder(EXT, 0, 0, diagcov=True, estwin=216)
    L.fit()
    rows, summary = L.recursive(24)
    ref = {}
    with open(os.path.join(D, "fue_recursive_reference.csv")) as fh:
        for r in csv.reader(l for l in fh if not l.startswith("#")):
            if r[0] != "origin":
                ref[(r[0], r[1], int(r[2]))] = float(r[3])
    ours = {(f"{p}/{y}", s, h): v for (y, p), s, h, v, _a in rows}
    assert len(ref) == 3456 and set(ref) <= set(ours)
    worst = max(abs(ours[k] - v) / v for k, v in ref.items())
    assert worst < 1e-5
    # and the C's evaluation of the same run (.out of tests/escalera)
    assert summary[("IPC_ES", 1)]["MAE"] == pytest.approx(0.618542, abs=2e-6)
    assert summary[("IPC_ES", 1)]["MAPE"] == pytest.approx(0.5824, abs=1e-4)


def test_the_bands_are_the_cs():
    L = Ladder(EXT, 0, 0, diagcov=True, estwin=216)
    L.fit()
    es = L.forecast(2)[0]
    assert es["origin"] == (2019, 12)
    assert es["level"][0] == pytest.approx(97.0785, abs=1e-4)
    assert es["low95"][0] == pytest.approx(96.6033, abs=1e-4)
    assert es["high95"][0] == pytest.approx(97.5560, abs=1e-4)
    assert es["sd"][1] == pytest.approx(0.4313, abs=1e-4)


# -- one format -------------------------------------------------------------- #

def test_split_is_the_old_full_varma(tmp_path):
    """The ladder on split(-mean, -ar 1) with p = 1 is the .inp path's VAR(1)."""
    from drvarma.inp import load as load_mv
    from drvarma.model import Model
    files = split(os.path.join(D, "IPC3.inp"), mean=True, ar=1, out_dir=str(tmp_path))
    assert len(files) == 3
    for f in files:                              # fue reads them
        ts, _m = fue.load(f)
        assert ts.nobs == 216
    r = Ladder(files, 1, 0).fit()
    ser, sp = load_mv(os.path.join(D, "IPC3.inp"))
    old = Model(ser, lam=sp.lam, d=sp.d, D=sp.D, p=1, include_mean=True)
    old.fit()
    np.testing.assert_allclose(r.mu, np.asarray(old.mu), atol=1e-5)
    np.testing.assert_allclose(r.phi[0], np.asarray(old.phi)[0], atol=1e-5)
    with pytest.raises(LadderError, match="not overwritten"):
        split(os.path.join(D, "IPC3.inp"), out_dir=str(tmp_path))


def test_a_multivariate_inp_is_refused_with_the_way_out():
    with pytest.raises(LadderError, match="split"):
        load([os.path.join(D, "IPC3.inp")])


def test_a_fue_inp_specification_is_taken(tmp_path):
    files = split(os.path.join(D, "IPC3.inp"), mean=True, out_dir=str(tmp_path))
    L = Ladder(files[:2], 0, 0, diagcov=True)
    L.fit()
    assert L.gate["passed"]


# -- BUG-0007: the MA coefficient in its invertible form ------------------- #

def test_m6_reports_the_invertible_ma_as_the_c_does():
    """fue's cast flips a regular MA(1) with |theta| > 1 to 1/theta, so the
    likelihood is the same on both sides and the optimiser may stop on the
    non-invertible one. On m6 it stopped on theta_EU = 1.0722: the model was
    right (1/1.0722 = 0.9327) and the report was not. The C: 0.932651."""
    M6 = [os.path.join(D, "m6", f"M6_{n}.pre") for n in ("EP", "EI", "EU", "EC", "EA", "P")]
    L = Ladder(M6, 0, 0, diagcov=True)
    L.fit()
    got = dict(zip(L.result.names, zip(L.result.x, L.result.std_errors)))
    assert got["theta_EU[B^1]"][0] == pytest.approx(0.932651, abs=2e-6)
    assert got["theta_EU[B^1]"][1] == pytest.approx(0.041682, abs=5e-6)
    moves = {r["series"]: r["move"] for r in L.gate["rows"]}
    assert moves["EU"] == pytest.approx(0.0527, abs=1e-4)       # the C's gate
    # The gate's difference on m6 is -0.000428 in the C too: not exact, as it
    # is on the CPI trio (1e-13). It passes the relative tolerance; see TODO.md.
    assert L.gate["difference"] == pytest.approx(-0.000428, abs=2e-6)
