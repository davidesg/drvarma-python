"""The re-homologation with the C of November 2026, second part: the mask of
the deterministic terms (the C's -det and -detbase, atsw-gui #209). The
numbers are those of the C's battery, engines/drvarma/tests/escalera
(section 10e)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.ladder import Ladder, LadderError  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
ES, FR = os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")
IPC, WTI = os.path.join(D, "IPC_m07.pre"), os.path.join(D, "WTI_m03.pre")


def test_det_all_is_redet():
    a = Ladder([ES, FR], 1, 0, redet=True)
    ra = a.fit()
    b = Ladder([ES, FR], 1, 0, det="all")
    rb = b.fit()
    assert a.npar() == b.npar() == 30
    assert ra.logL == pytest.approx(rb.logL, abs=1e-9)
    assert ra.logL == pytest.approx(92.584954, abs=1e-5)             # the C's
    assert a.det_report() == b.det_report() == "re-estimated"


def test_a_series_class():
    L = Ladder([ES, FR], 1, 0, det="IPC_FR:seasonal")
    r = L.fit()
    assert L.npar() == 19
    assert r.logL == pytest.approx(92.571461, abs=1e-5)              # the C's
    assert L.det_report() == ("re-estimated: IPC_FR:seasonal (11 of 22); "
                              "the rest fixed at the file")


def test_detbase_gate_on_genuine_files_is_a_fixed_point():
    a = Ladder([ES, FR], 1, 0)
    ra = a.fit()
    b = Ladder([ES, FR], 1, 0, detbase="gate")
    rb = b.fit()
    assert a.npar() == b.npar() == 8
    assert ra.logL == pytest.approx(92.566119, abs=1e-5)             # the C's
    assert rb.logL == pytest.approx(ra.logL, abs=1e-5)
    assert a.det_report() == "fixed at the file"
    assert b.det_report() == ("fixed at the gate's values (22 estimated there, "
                              "series by series)")


def _es_2003(tmp_path):
    """Spain's file a year shorter (the C battery's case 7): France is then
    trimmed to the common window."""
    L = open(ES, errors="replace").read().split("\n")
    L[8] = L[8].replace("216  1 2002", "204  1 2003")
    L = L[:72] + L[84:]
    p = tmp_path / "ES_2003.pre"
    p.write_text("\n".join(L))
    return str(p)


def test_the_trimmed_window(tmp_path):
    """By default the gate re-estimates the deterministic terms of the series
    it trims and holds them; detbase="file" keeps the file's. The C's
    numbers: 43.308275 (file), 43.743189 (default), 43.946694 (gate, and
    redet on the diagonal)."""
    es = _es_2003(tmp_path)
    kw = dict(p=0, q=0, diagcov=True)
    f = Ladder([es, FR], detbase="file", **kw)
    d = Ladder([es, FR], **kw)
    g = Ladder([es, FR], detbase="gate", **kw)
    r = Ladder([es, FR], redet=True, **kw)
    lf, ld, lg, lr = (x.fit().logL for x in (f, d, g, r))
    assert lf == pytest.approx(43.308275, abs=1e-5)
    assert ld == pytest.approx(43.743189, abs=1e-5)
    assert lg == pytest.approx(43.946694, abs=1e-5)
    assert lr == pytest.approx(lg, abs=1e-5)
    assert (f.npar(), d.npar(), g.npar(), r.npar()) == (5, 5, 5, 27)
    assert d.det_report() == ("fixed at the gate's values (11 estimated there, "
                              "series by series)")


def test_interventions_by_class_and_by_name():
    """IPC and WTI of the US monetary VAR: the Gulf step and Katrina in the
    IPC, the Gulf impulse in the WTI."""
    fits = {}
    for key, det in (("none", ""), ("interv", "interv"), ("all", "all"),
                     ("katrina", "w0_IPC[impulse 9 2005]"),
                     ("gulf", "w0_IPC[step 8 1990],w0_WTI[impulse 8 1990]")):
        L = Ladder([IPC, WTI], 1, 0, det=det)
        fits[key] = (L, L.fit())
    npar = {k: v[0].npar() for k, v in fits.items()}
    ll = {k: v[1].logL for k, v in fits.items()}
    assert npar == {"none": 9, "interv": 21, "all": 32, "katrina": 12, "gulf": 18}
    assert ll["none"] == pytest.approx(-631.112257, abs=1e-5)        # the C's
    assert ll["interv"] == pytest.approx(-628.995175, abs=1e-5)
    assert ll["all"] == pytest.approx(-626.987885, abs=1e-5)
    assert ll["katrina"] == pytest.approx(-630.230451, abs=1e-5)
    assert ll["none"] <= ll["gulf"] + 1e-6 <= ll["interv"] + 2e-6 <= ll["all"] + 3e-6
    names = fits["gulf"][0].names()
    assert any("step 8" in n for n in names) and not any("impulse 9" in n for n in names)


def test_a_selector_that_matches_nothing():
    with pytest.raises(LadderError, match="does not match any deterministic term"):
        Ladder([IPC, WTI], 1, 0, det="estacional")


def test_det_and_redet_together():
    with pytest.raises(ValueError):
        Ladder([IPC, WTI], 1, 0, redet=True, det="interv")
