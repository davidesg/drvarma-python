"""The record of the homologation with the C engine (atsw-gui
``engines/drvarma``): ``tests/data/homologation.json``.

It names the C's commit and a set of cases, each with the C's command line,
this port's call and the number both give. Three checks:

* the port reproduces every recorded number (always);
* the C reproduces them too, when it is named: ``ATSW_GUI`` (the checkout
  of atsw-gui, built) and, optionally, ``DRVARMA_C_BIN``. Skipped otherwise.
  No folder is guessed: a checkout that happens to be nearby may be at any
  commit, and then the comparison says nothing;
* the vendored C sources (``csrc/``) are the recorded ones: changing them
  means homologating again and updating the record.

docs/HOMOLOGATION.md."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.ladder import Ladder  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REC = json.load(open(os.path.join(ROOT, "tests", "data", "homologation.json"), encoding="utf-8"))
D = os.path.join(ROOT, "tests", "data", "ladder")
IDS = [c["name"] for c in REC["cases"]]


def _atsw_gui():
    """The checkout of atsw-gui, only if ATSW_GUI names it."""
    d = os.environ.get("ATSW_GUI")
    if d and os.path.isdir(os.path.join(d, "engines", "drvarma")):
        return os.path.abspath(d)
    return None


def _c_bin():
    b = os.environ.get("DRVARMA_C_BIN")
    if not b and _atsw_gui():
        b = os.path.join(_atsw_gui(), "engines", "drvarma", "bin",
                         "drvarma.exe" if sys.platform == "win32" else "drvarma")
    return b if b and os.path.exists(b) else None


def _port_kwargs(d):
    kw = dict(d)
    if "echelon" in kw:
        kw["echelon"] = tuple(kw["echelon"])
    if "own_zeros" in kw:
        kw["own_zeros"] = {s: {k: set(v) for k, v in z.items()} for s, z in kw["own_zeros"].items()}
    return kw


def test_the_record_names_a_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", REC["c_commit"])
    assert REC["c_repository"] == "davidesg/atsw-gui"


@pytest.mark.parametrize("case", REC["cases"], ids=IDS)
def test_the_port_gives_the_recorded_number(case):
    files = [os.path.join(D, os.path.basename(f)) for f in case["files"]]
    L = Ladder(files, **_port_kwargs(case["port"]))
    r = L.fit()
    assert r.logL == pytest.approx(case["logL"], abs=1e-5)
    assert L.npar() == case["npar"]
    if "nit" in case:
        # a stop on the MA wall is not a maximum: the same point is reached
        # only by the same path, so the iterations are part of the record
        assert r.nit == case["nit"]


@pytest.mark.parametrize("case", REC["cases"], ids=IDS)
def test_the_c_gives_the_recorded_number(case):
    c, root = _c_bin(), _atsw_gui()
    if not c or not root:
        pytest.skip("no C binary: set DRVARMA_C_BIN and ATSW_GUI (docs/HOMOLOGATION.md)")
    files = [os.path.join(root, f) for f in case["files"]]
    with tempfile.TemporaryDirectory() as t:
        subprocess.run([c, *files, *case["c_args"].split(), "-o", "o"], cwd=t,
                       capture_output=True, text=True, timeout=1800)
        out = os.path.join(t, "o.out")
        txt = open(out, errors="replace").read() if os.path.exists(out) else ""
    m = re.search(r"^Exact log-likelihood:\s*(-?[\d.]+)", txt, re.M)
    n = re.search(r"^Number of parameters:\s*(\d+)", txt, re.M)
    assert m and n, "the C did not estimate"
    assert float(m.group(1)) == pytest.approx(case["logL"], abs=1e-5)
    assert int(n.group(1)) == case["npar"]
    if "nit" in case:
        it = re.search(r"OPTIMIZER .* after (\d+) iterations", txt)
        assert it and int(it.group(1)) == case["nit"]


def test_the_c_at_hand_is_at_the_recorded_commit():
    """The binary says its commit (-version). If the C has moved, the record
    has to move with it: homologate again (docs/HOMOLOGATION.md)."""
    c = _c_bin()
    if not c:
        pytest.skip("no C binary: set ATSW_GUI (docs/HOMOLOGATION.md)")
    v = subprocess.run([c, "-version"], capture_output=True, text=True).stdout
    m = re.search(r"-g([0-9a-f]{7,40})(-dirty)?\)", v) or re.search(r"\(git ([0-9a-f]{7,40})(-dirty)?\)", v)
    assert m, f"the C does not say its commit: {v.strip()}"
    assert REC["c_commit"].startswith(m.group(1)) and not m.group(2), (
        f"the C at hand is at {m.group(1)}{m.group(2) or ''} and the record at "
        f"{REC['c_commit'][:10]}: homologate again and update the record")


def test_the_fixtures_are_the_cs():
    """The files the port's cases run on are the C battery's, byte for byte."""
    root = _atsw_gui()
    if not root:
        pytest.skip("no checkout of atsw-gui: set ATSW_GUI")
    for f in sorted({f for c in REC["cases"] for f in c["files"]}):
        a = open(os.path.join(root, f), "rb").read()
        b = open(os.path.join(D, os.path.basename(f)), "rb").read()
        assert a == b, f"{os.path.basename(f)} differs from the C battery's"


def test_the_vendored_c_is_the_recorded_one():
    """csrc/ changed without homologating again? Update the record."""
    for path, rec in REC["csrc"].items():
        got = hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()
        assert got == rec["sha256"], (f"{path} changed: homologate again and update "
                                      "tests/data/homologation.json (docs/HOMOLOGATION.md)")
