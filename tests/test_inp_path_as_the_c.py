"""The deprecated .inp path against the C of record (BUG-0002).

``tests/data/inp_c/cases.json`` holds, for six cases of the C's battery
(atsw-gui ``engines/drvarma/tests/banco``, whose references are byte goldens),
the C's log-likelihood, how it got the standard errors and its parameter
table. The vendored C (``csrc/``) is the C's, so with the compiled engine the
port's table is the C's line for line. The pure-Python estimator is held to
the numbers where the standard errors are fdhess's; where the optimum is on
the boundary they are BFGS's, path-dependent, and the port's path is not yet
the C's (BUG-0033)."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import drvarma

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "inp_c")
REC = json.load(open(os.path.join(D, "cases.json"), encoding="utf-8"))
SRC = os.path.join(HERE, "..", "src")


def _run(case, tmp_path, no_engine):
    shutil.copy(os.path.join(D, case["file"]), tmp_path)
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if no_engine:
        env["DRVARMA_NO_ENGINE"] = "1"
    else:
        env.pop("DRVARMA_NO_ENGINE", None)
    base = case["file"][:-4]
    subprocess.run([sys.executable, "-m", "drvarma.cli", base, *case["c_args"].split()],
                   cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1800)
    t = open(os.path.join(tmp_path, base + ".out"), errors="replace").read()
    logl = float(re.findall(r"Log-likelihood = (-?[\d.]+)", t)[-1])
    se = re.search(r"^Standard errors: (.*)$", t, re.M).group(1).strip()
    rows = []
    for line in t[t.index("Std.Error   t-stat"):].splitlines()[2:]:
        if not line.strip() or line.startswith("---") or line.startswith("Standard errors"):
            break
        rows.append(line.rstrip())
    return logl, se, rows


def _numbers(rows):
    out = {}
    for line in rows:
        x = line.split()
        try:
            out[x[0]] = (float(x[1]), float(x[2]))
        except (ValueError, IndexError):
            pass
    return out


@pytest.mark.skipif(drvarma.engine_backend() != "c", reason="needs the compiled engine")
@pytest.mark.parametrize("case", REC["cases"], ids=[c["name"] for c in REC["cases"]])
def test_with_the_engine_the_table_is_the_cs(case, tmp_path):
    logl, se, rows = _run(case, tmp_path, no_engine=False)
    assert logl == pytest.approx(case["logL"], abs=1e-5)
    assert se == case["se_method"]
    assert rows == case["table"]


# The pure-Python estimator is not yet the C's on these (BUG-0033): on PSW
# ARMA(1,1) it stops 0.22 lower, on the twostep case 0.0016 higher, on the
# boundary its BFGS standard errors are another path's, and on c5 it reaches
# the same logL at other estimates (a flat direction).
BUG_0033 = {"c2_ipc_arma21", "c3_psw_arma11", "c5_eu_diag", "c7_ipc_twostep"}


@pytest.mark.parametrize("case", REC["cases"], ids=[c["name"] for c in REC["cases"]])
def test_the_pure_python_estimator_gives_the_cs_numbers(case, tmp_path):
    if case["name"] in BUG_0033:
        pytest.xfail("BUG-0033: the pure-Python .inp path is not yet the C's here")
    logl, se, rows = _run(case, tmp_path, no_engine=True)
    assert logl == pytest.approx(case["logL"], abs=1e-5)
    assert se == case["se_method"]
    got, want = _numbers(rows), _numbers(case["table"])
    assert got.keys() == want.keys()
    for k in want:
        assert got[k][0] == pytest.approx(want[k][0], abs=2e-6), k
        # fdhess by finite differences in Python and in C: the last digits
        assert got[k][1] == pytest.approx(want[k][1], rel=5e-4), k
