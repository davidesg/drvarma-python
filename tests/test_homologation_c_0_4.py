"""The re-homologation with the C of November 2026 (atsw-gui #203-#219),
first part: ``split`` and the default start. The numbers are the C's
(atsw-gui ``engines/drvarma/tests/escalera`` and ``bugs``)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from drvarma.ladder import Ladder, split  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "data", "ladder")
PAIR = [os.path.join(D, "IPC_ES_m10.pre"), os.path.join(D, "IPC_FR_msar.pre")]


def _mu_line(path):
    L = open(path).read().split("\n")
    return L[L.index("** Mean parameter (mu):") + 1]


def test_split_seeds_the_mean_at_the_sample_mean(tmp_path):
    """The C's BUG-0007: a free mean seeded at zero, on series in levels
    (around 10000 with the factor), left the level to the own AR. The seeds
    are the C's, to its six decimals."""
    files = split(os.path.join(D, "NIVEL100.inp"), mean=True, ar=1, out_dir=str(tmp_path))
    assert [_mu_line(f) for f in files] == ["10009.205394 1", "9998.267168 1"]


def test_the_ladder_on_split_files_in_levels_is_the_inp_path(tmp_path):
    """The same VAR(1) with a mean: the C gives -3620.221054 by the .inp path
    and by the ladder on these files (with the mean at zero, -3669.43)."""
    files = split(os.path.join(D, "NIVEL100.inp"), mean=True, ar=1, out_dir=str(tmp_path))
    r = Ladder(files, 1, 0).fit()
    assert r.logL == pytest.approx(-3620.221054, abs=1e-5)


def test_split_without_a_mean_writes_none(tmp_path):
    files = split(os.path.join(D, "NIVEL100.inp"), out_dir=str(tmp_path))
    assert [_mu_line(f) for f in files] == ["0", "0"]


def test_split_header_fits_fues_format(tmp_path):
    """The C's BUG-0001: a free header line of 89 characters or more counts as
    two for fue. The source's base name, and never more than 88."""
    import shutil
    deep = tmp_path / ("a_long_folder_name_" * 4) / ("another_long_folder_" * 3)
    deep.mkdir(parents=True)
    src = shutil.copy(os.path.join(D, "NIVEL100.inp"), deep / "NIVEL100.inp")
    out = tmp_path / "out"
    out.mkdir()
    files = split(str(src), mean=True, out_dir=str(out))
    head = open(files[0]).read().split("\n")[3]
    assert head.startswith("* Split from NIVEL100.inp by drvarma-python ")
    assert len(head) <= 88
    Ladder(files, 0, 0)                         # fue reads what split wrote


def test_the_default_start_is_the_cs():
    """ols with q = 0 (the original drvarma's least squares), zero with an MA
    and with no cross terms: atsw-gui #207."""
    assert Ladder(PAIR, 1, 0).start == "ols"
    assert Ladder(PAIR, 2, 0).start == "ols"
    assert Ladder(PAIR, 1, 1).start == "zero"
    assert Ladder(PAIR, 0, 1).start == "zero"
    assert Ladder(PAIR, 0, 0).start == "zero"
    assert Ladder(PAIR, 1, 0, start="zero").start == "zero"


def test_the_default_start_is_used_and_reaches_the_optimum_of_zero():
    a = Ladder(PAIR, 1, 0)
    ra = a.fit()
    rb = Ladder(PAIR, 1, 0, start="zero").fit()
    assert a.start_used == "ols"
    assert ra.logL == pytest.approx(rb.logL, abs=1e-5)
