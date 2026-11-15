"""BUG-0025: the ladder's forecast table says a percentage only with logs; in
levels, the change (as the C since its BUG-0005)."""
import numpy as np
import pytest

fue = pytest.importorskip("fue")

from drvarma.ladder import Ladder, write_forecast


def _pres(tmp_path, lam, factor):
    rng = np.random.default_rng(11)
    paths, ys = [], []
    for nm in ("Y", "X"):
        y = np.zeros(120)
        y[0] = 100.0
        for t in range(1, 120):
            y[t] = 100.0 + 0.6 * (y[t - 1] - 100.0) + rng.normal(0, 1.0)
        ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name=nm)
        w = np.log(y) if lam == 0.0 else y
        m = fue.Model(ts, d=0, D=0, boxlam=lam, ar=[[0.1]], ar_free=[[True]],
                      mu=factor * float(w.mean()), estimate_mu=True, refactor=factor)
        m.fit()
        p = tmp_path / f"{nm}.pre"
        m.write_pre(str(p))
        paths.append(str(p))
        ys.append(y)
    return paths, ys


def _tabla(tmp_path, lam, factor):
    paths, ys = _pres(tmp_path, lam, factor)
    L = Ladder(paths, 0, 0)
    L.fit()
    out = tmp_path / "t.forecast"
    write_forecast(L, L.forecast(3), str(out))
    text = out.read_text().splitlines()
    i = next(k for k, l in enumerate(text) if l.startswith("Series Y"))
    row = text[i + 2].split()
    return text, float(row[1]), float(row[4]), ys[0][-1]


def test_in_levels_the_variation_is_the_change(tmp_path):
    text, level, mon, last = _tabla(tmp_path, 1.0, 100.0)
    assert mon == pytest.approx(level - last, abs=2e-4)
    assert "otherwise delta/refactor" in text[1]


def test_in_logs_it_is_a_percentage_and_the_header_is_the_old_one(tmp_path):
    text, level, mon, last = _tabla(tmp_path, 0.0, 100.0)
    assert mon == pytest.approx(100.0 * np.log(level / last), abs=2e-4)
    assert text[1].endswith("(100*delta/refactor; % for lambda=0).")


def test_files_of_different_factors_are_warned_of_as_in_the_c(tmp_path):
    import warnings
    from drvarma.ladder import mixed_refactors_warning
    assert mixed_refactors_warning(["Y", "X"], [100.0, 100.0]) is None
    assert mixed_refactors_warning(["Y", "X"], [100.0, 1.0]).startswith(
        "the files do not share one refactor (Y 100, X 1).")
    a, _ = _pres(tmp_path, 0.0, 100.0)
    (tmp_path / "b").mkdir()
    b, _ = _pres(tmp_path / "b", 0.0, 1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Ladder([a[0], b[1]], 0, 0)
    assert any("do not share one refactor" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Ladder(a, 0, 0)
    assert not any("refactor" in str(x.message) for x in w)
