"""BUG-0005 — when the C engine does not load, drvarma says so, once.

The `.so` is made to fail as it does for real (no binary wheel): the module is
blocked in sys.modules and `_engine`'s cache is reset, so nothing outside the
test sees it.
"""
import sys
import warnings

import numpy as np
import pytest

import drvarma
from drvarma import _engine


@pytest.fixture
def sin_motor(monkeypatch):
    monkeypatch.setitem(sys.modules, "drvarma._drvarma_engine", None)
    monkeypatch.setattr(_engine, "_C_ERROR", None)
    monkeypatch.setattr(_engine, "_WARNED", False)
    monkeypatch.delenv("DRVARMA_NO_ENGINE", raising=False)


def _w():
    return np.random.default_rng(0).standard_normal((200, 2))


def _avisos(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [x for x in w if "C engine did not load" in str(x.message)]


def test_el_respaldo_avisa_una_sola_vez(sin_motor):
    av = _avisos(lambda: (_engine.estimate_w(_w(), 1, 0),
                          _engine.estimate_w(_w(), 1, 0)))
    assert len(av) == 1 and av[0].category is RuntimeWarning
    assert "drvarma._drvarma_engine" in str(av[0].message)
    assert drvarma.engine_backend() == "python"
    assert "drvarma._drvarma_engine" in drvarma.engine_load_error()


def test_elf_y_marma_tambien_avisan(sin_motor):
    m, n, w = 2, 200, _w()
    z = np.zeros(m)
    phi = np.zeros((1, m, m))
    av = _avisos(lambda: (_engine.elf_c(m, n, 1, 0, z, phi, np.zeros((0, m, m)), np.eye(m), w),
                          _engine.marma_c(m, n, 1, 0, z, phi, np.zeros((0, m, m)), np.eye(m), w)))
    assert len(av) == 1


def test_con_DRVARMA_NO_ENGINE_no_avisa(sin_motor, monkeypatch):
    monkeypatch.setenv("DRVARMA_NO_ENGINE", "1")
    assert _avisos(lambda: _engine.estimate_w(_w(), 1, 0)) == []
    assert drvarma.engine_backend() == "python"


def test_con_el_motor_no_avisa(monkeypatch):
    pytest.importorskip("drvarma._drvarma_engine")
    monkeypatch.delenv("DRVARMA_NO_ENGINE", raising=False)
    assert drvarma.engine_backend() == "c"
    assert drvarma.engine_load_error() is None
    assert _avisos(lambda: _engine.estimate_w(_w(), 1, 0)) == []
