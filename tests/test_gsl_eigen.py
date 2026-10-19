"""GSL's eigensolvers from the compiled engine: `gensymmv` and `eigen_nonsymm`.

They exist for drvec's Johansen start (canonical_b2 calls gsl_eigen_gensymmv +
gsl_eigen_gensymmv_sort) and its reports (eigenvalues in gsl_eigen_nonsymm's
order): the Python port must get GSL's bits, which scipy's `eigh` does not give
(a few ulps off, and on Lutkepohl's E1 that moves an optimum by 0.47 in logL).

Three things are pinned:
1. Bit for bit with the same GSL called through ctypes, the way drvec did it
   before (only on Linux x86_64 with a system libgsl: that is where the engine
   is built against the same library).
2. To tolerance with scipy / numpy, on every platform with the engine.
3. Without the engine, an EngineUnavailableError: no silent fallback.
"""

import ctypes
import ctypes.util
import platform
import sys

import numpy as np
import pytest
import scipy.linalg as sl

from drvarma import _engine as E

needs_engine = pytest.mark.skipif(not E.gsl_available(),
                                reason="the compiled engine is not built")


def _cases(seed=20261017, sizes=range(1, 9), each=40):
    rng = np.random.default_rng(seed)
    for n in sizes:
        for _ in range(each):
            X = rng.standard_normal((n, n))
            Y = rng.standard_normal((n + 3, n))
            yield X + X.T, Y.T @ Y, rng.standard_normal((n, n))


# -- 1. the same bits as a C program calling GSL ----------------------------- #

def _system_gsl():
    """libgsl through ctypes, as drvec's `_gsl.py` loaded it, or None."""
    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        return None
    for c in (ctypes.util.find_library("gslcblas"), "libgslcblas.so.0"):
        if c:
            try:
                ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                pass
    for nm in (ctypes.util.find_library("gsl"), "libgsl.so.27", "libgsl.so"):
        if not nm:
            continue
        try:
            g = ctypes.CDLL(nm)
        except OSError:
            continue
        P, S, D = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
        for f, res, args in (
                ("gsl_matrix_alloc", P, [S, S]), ("gsl_vector_alloc", P, [S]),
                ("gsl_eigen_gensymmv_alloc", P, [S]),
                ("gsl_matrix_set", None, [P, S, S, D]), ("gsl_matrix_get", D, [P, S, S]),
                ("gsl_vector_get", D, [P, S]),
                ("gsl_eigen_gensymmv", ctypes.c_int, [P, P, P, P, P]),
                ("gsl_eigen_gensymmv_sort", ctypes.c_int, [P, P, ctypes.c_int]),
                ("gsl_eigen_gensymmv_free", None, [P]), ("gsl_matrix_free", None, [P]),
                ("gsl_vector_free", None, [P]),
                ("gsl_vector_complex_alloc", P, [S]), ("gsl_vector_complex_free", None, [P]),
                ("gsl_vector_complex_ptr", ctypes.POINTER(D), [P, S]),
                ("gsl_eigen_nonsymm_alloc", P, [S]), ("gsl_eigen_nonsymm_free", None, [P]),
                ("gsl_eigen_nonsymm_params", None, [ctypes.c_int, ctypes.c_int, P]),
                ("gsl_eigen_nonsymm", ctypes.c_int, [P, P, P]),
                ("gsl_set_error_handler_off", P, [])):
            fn = getattr(g, f)
            fn.restype, fn.argtypes = res, args
        g.gsl_set_error_handler_off()
        return g
    return None


def _ctypes_gensymmv_desc(g, A, B):
    n = A.shape[0]
    Ag, Bg, V, ev = (g.gsl_matrix_alloc(n, n), g.gsl_matrix_alloc(n, n),
                     g.gsl_matrix_alloc(n, n), g.gsl_vector_alloc(n))
    ws = g.gsl_eigen_gensymmv_alloc(n)
    try:
        for i in range(n):
            for j in range(n):
                g.gsl_matrix_set(Ag, i, j, float(A[i, j]))
                g.gsl_matrix_set(Bg, i, j, float(B[i, j]))
        assert g.gsl_eigen_gensymmv(Ag, Bg, ev, V, ws) == 0
        g.gsl_eigen_gensymmv_sort(ev, V, 1)
        return (np.array([g.gsl_vector_get(ev, i) for i in range(n)]),
                np.array([[g.gsl_matrix_get(V, i, j) for j in range(n)] for i in range(n)]))
    finally:
        g.gsl_eigen_gensymmv_free(ws)
        g.gsl_vector_free(ev)
        g.gsl_matrix_free(V)
        g.gsl_matrix_free(Bg)
        g.gsl_matrix_free(Ag)


def _ctypes_nonsymm(g, A):
    n = A.shape[0]
    Ag, ev, w = g.gsl_matrix_alloc(n, n), g.gsl_vector_complex_alloc(n), \
        g.gsl_eigen_nonsymm_alloc(n)
    try:
        for i in range(n):
            for j in range(n):
                g.gsl_matrix_set(Ag, i, j, float(A[i, j]))
        g.gsl_eigen_nonsymm_params(0, 1, w)
        g.gsl_eigen_nonsymm(Ag, ev, w)
        z = [g.gsl_vector_complex_ptr(ev, i) for i in range(n)]
        return np.array([p[0] for p in z]), np.array([p[1] for p in z])
    finally:
        g.gsl_eigen_nonsymm_free(w)
        g.gsl_vector_complex_free(ev)
        g.gsl_matrix_free(Ag)


@needs_engine
def test_bit_for_bit_with_gsl_through_ctypes():
    g = _system_gsl()
    if g is None:
        pytest.skip("the bit-for-bit check runs on Linux x86_64 with a system libgsl")
    for A, B, C in _cases():
        ev0, V0 = _ctypes_gensymmv_desc(g, A, B)
        ev1, V1 = E.gensymmv(A, B)
        assert np.array_equal(ev0, ev1) and np.array_equal(V0, V1)
        wr0, wi0 = _ctypes_nonsymm(g, C)
        wr1, wi1 = E.eigen_nonsymm(C)
        assert np.array_equal(wr0, wr1) and np.array_equal(wi0, wi1)


# -- 2. the same numbers as scipy / numpy, to tolerance ---------------------- #

@needs_engine
def test_gensymmv_against_scipy():
    for A, B, _ in _cases():
        ev, V = E.gensymmv(A, B)
        w, U = sl.eigh(A, B)
        assert np.allclose(ev, w[::-1], rtol=1e-9, atol=1e-10)
        assert np.all(np.diff(ev) <= 0)                       # VAL_DESC
        assert np.allclose(np.linalg.norm(V, axis=0), 1.0, atol=1e-12)  # GSL: unit norm
        assert np.allclose(A @ V, B @ V * ev, atol=1e-7 * (1 + np.abs(ev).max()))
        # eigenvectors agree up to sign wherever the eigenvalue is simple
        U = U[:, ::-1] / np.linalg.norm(U, axis=0)[::-1]
        gap = np.min(np.abs(np.diff(ev))) if len(ev) > 1 else 1.0
        if gap > 1e-6:
            s = np.sign(np.sum(U * V, axis=0))
            assert np.allclose(V, U * s, atol=1e-6)


@needs_engine
def test_gensymmv_sorts():
    A, B, _ = next(_cases(seed=3, sizes=[5], each=1))
    asc, Va = E.gensymmv(A, B, sort="val_asc")
    desc, Vd = E.gensymmv(A, B)
    raw, _ = E.gensymmv(A, B, sort=None)
    assert np.array_equal(asc, desc[::-1]) and np.array_equal(Va, Vd[:, ::-1])
    assert np.array_equal(np.sort(raw), asc)
    ab, _ = E.gensymmv(A, B, sort="abs_desc")
    assert np.all(np.diff(np.abs(ab)) <= 0)
    with pytest.raises(ValueError):
        E.gensymmv(A, B, sort="descending")


@needs_engine
def test_eigen_nonsymm_against_numpy():
    for _, _, C in _cases():
        wr, wi = E.eigen_nonsymm(C)
        got = np.sort_complex(wr + 1j * wi)
        ref = np.sort_complex(np.linalg.eigvals(C))
        assert np.allclose(got, ref, rtol=1e-8, atol=1e-9)


@needs_engine
def test_eigen_nonsymm_gsl_example():
    """GSL's manual (nonsymmetric eigensystems): the Vandermonde matrix."""
    C = np.array([[-1., 1., -1., 1.], [-8., 4., -2., 1.],
                  [27., 9., 3., 1.], [64., 16., 4., 1.]])
    wr, wi = E.eigen_nonsymm(C)
    got = np.sort_complex(wr + 1j * wi)
    ref = np.sort_complex([-6.41391, 2.32281, 5.54555 - 3.08545j, 5.54555 + 3.08545j])
    assert np.allclose(got, ref, atol=1e-5)


# -- 3. inputs, failures and no engine --------------------------------------- #

@needs_engine
def test_inputs_untouched_and_failures():
    A, B, C = next(_cases(seed=5, sizes=[4], each=1))
    A0, B0, C0 = A.copy(), B.copy(), C.copy()
    E.gensymmv(A, B)
    E.eigen_nonsymm(C)
    assert np.array_equal(A, A0) and np.array_equal(B, B0) and np.array_equal(C, C0)
    with pytest.raises(np.linalg.LinAlgError, match="positive definite"):
        E.gensymmv(np.eye(3), -np.eye(3))                 # no abort(): a return code
    with pytest.raises(ValueError):
        E.gensymmv(np.eye(3), np.eye(2))
    with pytest.raises(ValueError):
        E.eigen_nonsymm(np.zeros((2, 3)))


def test_no_engine_raises(monkeypatch):
    monkeypatch.setenv("DRVARMA_NO_ENGINE", "1")
    assert not E.gsl_available()
    with pytest.raises(E.EngineUnavailableError):
        E.gensymmv(np.eye(2), np.eye(2))
    with pytest.raises(E.EngineUnavailableError):
        E.eigen_nonsymm(np.eye(2))
