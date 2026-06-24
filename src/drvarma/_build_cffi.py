"""cffi build script for the drvarma estimation engine.

Compiles csrc/drvarma_api.c + csrc/internal/*.c against GSL and produces the
`drvarma._drvarma_engine` extension imported by `_engine.py`.

Build standalone:  python -m drvarma._build_cffi   (or run this file)
or via pip/setup through cffi_modules in pyproject.toml.
"""
import os
import sys
import cffi

_THIS = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.relpath(os.path.join(_THIS, "..", "..", "csrc"))
_INT = os.path.join(_CSRC, "internal")

_CDEF = """
typedef struct {
    int     m;
    int     nobs;
    double *w;
    int     p, q;
    int     include_mean;
    int     diag_ar, diag_ma, diag_cov;
    int     method;
    int     twostep;
    int     maxits;
    double  grtol, sptol;
} DrvarmaModelSpec;

typedef struct {
    int     ifault;
    int     npar;
    int     m;
    int     nresiduals;
    double *params;
    double *std_errors;
    double *cov_matrix;
    double *residuals;
    double  sigma2;
    double  logelf;
} DrvarmaResult;

void           drvarma_defaults(DrvarmaModelSpec *spec);
DrvarmaResult *drvarma_estimate(const DrvarmaModelSpec *spec);
void           drvarma_result_free(DrvarmaResult *r);
const char    *drvarma_strerror(int ifault);
"""

_SOURCES = [
    os.path.join(_CSRC, "drvarma_api.c"),
    os.path.join(_INT, "nlatools.c"),
    os.path.join(_INT, "elfvarma.c"),
    os.path.join(_INT, "multshea.c"),
    os.path.join(_INT, "qnewtopt.c"),
    os.path.join(_INT, "drvmlest.c"),
]

if sys.platform == "win32":
    _libs = ["gsl", "gslcblas"]
    _cargs = ["/O2"]
else:
    _libs = ["gsl", "gslcblas", "m"]
    _cargs = ["-O2", "-std=c99", "-w"]

ffi = cffi.FFI()
ffi.cdef(_CDEF)
ffi.set_source(
    "drvarma._drvarma_engine",
    r'#include "drvarma_api.h"',
    sources=_SOURCES,
    include_dirs=[_CSRC, _INT],
    libraries=_libs,
    extra_compile_args=_cargs,
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
