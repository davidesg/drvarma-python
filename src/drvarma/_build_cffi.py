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


def _discover_gsl_dirs():
    """Return (include_dirs, library_dirs) for GSL when it is outside the
    compiler's default search paths.

    Linux system installs put GSL in /usr/include (found automatically), but
    Homebrew on macOS uses /opt/homebrew (Apple Silicon) or /usr/local (Intel),
    which clang does not search by default — hence 'gsl/gsl_matrix.h not found'.
    Query `gsl-config --prefix` (GSL always ships gsl-config), then fall back to
    `brew --prefix gsl` and the common Homebrew prefixes. Only existing dirs are
    returned, so this is a harmless no-op on Linux.

    Lifted from fue's build script, which solved this first: the two engines
    have the same GSL dependency and there is no reason for them to discover it
    differently.
    """
    import subprocess
    prefixes = []
    for cmd in (["gsl-config", "--prefix"], ["brew", "--prefix", "gsl"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                prefixes.append(r.stdout.strip())
        except Exception:                                  # noqa: BLE001
            pass
    if sys.platform == "darwin":
        prefixes += ["/opt/homebrew", "/usr/local"]
    inc, lib = [], []
    for pref in prefixes:
        i, l = os.path.join(pref, "include"), os.path.join(pref, "lib")
        if os.path.isdir(i) and i not in inc:
            inc.append(i)
        if os.path.isdir(l) and l not in lib:
            lib.append(l)
    return inc, lib


_GSL_INC, _GSL_LIB = _discover_gsl_dirs()

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
    int     p, q;
    int     nresiduals;
    double *params;
    double *std_errors;
    double *cov_matrix;
    double *residuals;
    double *mu;
    double *phi;
    double *theta;
    double *sigma;
    double  sigma2;
    double  logelf;
    int     termcode;
    int     nit;
} DrvarmaResult;

void           drvarma_defaults(DrvarmaModelSpec *spec);
DrvarmaResult *drvarma_estimate(const DrvarmaModelSpec *spec);
void           drvarma_result_free(DrvarmaResult *r);
const char    *drvarma_strerror(int ifault);

int drvarma_elf(int m, int n, int p, int q,
                const double *mu, const double *phi, const double *theta,
                const double *qq, const double *w,
                double sigma2, double delta, int atf,
                double *a_out, double *f1, double *f2, double *logelf);
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
    include_dirs=[_CSRC, _INT] + _GSL_INC,
    library_dirs=_GSL_LIB,
    libraries=_libs,
    extra_compile_args=_cargs,
)

if __name__ == "__main__":
    ffi.compile(verbose=True)
