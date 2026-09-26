"""
BUG-0006 — la búsqueda lineal no volvía nunca con un objetivo NaN o infinito.

`lnsrch` (Dennis-Schnabel A6.3.1, en `csrc/internal/qnewtopt.c` y en su puerto
`_qnewt.py`) compara el valor del punto de prueba para aceptar el paso o
abandonarlo. Con un NaN toda comparación es falsa: el paso ni se aceptaba ni se
abandonaba, `tlambda` y luego `lambda` se volvían NaN, y el bucle giraba para
siempre, sin error ni salida. Un infinito llega al mismo bucle un paso después,
por inf/inf en el ajuste cúbico. drtran lo sufrió con datos reales —una hora y
media— y lo tapó en su función objetivo; el optimizador compartido seguía
expuesto para cualquier otro llamador.

Cada caso corre en un SUBPROCESO con límite de tiempo: si el defecto vuelve, la
prueba falla en segundos en vez de colgar la suite.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(RAIZ, "csrc", "internal")
LIMITE = 20  # segundos; el caso sano termina en milisegundos

PY_PROBE = textwrap.dedent("""
    import sys, numpy as np
    from drvarma import _qnewt
    kind = sys.argv[1]
    def f(x):
        if x[1] > 2.0:
            return float("nan") if kind == "nan" else float("inf")
        return 0.1 * (x[1] - 3) ** 2 / 0.9
    xk = np.zeros(2)
    fk, _b, nit, tc = _qnewt.raxopt(f, 1, xk, 100, 1e-6, 1e-8)
    print(repr((float(xk[1]), float(fk), int(tc))))
""")

C_PROBE = textwrap.dedent("""
    #include "main.h"
    real macheps; FILE *outputv; int quiet_mode = 1;
    void raxopt(real (*)(real []), real *, int, real *, real **, int, int, real, real);
    static int kind;
    real f(real *x) { if (x[1] > 2.0) return kind ? INFINITY : NAN;
                      return 0.1*(x[1]-3)*(x[1]-3)/0.9; }
    int main(int argc, char **argv) {
      kind = (argc > 1 && argv[1][0] == 'i');
      macheps = cmacheps(); outputv = stdout;
      real *x = vector(1,1); x[1] = 0.0; real **b = matrix(1,1,1,1), fk;
      raxopt(f, &fk, 1, x, b, 100, 1, 1e-6, 1e-8);
      printf("%.10g %.10g\\n", x[1], fk); return 0; }
""")


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_el_puerto_en_python_termina(kind):
    r = subprocess.run([sys.executable, "-c", PY_PROBE, kind],
                       capture_output=True, text=True, timeout=LIMITE)
    assert r.returncode == 0, r.stderr[-800:]
    x, fk, _tc = eval(r.stdout.strip())
    # Se queda en la región admisible y con un valor finito: el mínimo sin
    # restricción (x=3) está en la zona NaN/inf, así que acaba en su borde.
    assert x <= 2.0 + 1e-9 and fk == pytest.approx(0.1 * (x - 3) ** 2 / 0.9)


@pytest.fixture(scope="module")
def c_probe(tmp_path_factory):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("sin compilador de C")
    d = tmp_path_factory.mktemp("bug0006")
    src = d / "probe.c"
    src.write_text(C_PROBE)
    exe = d / "probe"
    r = subprocess.run(
        [cc, "-O0", "-w", f"-I{CSRC}", "-o", str(exe), str(src),
         os.path.join(CSRC, "qnewtopt.c"), os.path.join(CSRC, "nlatools.c"),
         "-lgsl", "-lgslcblas", "-lm"],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("no se pudo compilar la sonda (¿falta GSL?): " + r.stderr[-300:])
    return exe


@pytest.mark.parametrize("kind", ["nan", "inf"])
def test_el_motor_c_termina(c_probe, kind):
    try:
        r = subprocess.run([str(c_probe), kind], capture_output=True,
                           text=True, timeout=LIMITE)
    except subprocess.TimeoutExpired:
        pytest.fail(f"lnsrch en C sigue sin volver con un objetivo {kind} (BUG-0006)")
    x, fk = map(float, r.stdout.split())
    assert x <= 2.0 + 1e-9 and fk == pytest.approx(0.1 * (x - 3) ** 2 / 0.9)
