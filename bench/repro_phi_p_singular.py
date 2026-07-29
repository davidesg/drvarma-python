"""Repro: `elf` rechaza un VARMA estacionario cuando Φ_p es singular.

Encontrado el 2026-07-29 portando el cast EMPOTRADO de drtran, que produce
Φ_p de rango deficiente por construcción.

    DRVARMA_NO_ENGINE=1 python bench/repro_phi_p_singular.py

El problema
-----------
Un VARMA cuyas ecuaciones tienen ÓRDENES DISTINTOS tiene, por construcción, ceros
en los retardos altos de las filas de menor orden — es decir, Φ_p singular. Es la
situación normal de cualquier VARMA no balanceado, y exactamente lo que produce:

  * el cast empotrado de drtran (`build_embedded_varma`): el orden de la fila i es
    deg(φ_i·D_i), distinto por fila;
  * cualquier forma ECHELON con índices de Kronecker desiguales.

`_elf_f1f2` devuelve **ifault=3** ("non-stationary") sobre estos modelos, aunque
todas las raíces de det(Φ(B)) estén fuera del círculo unidad. El origen está en
`cgamma` (port de `elfvarma.c:cgamma`), que devuelve fallo cuando el sistema de
Yule-Walker para las autocovarianzas sale singular.

La prueba de que no es degradación numérica: perturbar el cero con **1e-8** hace
que el mismo modelo pase. El fallo es exactamente en el caso singular.

ES DEL PORT, NO DEL C — comprobado (2026-07-29)
-----------------------------------------------
El `drtran` compilado evalúa sin problema el cast EMPOTRADO, que produce Φ_p
singular por construcción, sobre el mismo caso canónico:

    drtran ES_CPI_m10.pre WTI_ar1.pre -b 0 -r 1 -s 0 -V   ->  logL = -721.801539
    drtran ES_CPI_m10.pre WTI_ar1.pre -b 0 -r 0 -s 1 -V   ->  logL = -718.287406
    drtran ES_CPI_m10.pre WTI_ar1.pre -b 1 -r 1 -s 1 -V   ->  logL = -756.602851

Los tres tienen órdenes de fila distintos ⇒ Φ_p de rango deficiente, y el C los
estima. La ruta pure-Ython devuelve ifault=3 sobre la misma estructura. El
arreglo va, por tanto, en `_as311.cgamma` (o en cómo se monta el sistema de
Yule-Walker), no en el algoritmo.

Impacto: bloquea el cast empotrado del puerto de drtran — que es el cast POR
DEFECTO del C — y cualquier VARMA con órdenes desiguales por ecuación, incluida
la forma echelon. El cast por resta no está afectado (se validó contra el C a
1e-7 en cuatro combinaciones de b/r/s).
"""
import os

import numpy as np

os.environ.setdefault("DRVARMA_NO_ENGINE", "1")

from drvarma.estimate_py import _elf_f1f2  # noqa: E402


def raices_ar(phi):
    """Módulos de las raíces de det(Φ(B)) = 0, por la matriz compañera."""
    phi = np.asarray(phi, float)
    p, m, _ = phi.shape
    comp = np.zeros((p * m, p * m))
    comp[:m] = np.hstack([phi[k] for k in range(p)])
    if p > 1:
        comp[m:, :-m] = np.eye((p - 1) * m)
    ev = np.abs(np.linalg.eigvals(comp))
    ev = ev[ev > 1e-12]
    return np.sort(1.0 / ev)              # raíces de det(Φ(B)), no autovalores


def probar(tag, phi, theta=None):
    rng = np.random.default_rng(0)
    n, m = 215, 2
    w = rng.normal(0, 1, (n, m))
    phi = np.asarray(phi, float)
    theta = np.zeros((0, m, m)) if theta is None else np.asarray(theta, float)
    f1, _f2, ifa = _elf_f1f2(w, np.zeros(m), phi, theta, np.eye(m), -1e-3)
    r = raices_ar(phi)
    est = "estacionario" if (len(r) == 0 or r.min() > 1.0) else "NO estacionario"
    sing = "singular" if abs(np.linalg.det(phi[-1])) < 1e-14 else "no singular"
    print(f"  {tag:34} Phi_p {sing:12} raiz|min|={r.min():6.3f} ({est:15}) "
          f"-> ifault={ifa}")
    return ifa


if __name__ == "__main__":
    print(__doc__.split("El problema")[0].strip())
    print()
    print("Todos estos modelos son ESTACIONARIOS (raiz minima > 1):")
    print()
    bad = 0
    # (1-0.7B+0.12B^2)(1-0.3B) = (1-0.4B)(1-0.3B)(1-0.3B)
    bad += probar("Phi2 fila 2 nula",
                  [[[0.7, 0], [0, 0.3]], [[-0.12, 0], [0, 0.0]]]) != 0
    bad += probar("Phi2 fila 1 nula",
                  [[[0.7, 0], [0, 0.3]], [[0.0, 0], [0, -0.05]]]) != 0
    bad += probar("Phi2 de rango completo",
                  [[[0.7, 0], [0, 0.3]], [[-0.12, 0], [0, -0.05]]]) != 0
    print()
    print("La discontinuidad: el mismo modelo con el cero perturbado")
    for e in (1e-2, 1e-6, 1e-8, 1e-12, 0.0):
        probar(f"Phi2[1][1] = {e:g}",
               [[[0.7, 0], [0, 0.3]], [[-0.12, 0], [0, e]]])
    print()
    print(f"=> {bad} modelo(s) estacionario(s) rechazado(s) con ifault=3.")
    print("   Un VARMA no balanceado (ordenes distintos por ecuacion) SIEMPRE")
    print("   tiene Phi_p singular. Esto bloquea el cast empotrado de drtran y")
    print("   cualquier forma echelon con indices de Kronecker desiguales.")
