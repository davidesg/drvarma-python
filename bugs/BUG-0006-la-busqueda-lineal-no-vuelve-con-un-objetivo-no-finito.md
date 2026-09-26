---
id: BUG-0006
title: La búsqueda lineal no vuelve nunca si el objetivo es NaN o infinito — el optimizador (C y Python) se queda girando, sin error ni salida
status: fixed
severity: high
component: optimizer
found_in: 0.1.6
fixed_in: 0.1.7
reported: 2026-09-26
reporter: David / Claude — trabajo pendiente de drvarma tras la prueba en frío de atsw 1.5.0
tags: [optimizer, lnsrch, cuelgue, nan, no-finito]
references:
  - csrc/internal/qnewtopt.c (lnsrch)
  - src/drvarma/_qnewt.py (lnsrch, el puerto)
  - csrc/internal/drvmlest.c (objcfunc)
  - src/drvarma/estimate_py.py (objcfunc del respaldo en Python)
  - drvarma_v.04.1/BUGS.md, «the line search never returns when the objective is NaN» (registrado allí el 2026-09-24, sin tocar src/)
  - drtran/src/drvmlest.c (la guarda que drtran puso en su objetivo)
  - tests/test_bug0006_lnsrch_no_finito.py
---

## Summary

`lnsrch` (Dennis-Schnabel A6.3.1) decide con comparaciones si acepta el paso de
prueba o lo abandona. Con un **NaN** todas son falsas: el paso ni se acepta ni se
abandona, `tlambda` sale NaN, luego `lambda` es NaN, `NaN < minlam` es falso, y el
bucle gira para siempre. Con un **infinito** se llega al mismo sitio un paso
después: el ajuste cúbico calcula inf/inf.

Está en las DOS implementaciones del optimizador que distribuye drvarma: la C
empotrada (`qnewtopt.c`, la que usa el motor por defecto) y su puerto en Python
(`_qnewt.py`), que además es **el optimizador de drtran** (`drtran/estimate.py`
importa `drvarma._qnewt.raxopt`).

## Impact

Alto: un cuelgue en vez de un error. drtran lo sufrió con datos reales —una
transferencia racional de memoria larga (δ ≈ 0.95) sobre 69 observaciones, hora
y media girando— y lo tapó en SU función objetivo (`if (!isfinite(f)) return
1.0`); el optimizador compartido seguía expuesto para cualquier otro llamador.

En drvarma, el objetivo del motor C (`objcfunc`) no tenía ninguna guarda: devuelve
`pow(π₁/π₁₀, m)·(π₂/π₂₀)`, y π₁ desbordado con π₂ subdesbordado da `inf·0 = NaN`.
El del respaldo en Python comprobaba f1 y f2, pero no el producto. No se ha
observado en una corrida real de drvarma; el camino existe.

## Reproduction

El `raxopt` real con un objetivo que es NaN (o inf) pasado un umbral:

```c
real f(real *x) { if (x[1] > 2.0) return NAN;          /* o INFINITY */
                  return 0.1*(x[1]-3)*(x[1]-3)/0.9; }
... raxopt(f, &fk, 1, x, b, 100, 1, 1e-6, 1e-8);
```

```python
def f(x): return float("nan") if x[1] > 2.0 else 0.1*(x[1]-3)**2/0.9
_qnewt.raxopt(f, 1, np.zeros(2), 100, 1e-6, 1e-8)
```

Con 0.1.6, las cuatro combinaciones (C/Python × NaN/inf) seguían en el bucle a
los 10 s (`timeout` → 124). Las sondas completas están en el test.

## Root cause

Un valor no finito no es un valor: es un punto inadmisible, y `lnsrch` lo trataba
como un número que comparar e interpolar.

## Fix

* **`lnsrch`, en C y en Python:** si el objetivo del punto de prueba no es finito,
  se acorta el paso (`λ ← 0.1·λ`) sin interpolar; si `λ < minlam`, se rinde como
  con un paso que nunca desciende (`retcode = 1`, vuelve al punto de partida y a
  su valor). La interpolación cúbica se usa sólo cuando ya hay un punto de prueba
  FINITO anterior (`haveprev`), porque tras un primer paso no finito
  `prelam = 0` dividía por cero —la misma variable de la que avisaba el
  compilador («may be used uninitialized»)—. **En un camino de valores finitos
  nada cambia:** `haveprev` es falso exactamente cuando `λ == 1` lo era.
* **El puerto en Python** divide `steptol/rellen` como C: con `rellen = 0`, `+inf`
  en vez de `ZeroDivisionError`.
* **Los dos objetivos** (`objcfunc` en C y en `estimate_py`) devuelven 1.0 —punto
  inadmisible, como un `ifault`— si el producto no es finito. En C es la línea de
  drtran.

Con el arreglo, las cuatro sondas terminan en el borde de la región admisible
(x ≈ 2) con un valor finito, en 92 evaluaciones las dos implementaciones.

## Validation

`tests/test_bug0006_lnsrch_no_finito.py`: C y Python × NaN e inf, cada uno en un
subproceso con límite de tiempo. Los cuatro fallan (por tiempo) sin el arreglo.
La suite completa comprueba que en el uso normal —valores finitos— no se mueve
ningún número.

**Pendiente:** llevar el mismo arreglo a las otras copias del optimizador —
`drvarma_v.04.1/src` (el ejecutable autónomo, donde está registrado) y
`drvec/src`— (BUG-0002).
