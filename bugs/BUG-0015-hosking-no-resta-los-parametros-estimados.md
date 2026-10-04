---
id: BUG-0015
title: El Q de Hosking usa gl = m²·s sin restar los parámetros ARMA estimados, así que casi nunca rechaza y da por buenos modelos cortos
status: in-progress
severity: high
component: diagnostics
found_in: 0.1.0
fixed_in:
reported: 2026-10-04
reporter: David / Claude — al decidir BUG-0012 (los armónicos tampoco cuentan)
tags: [hosking, portmanteau, grados-de-libertad, sima, motor-c]
references:
  - BUG-0012
  - src/drvarma/diagnostics.py (hosking_q)
  - src/drvarma/model.py (Model.diagnostics)
  - src/drvarma/report.py
  - sima-python/src/sima/evidence.py (estimation_text)
  - atsw-gui/engines/drvarma/src/diagnose.c:930 (hosking_test)
  - tools/hosking_size.py
---

## Summary

`diagnostics.hosking_q(res, s)` devuelve `df = m²·s`, y todos los que lo llaman
usan ese gl tal cual, sobre los residuos de un modelo ESTIMADO. Hosking (1980)
da para un VARMA(p,q) ajustado `m²·(s − p − q)`; en general, `m²·s − k` con
`k` el número de coeficientes ARMA estimados. Con gl de más, el p-valor sale
alto: el contraste casi nunca rechaza y el veredicto «sin autocorrelación» /
«white noise not rejected» sale de más.

Lo mismo en el C: `diagnose.c:930`, `int df = m * m * s;`. El porte en Python
lo heredó fielmente, así que es un defecto del motor, no del porte.

## Impact

Alto, porque es EL contraste de adecuación del sistema: «un Q significativo
significa que el orden (p o q) es demasiado bajo», dice `diagnose`. Con estos
gl, un orden demasiado bajo pasa más de lo debido.

Afecta a:
- `Model.diagnostics` → `diagnose` del MCP de drvarma y `report.py`;
- sima, `evidence.estimation_text`: ahí `k` serían los coeficientes cruzados y
  los ARMA univariantes de la diagonal, y no se resta ninguno;
- el motor C (`hosking_test`).

## Reproduction

`python tools/hosking_size.py`: 400 réplicas por fila, sistema bivariante
(m=2), el modelo verdadero ajustado con `estimate_w` (motor C), Q de Hosking
al 5 % nominal.

| modelo | n | s | Q medio | tamaño con m²·s | tamaño con m²·(s−p−q) |
|---|---|---|---|---|---|
| VAR(1) | 150 | 6 | 19.63 | 0.007 (gl 24) | 0.037 (gl 20) |
| VAR(1) | 300 | 12 | 42.97 | 0.015 (gl 48) | 0.045 (gl 44) |
| VARMA(1,1) | 150 | 6 | 15.64 | 0.003 (gl 24) | 0.043 (gl 16) |
| VARMA(1,1) | 300 | 12 | 38.59 | 0.003 (gl 48) | 0.030 (gl 40) |

El Q medio sigue a `m²·(s−p−q)`, no a `m²·s`. Con el gl actual, el contraste
del 5 % rechaza entre el 0,3 % y el 1,5 % de las veces.

## Root cause

`hosking_q` no sabe cuántos parámetros se estimaron, y nadie se lo dice. En el
C, igual.

## Fix

`hosking_q(res, s, k=0)` con `df = m²·s − k`, y que cada llamador pase su `k`:
- `Model`: `m²·(p + q)`, o los libres si hay restricciones diagonales;
- sima: los coeficientes ARMA libres del sistema (cruzados y de la diagonal),
  sin los de la covarianza ni las medias.

Decidir aparte si los armónicos de BUG-0012 entran en `k`: a primer orden, los
coeficientes de regresores deterministas no cambian la distribución asintótica
de las autocorrelaciones residuales, así que en principio no. Sí cuentan en los
criterios de información y en las bandas.

El C, en la rama drvarma-v5 de atsw-gui.

## Validation

`tools/hosking_size.py` con el gl corregido: tamaño real cerca del 5 % en las
cuatro filas. Un test que fije `df = m²·(s−p−q)` en `Model.diagnostics` y en sima.

## Arreglo en Python (2026-10-04); el C, pendiente

- `diagnostics.hosking_q(res, s, k=0)`: `df = m²·s − k`; con `df < 1` el
  p-valor es `nan` (hay que subir s). `qccf(w1, w2, lags, k=0)` igual. Con
  `k=0` (una serie, no residuos) no cambia nada.
- `Model.n_arma()`: `p·(m o m²) + q·(m o m²)` según las restricciones
  diagonales. Lo usan `Model.diagnostics` (y con él el `diagnose` del MCP) y
  el `.out` de `report.py`, que ahora imprime `Q(m²·(s−p−q))`.
- La escalera: `LadderSeries.n_arma()` (los ARMA propios de cada `.pre`, que
  cuentan aunque la escalera los tenga fijos: se estimaron con estos datos) y
  `Ladder.n_arma()` (los de las series activas, más los cruzados libres).
  sima los pasa en `estimation_text` (N3) y, por pareja, en
  `cross_identification` (N2).
- No se toca la figura de la CCF residual de sima: su P es el de GraphMaker,
  `4(K − (p + q))`, con p y q los órdenes máximos de los operadores. Es la
  fórmula de Hosking para matrices completas, legado de Treadway, y ya
  restaba.

Validación: `tests/test_bug_0015_hosking_gl.py`. Comprueba el `k` en
`hosking_q` y `qccf`, el p-valor `nan` sin grados de libertad, y el recuento de
la escalera (IPC_ES + IPC_FR, VARMA(1,1) cruzado: 1 + 1 + 4 = 6, también con
`fixarma`). Y el tamaño: 150 VARMA(1,1) simulados con el modelo verdadero,
`Model.diagnostics` rechaza al 5 % entre el 1 % y el 10 %; antes, el 0,3 %.

Al escribir ese test apareció otra cosa: `Model(lam=1.0, include_mean=False)`
sobre datos de media cero estima peor (logelf 11 unidades por debajo de
`estimate_w`), porque λ=1 es `y − 1` y el modelo sin media no lo recoge. Es el
convenio del C (drtran BUG-22) y no es este defecto; el test estima con media.

**Un caso real.** El VAR(3) de IPC3 (m=3, n=216, desestacionalizado), el
mismo de los tests de paridad con el C:

    C  :  Q(126) = 151.2169, p-value = 0.0624   Cannot reject H0: white noise
    Py :  Q(99)  = 151.2169, p-value = 0.0006   *** REJECT H0

El mismo Q, y el veredicto se invierte: con 27 coeficientes AR estimados, el
C daba por ruido blanco unos residuos que no lo son.

Los tests que comparan el `.out` byte a byte con el binario
(`test_report.py`, `test_pure_python_out.py`) declaran esa divergencia en
`tests/_bug0015.py`: exigen el mismo Q y el gl del C menos k, quitan esa
línea y su veredicto, y el resto de la sección sigue byte a byte. Se retira
cuando el C se arregle.

Queda el C (`diagnose.c:930`), en la rama drvarma-v5 de atsw-gui. Por eso el
informe sigue `in-progress`.

