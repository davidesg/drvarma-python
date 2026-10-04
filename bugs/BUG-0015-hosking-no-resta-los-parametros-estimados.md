---
id: BUG-0015
title: El Q de Hosking usa gl = m²·s sin restar los parámetros ARMA estimados, así que casi nunca rechaza y da por buenos modelos cortos
status: open
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
