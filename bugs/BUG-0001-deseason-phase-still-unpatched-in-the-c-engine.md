---
id: BUG-0001
title: The seasonal-adjustment phase off-by-one is fixed in the Python port but STILL LIVE in the C engine, which the standalone executable uses
status: open
severity: high
component: deseason
found_in: 0.1.3
fixed_in:
reported: 2026-07-28
reporter: David / ejercicio de pass-through del petróleo
tags:
  - deseason
  - c-engine
  - silent-failure
  - copy-drift
references:
  - drvarma_v.04.1/src/deseason.c:82-90 (el defecto, con el comentario que lo declara)
  - drvarma_v.04.1/BUGS.md (causa raíz, reproducción y el parche de un bucle)
  - src/drvarma/deseason.py (el arreglo, ya aplicado en el port)
  - tests/test_regression_bugs.py (36 tests de fase + invariancia al recortar cabecera)
  - BUG-0002 (la deriva entre copias, de la que esto es un caso)
---

## Summary

`deseasonalize_raw` estima las amplitudes con un diseño indexado por **desplazamiento
desde el inicio** de la serie (`t = i + d + 1`, sin `start_sub`) y luego las aplica en
**fase de subperiodo absoluto** (`(i + start_sub - 1) % s`). Las dos indexaciones solo
coinciden cuando `start_sub == 1`. Para cualquier otra, el patrón se resta desplazado
`start_sub - 1` meses y **el ajuste añade varianza estacional en vez de quitarla**,
sin un solo aviso.

**Está arreglado en el port de Python desde el 28-jul y sigue vivo en el C**
(`deseason.c:82-90`), que es el que usa el ejecutable autónomo (`src/drvarma.c:356`).
El propio fichero C lleva el comentario que lo declara — y ese comentario es la única
barrera hoy entre el defecto y un usuario.

## Impact

Alto y silencioso. Contamina la tubería entera: semilla de caracterización, CCM,
Tiao-Box, búsqueda de orden, estimación, IRF y FEVD. En el ejercicio que lo destapó
produjo ACF(12) ≈ 0.7 en los residuos de la ecuación del IPC, una persistencia AR(1)
de −0.707 para la inflación alemana y un coeficiente de −12.7 sobre el IPC retardado
en la ecuación del WTI de EE.UU. — todo con aspecto de problema de modelización y
nada de ello real.

Medido, autocorrelación en el retardo 12 de `dlog(IPC)`:

| serie | sin ajustar | ajustado, start_sub=2 | ajustado, start_sub=1 |
|---|---|---|---|
| US | +0.328 | +0.245 | **−0.140** |
| ES | +0.798 | **+0.866** | **+0.098** |
| FR | +0.728 | **+0.862** | **+0.250** |
| DE | +0.608 | **+0.808** | **+0.177** |

Las columnas del medio son peores que no hacer nada.

**Alcance real acotado:** `start_sub` vale 1 por defecto y la mayoría de las series
empiezan en enero, así que la nota de pass-through publicada no está afectada. Eso
explica por qué sobrevivió, no por qué puede quedarse.

## Reproduction

`bench/repro_multiart_passthrough.py` (BUG 4), con series que empiezan en febrero.
Barrer `start_sub` de 1 a 12 pone el mínimo de |ACF(12)| en **1 para los cuatro
países**, donde la salida coincide con una regresión OLS de dummies mensuales a ~0.04.
El algoritmo armónico es correcto; la convención de fase entre llamador y llamado
está desplazada en uno.

## Root cause

`harmonic_regression_differenced_basis` no recibe `start_sub`. Su matriz de diseño se
construye desde `t = i + d + 1`, así que `level` sale indexado por desplazamiento
desde el inicio; se aplica en fase absoluta.

## Fix

El mismo que en el port: rotar las dummies estimadas a indexación de subperiodo
absoluto antes de guardarlas o aplicarlas —
`level[(np.arange(s) + start_sub - 1) % s] = level_rel` en Python, un bucle
equivalente en C. Es la identidad en `start_sub == 1`, así que los goldens de paridad
con el C quedan intactos.

## Validation

En el port se verificó de dos maneras y las dos valen para el C: (a) serie sintética
con patrón conocido arrancada en cada uno de los 12 subperiodos, que ahora lo recupera
con error < 0.1; (b) datos reales de IPC, donde estimar desde enero frente a febrero
cambiaba el patrón entre el 36 % y el 83 % de su amplitud antes del arreglo y menos
del 1 % después.

## Nota separada, de método y no de fase

El componente estacional se estima sobre la **diferencia simple de niveles** y se
resta en niveles, **antes** del logaritmo de Box-Cox en `transform`. Para un patrón
estacional multiplicativo sobre un índice con tendencia (el IPC de España va de 69 a
98) un ajuste aditivo en niveles es la escala equivocada. Ver BUG-0003.
