---
id: BUG-0003
title: The seasonal component is estimated and subtracted in LEVELS, before the Box-Cox log, so a multiplicative pattern on a trending index is adjusted at the wrong scale
status: open
severity: medium
component: deseason
found_in: 0.1.3
fixed_in:
reported: 2026-07-28
reporter: David / ejercicio de pass-through del petróleo
tags:
  - deseason
  - box-cox
  - method
references:
  - src/drvarma/deseason.py (estimación sobre la diferencia simple de niveles)
  - src/drvarma/transform.py (el Box-Cox va DESPUÉS)
  - BUG-0001 (el defecto de fase, del que este quedó como nota separada)
---

## Summary

El componente estacional se estima sobre la **diferencia simple de niveles** y se
resta en niveles, y solo después `transform` aplica el logaritmo de Box-Cox.

Para un patrón estacional **multiplicativo** sobre un índice con tendencia, un ajuste
aditivo en niveles es la escala equivocada: la amplitud del patrón crece con el nivel
y una única dummy por subperiodo no puede seguirla. En el IPC de España, que va de 69
a 98 en la ventana, la amplitud al final del recorrido es del orden de 1.4 veces la
del principio, y la dummy es la misma.

Es un defecto de método, no de programación: la rutina hace correctamente lo que se le
pidió.

## Impact

Medio. Deja estacionalidad residual en la parte alta o baja del recorrido según dónde
caiga el ajuste medio, y esa residual entra en todo lo que viene después. No es
silencioso del todo — se ve en la ACF estacional de los residuos — pero se confunde
con facilidad con estacionalidad estocástica genuina, que es exactamente la decisión
que la desestacionalización pretende dejar limpia.

Importa más de lo que su severidad sugiere porque **la desestacionalización es la
decisión más consecuente de `sima`**: medida en el par del pass-through, mueve la
correlación contemporánea de 0.23 a 0.51, y la descomposición de varianza depende de
esa correlación.

## Reproduction

Estimar el patrón sobre la primera y la última mitad de `IPC_ES` por separado y
comparar amplitudes. La razón entre ellas debería ser ~1 si el patrón fuera aditivo
en niveles.

## Root cause

El orden de las operaciones: desestacionalizar y luego transformar. Para un patrón
multiplicativo el orden correcto es el inverso — en logaritmos un patrón
multiplicativo es aditivo, que es lo que la regresión de dummies sabe estimar.

## Fix

Desestacionalizar **después** de la transformación de Box-Cox. Cuidado: cambia los
números de todo el que use `deseason`, así que necesita su propia línea base medida
antes de tocarlo, y la comparación honesta es la ACF estacional de los residuos, no la
verosimilitud.

## Validation

- Serie sintética con patrón multiplicativo conocido sobre tendencia: recuperarlo con
  error decreciente respecto a hoy.
- En datos reales, la razón de amplitudes entre mitades debe acercarse a 1 después.
- Y la comprobación que importa: la ACF(12) de los residuos del modelo estimado.
