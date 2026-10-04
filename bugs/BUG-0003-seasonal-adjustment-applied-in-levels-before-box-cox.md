---
id: BUG-0003
title: The seasonal component is estimated and subtracted in LEVELS, before the Box-Cox log, so a multiplicative pattern on a trending index is adjusted at the wrong scale
status: in-progress
severity: medium
component: deseason
found_in: 0.1.3
fixed_in: 0.2.0 (Python; the C pending)
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

## Medición (2026-10-04)

**El orden, comprobado en el código** (`Model._prepare` →
`deseason.deseasonalize_raw` → `transform.transform`), igual en el C y en el
puerto:

1. d=1 sobre los niveles en bruto, sin log (`raw[1:] − raw[:-1]`);
2. regresión de ∇y sobre los armónicos diferenciados, paso a dummies en
   NIVELES y resta de los niveles;
3. log (Box-Cox) y ∇^d para el modelo.

O sea: **d=1 → desestacionalización → log.** El orden correcto, decidido
por el analista, es **log → d=1 → desestacionalización**: el patrón se estima
sobre las diferencias del log y se resta en logs.

**La medición** (`tools/deseason_order.py`, IPC3 = IPC_ES, IPC_FR, IPC_DE,
2002–2019).

Razón de amplitudes estacionales, segunda mitad / primera:

| serie | niveles (razón) | estimada en niveles | estimada en logs |
|---|---|---|---|
| IPC_ES | 1,17 | 0,75 | 0,64 |
| IPC_FR | 1,13 | 0,88 | 0,78 |
| IPC_DE | 1,14 | 1,92 | 1,69 |

ACF(12) de los residuos de un VAR(3) con media (banda ±0,136):

| serie | niveles | logs |
|---|---|---|
| IPC_ES | +0,021 | −0,025 |
| IPC_FR | **+0,147** | +0,106 |
| IPC_DE | +0,122 | +0,119 |
| logelf | 141,5 | 151,6 |

**El diagnóstico de arriba se corrige.** El Summary suponía que la amplitud
crece con el nivel y que en logs se estabiliza. No es así: la amplitud cambia
mucho entre mitades en las dos escalas, y en direcciones opuestas según el
país. Pasar a logs solo la reescala por el factor del nivel. Lo que domina es
una estacionalidad que EVOLUCIONA, y eso un patrón determinista fijo no lo
sigue en ningún orden.

El orden correcto sí es algo mejor: IPC_FR vuelve dentro de la banda y la
verosimilitud sube ~10 (las dos en escala 100·log).

## Arreglo en Python (2026-10-04); el C, documentado y sin tocar

La decisión inicial fue documentar. El analista la corrigió: el log va
primero, como en art, y el sistema tiene que ser coherente con art. Las
previsiones se leen en los informes como TLVA y tasa mensual, y en logs las
dummies del mismo mes se cancelan exactamente en la tasa anual; en niveles,
no.

**El orden ahora: log → d=1 → desestacionalización.** Todo se hace en la
escala de Box-Cox (el log, con λ=0) y después se recupera el valor original:

- `deseason.deseasonalize(levels, lam, ...)`: Box-Cox, y sobre eso
  `deseasonalize_raw` (∇ y regresión armónica, resta en esa escala). Devuelve
  la serie desestacionalizada EN NIVELES (exp), así que el resto del pipeline
  no cambia, y las dummies en la escala de Box-Cox, sin `scale`.
- `deseason.reseasonalize(lev_des, dseas, lam)`: suma las dummies en esa
  escala y deshace el Box-Cox. `seasonal_path` da las dummies de los
  periodos previstos.
- Usan ese par: `Model.prepare` y `Model.forecast` (también las bandas, que
  suman la dummy antes de deshacer el log), `report.forecast_report`, el
  informe HTML (`report_forecast`), la previsión recursiva y la comprobación
  de la desestacionalización del MCP.

**Compatible con art, comprobado.** `art.seasonal_detection.detect_seasonality`
hace 100·log → d=1 → regresión armónica. Sobre IPC_ES, IPC_FR e IPC_DE, sus
dummies y las de drvarma (×100) coinciden hasta 1e-14.

**El C no se toca** (el analista trabaja en atsw-gui): `drvarma.c` y
`deseason.c` siguen desestacionalizando los niveles antes del log. Los 13
tests que comparan con el binario usando `-deseason` quedan marcados como
`xfail(strict=True)` en `tests/_bug0003.py`. Cuando el C cambie en
drvarma-v5 volverán a pasar, y el `strict` hará que fallen para recordar que
hay que quitar la marca. Los de paridad sin `-deseason` siguen byte a byte.

Validación: `tests/test_bug_0003_log_primero.py`.
- En una serie con patrón multiplicativo conocido, las dummies son el patrón
  en logs (a 2e-3).
- `reseasonalize` deshace `deseasonalize` (1e-12).
- Las dummies son las de art (1e-9).
- La TLVA de la previsión es exactamente la de los niveles
  reestacionalizados.

(Superado por lo anterior) **Decisión inicial: documentar y cerrar.** `Model(deseason=...)` es el
camino v1, que sima sustituye. sima lleva los armónicos dentro del `.pre` de
cada serie, donde fue los estima ya en la escala transformada. El orden queda
escrito en `deseason.py` y en `Model._prepare`. Si el C se rehace en la rama
drvarma-v5 (junto con BUG-0001, la fase), el orden a implementar es
log → d=1 → desestacionalización.

