---
id: BUG-0027
title: Ladder.fit desde el arranque cero se para a las 2 iteraciones (termcode 3) en un VAR(2) de ruta U, con logL por debajo del VAR(1) anidado y sin decir que el punto es imposible
status: open
severity: high
component: ladder
found_in: 0.3.4
fixed_in:
reported: 2026-11-19
reporter: David / Claude — rodaje de sima 0.2.5, VAR monetario de EE. UU. 1990–2007 (IPI, IPC, WTI, GS1)
tags:
  - ladder
  - optimiser
  - raxopt
  - start
  - escala
references:
  - src/drvarma/ladder.py Ladder._estimate — objective(): «if ifa: return 1.0» / «f1 <= 0 ... return 1.0»; f normalised so that f(x0) = 1
  - src/drvarma/ladder.py Ladder.fit — start "zero" (cross terms and _qcov at 0)
  - src/drvarma/_qnewt.py raxopt — termcode 3, «last global step failed to locate a lower point»
  - sima-python bugs/BUG-0019 (the cache of estimate hid the preliminary start; sima now catches the symptom)
  - atsw-gui engines/drvarma/src/drvmlest.c objcfunc: «return( 1.0 )» on ifault and on a non-finite f (the C does the same)
  - bugs/BUG-0027-repro/repro.py
---

## Summary

En ruta U, un VARMA(2,0) con IPI, IPC, WTI y GS1 (los `.pre` de art) estimado
con `Ladder(files, 2, 0)` y el arranque por defecto (cero) se comporta así:

- **El optimizador se para a las 2 iteraciones** con termcode 3: «last global
  step failed to locate a lower point».
- **Todos los coeficientes cruzados quedan en ≈ 1e-6** y la covarianza de las
  innovaciones sale diagonal (Q[i,j] ≈ 0).
- **El logL es −1793,05.** Está por debajo del VAR(1) anidado (−1744,88) y
  apenas por encima del sistema diagonal. Un modelo que anida a otro no puede
  quedar por debajo de él.
- **La matriz hessiana por diferencias no es definida positiva.** Los errores
  estándar vienen de la BFGS inicial: casi todos valen 0,095075 y algunos son
  absurdos, como 355,0 o 559,6.

El mismo modelo tiene su óptimo en −1741,2755, y se llega a él de dos maneras:

| camino | logL | iteraciones | termcode |
|---|---|---|---|
| elf, arranque cero (por defecto) | −1793,0545 | 2 | 3 |
| elf, `start="preliminary"` | −1741,2755 | 152 | 2 |
| shea (AS 242), arranque cero (`study_estimation` de sima) | −1741,2755 | 167 | — |

Con FF, GS2 o GS3 en lugar de GS1, el mismo VAR(2) converge desde cero, y de
los dos arranques sale el mismo logL.

**Segundo caso: FF con p = 3.** El arranque cero se para a las 2 iteraciones
(termcode 3, Hessiana no definida positiva) con logL −1725,79, por debajo del
VAR(2) anidado (−1666,33). Desde `start="preliminary"` llega a −1659,89 en
183 iteraciones, con termcode 1 y fdhess. El fallo no es propio de GS1:
aparece cuando crece el número de parámetros (46–58).

Resto de los VAR(3), los dos arranques en el mismo punto:

| tipo | logL |
|---|---|
| GS1 | −1735,95 |
| GS2 | −1771,40 |
| GS3 | −1778,30 |

## Impact

Grave. El `Fit` vuelve sin error y con un informe aparentemente normal. sima lo
presenta así:

> «The cross terms are NOT significant in sample against the diagonal system
> (LR = 6.17, df = 30, p = 1.0000)»

Es decir, la conclusión opuesta a la verdadera: LR ≈ 103 frente al diagonal y
LR = 7,2 frente al VAR(1). Cualquier selección de orden o test de Wald sobre
ese ajuste es falso.

El único rastro es el termcode 3 con nit = 2. sima lo traduce como «the last
step found no lower point», un aviso que también aparece en ajustes buenos (FF
p = 2: termcode 3 tras 159 iteraciones, en el óptimo).

## Reproduction

```
cd drvarma-escala
python3 bugs/BUG-0027-repro/repro.py    # sale con 1 mientras siga abierto (~5 min)
```

El script ajusta tres modelos y sale con 1 si logL(p = 2) < logL(p = 1):

- p = 1 con elf;
- p = 2 con elf;
- p = 2 con shea.

Salida, drvarma 0.3.4:

```
p=1 lik=elf   start=zero         logL -1744.8766  nit 114  termcode 2  se fdhess  (76s)
p=2 lik=elf   start=zero         logL -1793.0545  nit   2  termcode 3  se bfgs (fdhess: the Hessian is not positive definite)  (13s)
p=2 lik=shea  start=zero         logL -1741.2755  nit 167  termcode 3  se bfgs (fdhess: the Hessian is not positive definite)  (342s)
VAR(2) elf below the nested VAR(1): True
Shea reaches above elf by 51.78
```

Note: from zero, Shea reaches the optimum, but its finite-difference Hessian
also fails to come out positive definite. From `start="preliminary"`, elf
converges with `se fdhess`, so with a valid Hessian. Both point to a poorly
conditioned likelihood surface in this system.

## Root cause

Hipótesis, en parte comprobada (19-nov-2026):

- **El C hace lo mismo.** `drvmlest.c` (`objcfunc`) devuelve `1.0` cuando
  `ifault > 0` y cuando f no es finita: «the article's strategy, sec. 3».
  El porte es fiel; no es una divergencia del porte, sino un defecto latente
  del motor homologado. Se arregla primero en el C (con confirmación del
  analista) y los dos se mueven juntos.
- **Contadas las evaluaciones del ajuste que falla:** de 1830 llamadas, 67
  fallan en `_elf` (ninguna en `cast`). Encaja con la hipótesis, pero falta
  ver que caen en la búsqueda lineal de las dos primeras iteraciones y no en
  el hessiano final.

1. **La función objetivo es plana para el optimizador.** `objective` está
   normalizada de modo que f(x0) = 1 y devuelve **exactamente 1.0** cuando
   `cast` o `_elf` fallan (`ifa`, o f1/f2 no finitos o ≤ 0). Si los primeros
   pasos de prueba desde x0 caen en puntos que fallan, el optimizador ve el
   mismo valor que en el arranque y la búsqueda lineal no encuentra un punto
   más bajo. Por ejemplo:
   - una raíz AR fuera del círculo, lo que es fácil con 46 parámetros y una
     escala muy dispar (GS1 en pb, con varianza ≈ 345, frente a IPC en
     100·log, con varianza ≈ 0,03);
   - un fallo de elf.
2. **Por qué solo con GS1.** GS1 es el único tipo cuyo univariante es AR(3) con
   φ₂ < 0. Junto con el factor estacional del IPI, la primera dirección
   cuasi-Newton (BFGS = identidad escalada) puede llevar fuera de la región
   admisible.
3. **Cómo comprobarlo:** contar las llamadas a `objective` que devuelven el
   1.0 de penalización en las dos primeras iteraciones.

## Fix

Propuestas, de menor a mayor alcance:

1. **Penalización que no empate con el arranque.** Que `objective` devuelva un
   valor mayor que f(x0) en un punto inadmisible, por ejemplo 1e10 o
   `2 * max(1, f_actual)`, para que la búsqueda lineal retroceda en lugar de
   rendirse.
2. **Detectar el atasco.** Si termcode = 3 con nit ≤ 5, reintentar desde
   `start="preliminary"` y quedarse con el mayor logL, diciéndolo en
   `start_used`.
3. **Avisar en el `Fit`/informe** cuando nit sea muy pequeño y se pare por
   termcode 3: «the optimiser did not move from its start».
4. **(sima) Contrastar contra el anidado.** Hecho en sima (BUG-0019,
   0.2.5): un ajuste por debajo de un anidado ya ajustado, o atascado
   (termcode 3 con nit ≤ 5), se reintenta desde el arranque preliminar y el
   informe lo dice. Es un parche de la capa del asistente; el defecto del
   motor sigue aquí, abierto, y su arreglo empieza en el C.

## Validation

- `repro.py` sale con 0: logL(p = 2, elf, por defecto) ≥ logL(p = 1), y en
  este caso ≈ −1741,28.
- Test de regresión con estos cuatro `.pre`: el VAR(2) desde cero alcanza el
  mismo logL que el arranque preliminar y que shea, con tolerancia 1e-3.
- Comprobar que FF, GS2 y GS3 no cambian. Valores de referencia del VAR(2):

| tipo | logL |
|---|---|
| FF | −1666,3269 |
| GS2 | −1778,2918 |
| GS3 | −1786,7790 |
