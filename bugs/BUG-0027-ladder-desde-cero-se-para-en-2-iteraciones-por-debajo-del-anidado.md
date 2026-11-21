---
id: BUG-0027
title: Ladder.fit desde el arranque cero se para a las 2 iteraciones (termcode 3) en un VAR(2) de ruta U, con logL por debajo del VAR(1) anidado y sin decir que el punto es imposible
status: fixed
severity: high
component: ladder
found_in: 0.3.4
fixed_in: 0.4.0
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

**The scale of the cross terms, not the penalty.** The first hypothesis of
this report (the objective returns exactly 1.0 = f(x0) at a rejected point,
so the search cannot tell a rejected trial from "no progress") was measured
and is **false**: no evaluation of the search is rejected in these fits. The
67 rejections seen belong to the finite-difference Hessian computed
afterwards.

What happens: the rate is in basis points (lambda = 1, factor 100) beside
prices in %. The stationary GS1 has variance 461.7 and the IPC's is 0.039.
The optimiser (raxopt) works with typical sizes of 1, so from zero cross
terms its first steps are badly scaled, the objective moves by about 5e-9,
and the line search gives up after two iterations. With GS1 at factor 1 the
same fit converges from zero.

The same happens with other systems of the same data: GS3 at p = 2 and
p = 3, FF at p = 3 (atsw-gui `engines/drvarma/tests/rutas`). The earlier
statement that only GS1 was affected was wrong.

## Fix

Neither the penalty nor the optimiser is touched (they are J. A. Mauricio's
and are not modified: atsw-gui `docs/MOTORES.md`). The start is.

- 0.3.5: `start="ols"`, the original drvarma's `init_varma` in the ladder
  (the C's `-start ols`, atsw-gui #199 and #200): least squares of the
  VAR(p) seed the cross AR terms and Q's covariances. From it the monetary
  VAR(2) reaches −1741.275477 in both engines.
- 0.4.0: **it is the default with q = 0**, as in the C (atsw-gui #207): the
  16 monetary VARs and the 29 golden models of the C's battery reach their
  optimum from it. With an MA the default stays zero: no start seeds the MA,
  and `ols` is not better than zero there.

## What remains: a known limit

Asking for `start="zero"` on series whose scales differ by orders of
magnitude still stalls. It is listed as a limit, not as a defect, in the C's
battery (`tests/rutas/limites.txt`), which checks that the default never
ends below the better of the two starts.

A second limit found with the default change: the stopping point depends on
the start, because the objective is normalised to f(x0) = 1. On the muskrat
and mink VAR(2) with one cross coefficient at zero, zero ends at
−559.001068 and `ols` at −559.002248, with the parameters equal to the third
decimal.

## Validation

`bugs/BUG-0027-repro/repro.py` exits 0: with the default start the VAR(2) is
above its nested VAR(1). The C's `tests/rutas` pins the four monetary
systems by route U from both starts and without `-start`.
