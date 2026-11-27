---
id: BUG-0002
title: The C sources are three unsynchronised COPIES, and they have already drifted — the termcode fix lives in the embedded copy only
status: open
severity: high
component: csrc
found_in: 0.1.3
fixed_in:
reported: 2026-08-12
reporter: Revisión de defectos del ecosistema
tags:
  - csrc
  - copy-drift
  - structural
  - c-engine
references:
  - csrc/internal/qnewtopt.c (tiene `qn_last_termcode` / `qn_last_nit`)
  - drvarma_v.04.1/src/qnewtopt.c (NO los tiene — 17 líneas de diferencia)
  - drtran/src/nlatools.c (la tercera copia)
  - TODO.md ("they are COPIES, and a fix in one does not travel by itself")
  - BUG-0001 (el mismo patrón: arreglado en el port, vivo en el C)
---

## Summary

El motor C existe **tres veces**: `csrc/internal/` (empotrada en el paquete Python),
`drvarma_v.04.1/src/` (el ejecutable autónomo) y `drtran/src/`. No hay mecanismo que
las sincronice: la regla es que quien toque una propague a mano y lo diga en el
mensaje del commit.

**Ya han derivado, y se ha medido hoy:**

| fichero | líneas distintas | naturaleza |
|---|---|---|
| `elfvarma.c` | 0 | — |
| `drvmlest.c` | 0 | — |
| `multshea.c` | 0 | — |
| `nlatools.c` | 8 | solo la cabecera de identidad; **benigna, por diseño** |
| `qnewtopt.c` | **17** | **divergencia real de código** |

La de `qnewtopt.c` es el registro de `qn_last_termcode` / `qn_last_nit` — el arreglo
de "el motor C no devuelve termcode/nit, así que la diagnosis de convergencia está
inerte". **Se aplicó a la copia empotrada y no viajó al C autónomo.** El ejecutable
sigue con la diagnosis de convergencia inerte.

## Impact

Alto y estructural, porque no es un defecto sino una fábrica de defectos.

Ya ha costado **dos diagnósticos completos del mismo bug de heap en `nlatools.c`**:
fue lo arregló el 3 de julio, drvarma lo redescubrió el 28 y lo atribuyó mal. Y hoy
produce un segundo caso: dos ejecutables de la misma versión nominal dan distinta
diagnosis de convergencia sobre el mismo ajuste.

Es además el motor del que depende `drtran` por el principio de «`elf` sin parches»,
que es inverificable si no está claro *qué* copia de `elf` corre.

## Reproduction

```bash
cd drvarma_source/drvarma
for f in nlatools elfvarma qnewtopt drvmlest multshea; do
    echo "$f.c: $(diff csrc/internal/$f.c ../drvarma_v.04.1/src/$f.c | grep -c '^[<>]') líneas"
done
```

Hoy: 8, 0, 17, 0, 0.

## Root cause

Tres copias y ningún control. La comprobación existe (un `diff`) y no la corre nadie
ni nada.

## Fix

Dos piezas, y la segunda es la que importa:

1. **Propagar el arreglo pendiente:** llevar `qn_last_termcode` / `qn_last_nit` a
   `drvarma_v.04.1/src/qnewtopt.c`. Solo REGISTRAN lo que `raxopt` ya calculó — ni
   criterio, ni anuncio, ni comportamiento numérico —, así que no toca el algoritmo
   publicado.
2. **Un test que corra el `diff`**, con lista explícita de excepciones permitidas
   (hoy: la cabecera de identidad de `nlatools.c`). Una divergencia no declarada
   falla la batería. Sin esto, propagar el arreglo de (1) solo repone el estado hasta
   la próxima vez.

La copia de `fue` es una limpieza SEPARADA — 33 funciones frente a 42, y su
disposición de matrices no es intercambiable —, así que necesita juicio propio y no
entra en el `diff` automático.

## Validation

El test de (2) pasa hoy solo después de aplicar (1). Es su propia comprobación.

## Update (2026-11-28): two copies now, and the drift is the `.inp` path's

`drvarma_v.04.1` is retired and drtran keeps its own engine: the copies are
two, atsw-gui's `engines/drvarma` (the C of record) and the embedded
`csrc/internal/` the wheel builds. The original symptom, an executable with
an inert convergence diagnosis, is solved in atsw-gui in its own way
(`opt_termcode`, `opt_iters`).

Measured against atsw-gui main today:

| file | lines different | what |
|---|---|---|
| `elfvarma.c` | 0 | the likelihood: the ladder's C (`elf_c`) is the C's |
| `drvmlest.c` | ~200 | the C's `.inp` path moved on: fdhess standard errors, the MA wall report, `objcfunc`'s boundary sentinel |
| `qnewtopt.c` | ~200 | the verdict recorded differently, the report deferred |
| `nlatools.c` | 34 | |

So the ladder (what sima uses) runs the C's likelihood, and its numbers are
homologated case by case (`tests/test_c_oracle.py`, `docs/HOMOLOGATION.md`).
What has drifted is `estimate_w`, the deprecated `.inp` path (`model.py`,
`forecast.py`), which the record pins by sha256 and declares pending; the
path goes in drvarma 6.0.

Two ways to close it: bring `drvmlest.c`/`qnewtopt.c`/`nlatools.c` to the
C's and regenerate the `.inp` path's references (its standard errors become
fdhess's), or leave it until 6.0 removes the path. A decision for David.
