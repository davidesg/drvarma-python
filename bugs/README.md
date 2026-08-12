# Bug reports — drvarma (port de Python)

Un fichero Markdown por defecto (`BUG-NNNN-slug.md`), mismo formato que los
registros de `art-tseries` y `fue`. Plantilla en `TEMPLATE.md`.

**Creado el 12-ago-2026.** Hasta entonces `drvarma` era el único paquete de la
suite sin registro de defectos: lo que había vivía mezclado con tareas en
`TODO.md`, que es donde se encontraron estas entradas.

**3 informes, 3 abiertos.**

| id | estado | sev | componente | título |
|----|--------|-----|------------|--------|
| [BUG-0001](BUG-0001-deseason-phase-still-unpatched-in-the-c-engine.md) | open | high | deseason | El desfase de fase estacional está arreglado en el port y sigue vivo en el motor C, que usa el ejecutable autónomo |
| [BUG-0002](BUG-0002-c-sources-are-copies-and-have-already-drifted.md) | open | high | csrc | Las fuentes C son tres copias sin sincronizar, y ya han derivado |
| [BUG-0003](BUG-0003-seasonal-adjustment-applied-in-levels-before-box-cox.md) | open | medium | deseason | El componente estacional se resta en NIVELES, antes del Box-Cox |

## Qué NO va aquí

**El optimizador.** `raxopt` / `qnewtopt` son el algoritmo publicado y arbitrado
de Mauricio (JASA). Un cambio en sus criterios de parada no es un arreglo de bug:
es una modificación de trabajo publicado, y cambia qué modelos se declaran
convergidos en todo lo que se haya estimado con esta herramienta. Fichar eso como
defecto ya presupone la conclusión.

Va como **estudio** en `TODO.md` §«El criterio de parada del optimizador», con lo
que haría falta para sostener una afirmación así. Lo único que sí entra en este
registro es que la copia empotrada y la autónoma difieran (BUG-0002), porque eso
es deriva de copias y no una cuestión de criterio.

## Registros hermanos

- `drvarma_v.04.1/BUGS.md` — el motor C autónomo. Ya existía.
- `art-python/bugs/`, `fue/bugs/` — mismo formato.
- `drtran-python/docs/BUGS.md` — formato en prosa, distinto.
