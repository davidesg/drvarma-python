---
id: BUG-0004
title: El docstring de `estimate_w` dice que el respaldo en Python sólo hace q=0, y hace VARMA completo desde junio
status: open
severity: low
component: engine
found_in: 0.1.6
fixed_in:
reported: 2026-09-12
reporter: David / Claude — revisión de las ruedas de la suite tras publicar art 0.2.1
tags: [documentacion, pure-python]
references: [docs/PURE_PYTHON_PLAN.md, docs/DEVELOPER_GUIDE.md]
---

## Summary

`src/drvarma/_engine.py:18`, en el docstring de `estimate_w`:

> *«otherwise falls back to the pure-Python exact-ML estimator
> (`estimate_py.estimate_w_py`, **q=0 only**).»*

Es falso. El respaldo estima VARMA(p,q) completo, parte MA incluida, y
reproduce el motor C.

## Impact

Documentación, no números: nadie obtiene un resultado distinto. Pero engaña a
quien decide si puede fiarse del respaldo. En la revisión del 12-sep llevó a
afirmar que «el respaldo sólo sabe q=0» y a proponer como defecto algo que no lo
era; lo que lo deshizo fue medirlo.

## Reproduction

Con drvarma 0.1.6 instalado desde PyPI, mismos datos (VARMA(1,1) bivariante,
T=300), `_engine.estimate_w` (motor C) frente a `estimate_py.estimate_w_py`:

| modelo | Δ logelf | máx |Δ parámetro| | iteraciones C / Python | errores típicos |
|---|---|---|---|---|
| VAR(1) | 4,5e-13 | 4,0e-10 | 14 / 14 | ≤ 2,2e-4 relativo |
| VARMA(1,1) | 0 | 5,5e-10 | 17 / 17 | ≤ 1,4e-4 relativo |
| VMA(1) | 2,3e-13 | 1,1e-10 | 14 / 14 | — |

Σ coincide a ≤ 7,4e-11. Que las iteraciones coincidan una a una es la firma de
un porte fiel del optimizador; Python sólo es entre 20 y 100 veces más lento.

## Root cause

La línea la escribió `fcbc4a0` (25-jun, «P3 (q=0): pure-Python exact VAR
likelihood»), cuando el porte sólo tenía el VAR. Ese mismo día `836f22d` portó la
verosimilitud exacta AS 311 de Mauricio con la parte MA, y el docstring se quedó
como estaba. `docs/DEVELOPER_GUIDE.md` sí lo dice bien, y los cinco huecos de
`docs/PURE_PYTHON_PLAN.md` (G1–G5) constan cerrados.

## Fix

Una línea: *«…falls back to the pure-Python exact-ML estimator
(`estimate_py.estimate_w_py`), a faithful port of the C engine: same logelf and
estimates to ~1e-10, std errors to ~1e-4 (the finite-difference Hessian).»*

## Validation

Un test que fije la afirmación y no sólo el texto: VARMA(1,1) con el motor y sin
él (`DRVARMA_NO_ENGINE=1`), logelf igual a 1e-8 y el mismo número de
iteraciones.
