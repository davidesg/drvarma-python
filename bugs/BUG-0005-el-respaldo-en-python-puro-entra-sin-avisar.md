---
id: BUG-0005
title: Donde no hay rueda binaria se instala la de Python puro y el motor C no se usa, sin ningún aviso
status: open
severity: medium
component: packaging
found_in: 0.1.6
fixed_in:
reported: 2026-09-12
reporter: David / Claude — revisión de las ruedas de la suite tras publicar art 0.2.1
tags: [pure-python, ruedas, aviso]
references: [BUG-0004]
---

## Summary

drvarma 0.1.6 publica ruedas binarias para cp310–cp313 en Linux (x86_64,
aarch64, glibc y musl), macOS 14 arm64 y Windows amd64, **y además una rueda
`py3-none-any`**. pip instala esa en cualquier sistema sin rueda binaria, y ahí
`_engine.estimate_w` cae a `estimate_w_py` en un `except ImportError:` sin decir
nada.

## Impact

Los resultados son los mismos (BUG-0004: el porte es fiel), así que no es un
número mal. Pero el usuario no sabe que está en el camino lento —entre 20 y 100
veces— ni por qué su análisis tarda. Afecta a:

- **Mac con Intel** (sólo hay rueda arm64);
- **macOS anterior al 14** (la arm64 exige 14.0);
- **Python 3.14** (hay ruedas hasta cp313);
- **Windows sobre ARM**.

**fue tiene exactamente la misma forma** (rueda `py3-none-any` y respaldo
silencioso en `fue/_engine.py`); se anota aquí por decisión del analista, y habría
que llevarlo también a fue.

## Reproduction

Simulando lo que pasa de verdad —que el `.so` no carga—, sin tocar la variable
de entorno (con `DRVARMA_NO_ENGINE` el respaldo lo pide el usuario y ahí el
silencio es correcto):

    python -W always -c "import sys; sys.modules['drvarma._drvarma_engine'] = None; \
        import numpy as np, drvarma._engine as E; \
        print(E.estimate_w(np.random.default_rng(0).standard_normal((200,2)), 1, 0)['logelf'])"

Estima con el porte en Python y no emite ningún aviso.

## Root cause

El respaldo se diseñó para que el paquete funcione sin compilador, y eso está
bien; lo que falta es decirlo.

## Fix

Un aviso único por proceso cuando se cae al respaldo —«drvarma: sin motor C
(`_drvarma_engine` no carga); se usa el porte en Python, mismos resultados, más
lento»—, que no salte si el usuario lo pidió con `DRVARMA_NO_ENGINE`. Y valorar
publicar ruedas para cp314 y macOS x86_64.

## Validation

Un test que simule la ausencia del `.so` y compruebe que el aviso sale una vez, y
que no sale con `DRVARMA_NO_ENGINE`.
