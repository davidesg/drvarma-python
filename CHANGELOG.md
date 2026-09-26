# Changelog — drvarma

Exact maximum-likelihood estimation, forecasting and diagnostics of multivariate
VARMA models (Mauricio 1995 JASA / 1997 AS 311), pure-Python with an optional
compiled C engine.

## 0.1.7 — 2026-09-26

**El optimizador se colgaba para siempre con un objetivo NaN o infinito**
(BUG-0006). La búsqueda lineal (`lnsrch`, Dennis-Schnabel) decide con
comparaciones si acepta o abandona el paso, y con un NaN todas son falsas: el
bucle giraba sin error ni salida. Un infinito llegaba al mismo sitio un paso
después. Estaba en el motor C empotrado y en su puerto en Python (`_qnewt`), que
es también el optimizador de drtran; drtran lo sufrió con datos reales —hora y
media— y lo había tapado sólo en su función objetivo. Ahora un valor no finito
es un punto inadmisible: el paso se acorta sin interpolar y la búsqueda se
rinde en el paso mínimo. **En el uso normal no cambia ningún número**: la suite
de drvarma y la de drtran, que optimiza con este código, dan los mismos
resultados.

**`chisq()` del motor C** (BUG-13 del conjunto): la copia empotrada era la única
de las cuatro vigentes sin el arreglo, y con 30 o más grados de libertad y el
estadístico por debajo de su media devolvía el complemento. Sin efecto en el
paquete Python —ningún código la llama; los p-valores salen de scipy—: es la
sincronización del fichero compartido.

**Empaquetado:** `pyproject.toml` volvió a ser TOML válido —en `master` el
paquete había dejado de poder construirse— y declara que trae el servidor MCP
`sima`.

**Pruebas:** diez tests de `test_regression_bugs.py` construían su JSON con el
`repr` de escalares de numpy, que con numpy 2 no es JSON: fallaban en toda
instalación limpia. Encontrado probando la 0.1.6 en contenedores limpios de
Python 3.10–3.13 y Alpine.

**Registro de defectos** (`bugs/`): nuevo, con BUG-0001…0006. Siguen ABIERTOS
BUG-0001 (la fase estacional en el motor C autónomo), BUG-0002 (las copias de
las fuentes C derivan), BUG-0003 (el ajuste estacional en niveles), BUG-0004
(un docstring) y BUG-0005 (el respaldo en Python entra sin avisar).

## 0.1.6 — 2026-08-10

Corrige el silenciado de 0.1.5: el aviso salta al CONSTRUIR FastMCP, no al
importarlo. Filtro a nivel de módulo, comprobado con -W always.

## 0.1.5 — 2026-08-10

Limpieza para una versión estable. Sin cambios en el motor ni en sima.

- **`__version__` se lee de los metadatos** en vez de una constante escrita a
  mano, que decía `0.1.1` con la 0.1.4 instalada.
- **Silenciado el aviso de `pydantic_settings` al arrancar sima**, acotado a ese
  aviso: no toca el protocolo, pero puede leerse como un fallo.

## 0.1.4 — 2026-08-10

Documentación y metadatos. Sin cambios en el motor ni en sima.

- **`docs/TOOLS.md`, generado de los docstrings**: las **15** herramientas de
  sima, que no tenían referencia. Documento e instrucción del modelo son el
  mismo texto por construcción.
- `[project.urls]` gana **Documentation** --el campo que PyPI muestra más
  arriba-- y **Changelog**, comprobados vivos antes de declararlos.
- El `MANIFEST.in` deja de distribuir `STATUS.md` y `PURE_PYTHON_PLAN.md`, que
  son notas de trabajo.

## 0.1.1 — 2026-07-27

Release-infrastructure homologation with the ATSW suite. No functional changes.

- First release built and published by **GitHub Actions trusted publishing**
  (OIDC, `publish.yml` on `v*` tags), matching `fue` / `art-tseries` / `atsw`.
  The 0.1.0 artifacts were uploaded by hand; 0.1.1 are CI-built and reproducible.
- Repository wired to `github.com/davidesg/drvarma`.

## 0.1.0 — 2026-06

Initial PyPI release. Pure-Python port of Mauricio's exact-likelihood VARMA
algorithm: `Model(series, p, q).fit()`, forecasting with error bands, impulse
responses, FEVD, residual diagnostics (Hosking Q, Jarque–Bera), volatility, HTML
reports, plots and a CLI. Optional CFFI C engine (`drvarma[c-engine]`).
