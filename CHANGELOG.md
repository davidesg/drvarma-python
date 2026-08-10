# Changelog — drvarma

Exact maximum-likelihood estimation, forecasting and diagnostics of multivariate
VARMA models (Mauricio 1995 JASA / 1997 AS 311), pure-Python with an optional
compiled C engine.

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
