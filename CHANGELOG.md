# Changelog — drvarma

Exact maximum-likelihood estimation, forecasting and diagnostics of multivariate
VARMA models (Mauricio 1995 JASA / 1997 AS 311), pure-Python with an optional
compiled C engine.

## 0.2.0 — unreleased

**Jenkins and Alavi's (1981) identification statistics**, `drvarma.identification_mv`:
the correlation matrices R_k (Bartlett's standard errors, or 1/sqrt(n) for
prewhitened series), the partial correlation matrices S_k by the multivariate
Yule-Walker equations, Alavi's q-conditioned S_k(q), their determinants and the
+ - . table. Checked: with one series S_k is the PACF (6e-16); S_k cuts off
after 2 on a VAR(2), R_k after 1 on a VMA(1), and on a VARMA(1,1) S_k(1) cuts
off after 1 and S_1(1) estimates Phi_1 while S_k does not cut off. Phase 1 of
sima's `docs/DESIGN-jenkins-alavi.md`, Python first.

**The bug register ships inside the package** (`drvarma/material/bugs/`,
copied by `tools/sync_material.py`, which the publish workflow runs before each
build) and `drvarma.register.bugs_dir()` finds it, installed or in the working
tree. sima serves it as `sima://engine-defects`.

**`Ladder.irf_fevd_bands` also returns the FEVD band at every horizon**
(`fevd_lo_h`, `fevd_hi_h`, H x m x m), for sima's figure; `fevd_lo`/`fevd_hi`
(the last horizon) are unchanged.

**Restricted cross terms: `Ladder(links=)` and the CLI's `-links`**, as drvarma
C's `-links` (atsw-gui e08bb23). `links="A<-B, C<-A"` (or a list of pairs)
keeps the cross dynamics only on those pairs; the others are zero and are not
parameters, and `lr_test` counts only them. Every pair named is the
unrestricted model to 1e-10; one link on the ES/FR pair gives the C's
logL 81.114461 with 8 parameters, reached again from the full model's
optimum (`tests/test_links.py`).

**termcode 3, one reading, and the MA wall in `Model`** (BUG-0010). The package
read termcode 3 two ways: "AT the optimum" in `estimate_py`, "not a maximum"
in the report. The report now states the fact and how to tell an optimum from
a stall (re-estimate from the values). `Model.ma_boundary`/`ma_nroots` report
MA inverse roots within 5e-5 of the unit circle, as the ladder and the C;
`Model.converged` is False there, and the report writes the C's "OPTIMIZER
STOPPED at the MA invertibility boundary". BUG-0009, and the old server's half
of BUG-0008, are not patched: that server is `sima-legacy` from 0.2.0.

**The MA wall: one tolerance for both sides.** A stop with an MA inverse root
within 5e-5 of the unit circle is reported as on the wall, as in the C
(atsw-gui lib/lik MA_WALL_TOL). chekma already accepted up to 1 + 5e-5, so a
root at 1.000048 was reported and one at 0.99999999 was not, though both are
the same fact. The note now reads "within 5e-5 of the unit circle".

**Shea's likelihood in pure Python.** `_as242.py` ports multshea.c (AS 242)
line by line, as `_as311.py` ports elf. `marma_c` uses it when the compiled
engine is not built, so `Ladder(lik="shea"|"both")` also works without it. It
reproduces `marma_c` to 1e-14 relative on the logL, with the same refusals.

**The ladder: fue's univariate models as the input of a VARMA.** The Python
port of drvarma 5.0's ladder mode (C, in the atsw-gui monorepo), with the C as
the oracle. `drvarma.ladder` and the CLI:

    drvarma F1.pre F2.pre [...] p q [-diagcov] [-redet] [-fixarma] [-m M] [-o NAME]
                                    [-forecast H [-estwin N]]
    drvarma -split FILE[.inp] [-mean] [-harmonics] [-ar P] [-ma Q] [-scale F] [-dir D]

- **What it is.** Each series brings its univariate model from a fue file, a
  `.pre` (an optimum) or an `.inp` (a specification), recognised by its
  content. The model sits on the diagonal of the VARMA; p and q are the orders
  of the cross dynamics. Nothing about a univariate model is re-implemented:
  - `fue.load` reads the file;
  - `fue.cast_us.cast_us_py` gives each series its stationary series and
    its ARMA polynomials;
  - `fue.forecast`'s pieces take a forecast back to the level.
- **The diagonal gate stops the program.** Each series is fitted alone, and
  the diagonal system is evaluated at those optima: the two must agree
  exactly (`GateError`). BUG-2: series with different frequency or last
  date are refused.
- **Forecasting.** `forecast(L)` and, with `estwin`, fixed-parameter
  forecasts from every origin (`recursive(H)`), with an out-of-sample
  evaluation. The univariate models are the yardstick a VARMA has to beat.
- **One format.** `split` converts the multivariate `.inp`, now
  **deprecated**, into one fue `.inp` per series. The old CLI path prints a
  one-line note.
- **Parity with the C** (`tests/test_ladder.py`, 17 tests, ~7 s with the
  compiled engine):
  - the gate (1e-13), and σ² and coefficients equal to fue's `.out`;
  - the log-likelihood of the full-covariance, VAR(1), pass-through and
    `-fixarma` models to 1e-6;
  - fue's fixed-parameter forecasts, 3456 values within 2.3e-6;
  - the C's bands and recursive evaluation;
  - the ladder on `split` files equals the old path's VAR(1) with mean.
- **Speed.** The system's likelihood is the compiled `elf` (`elf_c`, in the
  binary wheels). The stationary series of each model is cached, and only
  its polynomials are recomputed at each evaluation.
- **New dependency: `fue>=0.1.16`,** the reference parser, with wheels.
- **Shea's exact likelihood beside elf: `Ladder(lik="elf"|"shea"|"both")`**
  (CLI `-lik`), and `drvarma._engine.marma_c`.
  - **What it is.** Shea's AS 242 (1989) had been in `csrc` since 1996,
    compiled and never called. It is the independent benchmark in
    Mauricio's papers.
  - **The same rules as the C engines' `-lik`** (atsw-gui `lib/lik`; the
    `csrc` copy of `multshea.c` is synced from there, with its underflow
    fix):
    - Shea is always exact;
    - the MA frontier is elf's;
    - the residuals stay elf's, because Shea's are innovations and the
      forecasts need exact residuals.
  - **`both`** optimises with elf and evaluates Shea at every point, and
    reports the largest |ΔlogL| in `Ladder.lik_check` and in the report.
  - **Measured.** Python and the C oracle give the same ℓ (92.566119 on the
    ES/FR pair; −1752.522249 on m6). With Shea the m6 gate closes to 0.
    With `method=2`, elf and Shea agree to 7e-13 at every point.
  - **Needs the compiled engine.** The pure-Python port of Shea is still to
    be written, and will be validated against `marma_c`.
  - Pinned in `tests/test_shea.py`.
- **The optimiser's stop, as the C writes it.**
  - **The report block.** The ladder report now has the C's
    `OPTIMIZER … after N iterations` block, line for line.
  - **A stop on the MA wall.** With MA inverse roots at modulus >= 1, it
    says `OPTIMIZER STOPPED at the MA invertibility boundary` and
    `MA boundary: k of n inverse roots at modulus >= 1`.
  - **The data.** `Fit.ma_boundary`, `Fit.ma_nroots` and `Fit.fk`. It
    states facts and gives no verdict: studying the situation is sima's job.
- **The gate is evaluated with the untruncated likelihood,** whatever
  `method` says, as the C does since atsw-gui 2026-09-28. Pairs of m6 failed
  it by up to 0.0018 because of elf's ξ truncation; they now close to 1e-13.
  The m6 pin moves from −0.000428 to exact.
- **IRF/FEVD bands for the ladder model: `Ladder.irf_fevd_bands(H)`.**
  - Monte Carlo from N(estimate, covariance): each draw is rebuilt through
    the ladder's cast, univariate factors and cross terms together.
  - Draws that are non-stationary, non-invertible or have a Sigma that is
    not positive definite are discarded and counted.
  - The covariance of the estimates is kept as `Fit.cov`, from the same
    Hessian as the SEs.
- **`Ladder.refit(x)`** re-optimises the requested model from a given point,
  without the gate. With it a study can restart from where a fit stopped,
  e.g. with the MA roots pulled inside the wall.
- **Standard errors from fdhess by default** (`drvarma.stderr`). This is
  Mauricio's finite-difference Hessian at the optimum, from the published code,
  where it had been left commented out. It replaces the BFGS Hessian the search
  accumulated, in both paths. `hessian="bfgs"` / `-hessian bfgs` keeps the old
  behaviour.
  - Decided by `docs/STUDY-standard-errors.md`: fdhess matches the exact GLS
    within 0.35 % and OLS within 1 %, including series of very different
    scales. BFGS was off by up to 1483 % and depended on the start (t = −99 in
    the review's IPC/WTI case, instead of about −1.1).
  - The method in use is written in every report and in `export_fit`. If the
    optimum is on the boundary or the Hessian is not positive definite, BFGS
    is kept and the report says why.
  - In the `.inp` path, `qq[1,1]` (the flat direction Q → cQ) is held fixed
    and is reported as `(normalised)`, with no standard error.
  - The BFGS fallback applies only if the search iterated. raxopt starts at
    the identity, so a search that did not move has no BFGS Hessian. Then
    there are no standard errors, and the method reads `none (…)`.
- **The assistant leaves the engine.** The MCP assistant is now its own
  package, **sima-tseries**, built on the ladder, and it owns the `sima`
  command. The old server on raw series stays here, deprecated, as
  `sima-legacy`. The engine API it calls is declared:
  - `drvarma.ladder`;
  - `diagnostics.ccf`, `qccf` and `hosking_q`;
  - `irf.oirf` and `fevd`.

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
