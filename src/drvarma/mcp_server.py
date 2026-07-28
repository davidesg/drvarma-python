"""multiart — MCP server for multivariate VARMA analysis with drvarma.

The multivariate counterpart of ART's univariate Box-Jenkins MCP. See
`docs/DESIGN_MCP.md` for the full design. This is the v1 scaffold: the guided
protocol shell plus the first tools (load_data, series_info, seed_from_art,
identify_varma_order, confirm_and_estimate).

Rescaling note (docs/DESIGN_MCP.md §5b): drvarma's `scale` (default 100) is the
single source of truth. Never hardcode 100; use drvarma's native, level-unit
forecast bands; carry `scale` through any rebuild. This avoids the ATSW
BUG-0007/0008 class.
"""
from __future__ import annotations

import json
import numpy as np
from mcp.server.fastmcp import FastMCP

from .series import MultiSeries
from .model import Model
from . import diagnostics, transform

# ---------------------------------------------------------------------------
# In-process session state: named multivariate datasets and fitted models.
# The server is a single long-running process, so a module dict is enough for
# a session; file persistence (.inp/.pre-style) is a later refinement.
# ---------------------------------------------------------------------------
_DATA: dict[str, MultiSeries] = {}
_FITS: dict[str, Model] = {}


_INSTRUCTIONS = """
Eres multiart — asistente de análisis de series temporales MULTIVARIANTE (modelos
VARMA por máxima verosimilitud exacta, motor drvarma). Contraparte multivariante de
ART (univariante).

══════════════════════════════════════════════════════
IDIOMA / LANGUAGE
══════════════════════════════════════════════════════
Responde SIEMPRE en el idioma del usuario (inglés por defecto si es ambiguo). Estas
instrucciones y las salidas de las herramientas pueden venir en español: tradúcelas
al idioma del usuario; nunca pegues español a un usuario que escribe en inglés.
── Always respond in the user's language (default English). Tool output may be in
Spanish; translate it — never paste Spanish to an English-speaking user.

══════════════════════════════════════════════════════
PREGUNTA INICIAL OBLIGATORIA
══════════════════════════════════════════════════════
Al iniciar, pregunta SIEMPRE:
  "¿Cómo deseas proceder?
   1) GUIADO (paso a paso, con confirmación en cada etapa)
   2) AUTÓNOMO (identificación + estimación automáticas)"

══════════════════════════════════════════════════════
FLUJO (VARMA clásico estacionario — v1)
══════════════════════════════════════════════════════
1. load_data — carga las m series (CSV o JSON).
2. SIEMBRA de la especificación (sin esto, un VARMA es "palos de ciego"):
   • seed_from_art  → exploración univariante automática de ART por serie
     (λ, diferenciación d, estacionalidad D, techo de órdenes). RECOMENDADO.
   • o el usuario informado responde y se salta la siembra.
3. Identificación del orden sobre las series preparadas — combina:
   • cross_correlation_matrices (CCM) → corte tras lag q ⇒ MA(q);
   • partial_autoregression_matrices (Tiao-Box) → último lag signif. ⇒ AR(p);
   • identify_varma_order → rejilla AIC/BIC/HQ como contraste cuantitativo.
   Propón (p,q) razonado y ESPERA confirmación.
4. confirm_and_estimate — Model(p,q).fit() por ML exacta.
5. diagnose — Hosking Q (autocorrelación cruzada de residuos) + Jarque-Bera. Si Q
   es significativo, sube el orden y reestima.
6. impulse_response (OIRF) y variance_decomposition (FEVD) — análisis estructural.
7. generate_forecast — previsión con bandas 95% (en unidades de nivel, nativas).
[Pendiente: CCM + matrices de autocorrelación parcial (Tiao-Box) en la identificación;
full_report; SPS update_and_forecast.]

NOTA (cointegración): v1 asume series llevadas a estacionariedad por diferenciación.
Si las series parecen I(1) que se mueven juntas (posible cointegración), AVISA de que
lo estás dejando de lado — es alcance de v2.
"""

mcp = FastMCP("multiart — Multivariate VARMA Analysis (drvarma)",
              instructions=_INSTRUCTIONS)


def _require(name: str) -> MultiSeries:
    if name not in _DATA:
        raise ValueError(f"no dataset named {name!r}; call load_data first "
                         f"(known: {sorted(_DATA)})")
    return _DATA[name]


def _npar(m: int, p: int, q: int, include_mean: bool) -> int:
    """Free parameters of a full VARMA(p,q): AR+MA blocks, mean, covariance."""
    return m * m * (p + q) + (m if include_mean else 0) + m * (m + 1) // 2


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def load_data(name: str, csv_path: str = "", values_json: str = "",
              freq: int = 12, start_year: int = 2000, start_period: int = 1,
              series_names: str = "") -> str:
    """Load an m-variate time series into the session under `name`.

    Provide EITHER csv_path (one column per series, optional header row) OR
    values_json (a JSON list of rows, each row a list of m values). freq is
    observations/year (1/4/12); start is (start_year, start_period). series_names
    is an optional comma-separated list of labels.
    """
    names = [s.strip() for s in series_names.split(",") if s.strip()] or None
    if csv_path:
        try:
            arr = np.genfromtxt(csv_path, delimiter=",", names=None)
            if arr.dtype.names or np.isnan(arr).all(axis=1)[0]:
                arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        except Exception as e:  # noqa: BLE001
            return f"Error leyendo {csv_path}: {e}"
        data = np.atleast_2d(arr)
    elif values_json:
        data = np.asarray(json.loads(values_json), dtype=float)
    else:
        return "Falta csv_path o values_json."
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    try:
        ms = MultiSeries(data, freq=freq, start=(start_year, start_period), names=names)
    except Exception as e:  # noqa: BLE001
        return f"Error construyendo la serie: {e}"
    _DATA[name] = ms
    return (f"Cargado {name!r}: {ms.nobs} obs × {ms.m} series {ms.names}, "
            f"freq={ms.freq}, inicio={ms.start}. Siguiente: seed_from_art({name!r}) "
            f"o series_info({name!r}).")


@mcp.tool()
def series_info(name: str) -> str:
    """Per-series descriptive statistics (mean/var/std/skew/kurt/min/max)."""
    ms = _require(name)
    lines = [f"# {name}: {ms.nobs} obs × {ms.m} series (freq={ms.freq}, inicio={ms.start})", ""]
    for j, lab in enumerate(ms.names):
        try:
            st = diagnostics.series_stats(ms.data[:, j])
            lines.append(f"- {lab}: " + "  ".join(f"{k}={v:.4g}" for k, v in
                         (st.items() if hasattr(st, "items") else [])) or f"- {lab}: {st}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- {lab}: (series_stats no disponible: {e})")
    return "\n".join(lines)


@mcp.tool()
def seed_from_art(name: str) -> str:
    """Seed the VARMA spec from ART's automatic univariate identification.

    Runs ART/fue per component to propose λ (Box-Cox), differencing d, seasonal D
    and a sensible ARMA ceiling — so the multivariate order search is not blind.
    Defensive: falls back to log/d=1 when a component analysis is unavailable.
    """
    ms = _require(name)
    try:
        import fue
        from art import identification as _id
    except Exception as e:  # noqa: BLE001
        return (f"ART/fue no disponibles para la siembra ({e}). Instala "
                f"`drvarma[mcp]` (arrastra art-tseries) o usa la ruta 'usuario "
                f"informado' pasando lam/d/D a identify_varma_order.")
    rows, lam_all, d_all, D_all = [], [], [], []
    for j, lab in enumerate(ms.names):
        ts = fue.TimeSeries.from_array(ms.data[:, j].tolist(), freq=ms.freq,
                                       start=ms.start, name=lab)
        lam = 0.0
        try:
            lam = float(_id.boxcox_selection(ts).lam)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        d, D = 1, 0   # v1 default; per-series ADF/seasonal wiring is a refinement
        rows.append(f"- {lab}: λ={lam:.2f}, d={d}, D={D}")
        lam_all.append(lam); d_all.append(d); D_all.append(D)
    lam_c = 0.0 if all(abs(x) < 0.25 for x in lam_all) else float(np.median(lam_all))
    d_c, D_c = int(np.round(np.median(d_all))), int(np.round(np.median(D_all)))
    return ("Siembra univariante (ART) por serie:\n" + "\n".join(rows) +
            f"\n\nConsenso para el VARMA: λ={lam_c:.2f}, d={d_c}, D={D_c}. "
            f"Siguiente: identify_varma_order({name!r}, lam={lam_c}, d={d_c}, D={D_c}). "
            f"(Nota v1: d/D por serie usan el default; el cableado ADF/estacional "
            f"fino queda como refinamiento.)")


@mcp.tool()
def identify_varma_order(name: str, lam: float = 0.0, d: int = 1, D: int = 0,
                         p_max: int = 2, q_max: int = 2,
                         include_mean: bool = True) -> str:
    """Rank VARMA(p,q) orders by information criteria (AIC/BIC/HQ).

    Fits every (p,q) with p≤p_max, q≤q_max by exact ML and ranks them. v1 uses the
    IC grid; CCM and Tiao-Box partial-autoregression matrices are pending. Returns
    a table and a recommendation to confirm.
    """
    ms = _require(name)
    n, m = ms.nobs, ms.m
    results = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                mod = Model(ms, lam=lam, d=d, D=D, p=p, q=q,
                            include_mean=include_mean).fit()
                ll = float(mod.loglik)
                k = _npar(m, p, q, include_mean)
                aic = -2 * ll + 2 * k
                bic = -2 * ll + k * np.log(n)
                hq = -2 * ll + 2 * k * np.log(np.log(n))
                results.append((p, q, ll, k, aic, bic, hq))
            except Exception as e:  # noqa: BLE001
                results.append((p, q, None, None, None, None, str(e)[:40]))
    ok = [r for r in results if r[4] is not None]
    if not ok:
        return "Ninguna especificación convergió. Revisa la preparación (lam/d/D)."
    ok.sort(key=lambda r: r[5])   # by BIC
    hdr = f"VARMA order search (lam={lam}, d={d}, D={D}, mean={include_mean}) — ranked by BIC\n"
    hdr += "| p | q | loglik | k | AIC | BIC | HQ |\n|---|---|--------|---|-----|-----|----|"
    body = "\n".join(f"| {p} | {q} | {ll:.2f} | {k} | {aic:.1f} | {bic:.1f} | {hq:.1f} |"
                     for (p, q, ll, k, aic, bic, hq) in ok)
    bp, bq = ok[0][0], ok[0][1]
    return (hdr + "\n" + body +
            f"\n\nRecomendación (mín. BIC): **VARMA({bp},{bq})**. Confirma y llama a "
            f"confirm_and_estimate({name!r}, lam={lam}, d={d}, D={D}, p={bp}, q={bq}, "
            f"include_mean={include_mean}).")


@mcp.tool()
def confirm_and_estimate(name: str, lam: float = 0.0, d: int = 1, D: int = 0,
                         p: int = 1, q: int = 0, include_mean: bool = True,
                         diag_ar: bool = False, diag_ma: bool = False,
                         diag_cov: bool = False) -> str:
    """Estimate the final VARMA(p,q) by exact ML and store the fit under `name`.

    diag_ar/diag_ma/diag_cov impose diagonal AR/MA/covariance (as drvarma's C
    flags). Returns log-likelihood, the Φ/Θ/Σ summary and residual diagnostics.
    """
    ms = _require(name)
    try:
        mod = Model(ms, lam=lam, d=d, D=D, p=p, q=q, include_mean=include_mean,
                    diag_ar=diag_ar, diag_ma=diag_ma, diag_cov=diag_cov).fit()
    except Exception as e:  # noqa: BLE001
        return f"La estimación de VARMA({p},{q}) falló: {e}"
    _FITS[name] = mod
    m = ms.m
    k = _npar(m, p, q, include_mean)
    out = [f"# VARMA({p},{q}) estimado — {name} ({m} series, {ms.nobs} obs)",
           f"log-likelihood = {mod.loglik:.4f}   (k={k}, "
           f"AIC={-2*mod.loglik + 2*k:.1f}, BIC={-2*mod.loglik + k*np.log(ms.nobs):.1f})",
           f"scale (reescalado) = {mod.scale}"]
    try:
        phi = np.asarray(mod.phi)
        out.append("Φ₁ =\n" + np.array2string(phi[0], precision=3, suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:
        out.append("Σ =\n" + np.array2string(np.asarray(mod.sigma), precision=4,
                                              suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    out.append("\nSiguiente (pendiente en el scaffold): diagnose, impulse_response, "
               "fevd, generate_forecast.")
    return "\n".join(out)


def _prepared_w(ms: MultiSeries, lam: float, d: int, D: int) -> np.ndarray:
    """The stationary differenced series w = ∇^d ∇_s^D BoxCox_λ(level).

    scale is irrelevant for correlations/partial-autoregression (scale-invariant),
    so we transform at scale=1.
    """
    w, _ = transform.transform(ms.data, lam=lam, d=d, D=D, s=ms.freq, scale=1.0)
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError(
            f"la serie transformada tiene valores no finitos (λ={lam}): λ=0 (log) "
            "exige datos positivos. Usa λ=1 o revisa los datos. (No devuelvo un "
            "resultado silencioso con NaN.)")
    return w


@mcp.tool()
def cross_correlation_matrices(name: str, lam: float = 0.0, d: int = 1, D: int = 0,
                               n_lags: int = 6) -> str:
    """Sample cross-correlation matrices (CCM) of the differenced series.

    Tiao-Box +/-/. display (bound 2/√n). CCM that **cut off** after lag q point to
    a pure MA(q); a slow decay points to AR terms. Complements
    partial_autoregression_matrices (AR order) for VARMA order identification.
    """
    ms = _require(name)
    w = _prepared_w(ms, lam, d, D)
    n, m = w.shape
    wc = w - w.mean(0)
    sd = np.sqrt(np.diag((wc.T @ wc) / n))
    bound = 2.0 / np.sqrt(n)
    out = [f"# Matrices de correlación cruzada (CCM) — {name} (n={n}, m={m})",
           f"Símbolos: + (ρ>2/√n={bound:.3f}), - (ρ<-2/√n), . (no signif.). "
           f"Series: {ms.names}", ""]
    for k in range(1, n_lags + 1):
        Rk = ((wc[k:].T @ wc[:-k]) / n) / np.outer(sd, sd)   # corr(w_i(t), w_j(t-k))
        out.append(f"lag {k}:")
        for i in range(m):
            out.append("  " + " ".join(
                "+" if Rk[i, j] > bound else "-" if Rk[i, j] < -bound else "."
                for j in range(m)))
    return "\n".join(out)


@mcp.tool()
def partial_autoregression_matrices(name: str, lam: float = 0.0, d: int = 1,
                                    D: int = 0, max_order: int = 6) -> str:
    """Tiao-Box partial autoregression matrices — AR-order identification.

    Fits VAR(k) by OLS for k=1..max_order; the partial autoregression matrix is the
    last coefficient block Φ_kk. For a VAR(p) it is ≈0 (all '.') for k>p, so the AR
    order p is the last lag with significant symbols. Symbols from t-ratios (|t|>1.96).
    """
    ms = _require(name)
    w = _prepared_w(ms, lam, d, D)
    n0, m = w.shape
    out = [f"# Matrices de autocorrelación parcial (Tiao-Box) — {name} (m={m})",
           f"Símbolos por t-ratio: + (t>1.96), - (t<-1.96), . (no signif.). "
           f"Series: {ms.names}",
           "AR order p = último lag con símbolos significativos.", ""]
    for k in range(1, max_order + 1):
        Y = w[k:]
        T = len(Y)
        if T <= 1 + k * m:
            out.append(f"lag {k}: (muestra insuficiente)"); continue
        X = np.column_stack([np.ones(T)] + [w[k - l:n0 - l] for l in range(1, k + 1)])
        try:
            XtXi = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            out.append(f"lag {k}: (singular)"); continue
        B = XtXi @ X.T @ Y
        resid = Y - X @ B
        Sig = (resid.T @ resid) / (T - X.shape[1])
        r0 = 1 + (k - 1) * m                       # first row of the Φ_kk block
        out.append(f"lag {k}:")
        for i in range(m):
            syms = []
            for j in range(m):
                se = np.sqrt(Sig[i, i] * XtXi[r0 + j, r0 + j])
                t = B[r0 + j, i] / se if se > 0 else 0.0
                syms.append("+" if t > 1.96 else "-" if t < -1.96 else ".")
            out.append("  " + " ".join(syms))
    return "\n".join(out)


def _require_fit(name: str) -> Model:
    if name not in _FITS:
        raise ValueError(f"no fitted model for {name!r}; call confirm_and_estimate first")
    return _FITS[name]


@mcp.tool()
def diagnose(name: str, lag: int = 0) -> str:
    """Multivariate residual diagnostics of the fitted VARMA.

    Hosking's multivariate portmanteau Q (residual cross-correlation) and the
    multivariate Jarque-Bera. `lag` defaults to freq+2. A significant Q means the
    order (p or q) is too low.
    """
    mod = _require_fit(name)
    d = mod.diagnostics(lag or None)
    ok_q = d["hosking_p"] > 0.05
    ok_jb = d["JB_p"] > 0.05
    return (f"# Diagnóstico de residuos — {name}\n"
            f"- Hosking Q({d['hosking_lag']}) = {d['hosking_Q']:.2f}  df={d['hosking_df']}  "
            f"p={d['hosking_p']:.3f} → "
            f"{'sin autocorrelación cruzada ✓' if ok_q else 'AUTOCORRELACIÓN residual ⚠ (sube p/q)'}\n"
            f"- Jarque-Bera = {d['JB']:.2f}  df={d['JB_df']}  p={d['JB_p']:.3f} → "
            f"{'normalidad ok ✓' if ok_jb else 'no-normal ⚠'}\n"
            + ("Modelo adecuado." if ok_q else "Revisa la especificación (orden)."))


@mcp.tool()
def generate_forecast(name: str, horizon: int = 12) -> str:
    """Forecast the fitted VARMA `horizon` steps ahead with 95% bands.

    Uses drvarma's native, level-unit bands (scale-correct — never a hand-rolled
    level ± 1.96·std, which would resurrect the ATSW BUG-0008 relative-std band).
    """
    mod = _require_fit(name)
    levels, low, high = mod.forecast(horizon, bands=True)
    ms = mod.series
    out = [f"# Previsión VARMA — {name} ({horizon} pasos, bandas 95%, scale={mod.scale})"]
    for j, lab in enumerate(ms.names):
        out += [f"\n## {lab}", "| h | previsión | IC 95% |", "|---|-----------|--------|"]
        out += [f"| {h+1} | {levels[h, j]:.3f} | [{low[h, j]:.3f}, {high[h, j]:.3f}] |"
                for h in range(horizon)]
    return "\n".join(out)


@mcp.tool()
def impulse_response(name: str, horizon: int = 12) -> str:
    """Orthogonalised impulse responses (OIRF) of the fitted VARMA."""
    mod = _require_fit(name)
    irf = np.asarray(mod.irf(horizon, orthogonalized=True))   # (H+1, m, m)
    nm = mod.series.names
    hs = sorted({h for h in (0, 1, 2, 4, 8, horizon) if 0 <= h <= horizon})
    out = [f"# Respuestas al impulso ortogonalizadas (OIRF) — {name}"]
    for i in range(mod.series.m):        # response variable
        for k in range(mod.series.m):    # orthogonal shock
            path = irf[:, i, k]
            out.append(f"- {nm[i]} ← shock {nm[k]}: " +
                       ", ".join(f"h{h}={path[h]:+.3f}" for h in hs))
    return "\n".join(out)


@mcp.tool()
def variance_decomposition(name: str, horizon: int = 12) -> str:
    """Forecast-error variance decomposition (FEVD, %) at the given horizon."""
    mod = _require_fit(name)
    fevd = np.asarray(mod.fevd(horizon))   # (H, m, m)
    nm = mod.series.names
    dec = fevd[-1]                          # at the final horizon
    out = [f"# FEVD — {name}, h={horizon} (fila = % de la varianza de esa serie por cada shock)",
           "| serie ↓ / shock → | " + " | ".join(nm) + " |",
           "|" + "---|" * (mod.series.m + 1)]
    for i in range(mod.series.m):
        out.append(f"| {nm[i]} | " +
                   " | ".join(f"{dec[i, k]:.1f}" for k in range(mod.series.m)) + " |")
    return "\n".join(out)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
