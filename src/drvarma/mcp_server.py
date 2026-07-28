"""multiart — MCP server for multivariate VARMA analysis with drvarma.

The multivariate counterpart of ART's univariate Box-Jenkins MCP. See
`docs/DESIGN_MCP.md`. v1: classic stationary VARMA, ART-seeded.

Protocol (important): the analyst loads the **original, untransformed** series.
`characterize_series` runs ART's univariate identification per component, saves the
characteristics (λ, d, seasonality, orders) to the session, and derives a joint
consensus (λ, d, deseason). Every VARMA tool then reads that seed and applies the
transformation **once, internally** — the caller never pre-transforms the data.

Rescaling (docs/DESIGN_MCP.md §5b): drvarma's `scale` (default 100) is the single
source of truth; never hardcode 100; use drvarma's native level-unit forecast bands;
carry `scale` on any rebuild. Avoids the ATSW BUG-0007/0008 class.
"""
from __future__ import annotations

import json
import numpy as np
from mcp.server.fastmcp import FastMCP

from .series import MultiSeries
from .model import Model
from . import diagnostics, transform

# In-process session state (single long-running process): datasets, fits, and the
# per-dataset univariate characterization ("seed") that drives the transformation.
_DATA: dict[str, MultiSeries] = {}
_FITS: dict[str, Model] = {}
_SEED: dict[str, dict] = {}


_INSTRUCTIONS = """
Eres multiart — asistente de análisis MULTIVARIANTE (VARMA por ML exacta, motor
drvarma). Contraparte multivariante de ART (univariante).

══════════════════════════════════════════════════════
IDIOMA / LANGUAGE
══════════════════════════════════════════════════════
Responde SIEMPRE en el idioma del usuario (inglés por defecto si es ambiguo). Las
salidas de las tools pueden venir en español: tradúcelas; nunca pegues español a un
usuario que escribe en inglés.
── Always respond in the user's language (default English); translate tool output.

══════════════════════════════════════════════════════
REGLA DE ORO — SERIES ORIGINALES
══════════════════════════════════════════════════════
Carga SIEMPRE las series ORIGINALES sin transformar (nivel, desde Excel/CSV).
multiart aplica la transformación (Box-Cox λ, diferenciación d, desestacionalización)
INTERNAMENTE, guiada por la caracterización univariante. **NUNCA pre-transformes ni
prediferencies los datos** — si lo haces, la identificación será incorrecta.

══════════════════════════════════════════════════════
PREGUNTA INICIAL OBLIGATORIA
══════════════════════════════════════════════════════
"¿Cómo deseas proceder?
   1) GUIADO (paso a paso, con confirmación en cada etapa)
   2) AUTÓNOMO (caracterización + identificación + estimación automáticas)"

══════════════════════════════════════════════════════
FLUJO (VARMA clásico estacionario — v1)
══════════════════════════════════════════════════════
1. load_data — carga las m series ORIGINALES (Excel .xlsx / CSV / JSON).
2. characterize_series — identificación univariante de ART por serie; GUARDA el
   resumen (λ, d, estacionalidad, órdenes) y deriva el consenso conjunto
   (λ, d, deseason). Es la SIEMBRA: sin ella el VARMA es "palos de ciego".
   (Si el usuario conoce sus datos, puede saltarla y pasar λ/d/D explícitos.)
3. Identificación del orden (usan el seed guardado por defecto):
   • cross_correlation_matrices (CCM) → corte tras lag q ⇒ MA(q);
   • partial_autoregression_matrices (Tiao-Box) → último lag signif. ⇒ AR(p);
   • identify_varma_order → rejilla AIC/BIC/HQ como contraste.
   Propón (p,q) razonado y ESPERA confirmación.
4. confirm_and_estimate — Model(p,q).fit() por ML exacta (usa el seed).
5. diagnose — Hosking Q + Jarque-Bera. Si Q es signif., sube el orden.
6. impulse_response (OIRF) y variance_decomposition (FEVD).
7. generate_forecast — previsión con bandas 95% (unidades de nivel, nativas).

ESTACIONALIDAD: si la caracterización la detecta, el consenso usa deseason="auto"
(desestacionalización armónica de drvarma). Todas las tools de identificación y
estimación desestacionalizan de forma coherente — no se usa diferencia estacional.

COINTEGRACIÓN (v1 fuera): si las series parecen I(1) que se mueven juntas, AVISA de
que lo dejas de lado (es alcance de v2).
"""

mcp = FastMCP("multiart — Multivariate VARMA Analysis (drvarma)",
              instructions=_INSTRUCTIONS)


# ── helpers ────────────────────────────────────────────────────────────────
def _require(name: str) -> MultiSeries:
    if name not in _DATA:
        raise ValueError(f"no dataset named {name!r}; call load_data first "
                         f"(known: {sorted(_DATA)})")
    return _DATA[name]


def _require_fit(name: str) -> Model:
    if name not in _FITS:
        raise ValueError(f"no fitted model for {name!r}; call confirm_and_estimate first")
    return _FITS[name]


def _npar(m: int, p: int, q: int, include_mean: bool) -> int:
    return m * m * (p + q) + (m if include_mean else 0) + m * (m + 1) // 2


def _resolve(name: str, lam: float, d: int, D: int, deseason: str) -> tuple:
    """Fill unset transform params (sentinels) from the saved characterization.

    Sentinels: lam<=-90, d<0, D<0, deseason=='seed' → take from _SEED[name]. This
    is what makes the tools operate on the ORIGINAL series with the transform the
    univariate pass decided, without the caller re-specifying it each call.
    """
    cons = _SEED.get(name, {}).get("consensus", {})
    if lam <= -90:
        lam = float(cons.get("lam", 0.0))
    if d < 0:
        d = int(cons.get("d", 1))
    if D < 0:
        D = int(cons.get("D", 0))
    if deseason == "seed":
        deseason = cons.get("deseason") or ""
    ds = None if deseason in ("", "none", "None", "0") else deseason
    return lam, d, D, ds


def _prepared_w(ms: MultiSeries, lam: float, d: int, D: int,
                deseason=None) -> np.ndarray:
    """Stationary series w = ∇^d ∇_s^D BoxCox_λ(level), deseasonalized like the
    Model does (so CCM/Tiao-Box identify on the SAME series the Model estimates).
    scale is irrelevant for correlations, so transform at scale=1.
    """
    levels = ms.data
    if deseason:
        from .deseason import deseasonalize_raw
        levels, _, _ = deseasonalize_raw(levels, s=ms.freq, start_sub=ms.start[1],
                                         mode=deseason)
    w, _ = transform.transform(levels, lam=lam, d=d, D=D, s=ms.freq, scale=1.0)
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError(
            f"la serie transformada tiene valores no finitos (λ={lam}): λ=0 (log) "
            "exige datos positivos. Usa λ=1 o revisa los datos. (No devuelvo NaN "
            "silencioso.)")
    return w


# ── tools ──────────────────────────────────────────────────────────────────
@mcp.tool()
def load_data(name: str, path: str = "", values_json: str = "",
              freq: int = 12, start_year: int = 2000, start_period: int = 1,
              series_names: str = "") -> str:
    """Load the m ORIGINAL (untransformed) series into the session under `name`.

    Provide EITHER `path` (an .xlsx/.xls Excel or a .csv, one column per series,
    optional header row) OR `values_json` (a JSON list of rows). freq = obs/year
    (1/4/12); start = (start_year, start_period). series_names = optional
    comma-separated labels. Do NOT pass pre-transformed or pre-differenced data.
    """
    names = [s.strip() for s in series_names.split(",") if s.strip()] or None
    if path:
        low = path.lower()
        try:
            if low.endswith((".xlsx", ".xls")):
                import pandas as pd
                df = pd.read_excel(path)
                if names is None:
                    names = [str(c) for c in df.columns]
                data = df.to_numpy(dtype=float)
            else:
                try:
                    data = np.genfromtxt(path, delimiter=",", skip_header=1)
                    if np.isnan(data).all():
                        data = np.genfromtxt(path, delimiter=",")
                except Exception:
                    data = np.genfromtxt(path, delimiter=",")
        except ImportError:
            return "Para .xlsx necesito pandas (pip install pandas) o exporta a CSV."
        except Exception as e:  # noqa: BLE001
            return f"Error leyendo {path}: {e}"
        data = np.atleast_2d(data)
    elif values_json:
        data = np.asarray(json.loads(values_json), dtype=float)
    else:
        return "Falta `path` (Excel/CSV) o `values_json`."
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    try:
        ms = MultiSeries(data, freq=freq, start=(start_year, start_period), names=names)
    except Exception as e:  # noqa: BLE001
        return f"Error construyendo la serie: {e}"
    _DATA[name] = ms
    _SEED.pop(name, None); _FITS.pop(name, None)
    return (f"Cargado {name!r} (series ORIGINALES): {ms.nobs} obs × {ms.m} series "
            f"{ms.names}, freq={ms.freq}, inicio={ms.start}. "
            f"Siguiente: characterize_series({name!r}).")


@mcp.tool()
def series_info(name: str) -> str:
    """Per-series descriptive statistics (mean/var/std/skew/kurt/min/max)."""
    ms = _require(name)
    lines = [f"# {name}: {ms.nobs} obs × {ms.m} series (freq={ms.freq}, inicio={ms.start})", ""]
    for j, lab in enumerate(ms.names):
        try:
            st = diagnostics.series_stats(ms.data[:, j])
            body = "  ".join(f"{k}={v:.4g}" for k, v in st.items()) if hasattr(st, "items") else str(st)
            lines.append(f"- {lab}: {body}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- {lab}: (series_stats no disponible: {e})")
    return "\n".join(lines)


@mcp.tool()
def characterize_series(name: str) -> str:
    """SEED step — ART's univariate identification per component, saved to session.

    For each series computes λ (Box-Cox), d (unit-root recommended differencing),
    seasonality (HAC F-test) and rough ARMA orders, then derives a JOINT consensus
    (λ, d, deseason) that the VARMA tools use by default. Without this, a VARMA
    spec is blind guessing. Defensive: falls back to log/d=1 per piece if a
    component analysis is unavailable.
    """
    ms = _require(name)
    try:
        import fue
        from art import identification as _id
        from art import seasonal_detection as _sd
        from art import model_detection as _md
    except Exception as e:  # noqa: BLE001
        return (f"ART/fue no disponibles para caracterizar ({e}). Instala "
                f"`drvarma[mcp]` o pasa λ/d/D explícitos a las tools.")

    per, lams, ds, seas, pcs, qcs = [], [], [], [], [], []
    for j, lab in enumerate(ms.names):
        ts = fue.TimeSeries.from_array(ms.data[:, j].tolist(), freq=ms.freq,
                                       start=ms.start, name=lab)
        lam = 0.0
        try:
            lam = float(_id.boxcox_selection(ts).lam)
        except Exception:  # noqa: BLE001
            pass
        d = 1
        try:
            d = int(_id.recommended_d(_id.unit_root_tests(ts, lam=lam)))
        except Exception:  # noqa: BLE001
            pass
        is_seas = False
        try:
            is_seas = bool(_sd.detect_seasonality(ts, d=d, lam=lam).seasonal_detected)
        except Exception:  # noqa: BLE001
            pass
        p, q = 1, 0
        try:
            specs = _md.suggest_orders(ts, d=d, D=(1 if is_seas else 0), lam=lam)
            if specs:
                p = int(getattr(specs[0], "p", 1)); q = int(getattr(specs[0], "q", 0))
        except Exception:  # noqa: BLE001
            pass
        per.append(dict(name=lab, lam=lam, d=d, seasonal=is_seas, p=p, q=q))
        lams.append(lam); ds.append(d); seas.append(is_seas); pcs.append(p); qcs.append(q)

    lam_c = 0.0 if all(abs(x) < 0.25 for x in lams) else float(np.median(lams))
    d_c = int(max(ds)) if ds else 1          # difference all to stationarity (v1, no cointegration)
    deseason_c = "auto" if any(seas) else None
    p_ceil = int(min(3, max(pcs) if pcs else 1))
    q_ceil = int(min(3, max(qcs) if qcs else 1))
    _SEED[name] = dict(
        per_series=per,
        consensus=dict(lam=lam_c, d=d_c, D=0, deseason=deseason_c,
                       p_ceiling=p_ceil, q_ceiling=q_ceil),
    )

    rows = "\n".join(
        f"| {r['name']} | {r['lam']:.2f} | {r['d']} | "
        f"{'sí' if r['seasonal'] else 'no'} | ({r['p']},{r['q']}) |" for r in per)
    return (f"# Caracterización univariante (ART) — {name}\n"
            f"| serie | λ | d | estacional | ARMA≈ |\n|---|---|---|---|---|\n{rows}\n\n"
            f"**Consenso para el VARMA (guardado):** λ={lam_c:.2f}, d={d_c}, "
            f"deseason={deseason_c or 'no'}, techo (p,q)≤({p_ceil},{q_ceil}).\n"
            f"{'⚠ Estacionalidad detectada → se usará desestacionalización armónica (deseason=auto).' if deseason_c else ''}\n"
            f"Siguiente: cross_correlation_matrices({name!r}) y "
            f"partial_autoregression_matrices({name!r}) (usan este seed), luego "
            f"identify_varma_order({name!r}).\n"
            f"(Nota: las series se dejaron ORIGINALES; la transformación la aplican las tools.)")


@mcp.tool()
def cross_correlation_matrices(name: str, lam: float = -99.0, d: int = -1,
                               D: int = -1, deseason: str = "seed",
                               n_lags: int = 6) -> str:
    """Sample cross-correlation matrices (CCM) of the prepared series.

    Uses the saved characterization (λ/d/deseason) by default. Tiao-Box +/-/.
    (bound 2/√n). CCM that CUT OFF after lag q ⇒ pure MA(q); slow decay ⇒ AR terms.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    w = _prepared_w(ms, lam, d, D, ds)
    n, m = w.shape
    wc = w - w.mean(0)
    sd = np.sqrt(np.diag((wc.T @ wc) / n))
    bound = 2.0 / np.sqrt(n)
    out = [f"# CCM — {name} (n={n}, m={m}; λ={lam}, d={d}, deseason={ds or 'no'})",
           f"Símbolos: + (ρ>2/√n={bound:.3f}), - (<-2/√n), . (no signif.). Series: {ms.names}", ""]
    for k in range(1, n_lags + 1):
        Rk = ((wc[k:].T @ wc[:-k]) / n) / np.outer(sd, sd)
        out.append(f"lag {k}:")
        for i in range(m):
            out.append("  " + " ".join(
                "+" if Rk[i, j] > bound else "-" if Rk[i, j] < -bound else "."
                for j in range(m)))
    return "\n".join(out)


@mcp.tool()
def partial_autoregression_matrices(name: str, lam: float = -99.0, d: int = -1,
                                    D: int = -1, deseason: str = "seed",
                                    max_order: int = 6) -> str:
    """Tiao-Box partial autoregression matrices — AR-order identification.

    Uses the saved characterization by default. Fits VAR(k) by OLS; the partial
    autoregression matrix is the last block Φ_kk. For a VAR(p) it is ≈0 (all '.')
    for k>p, so p = last lag with significant symbols (t-ratios, |t|>1.96).
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    w = _prepared_w(ms, lam, d, D, ds)
    n0, m = w.shape
    out = [f"# Matrices de autocorrelación parcial (Tiao-Box) — {name} "
           f"(m={m}; λ={lam}, d={d}, deseason={ds or 'no'})",
           f"Símbolos por t-ratio: + (>1.96), - (<-1.96), . (no signif.). Series: {ms.names}",
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
        Sig = ((Y - X @ B).T @ (Y - X @ B)) / (T - X.shape[1])
        r0 = 1 + (k - 1) * m
        out.append(f"lag {k}:")
        for i in range(m):
            syms = []
            for j in range(m):
                se = np.sqrt(Sig[i, i] * XtXi[r0 + j, r0 + j])
                t = B[r0 + j, i] / se if se > 0 else 0.0
                syms.append("+" if t > 1.96 else "-" if t < -1.96 else ".")
            out.append("  " + " ".join(syms))
    return "\n".join(out)


@mcp.tool()
def identify_varma_order(name: str, lam: float = -99.0, d: int = -1, D: int = -1,
                         deseason: str = "seed", p_max: int = 0, q_max: int = 0,
                         include_mean: bool = True) -> str:
    """Rank VARMA(p,q) orders by information criteria (AIC/BIC/HQ).

    Uses the saved characterization (λ/d/deseason and the p,q ceilings) by default;
    p_max/q_max=0 means "use the seed's ceiling". Fits every (p,q) by exact ML and
    ranks by BIC. Complements the CCM / Tiao-Box evidence.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    cons = _SEED.get(name, {}).get("consensus", {})
    if p_max <= 0:
        p_max = int(cons.get("p_ceiling", 2))
    if q_max <= 0:
        q_max = int(cons.get("q_ceiling", 1))
    n, m = ms.nobs, ms.m
    results = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                mod = Model(ms, lam=lam, d=d, D=D, p=p, q=q, deseason=ds,
                            include_mean=include_mean).fit()
                ll = float(mod.loglik); k = _npar(m, p, q, include_mean)
                results.append((p, q, ll, k, -2*ll + 2*k, -2*ll + k*np.log(n),
                                -2*ll + 2*k*np.log(np.log(n))))
            except Exception as e:  # noqa: BLE001
                results.append((p, q, None, None, None, None, str(e)[:40]))
    ok = [r for r in results if r[4] is not None]
    if not ok:
        return "Ninguna especificación convergió. Revisa el seed (λ/d/deseason)."
    ok.sort(key=lambda r: r[5])
    hdr = (f"VARMA order search (λ={lam}, d={d}, deseason={ds or 'no'}, "
           f"mean={include_mean}) — ranked by BIC\n"
           "| p | q | loglik | k | AIC | BIC | HQ |\n|---|---|--------|---|-----|-----|----|")
    body = "\n".join(f"| {p} | {q} | {ll:.2f} | {k} | {aic:.1f} | {bic:.1f} | {hq:.1f} |"
                     for (p, q, ll, k, aic, bic, hq) in ok)
    bp, bq = ok[0][0], ok[0][1]
    return (hdr + "\n" + body +
            f"\n\nRecomendación (mín. BIC): **VARMA({bp},{bq})**. Confirma → "
            f"confirm_and_estimate({name!r}, p={bp}, q={bq}).")


@mcp.tool()
def confirm_and_estimate(name: str, p: int, q: int, lam: float = -99.0,
                         d: int = -1, D: int = -1, deseason: str = "seed",
                         include_mean: bool = True, diag_ar: bool = False,
                         diag_ma: bool = False, diag_cov: bool = False) -> str:
    """Estimate the final VARMA(p,q) by exact ML and store the fit under `name`.

    Uses the saved characterization (λ/d/deseason) by default. diag_ar/diag_ma/
    diag_cov impose diagonal AR/MA/covariance. Returns loglik, Φ/Σ and lets you
    diagnose/forecast next.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    try:
        mod = Model(ms, lam=lam, d=d, D=D, p=p, q=q, deseason=ds,
                    include_mean=include_mean, diag_ar=diag_ar, diag_ma=diag_ma,
                    diag_cov=diag_cov).fit()
    except Exception as e:  # noqa: BLE001
        return f"La estimación de VARMA({p},{q}) falló: {e}"
    _FITS[name] = mod
    m = ms.m; k = _npar(m, p, q, include_mean)
    out = [f"# VARMA({p},{q}) estimado — {name} ({m} series, {ms.nobs} obs; "
           f"λ={lam}, d={d}, deseason={ds or 'no'})",
           f"log-likelihood = {mod.loglik:.4f}  (k={k}, "
           f"AIC={-2*mod.loglik + 2*k:.1f}, BIC={-2*mod.loglik + k*np.log(ms.nobs):.1f})",
           f"scale (reescalado) = {mod.scale}"]
    try:
        out.append("Φ₁ =\n" + np.array2string(np.asarray(mod.phi)[0], precision=3,
                                               suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:
        out.append("Σ =\n" + np.array2string(np.asarray(mod.sigma), precision=4,
                                              suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    out.append("\nSiguiente: diagnose, impulse_response, variance_decomposition, generate_forecast.")
    return "\n".join(out)


@mcp.tool()
def diagnose(name: str, lag: int = 0) -> str:
    """Multivariate residual diagnostics: Hosking Q + Jarque-Bera. A significant Q
    means the order (p or q) is too low."""
    mod = _require_fit(name)
    dd = mod.diagnostics(lag or None)
    ok_q = dd["hosking_p"] > 0.05
    return (f"# Diagnóstico de residuos — {name}\n"
            f"- Hosking Q({dd['hosking_lag']}) = {dd['hosking_Q']:.2f}  df={dd['hosking_df']}  "
            f"p={dd['hosking_p']:.3f} → "
            f"{'sin autocorrelación cruzada ✓' if ok_q else 'AUTOCORRELACIÓN residual ⚠ (sube p/q)'}\n"
            f"- Jarque-Bera = {dd['JB']:.2f}  df={dd['JB_df']}  p={dd['JB_p']:.3f} → "
            f"{'normalidad ok ✓' if dd['JB_p'] > 0.05 else 'no-normal ⚠'}\n"
            + ("Modelo adecuado." if ok_q else "Revisa la especificación (orden)."))


@mcp.tool()
def generate_forecast(name: str, horizon: int = 12) -> str:
    """Forecast the fitted VARMA `horizon` steps ahead with 95% bands (drvarma's
    native, level-unit, scale-correct bands — never a hand-rolled relative-std band)."""
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
    irf = np.asarray(mod.irf(horizon, orthogonalized=True))
    nm = mod.series.names
    hs = sorted({h for h in (0, 1, 2, 4, 8, horizon) if 0 <= h <= horizon})
    out = [f"# Respuestas al impulso ortogonalizadas (OIRF) — {name}"]
    for i in range(mod.series.m):
        for k in range(mod.series.m):
            out.append(f"- {nm[i]} ← shock {nm[k]}: " +
                       ", ".join(f"h{h}={irf[h, i, k]:+.3f}" for h in hs))
    return "\n".join(out)


@mcp.tool()
def variance_decomposition(name: str, horizon: int = 12) -> str:
    """Forecast-error variance decomposition (FEVD, %) at the given horizon."""
    mod = _require_fit(name)
    dec = np.asarray(mod.fevd(horizon))[-1]
    nm = mod.series.names
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
