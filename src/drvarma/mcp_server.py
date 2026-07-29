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
import os

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


# Engine model-adequacy codes (as the C `est`/`elf`), used to reject bad fits.
_IFAULT = {1: "Q no definida positiva", 2: "raíz unitaria AR", 3: "no estacionario",
           4: "MA no invertible", 5: "fallo numérico"}

# Optimiser termination codes (raxopt / qnewtopt.c). Only 1 is convergence on the
# gradient; 2 converges on the step and often signals ill-conditioning.
_TERMCODE = {1: "norma del gradiente escalado <= gradtol",
             2: "distancia escalada entre los dos últimos pasos <= steptol",
             3: "el último paso global no encontró un punto menor",
             4: "límite de iteraciones alcanzado",
             5: "cinco pasos consecutivos de longitud máxima"}


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
    # Delegate to Model.prepare() — the SAME code path fit() uses — instead of
    # replaying deseason+transform here. Replaying it kept the two in sync only by
    # hand, so any change to the estimation pipeline would silently make us identify
    # on a different series from the one we estimate on.
    w, _, _, _, _ = Model(ms, lam=lam, d=d, D=D, p=1, q=0,
                          deseason=deseason).prepare(scale=1.0)
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
                # Decide header vs no-header by SNIFFING the first line: if every
                # field parses as a float it is data, not a header. The old logic
                # skipped the first row unconditionally and only retried when the
                # result was ALL NaN, so a purely numeric header-less CSV silently
                # lost observation 1.
                with open(path) as fh:
                    first = fh.readline().strip()
                fields = [f.strip() for f in first.split(",") if f.strip() != ""]
                try:
                    [float(f) for f in fields]
                    has_header = False
                except ValueError:
                    has_header = True
                data = np.genfromtxt(path, delimiter=",",
                                     skip_header=1 if has_header else 0)
                if has_header and names is None:
                    names = fields
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
            # ART's order is lambda -> d -> seasonality, and d is therefore chosen
            # BEFORE seasonality is dealt with. ADF/KPSS have low power against a
            # seasonal series, so they can fail to reject at d=1 and escalate to
            # d=2 spuriously — over-differencing and injecting an MA unit root.
            # Cap the search at d=1 and let the seasonality step handle the rest:
            # on IPC_ES the uncapped search returns d=2 from January (n=216) but
            # d=1 from February (n=215), i.e. one observation flips the order of
            # integration. Capped, every series here is a stable d=1.
            d = int(_id.recommended_d(_id.unit_root_tests(ts, lam=lam, max_d=1)))
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


def _ccm_values(w):
    """Sample cross-correlation matrices R(k) of the prepared series.

    Returns (R, bound) with R[k-1] the m×m matrix at lag k and bound = 2/sqrt(n).
    Single source of truth: the CCM table and the CCM plot both call this, so a
    change to the statistic can never make them disagree.
    """
    n, m = w.shape
    wc = w - w.mean(0)
    sd = np.sqrt(np.diag((wc.T @ wc) / n))
    bound = 2.0 / np.sqrt(n)

    def R(k):
        return ((wc[k:].T @ wc[:-k]) / n) / np.outer(sd, sd)

    return R, bound, n, m


def _tiaobox_tratios(w, max_order):
    """Tiao-Box partial autoregression matrices: t-ratios of the last block Φ_kk.

    Returns a dict {k: (m, m) array of t-ratios}; lags that cannot be fitted
    (insufficient sample / singular) are absent. Shared by the table and the plot.
    """
    n0, m = w.shape
    out = {}
    for k in range(1, max_order + 1):
        Y = w[k:]
        T = len(Y)
        if T <= 1 + k * m:
            continue
        X = np.column_stack([np.ones(T)] + [w[k - l:n0 - l] for l in range(1, k + 1)])
        try:
            XtXi = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            continue
        B = XtXi @ X.T @ Y
        resid = Y - X @ B
        Sig = (resid.T @ resid) / (T - X.shape[1])
        r0 = 1 + (k - 1) * m
        t = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                se = np.sqrt(Sig[i, i] * XtXi[r0 + j, r0 + j])
                t[i, j] = B[r0 + j, i] / se if se > 0 else 0.0
        out[k] = t
    return out


def _plot_path(name, kind, path):
    """Resolve the output file for a plot tool."""
    if path:
        return os.path.abspath(os.path.expanduser(path))
    import tempfile
    return os.path.join(tempfile.gettempdir(), f"multiart_{name}_{kind}.png")


def _grid_axes(m, per=2.4):
    """m×m grid of axes for the identification plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(m, m, figsize=(per * m + 1.5, per * m),
                             squeeze=False, sharex=True)
    return plt, fig, axes


def _drvus_onesided(ax, vals, lags, band, freq, title, cmax=None):
    """One ONE-SIDED panel in the drvus house style (`x11plots.c`, `set border 2`).

    Same conventions as `plots._draw_ccf_panel`, which reproduces the C exactly:
    borderless except a thick left axis, solid black seasonal grid lines at
    freq/2·freq/3·freq, a solid zero line, dashed ±band lines and thick black
    impulses. Kept visually identical so these plots sit alongside the residual
    CCFs of the .out report rather than looking like a different program.
    """
    from .plots import _snap_cmax
    if cmax is None:
        cmax = _snap_cmax(max(float(np.max(np.abs(vals))), band))
    seas = [s for s in range(freq, int(max(lags)) + 1, freq)] if freq > 1 else []
    for xx in seas:
        ax.plot([xx, xx], [-cmax, cmax], color="k", lw=0.8, zorder=1)
    ax.axhline(band, color="k", ls="--", lw=1.0, zorder=2)
    ax.axhline(-band, color="k", ls="--", lw=1.0, zorder=2)
    ax.axhline(0.0, color="k", lw=1.4, zorder=2)
    ax.vlines(lags, 0.0, vals, color="k", lw=3.0, zorder=3)
    ax.set_ylim(-cmax, cmax)
    ax.set_yticks([-cmax, -cmax / 2.0, 0.0, cmax / 2.0, cmax])
    ax.set_xlim(0.5, int(max(lags)) + 0.5)
    if seas:
        ax.set_xticks(seas)                      # drvus: labels on the seasonal grid
    else:
        # No seasonal line falls inside the window (e.g. 5 lags of monthly data):
        # fall back to readable integer ticks instead of a single label.
        top = int(max(lags))
        step = max(1, top // 6)
        ax.set_xticks(list(range(step, top + 1, step)))
    ax.tick_params(axis="y", direction="out", length=4, labelsize=9)
    ax.tick_params(axis="x", length=0, labelsize=9)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.6)
    ax.set_title(title, fontsize=10)


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
    R, bound, n, m = _ccm_values(w)
    out = [f"# CCM — {name} (n={n}, m={m}; λ={lam}, d={d}, deseason={ds or 'no'})",
           f"Símbolos: + (ρ>2/√n={bound:.3f}), - (<-2/√n), . (no signif.). Series: {ms.names}", ""]
    for k in range(1, n_lags + 1):
        Rk = R(k)
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
    tt = _tiaobox_tratios(w, max_order)
    for k in range(1, max_order + 1):
        if k not in tt:
            out.append(f"lag {k}: (muestra insuficiente o singular)"); continue
        out.append(f"lag {k}:")
        for i in range(m):
            out.append("  " + " ".join(
                "+" if tt[k][i, j] > 1.96 else "-" if tt[k][i, j] < -1.96 else "."
                for j in range(m)))
    return "\n".join(out)


@mcp.tool()
def plot_cross_correlation_matrices(name: str, lam: float = -99.0, d: int = -1,
                                    D: int = -1, deseason: str = "seed",
                                    n_lags: int = 0, path: str = "") -> str:
    """PLOT of the CCM: m×m grid, cell (i,j) = ρ_ij(k) for k=1..K, with ±2/√n bands.

    The visual counterpart of `cross_correlation_matrices` (same statistic, shared
    code). Row i = series i, column j = series j; a bar at lag k in cell (i,j) means
    series j lagged k correlates with series i now. A cell that CUTS OFF after lag q
    points to MA(q); slow decay points to AR terms. Writes a PNG and returns its path.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    w = _prepared_w(ms, lam, d, D, ds)
    R, bound, n, m = _ccm_values(w)
    if n_lags <= 0:
        n_lags = min(max(2 * int(ms.freq), 8), max(4, n // 4))
    lags = np.arange(1, n_lags + 1)
    vals = np.array([R(k) for k in lags])                 # (K, m, m)
    from .plots import _snap_cmax
    plt, fig, axes = _grid_axes(m)
    cmax = _snap_cmax(max(float(np.abs(vals).max()), bound))   # common scale
    for i in range(m):
        for j in range(m):
            _drvus_onesided(axes[i][j], vals[:, i, j], lags, bound, int(ms.freq),
                            f"{ms.names[i]} ← {ms.names[j]}(-k)", cmax=cmax)
            if j != 0:
                axes[i][j].tick_params(labelleft=False)
    fig.suptitle(f"CCM — {name}  (n={n}, λ={lam}, d={d}, deseason={ds or 'no'}; "
                 f"banda ±{bound:.3f})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _plot_path(name, "ccm", path)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    nsig = int((np.abs(vals) > bound).sum())
    return (f"CCM guardada en {out}\n"
            f"Rejilla {m}×{m}, retardos 1..{n_lags}, banda ±{bound:.3f}. "
            f"{nsig} de {vals.size} correlaciones superan la banda.\n"
            f"Fila = ecuación, columna = variable retardada. Corte tras el retardo q "
            f"en una celda ⇒ MA(q); decaimiento lento ⇒ términos AR.")


@mcp.tool()
def plot_cross_correlation_functions(name: str, lam: float = -99.0, d: int = -1,
                                     D: int = -1, deseason: str = "seed",
                                     n_lags: int = 0, path: str = "") -> str:
    """PLOT of the TWO-SIDED CCFs for every pair — reads DIRECTION, unlike the CCM.

    One panel per pair (i<j) over lags -K..K with ±2/√n bands. Bars on the RIGHT
    (k>0) mean series j leads series i; bars on the LEFT (k<0) mean i leads j. A
    strictly one-sided pattern is the visual signature of an exogenous variable;
    bars on both sides mean feedback. Writes a PNG and returns its path.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    w = _prepared_w(ms, lam, d, D, ds)
    from .diagnostics import ccf as _ccf, qccf as _qccf
    from .plots import _draw_ccf_panel
    n, m = w.shape
    freq = int(ms.freq)
    if n_lags <= 0:                       # drvus graphic window = 3·(freq+1)
        n_lags = 3 * (freq + 1) if freq > 1 else min(9, n // 4)
    n_lags = min(n_lags, n - 2)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    if not pairs:
        return "Se necesitan al menos 2 series para una CCF cruzada."
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 1, figsize=(11.0, 3.0 * len(pairs)),
                             squeeze=False, layout="constrained")
    band = 2.0 / np.sqrt(n)
    qfmt = "Q( %d ) = %.1f" if freq > 4 else "Q ( %d ) = %.1f"
    lines = []
    for ax, (i, j) in zip([a[0] for a in axes], pairs):
        # CANONICAL ORIENTATION (drvus / the .out report): for the pair titled
        # "A - B", k > 0 means A --> B, i.e. the FIRST name leads. `ccf(w1, w2)`
        # returns w2 leading at k > 0, so the arguments go in reversed.
        rho = _ccf(w[:, j], w[:, i], n_lags)
        Q, _df, _p = _qccf(w[:, i], w[:, j], n_lags)
        _draw_ccf_panel(ax, rho, n_lags, n, freq,
                        "%s - %s" % (ms.names[i], ms.names[j]),
                        qfmt % (n_lags, Q))
        # Report WHICH lags, not just how many: an isolated spike at the seasonal
        # lag means something very different from one at lag 1.
        lead = [k for k in range(1, n_lags + 1) if abs(rho[n_lags + k]) > band]
        lag_ = [-k for k in range(1, n_lags + 1) if abs(rho[n_lags - k]) > band]
        lines.append(
            f"- {ms.names[i]} - {ms.names[j]}: ρ(0)={rho[n_lags]:+.3f} | "
            f"k>0 ({ms.names[i]} → {ms.names[j]}): {lead or '—'} | "
            f"k<0 ({ms.names[j]} → {ms.names[i]}): {sorted(lag_) or '—'}")
    out = _plot_path(name, "ccf", path)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return ("CCF guardada en " + out +
            f"\nFormato drvus, ventana 3·(s+1)={n_lags}, banda ±{band:.3f}.\n"
            "Convención del informe .out: en el par «A - B», k>0 significa A → B.\n"
            + "\n".join(lines) +
            "\nPatrón unilateral ⇒ exogeneidad; barras a ambos lados ⇒ retroalimentación.")


@mcp.tool()
def plot_partial_autoregression_matrices(name: str, lam: float = -99.0, d: int = -1,
                                         D: int = -1, deseason: str = "seed",
                                         max_order: int = 6, path: str = "") -> str:
    """PLOT of the Tiao-Box partial autoregression matrices (t-ratios).

    The visual counterpart of `partial_autoregression_matrices` (same computation,
    shared code). m×m grid; cell (i,j) shows the t-ratio of Φ_kk[i,j] for k=1..K
    with the ±1.96 band. The AR order p is the last lag with bars outside the band.
    Writes a PNG and returns its path.
    """
    ms = _require(name)
    lam, d, D, ds = _resolve(name, lam, d, D, deseason)
    w = _prepared_w(ms, lam, d, D, ds)
    m = w.shape[1]
    tt = _tiaobox_tratios(w, max_order)
    if not tt:
        return "Ningún retardo pudo estimarse (muestra insuficiente)."
    lags = sorted(tt)
    vals = np.array([tt[k] for k in lags])
    import math
    plt, fig, axes = _grid_axes(m)
    # NOT _snap_cmax: that one caps at 1.0 because it is meant for correlations.
    # These are t-ratios and are unbounded — capping them clips the bars and hides
    # the ±1.96 band entirely.
    cmax = max(math.ceil(float(np.abs(vals).max())), 3.0)      # common scale
    for i in range(m):
        for j in range(m):
            _drvus_onesided(axes[i][j], vals[:, i, j], lags, 1.96, int(ms.freq),
                            f"{ms.names[i]} ← {ms.names[j]}(-k)", cmax=cmax)
            if j != 0:
                axes[i][j].tick_params(labelleft=False)
    fig.suptitle(f"Tiao-Box: autorregresión parcial (t-ratios) — {name}  "
                 f"(λ={lam}, d={d}, deseason={ds or 'no'}; banda ±1.96)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _plot_path(name, "tiaobox", path)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    # How MANY entries are significant at each lag, not just whether any is: with
    # m² × K t-tests at 5 %, isolated hits are expected by chance, so a lag with
    # one hit is not the same evidence as a lag with several.
    per_lag = {k: int((np.abs(tt[k]) > 1.96).sum()) for k in lags}
    expected = 0.05 * m * m
    last = [k for k in lags if per_lag[k] > 0]
    detail = ", ".join(f"lag {k}: {per_lag[k]}/{m * m}" for k in lags)
    msg = (f"Tiao-Box guardada en {out}\n"
           f"Rejilla {m}×{m}, retardos {lags[0]}..{lags[-1]}, banda ±1.96.\n"
           f"Elementos significativos por retardo — {detail}\n"
           f"(al 5 % se esperan ~{expected:.1f} por retardo sólo por azar)\n")
    if not last:
        return msg + "Ningún t significativo ⇒ sugiere p=0."
    solido = [k for k in lags if per_lag[k] > expected]
    msg += f"Último retardo con algún t significativo: {max(last)}."
    if solido:
        msg += (f" Retardos por encima de lo esperado por azar: {solido} ⇒ "
                f"sugiere p={max(solido)}.")
    else:
        msg += " Ningún retardo supera lo esperado por azar: sospecha de p bajo."
    return msg


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
    rejected = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                mod = Model(ms, lam=lam, d=d, D=D, p=p, q=q, deseason=ds,
                            include_mean=include_mean).fit()
                ll = float(mod.loglik); k = _npar(m, p, q, include_mean)
                fault = int(getattr(mod, "ifault", 0) or 0)
                # A non-finite loglik is a numerical failure, NOT a perfect fit: it
                # makes AIC/BIC/HQ -inf, which sorts BEST and would be emitted as the
                # recommendation. ifault != 0 flags an inadequate model (1 Q not PD,
                # 2 AR unit root, 3 non-stationary, 4 MA non-invertible, 5 numerical)
                # and the engine already computes it — so honour it here.
                tc = getattr(mod, "termcode", None)
                if not np.isfinite(ll):
                    rejected.append((p, q, "loglik no finito"))
                elif fault:
                    rejected.append((p, q, f"ifault={fault} ({_IFAULT.get(fault, '?')})"))
                elif tc is not None and tc not in (1, 2):
                    # Not converged (or never run): the estimates are not a maximum,
                    # so the likelihood — and every criterion built on it — is not
                    # comparable with the converged fits. Common on high orders,
                    # where the VARMA likelihood is ill conditioned.
                    rejected.append((p, q, "no convergió: "
                                     + ("optimizador no ejecutado" if tc == 0
                                        else _TERMCODE.get(tc, f"termcode={tc}"))))
                else:
                    results.append((p, q, ll, k, -2*ll + 2*k, -2*ll + k*np.log(n),
                                    -2*ll + 2*k*np.log(np.log(n))))
            except Exception as e:  # noqa: BLE001
                rejected.append((p, q, f"excepción: {type(e).__name__}"))
    ok = [r for r in results if r[4] is not None and np.isfinite(r[4])]
    if not ok:
        return ("Ninguna especificación convergió. Revisa el seed (λ/d/deseason).\n"
                + "\n".join(f"  ({p},{q}): {why}" for p, q, why in rejected))
    ok.sort(key=lambda r: r[5])
    hdr = (f"VARMA order search (λ={lam}, d={d}, deseason={ds or 'no'}, "
           f"mean={include_mean}) — ranked by BIC\n"
           "| p | q | loglik | k | AIC | BIC | HQ |\n|---|---|--------|---|-----|-----|----|")
    body = "\n".join(f"| {p} | {q} | {ll:.2f} | {k} | {aic:.1f} | {bic:.1f} | {hq:.1f} |"
                     for (p, q, ll, k, aic, bic, hq) in ok)
    bp, bq = ok[0][0], ok[0][1]
    out = [hdr, body]
    if rejected:
        out.append("\n**Descartadas** (no entran en el ranking ni en la "
                   "recomendación):\n" +
                   "\n".join(f"- ({p},{q}): {why}" for p, q, why in rejected))
    # Nesting sanity check: for pure-AR specs the exact likelihood must increase
    # with p, since VAR(p) is nested in VAR(p+1). A drop means the optimiser failed
    # to converge on the larger model while still reporting ifault=0, which would
    # silently corrupt every criterion computed from it.
    ar_only = sorted([(p, ll) for (p, q, ll, *_rest) in ok if q == 0])
    drops = [(p1, p2) for (p1, l1), (p2, l2) in zip(ar_only, ar_only[1:])
             if l2 < l1 - 1e-6]
    if drops:
        out.append("\n⚠ **Aviso de convergencia**: la log-verosimilitud EXACTA cae "
                   "al aumentar p en " +
                   ", ".join(f"VAR({a})→VAR({b})" for a, b in drops) +
                   ". Como VAR(p) está anidado en VAR(p+1), esto es imposible en el "
                   "óptimo: el ajuste mayor NO convergió. Sus criterios de "
                   "información no son fiables.")
    out.append(f"\nRecomendación (mín. BIC): **VARMA({bp},{bq})**. Confirma → "
               f"confirm_and_estimate({name!r}, p={bp}, q={bq}).")
    return "\n".join(out)


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
    # Convergence diagnosis. In multivariate VARMA this matters as much as the
    # estimates: WHY the optimiser stopped tells you whether the likelihood was
    # well conditioned. Only termcode 1 is convergence on the gradient.
    tc = getattr(mod, "termcode", None)
    nit = getattr(mod, "nit", None)
    if tc is None:
        out.append("Convergencia: no informada por el backend.")
    else:
        crit = _TERMCODE.get(tc, "?")
        estado = "CONVERGIÓ" if tc in (1, 2) else "SE DETUVO"
        out.append(f"Convergencia: {estado}"
                   + (f" tras {nit} iteraciones" if nit is not None else "")
                   + f" — criterio: {crit}")
        if tc == 2:
            out.append("  ⚠ paró por steptol, NO por el gradiente: el paso se "
                       "hizo pequeño mientras el gradiente puede seguir siendo "
                       "apreciable. Típico de verosimilitud mal condicionada "
                       "(casi no identificada / factores comunes). Desconfía de "
                       "los errores estándar; reestima desde otros valores "
                       "iniciales o baja el orden.")
        elif tc in (3, 4, 5):
            out.append("  ⚠ NO es convergencia: el optimizador se rindió. Las "
                       "estimaciones no son un máximo — no uses este ajuste ni "
                       "sus criterios de información.")
    fault = int(getattr(mod, "ifault", 0) or 0)
    if fault:
        out.append(f"⚠ ifault={fault} ({_IFAULT.get(fault, '?')}) — "
                   "el ajuste NO es adecuado; no lo uses para IRF/FEVD.")
    try:  # ALL AR matrices, not just the first
        for k, Pk in enumerate(np.asarray(mod.phi)[:p], start=1):
            out.append(f"Φ{k} =\n" + np.array2string(Pk, precision=3,
                                                     suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:  # MA matrices
        for k, Tk in enumerate(np.asarray(mod.theta)[:q], start=1):
            out.append(f"Θ{k} =\n" + np.array2string(Tk, precision=3,
                                                     suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:
        out.append("μ = " + np.array2string(np.asarray(mod.mu), precision=4,
                                            suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:
        out.append("Σ =\n" + np.array2string(np.asarray(mod.sigma), precision=4,
                                              suppress_small=True))
        S = np.asarray(mod.sigma); sd = np.sqrt(np.diag(S))
        R = S / np.outer(sd, sd)
        out.append("Corr(residuos) =\n" + np.array2string(R, precision=3,
                                                          suppress_small=True))
    except Exception:  # noqa: BLE001
        pass
    try:  # standard errors / t-ratios: the engine computes them, so report them
        se = np.asarray(mod.std_errors, dtype=float).ravel()
        pars = np.asarray(mod.params, dtype=float).ravel()
        if se.size == pars.size and se.size:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(se > 0, pars / se, np.nan)
            out.append("parámetros (est / s.e. / t):\n" + "\n".join(
                f"  θ[{i}] = {pv:9.4f}  {sv:9.4f}  {tv:8.2f}"
                + ("  *" if abs(tv) > 1.96 else "")
                for i, (pv, sv, tv) in enumerate(zip(pars, se, t))))
    except Exception:  # noqa: BLE001
        pass
    out.append("\nSiguiente: diagnose, impulse_response, variance_decomposition, generate_forecast.")
    return "\n".join(out)


@mcp.tool()
def diagnose(name: str, lag: int = 0) -> str:
    """Multivariate residual diagnostics: Hosking Q + Jarque-Bera. A significant Q
    means the order (p or q) is too low."""
    mod = _require_fit(name)
    ms = _require(name)
    dd = mod.diagnostics(lag or None)
    ok_q = dd["hosking_p"] > 0.05
    lines = [f"# Diagnóstico de residuos — {name}",
             f"- Hosking Q({dd['hosking_lag']}) = {dd['hosking_Q']:.2f}  "
             f"df={dd['hosking_df']}  p={dd['hosking_p']:.3f} → "
             + ("sin autocorrelación cruzada ✓" if ok_q
                else "AUTOCORRELACIÓN residual ⚠ (sube p/q)"),
             f"- Jarque-Bera = {dd['JB']:.2f}  df={dd['JB_df']}  "
             f"p={dd['JB_p']:.3f} → "
             + ("normalidad ok ✓" if dd['JB_p'] > 0.05 else "no-normal ⚠")]
    # Per-lag residual ACF. The aggregated Hosking statistic has little power
    # against an isolated spike at the seasonal lag: with df=56 a residual ACF of
    # 0.27 at lag 12 still gave p=0.45 and "modelo adecuado". Report the seasonal
    # lags explicitly so leftover seasonality cannot hide behind the aggregate.
    seasonal_flag = False
    try:
        r = np.asarray(mod.residuals, dtype=float)
        nr = len(r); rc = r - r.mean(0); sd = rc.std(0)
        bound = 2.0 / np.sqrt(nr)
        s = int(ms.freq)
        lags = sorted({1, 2, 3, s, 2 * s} - {0})
        lags = [L for L in lags if L < nr // 2]
        rows = []
        for L in lags:
            a = (rc[L:] * rc[:-L]).mean(0) / (sd ** 2)
            marks = "".join("*" if abs(v) > bound else " " for v in a)
            tag = "  ← estacional" if s and L % s == 0 else ""
            rows.append(f"    lag {L:2d}: " +
                        "  ".join(f"{nm}={v:+.3f}" for nm, v in zip(ms.names, a))
                        + f" {marks}{tag}")
            if s and L % s == 0 and np.any(np.abs(a) > bound):
                seasonal_flag = True
        lines.append(f"- ACF de residuos por retardo (banda ±{bound:.3f}):")
        lines.extend(rows)
    except Exception:  # noqa: BLE001
        pass
    if seasonal_flag:
        lines.append("⚠ ACF significativa en un retardo ESTACIONAL: queda "
                     "estacionalidad sin modelar. Subir p/q no es la solución — "
                     "revisa la desestacionalización (λ/d/deseason y el período "
                     "inicial declarado).")
    if ok_q and not seasonal_flag:
        lines.append("Modelo adecuado.")
    else:
        lines.append("Revisa la especificación.")
    return "\n".join(lines)


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
