"""HTML forecast report (SPS — Sistema de Previsión y Seguimiento).

Multivariate port of fue's ``report_forecast.py`` so drvarma forecasts are
*homologable* with the fuf reports.  drvarma is multivariate, so this writes one
self-contained HTML report **per series** (the univariate SPS layout fue produces
for each variable):

LEFT column  : forecast table (history + first-year forecast + the H=L row) +
               model details.
RIGHT column : a single SVG with two x-aligned panels —
                 top    — annual rate of change (history dots+line + forecast +
                          ±1σ bands + separator at the forecast origin);
                 bottom — ERR residuals as impulses + ±2σ bands, x-axis truncated
                          at the forecast origin.

The numbers reuse drvarma's own forecast machinery (``forecast_levels`` +
``forecast_level_variances``), so they match the ``.forecast`` report exactly.

Public API
----------
write_forecast_report(model, path_prefix, L=None, ...)  -> list of written paths

Requires jinja2 (``pip install "drvarma[forecast-report]"``).

License: GPL-2.0-or-later
"""

import io
import math
from datetime import date as _date

import numpy as np

from . import transform
from .report import obs_to_date


# ── HTML/CSS template (Jinja2), mirroring fue's SPS report ────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ page_title }}</title>
  <style>
    :root {
      --font: system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif;
      --mono: ui-monospace, "Cascadia Code", "Fira Code", monospace;
      --fg:#111827; --muted:#6b7280; --border:#e5e7eb;
      --fore-bg:#eff6ff; --accent:#1d4ed8; --max-w:1200px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0 }
    body { font-family: var(--font); font-size: 14px; color: var(--fg);
           background: #fff; padding: 2rem 1.5rem; }
    .container { max-width: var(--max-w); margin: 0 auto }
    header { border-bottom: 2.5px solid var(--fg); padding-bottom: .9rem;
             margin-bottom: 1.8rem; }
    .sps-label { font-size: .72rem; font-weight: 600; text-transform: uppercase;
                 letter-spacing: .1em; color: var(--accent); margin-bottom: .3rem; }
    header h1 { font-size: 1.55rem; font-weight: 700; line-height: 1.2 }
    .meta { display: flex; flex-wrap: wrap; gap: .25rem 1.6rem; margin-top: .55rem;
            font-size: .81rem; color: var(--muted); }
    .meta strong { color: var(--fg) }
    .report-body { display: grid;
                   grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
                   gap: 2rem 2.5rem; align-items: start; }
    .col-right figure { margin: 0 }
    .col-right figure svg { width: 100%; height: auto; display: block }
    .col-right figcaption { font-size: .73rem; color: var(--muted);
                            margin-top: .3rem; text-align: center; font-style: italic; }
    .table-wrap { overflow-x: auto }
    .data-table { width: 100%; border-collapse: collapse; font-size: .77rem; }
    .data-table caption { font-size: .72rem; font-weight: 600; text-transform: uppercase;
                          letter-spacing: .07em; color: var(--muted); text-align: left;
                          padding-bottom: .5rem; }
    .data-table thead tr:first-child th { padding: .3rem .45rem;
        border-bottom: 1px solid var(--border); font-size: .65rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
        text-align: center; }
    .data-table thead tr:first-child th:first-child { text-align: left }
    .data-table thead tr:last-child th { padding: .28rem .45rem;
        border-bottom: 2px solid var(--fg); font-size: .68rem; font-weight: 600;
        text-align: right; white-space: nowrap; }
    .data-table thead tr:last-child th:first-child { text-align: left }
    .data-table td { padding: .22rem .45rem; border-bottom: 1px solid var(--border);
        text-align: right; font-family: var(--mono); font-size: .74rem;
        font-variant-numeric: tabular-nums; white-space: nowrap; }
    .data-table td:first-child { text-align: left; font-family: var(--font) }
    .data-table tr.fore td { background: var(--fore-bg) }
    .data-table tr.sep  td { border-top: 2px solid var(--fg) }
    .data-table tr.blank td { padding-top: .04rem; padding-bottom: .04rem;
        border-bottom: none; background: #fff; height: 6px; }
    .data-table .na { color: var(--muted) }
    .model-details { margin-top: 1.2rem }
    details > summary { cursor: pointer; font-size: .72rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: .08em; color: var(--muted);
        padding: .4rem 0; border-top: 1px solid var(--border); list-style: none; }
    details > summary::after { content: " ▸"; font-size: .7rem }
    details[open] > summary::after { content: " ▾" }
    .model-grid { display: flex; flex-wrap: wrap; gap: .3rem 2rem; padding: .7rem 0;
                  font-size: .79rem; }
    .model-grid span { color: var(--muted) }
    .model-grid strong { font-family: var(--mono) }
    footer { margin-top: 2.5rem; padding-top: .7rem; border-top: 1px solid var(--border);
             font-size: .71rem; color: var(--muted); }
    @media print { body { padding: 0; font-size: 10pt }
        .report-body { grid-template-columns: 1fr 1.2fr } .data-table { font-size: 7.5pt }
        details { display: none } }
    @media (max-width: 860px) { .report-body { grid-template-columns: 1fr } }
  </style>
</head>
<body>
<div class="container">
<header>
  <div class="sps-label">SPS{% if sps_name %}: {{ sps_name }}{% endif %}</div>
  <h1>{{ page_title }}</h1>
  <div class="meta">
    {% if source %}<span>Data Source&#160;<strong>{{ source }}</strong></span>{% endif %}
    <span>Forecast Origin&#160;<strong>{{ origin }}</strong></span>
    <span>Horizon&#160;<strong>{{ horizon }}&#160;{{ freq_label }}</strong></span>
    <span>Generated&#160;<strong>{{ generated }}</strong></span>
  </div>
</header>
<div class="report-body">
  <div class="col-left">
    <div class="table-wrap">
    <table class="data-table">
      <caption>Forecast — {{ series_name }}</caption>
      <thead>
        <tr>
          <th rowspan="2">Date</th>
          <th colspan="2">Level</th>
          <th colspan="2">{{ diff1_label }}</th>
          <th colspan="2">Annual (%)</th>
          <th rowspan="2">ERR<br>(%)</th>
        </tr>
        <tr>
          <th>Value</th><th>Std&#160;(%)</th>
          <th>(%)</th><th>Std&#160;(%)</th>
          <th>(%)</th><th>Std&#160;(%)</th>
        </tr>
      </thead>
      <tbody>
        {% for row in hist_rows %}
        <tr>
          <td>{{ row.date }}</td><td>{{ row.level }}</td><td class="na">—</td>
          <td>{{ row.diff1 }}</td><td class="na">—</td>
          <td>{{ row.annual }}</td><td class="na">—</td><td>{{ row.err }}</td>
        </tr>
        {% endfor %}
        {% for row in fore_rows_main %}
        <tr class="fore{% if loop.first %} sep{% endif %}">
          <td>{{ row.date }}</td><td>{{ row.level }}</td><td>{{ row.level_std }}</td>
          <td>{{ row.diff1 }}</td><td>{{ row.diff1_std }}</td>
          <td>{{ row.annual }}</td><td>{{ row.annual_std }}</td><td class="na">—</td>
        </tr>
        {% endfor %}
        {% if fore_row_end %}
        <tr class="fore blank"><td colspan="8"></td></tr>
        <tr class="fore">
          <td>{{ fore_row_end.date }}</td><td>{{ fore_row_end.level }}</td>
          <td>{{ fore_row_end.level_std }}</td><td>{{ fore_row_end.diff1 }}</td>
          <td>{{ fore_row_end.diff1_std }}</td><td>{{ fore_row_end.annual }}</td>
          <td>{{ fore_row_end.annual_std }}</td><td class="na">—</td>
        </tr>
        {% endif %}
      </tbody>
    </table>
    </div>
    <div class="model-details">
      <details>
        <summary>Model details</summary>
        <div class="model-grid">
          <div><span>Model&#160;</span><strong>{{ stem }}</strong></div>
          <div><span>npar&#160;</span><strong>{{ npar }}</strong></div>
          <div><span>σ²&#160;</span><strong>{{ sigma2 }}</strong></div>
          <div><span>logLik&#160;</span><strong>{{ loglik }}</strong></div>
          <div><span>AIC&#160;</span><strong>{{ aic }}</strong></div>
          <div><span>BIC&#160;</span><strong>{{ bic }}</strong></div>
          <div><span>N&#160;</span><strong>{{ nobs }}</strong></div>
          <div><span>Sample&#160;</span><strong>{{ sample }}</strong></div>
        </div>
      </details>
    </div>
  </div>
  <div class="col-right">
    <figure>
      {{ charts_svg | safe }}
      <figcaption>Forecast bands ±1σ &nbsp;·&nbsp; ERR bands ±2σ</figcaption>
    </figure>
  </div>
</div>
<footer>drvarma {{ version }} · {{ generated }}</footer>
</div>
</body>
</html>
"""


# ── forecast numbers (reuse drvarma's machinery) ──────────────────────────────

def _forecast_arrays(model, L, b=0):
    """Per-(h, series) forecast quantities, identical to ``report.forecast_report``.

    Returns a dict of (L, m) arrays: level, level_std, diff1, diff1_std, annual,
    annual_std (variations in %), plus cf (L,m) and bc (nobs,m).
    """
    from .forecast import forecast_levels, forecast_level_variances
    res, m, freq = model.result, model.series.m, model.series.freq
    scale, lam = model.scale, model.lam
    bc = model._bc
    nobs_raw = bc.shape[0]
    lev_des, _ = forecast_levels(res, model._w, bc, lam=lam, scale=scale,
                                 d=model.d, D=model.D, s=freq, L=L, b=b)
    v_lvl, v_mon, v_ann = forecast_level_variances(
        res["phi"], res["theta"], res["sigma"], L, model.d, model.D, freq)
    origin = nobs_raw - b
    dseas = np.zeros((L, m))
    if model.deseason and model._dummies is not None:
        for l in range(1, L + 1):
            period = (origin + l + model.series.start[1] - 2) % freq
            dseas[l - 1] = model._dummies[:, period]
    cf = scale * transform.boxcox_fwd(lev_des, lam)
    level = lev_des + dseas
    sc = 100.0 / scale
    diff1 = np.zeros((L, m)); annual = np.zeros((L, m))
    lvl_std = np.zeros((L, m)); d1_std = np.zeros((L, m)); an_std = np.zeros((L, m))
    for l in range(1, L + 1):
        for i in range(m):
            g2 = cf[0, i] - bc[nobs_raw - 1, i] if l == 1 else cf[l - 1, i] - cf[l - 2, i]
            if l <= freq:
                idx = nobs_raw - freq + l
                g3 = (cf[l - 1, i] - bc[idx - 1, i]) if idx >= 1 else 0.0
            else:
                g3 = cf[l - 1, i] - cf[l - 1 - freq, i]
            diff1[l - 1, i] = sc * g2
            annual[l - 1, i] = sc * g3
            lvl_std[l - 1, i] = sc * np.sqrt(v_lvl[l, i, i])
            d1_std[l - 1, i] = sc * np.sqrt(v_mon[l, i, i])
            an_std[l - 1, i] = sc * np.sqrt(v_ann[l, i, i])
    return {"level": level, "level_std": lvl_std, "diff1": diff1,
            "diff1_std": d1_std, "annual": annual, "annual_std": an_std,
            "cf": cf, "bc": bc}


# ── helpers ───────────────────────────────────────────────────────────────────

def _f2(v):
    return "%.2f" % v


def _date_str(model, k):
    """Date label "subperiod/year" for 1-indexed observation k (as the .forecast)."""
    per, sub = obs_to_date(model.series.start[0], model.series.start[1], k,
                           model.series.freq)
    return "%d/%d" % (sub, per) if model.series.freq > 1 else "%d" % per


def _spines(ax, keep):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(s in keep)


def _fig_to_svg(fig):
    import matplotlib.pyplot as plt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches=None)
    plt.close(fig)
    buf.seek(0)
    svg = buf.read()
    idx = svg.find("<svg")
    return svg[idx:] if idx >= 0 else svg


def _year_ticks(model, nobs, L):
    freq = model.series.freq
    begyear, begtime = model.series.start

    def yr_per(obs1):
        total = begyear * freq + (begtime - 1) + (obs1 - 1)
        return int(total // freq), int(total % freq + 1)

    prevby = previndex = None
    for i in range(freq):
        yr, per = yr_per(nobs - L + 1 + i)
        if per == 1:
            prevby, previndex = yr, i
            break
    if prevby is None:
        prevby, _ = yr_per(nobs - L + 1)
        previndex = 0
    yr_step = 1 if freq == 12 else (2 if freq == 4 else 10)
    x_step = freq * yr_step if freq > 1 else 10
    pos, lbl = [], []
    cur_yr, cur_x = prevby, previndex
    while cur_x < 2 * L:
        pos.append(cur_x); lbl.append(str(cur_yr))
        cur_yr += yr_step; cur_x += x_step
    return pos, lbl


def _prevcmax(err_vals, sigma_plot):
    prevcmax = 4.0 * sigma_plot
    for v in np.abs(err_vals):
        if v >= prevcmax:
            prevcmax = float(v)
    if 4.0 * sigma_plot < prevcmax <= 6.0 * sigma_plot:
        prevcmax = 6.0 * sigma_plot
    elif 6.0 * sigma_plot < prevcmax <= 7.0 * sigma_plot:
        prevcmax = 7.0 * sigma_plot
    elif prevcmax > 7.0 * sigma_plot:
        prevcmax = 10.0 * sigma_plot
    return prevcmax


# ── table + chart per series ──────────────────────────────────────────────────

def _table_data(model, i, L, arr):
    freq = model.series.freq
    nobs = model.series.nobs
    raw = np.asarray(model.series.data)[:, i]
    scale, lam = model.scale, model.lam
    bc_i = arr["bc"][:, i]
    residuals = model.result["residuals"][:, i]
    ornsop = nobs - len(residuals)
    n_hist = freq + 1

    def bx(x):
        return scale * transform.boxcox_fwd(x, lam)

    hist_rows = []
    for k in range(nobs - n_hist + 1, nobs + 1):
        if k < 1:
            continue
        diff1 = 100.0 * (bx(raw[k - 1]) - bx(raw[k - 2])) / scale if k > 1 else 0.0
        if k > freq:
            annual = 100.0 * (bx(raw[k - 1]) - bx(raw[k - 1 - freq])) / scale
        else:
            annual = 100.0 * bx(raw[k - 1]) / scale
        r = k - ornsop - 1
        err = _f2(100.0 * residuals[r] / scale) if 0 <= r < len(residuals) else "—"
        hist_rows.append({"date": _date_str(model, k), "level": _f2(raw[k - 1]),
                          "diff1": _f2(diff1), "annual": _f2(annual), "err": err})

    all_fore = []
    for h in range(L):
        all_fore.append({
            "date": _date_str(model, nobs + h + 1),
            "level": _f2(arr["level"][h, i]), "level_std": _f2(arr["level_std"][h, i]),
            "diff1": _f2(arr["diff1"][h, i]), "diff1_std": _f2(arr["diff1_std"][h, i]),
            "annual": _f2(arr["annual"][h, i]), "annual_std": _f2(arr["annual_std"][h, i])})
    return hist_rows, all_fore[:freq], (all_fore[-1] if L > freq else None)


def _make_charts_svg(model, i, L, arr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.gridspec import GridSpec

    freq = model.series.freq
    nobs = model.series.nobs
    raw = np.asarray(model.series.data)[:, i]
    scale, lam = model.scale, model.lam

    def bx(x):
        return scale * transform.boxcox_fwd(x, lam)

    xtick_pos, xtick_lbl = _year_ticks(model, nobs, L)
    xlim = (-0.5, 2 * L - 0.5)
    fig = plt.figure(figsize=(7, 7.5))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.18,
                  left=0.10, right=0.97, top=0.96, bottom=0.07)
    ax_top, ax_bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # top: annual rate of change
    hist_annual = np.array([100.0 * (bx(raw[nobs - L + k]) - bx(raw[nobs - L + k - freq]))
                            / scale for k in range(L)])
    fore_annual = arr["annual"][:, i]
    fore_astd = arr["annual_std"][:, i]
    y_all = np.concatenate([hist_annual, fore_annual])
    x_all = np.arange(2 * L); x_hist = np.arange(L); x_fore = np.arange(L, 2 * L)
    _spines(ax_top, ("left", "bottom"))
    ax_top.plot(x_all, y_all, color="k", ls="--", marker="o", lw=1.2, ms=4.0, zorder=3)
    ax_top.plot(x_hist, hist_annual, "ko", ms=6.0, zorder=4)
    ax_top.plot(x_fore, fore_annual + fore_astd, "k--", lw=1.5, zorder=2)
    ax_top.plot(x_fore, fore_annual - fore_astd, "k--", lw=1.5, zorder=2)
    ax_top.axvline(L - 0.5, color="0.55", lw=0.9, zorder=1)
    ax_top.axhline(0, color="k", lw=0.7, zorder=1)
    ax_top.set_xlim(*xlim); ax_top.set_xticks(xtick_pos)
    ax_top.set_xticklabels(xtick_lbl, fontsize=9)
    ax_top.tick_params(direction="out", labelsize=9)
    ax_top.set_axisbelow(True)
    ax_top.grid(axis="x", color="0.75", lw=0.5, ls="-", zorder=0)
    ax_top.set_title("Annual rate of change (%)", loc="left", fontsize=9, pad=4)

    # bottom: ERR
    residuals = model.result["residuals"][:, i]
    err_L = min(L, len(residuals))
    err_vals = 100.0 * residuals[-err_L:] / scale
    x_err = np.arange(err_L)
    sigma_plot = math.sqrt(model.result["sigma"][i, i]) * 100.0 / scale
    prevcmax = _prevcmax(err_vals, sigma_plot)
    hist_tick_pos = [p for p in xtick_pos if p < err_L]
    hist_tick_lbl = xtick_lbl[:len(hist_tick_pos)]
    x_end = err_L - 0.5
    _spines(ax_bot, ("left", "bottom"))
    ax_bot.vlines(x_err, 0, err_vals, colors="k", lw=1.6, zorder=3)
    ax_bot.hlines(2 * sigma_plot, -0.5, x_end, colors="k", lw=1.0, ls="--", zorder=2)
    ax_bot.hlines(-2 * sigma_plot, -0.5, x_end, colors="k", lw=1.0, ls="--", zorder=2)
    ax_bot.hlines(0, -0.5, x_end, colors="k", lw=1.2, zorder=2)
    margin = 0.1 * sigma_plot
    ax_bot.set_ylim(-(prevcmax + margin), prevcmax + margin)
    yt = np.arange(0, prevcmax + 0.05 * sigma_plot, 2 * sigma_plot)
    ax_bot.set_yticks(np.concatenate([-yt[1:][::-1], yt]))
    ax_bot.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax_bot.set_xlim(*xlim); ax_bot.set_xticks(hist_tick_pos)
    ax_bot.set_xticklabels(hist_tick_lbl, fontsize=9)
    ax_bot.tick_params(direction="out", labelsize=9)
    ax_bot.set_axisbelow(True)
    ax_bot.grid(axis="x", color="0.75", lw=0.5, ls="-", zorder=0)
    ax_bot.spines["bottom"].set_bounds(-0.5, err_L - 0.5)
    ax_bot.set_title("ERR", loc="left", fontsize=9, pad=4)
    return _fig_to_svg(fig)


# ── public API ────────────────────────────────────────────────────────────────

def write_forecast_report(model, path_prefix, L=12, b=0, source=None,
                          sps_name=None, title=None):
    """Write one SPS HTML forecast report per series.

    `path_prefix` -> ``<path_prefix>_<series_name>.html``.  Returns the paths.
    """
    try:
        from jinja2 import Environment
    except ImportError:
        raise ImportError('HTML forecast report requires jinja2 — '
                          'pip install "drvarma[forecast-report]"')
    if model.result is None:
        raise RuntimeError("call fit() before write_forecast_report()")

    r, ts = model.result, model.series
    freq = ts.freq
    nobs = ts.nobs
    npar = r["npar"]
    logelf = r["logelf"]
    aic = 2 * npar - 2 * logelf
    bic = npar * math.log(nobs) - 2 * logelf
    if freq == 12:
        freq_label, diff1_label = "months", "Monthly (%)"
    elif freq == 4:
        freq_label, diff1_label = "quarters", "Quarterly (%)"
    else:
        freq_label, diff1_label = "periods", "Period (%)"
    sample = "%s – %s" % (_date_str(model, 1), _date_str(model, nobs))
    origin = _date_str(model, nobs - b)

    arr = _forecast_arrays(model, L, b=b)
    env = Environment(autoescape=False)
    tmpl = env.from_string(_HTML)
    paths = []
    for i in range(ts.m):
        name = ts.names[i]
        hist_rows, fore_main, fore_end = _table_data(model, i, L, arr)
        html = tmpl.render(
            page_title=title or name, sps_name=sps_name or "",
            series_name=name, stem="VARMA(%d,%d)" % (model.p, model.q),
            source=source or "", origin=origin, horizon=L,
            freq_label=freq_label, diff1_label=diff1_label,
            sigma2="%.6f" % r["sigma2"], loglik="%.3f" % logelf,
            aic="%.2f" % aic, bic="%.2f" % bic, npar=npar, nobs=nobs,
            sample=sample, hist_rows=hist_rows, fore_rows_main=fore_main,
            fore_row_end=fore_end, charts_svg=_make_charts_svg(model, i, L, arr),
            version="0.1 (Python port)", generated=_date.today().isoformat())
        path = "%s_%s.html" % (path_prefix, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        paths.append(path)
    return paths
