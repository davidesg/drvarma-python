"""Repros for three multiart/drvarma defects found building the oil pass-through
exercise (WTI -> CPI, monthly 2002:02-2019:12, m=2, n=215).

Real data, not synthetic: `levels_2002_2019.csv` (WTI + CPI levels for US/ES/FR/DE).
Run:  DRVARMA_NO_ENGINE=1 python bench/repro_multiart_passthrough.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("DRVARMA_NO_ENGINE", "1")

from drvarma import Model, MultiSeries  # noqa: E402

DATA = ("/home/david/Dropbox/Nivel de Precios y Energia/"
        "passthrough_multiart/data/levels_2002_2019.csv")
COLS = {"US": 2, "ES": 3, "FR": 4, "DE": 5}  # col 1 = WTI, col 0 = FECHA


def levels(country):
    raw = np.genfromtxt(DATA, delimiter=",", skip_header=1)
    return raw[:, [1, COLS[country]]]


def series(country):
    return MultiSeries(levels(country), freq=12, start=(2002, 2),
                       names=["WTI", f"CPI_{country}"])


# ── BUG 1 — load_data drops observation 1 on a header-less numeric CSV ────────
def bug1_load_data_drops_first_obs():
    """mcp_server.load_data: `np.genfromtxt(..., skip_header=1)` is applied
    unconditionally and only retried when the result is *all* NaN. A purely
    numeric CSV with no header row parses fine after skipping, so the retry never
    fires and observation 1 is dropped silently.

    Impact here: n=214 instead of 215, i.e. the first differenced observation is
    lost, so the estimation sample no longer matches the published one. No warning.
    """
    from drvarma import mcp_server as M

    path = ("/home/david/Dropbox/Nivel de Precios y Energia/"
            "passthrough_multiart/data/US_levels.csv")  # 215 rows, NO header
    on_disk = np.genfromtxt(path, delimiter=",")
    M.load_data(name="_bug1", path=path, freq=12, start_year=2002,
                start_period=2, series_names="WTI,CPI_US")
    got = M._DATA["_bug1"]

    print(f"  rows on disk        : {on_disk.shape[0]}")
    print(f"  rows after load_data: {got.nobs}")
    print(f"  first row on disk   : {on_disk[0].tolist()}")
    print(f"  first row loaded    : {got.data[0].tolist()}")
    ok = got.nobs == on_disk.shape[0]
    print(f"  -> {'OK' if ok else 'BUG: observation 1 silently dropped'}")

    # Same call on a CSV that *does* have a header: rows are right, names are not.
    path_h = ("/home/david/Dropbox/Nivel de Precios y Energia/"
              "passthrough_multiart/data/US_lv.csv")
    M.load_data(name="_bug1b", path=path_h, freq=12, start_year=2002,
                start_period=2)
    print(f"  header CSV, no series_names -> names={M._DATA['_bug1b'].names} "
          f"(expected ['WTI', 'CPI_USA'] from the header row)")
    return ok


# ── BUG 2 — inf loglik pollutes the order search AND its recommendation ───────
def bug2_inf_loglik_drives_recommendation():
    """Already filed as MEDIUM (pure-Python `inf` loglik for MA specs under
    deseason). Two additions from this exercise:

    (a) it is *systematic*, not data-specific: exactly (0,1), (0,2), (1,2) blow up
        on all four independent country datasets;
    (b) it corrupts the *recommendation*, not just the ranking. identify_varma_order
        filters with `r[4] is not None`, which keeps non-finite entries; AIC/BIC
        become -inf, sort first, and `ok[0]` is emitted as
        "Recomendacion (min. BIC): **VARMA(0,1)**".

    A pure MA(1) on monthly inflation is not a defensible specification and it
    contradicts the CCM evidence, so in autonomous mode this yields a wrong model
    with no visible failure. Suggested fix: filter on np.isfinite(ll).
    """
    bad = {}
    for c in COLS:
        ms = series(c)
        hits = []
        for (p, q) in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 1)]:
            try:
                fit = Model(ms, lam=0.0, d=1, p=p, q=q,
                            deseason="auto", include_mean=True).fit()
                ll, ifault = float(fit.loglik), getattr(fit, "ifault", "-")
            except Exception as e:  # noqa: BLE001
                ll, ifault = f"raised {type(e).__name__}", "-"
            finite = isinstance(ll, float) and np.isfinite(ll)
            hits.append(((p, q), ll, ifault, finite))
            if not finite:
                bad.setdefault(c, []).append((p, q))
        shown = ", ".join(
            f"({p},{q})={ll if isinstance(ll, str) else f'{ll:.2f}'}[ifault={f}]"
            for (p, q), ll, f, _ in hits)
        print(f"  {c}: {shown}")
    print(f"  -> non-finite specs by country: {bad}")
    print("  -> identical across all four datasets => systematic, not data-specific"
          if len({tuple(v) for v in bad.values()}) == 1 else "  -> varies by dataset")
    print("  -> NOTE: the failing fits already carry ifault=3 (non-stationary), so the")
    print("     diagnosis exists and is correct -- identify_varma_order just never")
    print("     reads it. Root cause of the inf itself: estimate_py.py:337-338,")
    print("     logelf = ... - 0.5*n*(m*log(f1) + log(f2)); f1 or f2 -> 0 gives")
    print("     log -> -inf and the leading minus flips it to +inf.")
    return not bad


# ── BUG 3 (NEW) — likelihood not monotone in p: VAR(3) worse than VAR(1) ──────
def bug3_nonmonotone_loglik_de():
    """NEW, not yet filed. For DE the VAR(3) fit returns a *lower* exact
    log-likelihood than the nested VAR(1). Since VAR(1) is nested in VAR(3), the
    maximised likelihood cannot decrease: this is an optimiser convergence failure
    reported as a valid fit. It silently corrupts every information criterion
    computed from it, and there is no convergence flag exposed to the caller.

    US/ES/FR are monotone on the same call, so it is specific to this fit.

    Worse than BUG 2: the bad fit reports ifault=0 ("model OK"), so unlike the inf
    case there is no existing flag to propagate. `termcode`/`nit` (the optimiser's
    own termination state, computed in estimate_py) are not exposed on the fitted
    Model either, so a caller has no way to detect this at all.
    """
    bad = False
    for c in COLS:
        lls, faults = [], []
        for p in (1, 2, 3):
            fit = Model(series(c), lam=0.0, d=1, p=p, q=0,
                        deseason="auto", include_mean=True).fit()
            lls.append(float(fit.loglik))
            faults.append(getattr(fit, "ifault", "-"))
        mono = lls[0] <= lls[1] + 1e-6 and lls[1] <= lls[2] + 1e-6
        flag = "" if mono else "   <-- BUG: not monotone in p"
        print(f"  {c}: VAR(1)={lls[0]:10.2f}  VAR(2)={lls[1]:10.2f}  "
              f"VAR(3)={lls[2]:10.2f}   ifault={faults}{flag}")
        bad |= not mono
    print("  -> the non-monotone fit reports ifault=0: no flag exists to propagate,")
    print("     and termcode/nit are not attributes of the fitted Model.")
    return not bad


# ── BUG 4 (NEW, WORST) — deseason phase off-by-one in `start_sub` ─────────────
def bug4_deseason_phase_offbyone():
    """`deseason="auto"` makes seasonality WORSE unless the series happens to start
    at subperiod 1. Our series start in February (start_sub=2, passed correctly by
    mcp_server._prepared_w as ms.start[1]); the lag-12 residual autocorrelation of
    dlog(CPI) then goes UP relative to not deseasonalising at all.

    Sweeping start_sub shows the minimum |ACF(12)| at start_sub=1 for all four
    countries, where the result matches a plain monthly-dummy OLS regression. So the
    harmonic algorithm is right and the PHASE convention between caller and callee is
    off by one. Everything downstream (characterization, CCM, Tiao-Box, estimation,
    IRF, FEVD) is silently corrupted for any series not starting at subperiod 1.
    """
    import statsmodels.api as sm
    import pandas as pd
    from drvarma import transform
    from drvarma.deseason import deseasonalize_raw

    def acf12(x):
        xc = x - x.mean()
        return float((xc[12:] * xc[:-12]).mean() / xc.var())

    ok = True
    print(f"  {'country':8}{'none':>9}{'ss=2 (used)':>13}{'ss=1':>9}"
          f"{'dummy OLS':>11}   best ss")
    for c in COLS:
        lv = levels(c)
        w_none, _ = transform.transform(lv, lam=0.0, d=1, D=0, s=12, scale=1.0)
        a_none = acf12(np.asarray(w_none)[:, 1])

        sweep = []
        for ss in range(1, 13):
            adj, _, _ = deseasonalize_raw(lv, s=12, start_sub=ss, mode="auto")
            w, _ = transform.transform(adj, lam=0.0, d=1, D=0, s=12, scale=1.0)
            sweep.append(acf12(np.asarray(w)[:, 1]))
        best = int(np.argmin(np.abs(sweep))) + 1

        y = np.asarray(w_none)[:, 1]
        X = sm.add_constant(pd.get_dummies((np.arange(len(y)) + 2) % 12,
                                           drop_first=True).astype(float).values)
        ref = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        print(f"  {c:8}{a_none:+9.3f}{sweep[1]:+13.3f}{sweep[0]:+9.3f}"
              f"{acf12(ref):+11.3f}      {best}")
        # correct behaviour would be: the phase actually used is the best one
        ok &= best == 2
    print("  -> series start in FEBRUARY, so start_sub=2 is what mcp_server passes,")
    print("     yet |ACF(12)| is minimised at start_sub=1 for every country, where it")
    print("     matches the dummy-OLS reference => off-by-one in the phase convention.")
    print("  -> at the phase actually used, deseason makes seasonality WORSE than not")
    print("     deseasonalising at all (ES .798->.866, FR .728->.862, DE .608->.808).")
    return ok


if __name__ == "__main__":
    results = {}
    for title, fn in [
        ("BUG 1 - load_data drops observation 1 (header-less CSV)",
         bug1_load_data_drops_first_obs),
        ("BUG 2 - inf loglik drives the order recommendation",
         bug2_inf_loglik_drives_recommendation),
        ("BUG 3 - loglik not monotone in p (DE VAR(3) < VAR(1))",
         bug3_nonmonotone_loglik_de),
        ("BUG 4 - deseason phase off-by-one (start_sub) makes seasonality worse",
         bug4_deseason_phase_offbyone),
    ]:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
        results[title] = fn()
    print(f"\n{'=' * 72}\nSUMMARY (True = behaved correctly)\n{'=' * 72}")
    for t, ok in results.items():
        print(f"  {'PASS' if ok else 'REPRODUCED'}  {t}")
    sys.exit(0)
