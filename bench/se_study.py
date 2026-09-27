"""The standard-error study: BFGS against Mauricio's fdhess, against the truth.

    python3 bench/se_study.py [--quick] [--out docs/STUDY-standard-errors-data.md]

Every section compares the two Hessians with something external:

  S1  the exact GLS of fue's ES_CPI_m10 (mu + 11 deterministic terms + AR(1)),
      the reference of drtran's battery §1d, from two starting points;
  S2  the 2026-09-27 review's real case (VAR(1) of IPC and WTI) against OLS;
  S3  the review's simulation: a bivariate VAR(1) whose series differ in scale,
      against OLS, over many seeds;
  S4  Monte Carlo: the dispersion of the estimates across replications IS the
      standard error; the coverage of the 95% intervals;
  S5  a sweep of the real cases (the .inp bench of the C, the ladder cases):
      how often the Hessian at the optimum is not positive definite, the ratio
      fd/bfgs, and the cost.

It writes a Markdown data file; docs/STUDY-standard-errors.md reads it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")

from drvarma._engine import estimate_w                 # noqa: E402
from drvarma.stderr import legacy_fd_std_errors         # noqa: E402
from drvarma.ladder import Ladder                       # noqa: E402

LADDER = os.path.join(ROOT, "tests", "data", "ladder")
DRTRAN_CASES = os.path.expanduser("~/Dropbox/SRC/atsw-gui/engines/drtran/tests/cases")
MONOREPO_DRVARMA = os.path.expanduser("~/Dropbox/SRC/atsw-gui/engines/drvarma")


def _ols_var1(w):
    """OLS of a VAR(1) with constant: (coef (m, 1+m), se (m, 1+m))."""
    Y, X = w[1:], np.column_stack([np.ones(len(w) - 1), w[:-1]])
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    k = X.shape[1]
    S = E.T @ E / (len(Y) - k)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.outer(np.diag(S), np.diag(XtXi)))
    return B.T, se


def _phi_block(v, m, include_mean=True):
    """The phi(1) entries of the packed vector, row-major."""
    o = m if include_mean else 0
    return np.asarray(v)[o:o + m * m]


def _fit(w, p, q, hessian):
    r = estimate_w(w, p, q, include_mean=True)
    if hessian == "fd":
        cov, std, how, info = legacy_fd_std_errors(r, w, p, q, include_mean=True)
        r = dict(r, std_errors=std, se_method=how, se_info=info)
    else:
        r = dict(r, se_method="bfgs")
    return r


# --------------------------------------------------------------------------- #
def s1_gls(out):
    out += ["## S1. The exact GLS (fue's ES_CPI_m10: mu + 11 deterministic + AR(1))", "",
            "Reference: drtran's battery §1d (exact GLS on the differenced design and",
            "the AR(1) theory). The model is fitted by the ladder with its deterministic",
            "terms estimated (`redet`), from the `.pre` and from a start 20% away.", ""]
    pre = os.path.join(DRTRAN_CASES, "ES_CPI_m10.pre")
    if not os.path.exists(pre):
        out += [f"(skipped: {pre} not found)", ""]
        return {}
    ref = {"mu": 0.028502, "phi": 0.062421, "cos1": 0.068328, "sin1": 0.068294,
           "cos2": 0.027692, "alter": 0.006094}
    rows, res = [], {}
    for hess in ("bfgs", "fd"):
        for start in ("file", "perturbed"):
            L = Ladder([pre], 0, 0, diagcov=True, redet=True, hessian=hess)
            if start == "perturbed":
                L.series[0].x *= 1.2
            t = time.time()
            r = L.fit()
            dt = time.time() - t
            se = dict(zip(r.names, r.std_errors))
            pick = {"mu": next(v for k, v in se.items() if k.startswith("mu_")),
                    "phi": next(v for k, v in se.items() if k.startswith("phi_")),
                    "cos1": next(v for k, v in se.items() if k.endswith("[cos 1]")),
                    "sin1": next(v for k, v in se.items() if k.endswith("[sin 1]")),
                    "cos2": next(v for k, v in se.items() if k.endswith("[cos 2]")),
                    "alter": next(v for k, v in se.items() if k.endswith("[alter]"))}
            res[(hess, start)] = pick
            rows.append((hess, start, r.se_method, pick, dt))
    out += ["| Hessian | start | method | " + " | ".join(ref) + " | seconds |",
            "|---|---|---|" + "---|" * len(ref) + "---|",
            "| **exact GLS** | | | " + " | ".join(f"{v:.6f}" for v in ref.values()) + " | |"]
    for hess, start, how, pick, dt in rows:
        out.append(f"| {hess} | {start} | {how} | "
                   + " | ".join(f"{pick[k]:.6f}" for k in ref) + f" | {dt:.1f} |")
    worst = {h: max(abs(res[(h, s)][k] / ref[k] - 1) for s in ("file", "perturbed") for k in ref)
             for h in ("bfgs", "fd")}
    out += ["", f"Largest relative error against the GLS: BFGS {100 * worst['bfgs']:.1f}%, "
            f"fdhess {100 * worst['fd']:.2f}%.", ""]
    return worst


def s2_review(out):
    out += ["## S2. The review's real case: VAR(1) of IPC and WTI, against OLS", ""]
    try:
        import pandas as pd
        import atsw
        a = pd.read_csv(atsw.example_path("IPC_ES.csv"))
        b = pd.read_csv(atsw.example_path("WTI.csv"))
    except Exception as e:                                  # pragma: no cover
        out += [f"(skipped: {e})", ""]
        return {}
    from drvarma import MultiSeries, Model
    ms = MultiSeries(np.column_stack([a.value, b.value]), freq=12, start=(2002, 1),
                     names=["IPC", "WTI"])
    fits = {h: Model(ms, lam=0.0, d=1, p=1, include_mean=True, deseason="auto",
                     hessian=h).fit() for h in ("bfgs", "fd")}
    w = fits["fd"]._w
    B, se_ols = _ols_var1(w)
    ols = se_ols[:, 1:].ravel()
    names = ["phi11 (IPC<-IPC)", "phi12 (IPC<-WTI)", "phi21 (WTI<-IPC)", "phi22 (WTI<-WTI)"]
    out += ["| parameter | estimate | SE bfgs | SE fdhess | SE OLS | fd/OLS |", "|---|---|---|---|---|---|"]
    ratios = []
    for i, nm in enumerate(names):
        e = _phi_block(fits["fd"].params, 2)[i]
        sb = _phi_block(fits["bfgs"].std_errors, 2)[i]
        sf = _phi_block(fits["fd"].std_errors, 2)[i]
        ratios.append(sf / ols[i])
        out.append(f"| {nm} | {e:.4f} | {sb:.4f} | {sf:.4f} | {ols[i]:.4f} | {sf / ols[i]:.3f} |")
    info = fits["fd"].result.get("se_info", {})
    out += ["", f"Hessian at the optimum: positive definite {info.get('positive_definite')}, "
            f"condition number {info.get('cond', float('nan')):.2e}. OLS SEs use n-k; ML",
            f"uses n, so ML/OLS ~ sqrt((n-k)/n) = {np.sqrt((len(w) - 4) / (len(w) - 1)):.3f}.", ""]
    return {"fd_over_ols": ratios}


def _sim_var1(rng, ratio, n=400):
    Phi = np.array([[0.3, 0.1], [0.5, 0.4]])
    L = np.linalg.cholesky(np.array([[1, 0.5 * ratio], [0.5 * ratio, ratio * ratio]]))
    w = np.zeros((n, 2))
    for t in range(1, n):
        w[t] = Phi @ w[t - 1] + L @ rng.standard_normal(2)
    return w


def s3_scales(out, seeds):
    out += ["## S3. Series of very different scale (the review's simulation), against OLS", "",
            "Bivariate VAR(1), Phi = [[0.3, 0.1], [0.5, 0.4]], innovation correlation 0.5,",
            f"n = 400, {seeds} seeds per ratio of standard deviations. Median over seeds of",
            "SE/OLS for (phi11, phi12, phi21, phi22), and the worst seed.", "",
            "| sd ratio | BFGS median | BFGS worst | fdhess median | fdhess worst | fd not PD |",
            "|---|---|---|---|---|---|"]
    summary = {}
    for ratio in (1, 10, 30, 100, 300):
        rb, rf, npd = [], [], 0
        for sd in range(seeds):
            w = _sim_var1(np.random.default_rng(1000 + sd), ratio)
            _B, se = _ols_var1(w)
            ols = se[:, 1:].ravel()
            b = _fit(w, 1, 0, "bfgs")
            f = _fit(w, 1, 0, "fd")
            rb.append(_phi_block(b["std_errors"], 2) / ols)
            if f["se_method"] == "fdhess":
                rf.append(_phi_block(f["std_errors"], 2) / ols)
            else:
                npd += 1
        rb, rf = np.array(rb), np.array(rf) if rf else np.full((1, 4), np.nan)
        worst = lambda r: r.flat[np.argmax(np.abs(np.log(np.where(r > 0, r, 1e-12))))]
        fmt = lambda v: " · ".join(f"{x:.2f}" for x in v)
        out.append(f"| {ratio} | {fmt(np.median(rb, 0))} | {worst(rb):.3f} | "
                   f"{fmt(np.median(rf, 0))} | {worst(rf):.3f} | {npd}/{seeds} |")
        summary[ratio] = {"bfgs_worst": float(worst(rb)), "fd_worst": float(worst(rf)), "not_pd": npd}
    out.append("")
    return summary


def _sim_varma11(rng, n=200):
    # Well identified: Theta far from Phi (no near-cancelling factors). A first
    # choice with Phi ~ Theta gave estimates spread over +-4 and no method
    # covered: that measures non-identification, not the Hessian.
    Phi = np.array([[0.7, 0.2], [-0.1, 0.5]])
    Th = np.array([[-0.4, 0.0], [0.2, -0.3]])
    L = np.linalg.cholesky(np.array([[1.0, 0.3], [0.3, 1.0]]))
    a = (L @ rng.standard_normal((2, n + 1))).T
    w = np.zeros((n + 1, 2))
    for t in range(1, n + 1):
        w[t] = Phi @ w[t - 1] + a[t] - Th @ a[t - 1]
    return w[1:], np.concatenate([Phi.ravel(), Th.ravel()])


def s4_montecarlo(out, reps):
    out += ["## S4. Monte Carlo: the dispersion of the estimates IS the standard error", "",
            f"{reps} replications each. For every AR/MA coefficient: the standard deviation",
            "of its estimates across replications (the truth), the mean reported SE of each",
            "method, and the coverage of the nominal 95% interval.", ""]
    res = {}
    def sim_arma11(r, n=200):
        e = r.standard_normal(n + 1)
        w = np.zeros(n + 1)
        for t in range(1, n + 1):
            w[t] = 0.6 * w[t - 1] + e[t] + 0.5 * e[t - 1]      # theta = -0.5 in BJ sign
        return w[1:].reshape(-1, 1), np.array([0.6, -0.5])

    for label, sim, p, q in (("VAR(1), sd ratio 30", lambda r: (_sim_var1(r, 30, 300), None), 1, 0),
                             ("VARMA(1,1), well identified", lambda r: _sim_varma11(r), 1, 1),
                             ("ARMA(1,1) univariate", sim_arma11, 1, 1)):
        est, seb, sef, covb, covf, npd = [], [], [], [], [], 0
        truth_par = None
        for k in range(reps):
            w, tp = sim(np.random.default_rng(5000 + k))
            if tp is not None:
                truth_par = tp
            b = _fit(w, p, q, "bfgs")
            if b["ifault"]:
                continue
            f = _fit(w, p, q, "fd")
            m = w.shape[1]
            sl = slice(m, m + m * m * (p + q))
            est.append(np.asarray(b["params"])[sl])
            seb.append(np.asarray(b["std_errors"])[sl])
            if f["se_method"] == "fdhess":
                sef.append(np.asarray(f["std_errors"])[sl])
            else:
                npd += 1
                sef.append(np.full(m * m * (p + q), np.nan))
        est, seb, sef = map(np.array, (est, seb, sef))
        sd = est.std(0, ddof=1)
        center = truth_par if truth_par is not None else est.mean(0)
        if truth_par is None:                  # VAR(1): the true Phi
            center = np.array([0.3, 0.1, 0.5, 0.4])
        cov_b = np.mean(np.abs(est - center) <= 1.96 * seb, 0)
        cov_f = np.nanmean(np.abs(est - center) <= 1.96 * sef, 0)
        out += [f"**{label}** ({len(est)} usable replications; fdhess not PD in {npd})", "",
                "| coefficient | sd of estimates | mean SE bfgs | mean SE fdhess | coverage bfgs | coverage fdhess |",
                "|---|---|---|---|---|---|"]
        for i in range(est.shape[1]):
            out.append(f"| {i + 1} | {sd[i]:.4f} | {np.mean(seb[:, i]):.4f} | "
                       f"{np.nanmean(sef[:, i]):.4f} | {cov_b[i]:.2f} | {cov_f[i]:.2f} |")
        out.append("")
        res[label] = {"sd": sd, "bfgs": np.mean(seb, 0), "fd": np.nanmean(sef, 0),
                      "cov_b": cov_b, "cov_f": cov_f, "not_pd": npd}
    return res


def s5_sweep(out):
    out += ["## S5. The real cases: positive definiteness, fd/bfgs, cost", ""]
    rows = []
    # the ladder cases
    ES, FR, DE = (os.path.join(LADDER, f) for f in ("IPC_ES_m10.pre", "IPC_FR_msar.pre", "IPC_DE_mar3sar.pre"))
    WTI = os.path.join(LADDER, "WTI_ar1.pre")
    M6 = [os.path.join(LADDER, "m6", f"M6_{n}.pre") for n in ("EP", "EI", "EU", "EC", "EA", "P")]
    cases = [("ladder CPI diag", [ES, FR, DE], 0, 0, True), ("ladder CPI full cov", [ES, FR, DE], 0, 0, False),
             ("ladder CPI VAR(1)", [ES, FR, DE], 1, 0, False), ("ladder CPI VARMA(1,1)", [ES, FR, DE], 1, 1, False),
             ("ladder WTI+ES VAR(1)", [WTI, ES], 1, 0, False), ("ladder m6 diag", M6, 0, 0, True)]
    for label, files, p, q, dc in cases:
        r = {}
        for h in ("bfgs", "fd"):
            L = Ladder(files, p, q, diagcov=dc, hessian=h)
            t = time.time()
            r[h] = (L.fit(), time.time() - t)
        b, f = r["bfgs"][0], r["fd"][0]
        ratio = f.std_errors / np.where(b.std_errors > 0, b.std_errors, np.nan)
        rows.append((label, f.npar, f.se_method, np.nanmin(ratio), np.nanmedian(ratio),
                     np.nanmax(ratio), r["bfgs"][1], r["fd"][1]))
    # the .inp bench of the C, where the Python can run it (no -volexp)
    inp_cases = [("inp IPC VAR(1)", "data/IPC.inp", 1, 0, {}),
                 ("inp IPC ARMA(2,1) diagcov", "data/IPC.inp", 2, 1, {"diag_cov": True}),
                 ("inp PSW ARMA(1,1)", "data/PSW.inp", 1, 1, {}),
                 ("inp IPC3 VAR(3)", "data/models_group1/IPC3.inp", 3, 0, {}),
                 ("inp WTI+ES VAR(1)", "data/passthrough/WTI_IPC_ES.inp", 1, 0, {}),
                 ("inp WTI+ES VAR(3)", "data/passthrough/WTI_IPC_ES.inp", 3, 0, {})]
    from drvarma import Model, load
    for label, path, p, q, kw in inp_cases:
        full = os.path.join(MONOREPO_DRVARMA, path)
        if not os.path.exists(full):
            continue
        ser, sp = load(full)
        r = {}
        for h in ("bfgs", "fd"):
            t = time.time()
            m = Model(ser, lam=sp.lam, d=sp.d, D=sp.D, p=p, q=q, include_mean=True,
                      deseason="auto", hessian=h, **kw).fit()
            r[h] = (m, time.time() - t)
        b, f = r["bfgs"][0], r["fd"][0]
        sb, sf = np.asarray(b.std_errors), np.asarray(f.std_errors)
        ratio = sf / np.where(sb > 0, sb, np.nan)
        rows.append((label, len(sf), f.se_method, np.nanmin(ratio), np.nanmedian(ratio),
                     np.nanmax(ratio), r["bfgs"][1], r["fd"][1]))
    out += ["| case | npar | method | fd/bfgs min | median | max | s bfgs | s fd |",
            "|---|---|---|---|---|---|---|---|"]
    for label, k, how, lo, md, hi, tb, tf in rows:
        out.append(f"| {label} | {k} | {how} | {lo:.2f} | {md:.2f} | {hi:.1f} | {tb:.1f} | {tf:.1f} |")
    nonpd = sum(1 for r in rows if r[2] != "fdhess")
    out += ["", f"Hessian not positive definite at the optimum in {nonpd} of {len(rows)} cases.", ""]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "STUDY-standard-errors-data.md"))
    a = ap.parse_args()
    seeds, reps = (5, 40) if a.quick else (20, 200)
    out = ["# Standard-error study — data", "",
           f"*Generated by `bench/se_study.py`{' --quick' if a.quick else ''}. Do not edit by hand.*", ""]
    t0 = time.time()
    s1_gls(out)
    s2_review(out)
    s3_scales(out, seeds)
    s4_montecarlo(out, reps)
    s5_sweep(out)
    out += [f"*Total time {time.time() - t0:.0f} s.*", ""]
    with open(a.out, "w") as fh:
        fh.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
