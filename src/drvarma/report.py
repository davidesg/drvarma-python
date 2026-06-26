"""Text report writers reproducing the C engine's ``.out`` / ``.forecast`` files.

The numbers all come from a fitted :class:`drvarma.model.Model`; this module is
presentation only.  Formats are byte-for-byte ports of the ``fprintf`` blocks in
the C ``drvarma.c`` so output can be diffed against the reference binary.
"""

import math

import numpy as np

from . import transform

EQ = "=" * 61
DASH = "-" * 68


def _normal_cdf(z):
    """Abramowitz & Stegun standard-normal CDF (port of nlatools.c normal_cdf)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * math.exp(-z * z / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779
                    + t * (-1.821256 + t * 1.330274))))
    return (1.0 - prob) if z > 0 else prob


def _sig_code(p):
    """Significance stars (port of sig_code); always 3 chars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    if p < 0.1:
        return ".  "
    return "   "


def obs_to_date(beg_year, beg_sub, obs_no, freq):
    """Map a 1-based observation number to (year, sub) — port of ObsToDate."""
    if obs_no + beg_sub - 1 <= freq:
        return beg_year, beg_sub + obs_no - 1
    quot, rem = divmod(obs_no - (freq - beg_sub + 1), freq)
    if rem > 0:
        return beg_year + quot + 1, rem
    return beg_year + quot, freq


# --------------------------------------------------------------------------- #
#  Forecast report (.forecast)                                                #
# --------------------------------------------------------------------------- #

def forecast_report(model, L, b=0):
    """Return the ``.forecast`` text for a fitted `model`, horizon `L`.

    Reproduces the re-seasonalized Level/Low95/High95 plus the monthly/annual
    variation rates (and their std) exactly as ``drvarma.c`` writes them.
    """
    from .forecast import forecast_levels, forecast_level_variances

    if model.result is None:
        raise RuntimeError("call fit() before forecast_report()")
    res = model.result
    m = model.series.m
    freq = model.series.freq
    sub0 = model.series.start[1]
    year0 = model.series.start[0]
    scale = model.scale
    lam = model.lam

    bc = model._bc                              # (nobs_raw, m), scale*BoxCox(level)
    nobs_raw = bc.shape[0]

    lev_des, _ = forecast_levels(res, model._w, bc, lam=lam, scale=scale,
                                 d=model.d, D=model.D, s=freq, L=L, b=b)
    v_lvl, v_mon, v_ann = forecast_level_variances(
        res["phi"], res["theta"], res["sigma"], L, model.d, model.D, freq)

    origin = nobs_raw - b
    # re-seasonalization dummies and forecast on the scale*BoxCox scale (cf)
    dseas = np.zeros((L, m))
    if model.deseason and model._dummies is not None:
        for l in range(1, L + 1):
            period = (origin + l + sub0 - 2) % freq
            dseas[l - 1] = model._dummies[:, period]
    cf = scale * transform.boxcox_fwd(lev_des, lam)             # (L, m)
    level = lev_des + dseas
    low = np.zeros((L, m))
    high = np.zeros((L, m))
    for l in range(1, L + 1):
        for i in range(m):
            sd = np.sqrt(v_lvl[l, i, i])
            low[l - 1, i] = transform.boxcox_inv(
                (cf[l - 1, i] - 1.96 * sd) / scale, lam) + dseas[l - 1, i]
            high[l - 1, i] = transform.boxcox_inv(
                (cf[l - 1, i] + 1.96 * sd) / scale, lam) + dseas[l - 1, i]

    deseason_str = (model.deseason if model.deseason else "no")
    reseas = " (re-seasonalized)" if model.deseason else ""
    sc = 100.0 / scale

    out = []
    out.append("Forecasts from VARMA(%d,%d) model\n" % (model.p, model.q))
    out.append("lambda=%g, scale=%g, d=%d, D=%d, freq=%d, horizon=%d, deseason=%s\n"
               % (lam, scale, model.d, model.D, freq, L, deseason_str))
    out.append("Level/Low95/High95 in original units%s. "
               "mon%%/ann%% = monthly/annual variation rate and std "
               "(100*delta/scale; %% for lambda=0).\n\n" % reseas)
    for k in range(1, m + 1):
        i = k - 1
        out.append("Series %d (%s):\n" % (k, model.series.names[i]))
        out.append("  date   %10s %10s %10s %8s %7s %8s %7s\n"
                   % ("Level", "Low95", "High95", "mon%", "std", "ann%", "std"))
        for l in range(1, L + 1):
            per, sub = obs_to_date(year0, sub0, nobs_raw + l, freq)
            # monthly variation on the scale*BoxCox scale
            if l == 1:
                g2 = cf[0, i] - bc[nobs_raw - 1, i]
            else:
                g2 = cf[l - 1, i] - cf[l - 2, i]
            # annual variation
            if l <= freq:
                idx = nobs_raw - freq + l            # 1-based index into bc
                g3 = (cf[l - 1, i] - bc[idx - 1, i]) if idx >= 1 else 0.0
            else:
                g3 = cf[l - 1, i] - cf[l - 1 - freq, i]
            out.append("%3d/%4d %10.4f %10.4f %10.4f %8.4f %7.4f %8.4f %7.4f\n"
                       % (sub, per, level[l - 1, i], low[l - 1, i], high[l - 1, i],
                          sc * g2, sc * np.sqrt(v_mon[l, i, i]),
                          sc * g3, sc * np.sqrt(v_ann[l, i, i])))
        out.append("\n")
    return "".join(out)


def write_forecast(model, L, path, b=0):
    """Write the ``.forecast`` report for `model` to `path`."""
    text = forecast_report(model, L, b=b)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
#  Estimation report (.out)                                                   #
# --------------------------------------------------------------------------- #

def _yesno(flag):
    return "yes" if flag else "no"


def _banner(title):
    return "%s\n%s\n%s\n" % (EQ, title, EQ)


def _header_block(model, input_path, output_path):
    s = model.series
    nobs_eff = model._w.shape[0]
    deseason = ("harmonic (force all)" if model.deseason == "force"
                else "harmonic (auto)" if model.deseason else "no")
    out = []
    out.append("Input Data File  : %s\n" % input_path)
    out.append("Output File      : %s\n" % output_path)
    out.append("Model: VARMA(%d,%d)\n" % (model.p, model.q))
    out.append("Include mean     : %s\n" % _yesno(model.include_mean))
    out.append("Diagonal AR      : %s\n" % _yesno(model.diag_ar))
    out.append("Diagonal MA      : %s\n" % _yesno(model.diag_ma))
    out.append("Diagonal Cov     : %s\n" % _yesno(model.diag_cov))
    out.append("Estimation method: %d\n" % model.method)
    out.append("Two-step init    : %s\n" % _yesno(model.twostep))
    out.append("Frequency        : %d\n" % s.freq)
    out.append("Start            : %d %d\n" % (s.start[1], s.start[0]))
    out.append("Box-Cox lambda   : %g\n" % model.lam)
    out.append("Rescale factor   : %g\n" % model.scale)
    out.append("Differences      : d=%d, D=%d (seasonal lag=%d)\n"
               % (model.d, model.D, s.freq))
    out.append("Deseasonalize    : %s\n" % deseason)
    out.append("Series names     :" + "".join(" %s" % n for n in s.names) + "\n")
    out.append("Observations     : %d (raw %d)\n\n" % (nobs_eff, s.nobs))
    out.append("Number of parameters: %d\n" % model.result["npar"])
    return "".join(out)


def _convergence_block(model):
    """Convergence banner.

    The C engine's internal iteration count and optimizer objective scalar are
    not exposed by the CFFI result, so this reports the exact log-likelihood
    instead (see docs/STATUS.md).
    """
    r = model.result
    status = "CONVERGED" if r["ifault"] == 0 else "FAILED (code %d)" % r["ifault"]
    return ("\n\n%s\n  OPTIMIZER %s\n  Log-likelihood = %.6f\n"
            "  Convergence criterion: norm of scaled gradient <= gradtol\n%s\n"
            % (EQ, status, r["logelf"], EQ))


def _param_labels(model):
    """Parameter (label, is_fixed) list in the C packing order."""
    m = model.series.m
    labels = []
    if model.include_mean:
        for i in range(1, m + 1):
            labels.append(("mu[%d]" % i, " " * 15, False))
    for k in range(1, model.p + 1):
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                fixed = model.diag_ar and i != j
                labels.append(("phi[%d]_%d%d" % (k, i, j), " " * 10, fixed))
    for k in range(1, model.q + 1):
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                fixed = model.diag_ma and i != j
                labels.append(("theta[%d]_%d%d" % (k, i, j), " " * 8, fixed))
    if model.diag_cov:
        for i in range(1, m + 1):
            labels.append(("cov[%d,%d]" % (i, i), " " * 11, False))
    else:
        for i in range(1, m + 1):
            for j in range(1, i + 1):
                labels.append(("cov[%d,%d]" % (i, j), " " * 11, False))
    return labels


def _parameters_block(model):
    r = model.result
    x = r["params"]
    dev = r["std_errors"]
    out = ["\n\n", _banner("  Estimated Parameters and Standard Deviations               ")]
    out.append("\n%33s %12s %12s %8s %6s\n"
               % ("Parameter", "Estimate", "Std.Error", "t-stat", "p-val"))
    out.append(DASH + "\n")
    idx = 0
    for label, pad, fixed in _param_labels(model):
        if fixed:
            out.append("%s%s(fixed 0.0)\n" % (label, pad))
            continue
        est = x[idx]
        se = dev[idx]
        t = est / se
        p = 2.0 * (1.0 - _normal_cdf(abs(t)))
        out.append("%s%s%12.6f %12.6f %8.3f %6.4f %s\n"
                   % (label, pad, est, se, t, p, _sig_code(p)))
        idx += 1
    out.append(DASH + "\n")
    out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n")
    return "".join(out)


# -- Wald joint hypothesis tests ------------------------------------------- #

def _wald(x, cov, indices):
    """chi2 = th' pinv(Sigma) th, df = rank (port of diagnose.c wald_test)."""
    from .diagnostics import _chisq_sf
    idx = np.asarray(indices, int)
    th = x[idx]
    Sig = cov[np.ix_(idx, idx)]
    u, s, vt = np.linalg.svd(Sig)
    tol = s[0] * math.sqrt(np.finfo(float).eps) if s.size else 0.0
    sinv = np.array([1.0 / v if v > tol else 0.0 for v in s])
    chi2 = float(th @ (vt.T @ (sinv * (u.T @ th))))
    df = int((s > tol).sum()) or 1
    return chi2, df, _chisq_sf(chi2, df)


def _wald_blocks(model):
    m = model.series.m
    p, q = model.p, model.q
    r = model.result
    x, cov = r["params"], r["cov"]
    names = model.series.names
    n_ar = m if model.diag_ar else m * m
    n_ma = m if model.diag_ma else m * m
    base = m if model.include_mean else 0      # 0-based offset to first AR param
    ar_start = base

    def ar_index(lag, i, j):                   # 0-based param index of phi[lag]_ij
        return ar_start + (lag - 1) * n_ar + (i - 1) * m + (j - 1)

    def ma_index(lag, i, j):
        return ar_start + p * n_ar + (lag - 1) * n_ma + (i - 1) * m + (j - 1)

    out = ["\n\n", _banner("           JOINT HYPOTHESIS TESTS (WALD)                    ")]

    def emit(title, extra, idxs, reject, accept):
        if not idxs:
            return
        chi2, df, pval = _wald(x, cov, idxs)
        out.append("\n%s\n" % title)
        for line in extra:
            out.append("  %s\n" % line)
        out.append("  Wald chi2(%d) = %.4f, p-value = %.4f\n" % (df, chi2, pval))
        out.append("  Conclusion: %s\n" % (reject if pval < 0.05 else accept))

    # last lag (AR and MA)
    if p > 0 or q > 0:
        idxs = []
        if p > 0:
            idxs += [ar_index(p, i, j) for i in range(1, m + 1) for j in range(1, m + 1)
                     if not (model.diag_ar and i != j)]
        if q > 0:
            idxs += [ma_index(q, i, j) for i in range(1, m + 1) for j in range(1, m + 1)
                     if not (model.diag_ma and i != j)]
        if p > 0:
            emit("Test of joint significance of last lag (AR(%d) and MA(%d)):" % (p, q),
                 [], idxs,
                 "REJECT H0 → last lag is statistically significant.",
                 "Cannot reject H0 → last lag is not significant.")
    # last AR lag
    if p > 0:
        idxs = [ar_index(p, i, j) for i in range(1, m + 1) for j in range(1, m + 1)
                if not (model.diag_ar and i != j)]
        emit("Test of joint significance of last AR lag (AR(%d)):" % p, [], idxs,
             "REJECT H0 → last AR lag is significant.",
             "Cannot reject H0 → last AR lag is not significant.")
    # last MA lag
    if q > 0:
        idxs = [ma_index(q, i, j) for i in range(1, m + 1) for j in range(1, m + 1)
                if not (model.diag_ma and i != j)]
        emit("Test of joint significance of last MA lag (MA(%d)):" % q, [], idxs,
             "REJECT H0 → last MA lag is significant.",
             "Cannot reject H0 → last MA lag is not significant.")
    # all cross effects
    cross = []
    if not model.diag_ar:
        cross += [ar_index(l, i, j) for l in range(1, p + 1)
                  for i in range(1, m + 1) for j in range(1, m + 1) if i != j]
    if not model.diag_ma and q > 0:
        cross += [ma_index(l, i, j) for l in range(1, q + 1)
                  for i in range(1, m + 1) for j in range(1, m + 1) if i != j]
    if cross:
        chi2, df, pval = _wald(x, cov, cross)
        out.append("\nTest of joint significance of all cross effects:\n")
        out.append("  H0: all cross coefficients = 0\n")
        out.append("  Wald chi2(%d) = %.4f, p-value = %.4f\n" % (df, chi2, pval))
        out.append("  Conclusion: %s\n" % (
            "REJECT H0 → cross effects are jointly significant."
            if pval < 0.05 else
            "Cannot reject H0 → cross effects are not jointly significant."))
    else:
        out.append("\nNo cross effects to test (model already diagonal).\n")

    # directional tests
    out.append("\n--- Directional cross-effects tests ---\n")
    for var in range(1, m + 1):
        vn = names[var - 1]
        on = []
        if not model.diag_ar and p > 0:
            on += [ar_index(l, var, j) for l in range(1, p + 1)
                   for j in range(1, m + 1) if j != var]
        if not model.diag_ma and q > 0:
            on += [ma_index(l, var, j) for l in range(1, q + 1)
                   for j in range(1, m + 1) if j != var]
        if on:
            chi2, df, pval = _wald(x, cov, on)
            out.append("\n%s: effects of other variables on it:\n" % vn)
            out.append("  H0: all coefficients in equation %s from other vars = 0\n" % vn)
            out.append("  Wald chi2(%d) = %.4f, p-value = %.4f\n" % (df, chi2, pval))
            out.append("  Conclusion: %s\n" % (
                "REJECT H0 → %s is influenced by others." % vn if pval < 0.05
                else "Cannot reject H0 → %s is not influenced by others." % vn))
        frm = []
        if not model.diag_ar and p > 0:
            frm += [ar_index(l, i, var) for l in range(1, p + 1)
                    for i in range(1, m + 1) if i != var]
        if not model.diag_ma and q > 0:
            frm += [ma_index(l, i, var) for l in range(1, q + 1)
                    for i in range(1, m + 1) if i != var]
        if frm:
            chi2, df, pval = _wald(x, cov, frm)
            out.append("\n%s: effects of it on other variables:\n" % vn)
            out.append("  H0: all coefficients of %s in other equations = 0\n" % vn)
            out.append("  Wald chi2(%d) = %.4f, p-value = %.4f\n" % (df, chi2, pval))
            out.append("  Conclusion: %s\n" % (
                "REJECT H0 → %s influences others." % vn if pval < 0.05
                else "Cannot reject H0 → %s does not influence others." % vn))
    return "".join(out)


# -- IRF / FEVD ------------------------------------------------------------- #

def _irf_blocks(model, horizon):
    from .irf import oirf
    r = model.result
    m = model.series.m
    names = model.series.names
    O = oirf(r["phi"], r["theta"], r["sigma"], horizon)   # (H+1, m, m)
    out = ["\n\n", _banner("           ORTHOGONALIZED IMPULSE RESPONSE FUNCTIONS        ")]
    out.append("Shocks are orthogonalized via Cholesky decomposition of Sigma Matrix.\n")
    out.append("Order of variables: as in the model.\n\n")
    for shock in range(m):
        out.append("Shock to %s:\n" % names[shock])
        out.append("Horizon" + "".join("%9s" % n for n in names) + "\n")
        for h in range(horizon + 1):
            out.append("%5d  " % h + "".join("%9.4f" % O[h][i, shock] for i in range(m)) + "\n")
        out.append("\n")

    out.append("\n" + _banner("        ACCUMULATED IMPULSE RESPONSE FUNCTIONS               "))
    out.append("Cumulative response of each variable to a one-s.d. shock\n")
    out.append("(sum of responses up to each horizon).\n\n")
    for shock in range(m):
        out.append("Shock to %s:\n" % names[shock])
        out.append("Horizon" + "".join("%9s" % n for n in names) + "\n")
        acc = np.zeros(m)
        for h in range(horizon + 1):
            acc += O[h][:, shock]
            out.append("%5d  " % h + "".join("%9.4f" % acc[i] for i in range(m)) + "\n")
        out.append("\n")

    out.append("\n" + _banner("  LONG‑RUN GAIN, SHOCK STD. DEVIATIONS AND MEAN LAG        "))
    out.append("\nStructural shock standard deviations (one‑s.d. shock size):\n")
    for j in range(m):
        out.append("  %-8s : %.4f\n" % (names[j], O[0][j, j]))
    for shock in range(m):
        out.append("\nShock to %s:\n" % names[shock])
        for var in range(m):
            resp = O[:, var, shock]
            gain = float(resp.sum())
            weighted = float((np.arange(horizon + 1) * resp).sum())
            gross = float(np.abs(resp).sum())
            if gross > 1e-12 and abs(gain) > 0.1 * gross:
                out.append("  %-8s : gain = %8.4f  mean lag = %6.2f months\n"
                           % (names[var], gain, weighted / gain))
            else:
                out.append("  %-8s : gain = %8.4f  mean lag = undefined\n"
                           % (names[var], gain))
    out.append(EQ + "\n\n")
    return "".join(out)


def _fevd_block(model, horizon):
    from .irf import oirf
    r = model.result
    m = model.series.m
    names = model.series.names
    O = oirf(r["phi"], r["theta"], r["sigma"], horizon)
    cum = np.cumsum(O ** 2, axis=0)            # cum[h, var, shock]
    out = ["\n\n", _banner("           FORECAST ERROR VARIANCE DECOMPOSITION             ")]
    out.append("Values are percentages of forecast error variance accounted for by each shock.\n")
    out.append("Shocks are orthogonalized via Cholesky decomposition.\n\n")
    for var in range(m):
        out.append("%s:\n" % names[var])
        out.append("Horizon" + "".join("%9s" % n for n in names) + "\n")
        for h in range(1, horizon + 1):
            c = cum[h - 1, var]
            tot = c.sum() or 1.0
            out.append("%5d  " % h + "".join("%9.2f" % (100.0 * c[j] / tot) for j in range(m)) + "\n")
        out.append("\n")
    return "".join(out)


def _diagnostics_block(model):
    from .diagnostics import hosking_q, jarque_bera_mv
    res = model.result["residuals"]
    n, m = res.shape
    s = int(math.sqrt(n))
    if s < 1:
        s = 1
    if s > n - 2:
        s = n - 2
    Q, qdf, qp = hosking_q(res, s)
    JB, jdf, jp = jarque_bera_mv(res)
    out = ["\n\n", _banner("           MULTIVARIATE RESIDUAL DIAGNOSTICS                ")]
    out.append("\nHosking's Multivariate Portmanteau Test (lag %d):\n" % s)
    out.append("  Q(%d) = %.4f, p-value = %.4f\n" % (qdf, Q, qp))
    out.append("  *** REJECT H0: residuals are not white noise.\n" if qp < 0.05
               else "  Cannot reject H0: residuals appear white noise.\n")
    out.append("\nMultivariate Jarque-Bera Test (normality):\n")
    out.append("  JB(%d) = %.4f, p-value = %.4f\n" % (jdf, JB, jp))
    out.append("  *** REJECT H0: residuals are not normally distributed.\n" if jp < 0.05
               else "  Cannot reject H0: residuals appear normal.\n")
    out.append(EQ + "\n")
    return "".join(out)


def _matrices_block(model):
    r = model.result
    m = model.series.m
    out = ["\nNormalized model:\n", "mu vector:\n"]
    for i in range(m):
        out.append("  %12.6f\n" % r["mu"][i])
    for k in range(model.p):
        out.append("phi(%d) matrix:\n" % (k + 1))
        for i in range(m):
            out.append("".join("  %12.6f" % r["phi"][k][i, j] for j in range(m)) + "\n")
    for k in range(model.q):
        out.append("theta(%d) matrix:\n" % (k + 1))
        for i in range(m):
            out.append("".join("  %12.6f" % r["theta"][k][i, j] for j in range(m)) + "\n")
    sigma2 = r["sigma2"]
    qq = r["sigma"] / sigma2
    out.append("Q matrix:\n")
    for i in range(m):
        out.append("".join("  %12.6f" % qq[i, j] for j in range(i + 1)) + "\n")
    out.append("Sigma = sigma2 * Q:\n")
    for i in range(m):
        out.append("".join("  %12.6f" % (sigma2 * qq[i, j]) for j in range(i + 1)) + "\n")
    out.append("\n")
    return "".join(out)


def _roots_block(model):
    """Inverse roots of |phi(B)|=0 / |theta(B)|=0.

    Computed as companion-matrix eigenvalues, sorted by descending modulus.
    (The C engine's chekma emits them in a different, QR-internal order.)
    """
    r = model.result
    m = model.series.m
    out = []

    def roots(coeffs, p):
        if p == 0:
            return []
        comp = np.zeros((m * p, m * p))
        for i in range(p):
            comp[:m, i * m:(i + 1) * m] = coeffs[i]
        if p > 1:
            comp[m:, :m * (p - 1)] = np.eye(m * (p - 1))
        ev = np.linalg.eigvals(comp)
        ev = [e for e in ev if abs(e) >= 0.00005]
        ev.sort(key=lambda z: (-abs(z), -z.real, z.imag))
        return ev

    if model.p > 0:
        out.append("Inverse roots of |phi(B)|=0:\n")
        for e in roots(r["phi"], model.p):
            out.append("  %12.6f  %12.6f i  (modulus %12.6f)\n"
                       % (e.real, e.imag, abs(e)))
    if model.q > 0:
        out.append("Inverse roots of |theta(B)|=0:\n")
        for e in roots(r["theta"], model.q):
            out.append("  %12.6f  %12.6f i  (modulus %12.6f)\n"
                       % (e.real, e.imag, abs(e)))
    out.append("\n")
    return "".join(out)


def out_report(model, input_path="", output_path="", irf_horizon=None,
               residuals="auto"):
    """Return the full ``.out`` estimation report for a fitted `model`.

    Reproduces the C engine's sections (header, parameters, Wald tests, IRF/FEVD,
    diagnostics, normalized model, inverse roots).  With ``residuals`` the
    per-series ASCII residual diagnostics section (`diagnose()`) is appended:
    ``"auto"`` (default) includes it when pyfug is importable, ``True`` requires
    pyfug, ``False`` omits it.  Not reproduced: the optimizer iteration/objective
    line (engine-internal; the log-likelihood is shown instead) and the
    inverse-roots ordering (modulus-sorted rather than chekma's QR order).
    """
    if model.result is None:
        raise RuntimeError("call fit() before out_report()")
    if model.result["ifault"] != 0:
        return (_header_block(model, input_path, output_path)
                + _convergence_block(model)
                + "\n**** ESTIMATION FAILED (code %d)\n" % model.result["ifault"])
    if irf_horizon is None:
        irf_horizon = 10 if model._w.shape[0] < 40 else 20
    parts = [
        _header_block(model, input_path, output_path),
        _convergence_block(model),
        _parameters_block(model),
        _wald_blocks(model),
        _irf_blocks(model, irf_horizon),
        _fevd_block(model, irf_horizon),
        _diagnostics_block(model),
        _matrices_block(model),
        _roots_block(model),
    ]
    if residuals is True or (residuals == "auto" and _have_pyfug()):
        parts.append(residual_report(model))
    return "".join(parts)


def _have_pyfug():
    from . import _pyfug
    return _pyfug.have_pyfug()


def write_out(model, path, input_path="", output_path="", irf_horizon=None,
              residuals="auto"):
    """Write the ``.out`` report for `model` to `path`."""
    text = out_report(model, input_path=input_path,
                      output_path=output_path or path, irf_horizon=irf_horizon,
                      residuals=residuals)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
#  Recursive fixed-parameter forecasts (.recursive)                           #
# --------------------------------------------------------------------------- #

def recursive_report(model, estwin, H):
    """Return the ``.recursive`` text: fixed-parameter forecasts from every origin.

    Port of the ``-estwin`` writer in drvarma.c (general VARMA(p,q)).
    """
    rows = model.recursive_forecast(estwin, H)        # (origin_raw, series, horizon, level)
    s = model.series
    freq = s.freq
    year0, sub0 = s.start
    off = model.d + model.D * freq
    origins = sorted({r[0] for r in rows})
    estwin_eff = origins[0] - off
    nobs_eff = origins[-1] - off
    deseason = model.deseason if model.deseason else "no"
    lut = {(a, b, c): d for a, b, c, d in rows}
    out = []
    out.append("Recursive fixed-parameter forecasts VARMA(%d,%d)\n" % (model.p, model.q))
    out.append("estwin_eff=%d nobs_eff=%d horizon=%d deseason=%s\n"
               % (estwin_eff, nobs_eff, H, deseason))
    out.append("origin series horizon level\n")
    for rraw in origins:
        oper, osub = obs_to_date(year0, sub0, rraw, freq)
        for j in range(s.m):
            for l in range(1, H + 1):
                out.append("%d/%d %s %d %.6f\n"
                           % (osub, oper, s.names[j], l, lut[(rraw, j, l)]))
    return "".join(out)


def write_recursive(model, estwin, H, path):
    """Write the ``.recursive`` report for `model` to `path`."""
    text = recursive_report(model, estwin, H)
    with open(path, "w") as fh:
        fh.write(text)
    return text


# --------------------------------------------------------------------------- #
#  Residual diagnostics section of the .out report                            #
# --------------------------------------------------------------------------- #

def _residual_stats_block(data, freq, start):
    """drvarma's File_StatSer stats block (own wording; no Jarque-Bera line)."""
    from .diagnostics import series_stats
    s = series_stats(data)
    n = s["n"]
    by, bt = start
    ey, et = obs_to_date(by, bt, n, freq)
    out = []
    out.append("Unconditional residuals (seasonal period: %d)\n" % freq)
    if freq > 1:
        out.append("%d observations: from %d/%d to %d/%d\n" % (n, bt, by, et, ey))
    else:
        out.append("%d observations: from %d to %d\n" % (n, by, ey))
    out.append("\n")
    out.append("                  Mean: %18.6f\n" % s["mean"])
    out.append("Standard error of mean: %18.6f\n" % s["std_error"])
    out.append("              Variance: %18.6f\n" % s["variance"])
    out.append("    Standard deviation: %18.6f\n" % s["std"])
    out.append("              Skewness: %18.6f\n" % s["skew"])
    out.append("              Kurtosis: %18.6f\n" % s["kurt"])
    miny, mint = obs_to_date(by, bt, s["min_idx"] + 1, freq)
    maxy, maxt = obs_to_date(by, bt, s["max_idx"] + 1, freq)
    if freq > 1:
        out.append("               Minimum: %18.6f at %2d/%d (observation %3d)\n"
                   % (s["min_val"], mint, miny, s["min_idx"] + 1))
        out.append("               Maximum: %18.6f at %2d/%d (observation %3d)\n"
                   % (s["max_val"], maxt, maxy, s["max_idx"] + 1))
    else:
        out.append("               Minimum: %18.6f at %d (observation %3d)\n"
                   % (s["min_val"], miny, s["min_idx"] + 1))
        out.append("               Maximum: %18.6f at %d (observation %3d)\n"
                   % (s["max_val"], maxy, s["max_idx"] + 1))
    return "".join(out)            # the blank line comes from the plot renderer


def residual_report(model):
    """Per-series residual diagnostics section of the ``.out`` (text).

    drvarma writes its own statistics block (File_StatSer wording); the intricate
    ASCII renderings (standardized plot, histogram, ACF/PACF correlograms) reuse
    pyfug's migrated `diagnose.c` renderers via the Tseries adapter (see
    docs/FUE_REUSE.md).  Requires pyfug.
    """
    import io
    from . import _pyfug, _ascii
    from .diagnostics import _acf_lags, series_stats, acf as _acf, pacf as _pacf
    _pyfug.require_pyfug()
    import pyfug.ascii as A

    if model.result is None:
        raise RuntimeError("call fit() before residual_report()")
    s = model.series
    freq = s.freq
    start = _pyfug.residual_start(model)
    res = model.residuals
    nobs, m = res.shape
    nlags = _acf_lags(nobs, freq)

    out = ["\n\n", _banner("                RESIDUAL DIAGNOSTICS                         ")]
    for j in range(m):
        col = res[:, j]
        out.append("\n--- Residual series a[%d] (%s) ---\n" % (j + 1, s.names[j]))
        out.append(_residual_stats_block(col, freq, start))
        # standardized time-series plot: reuse pyfug (matches), but map its
        # U+00AF/U+00AE outlier markers back to drvarma's '>'.
        ts = _pyfug.residual_to_tseries(model, j, lags=nlags)
        f = io.StringIO()
        A._write_ascii_plot(f, ts)
        out.append(f.getvalue().replace("¯", ">").replace("®", ">"))
        # histogram + ACF/PACF correlograms: drvarma's own renderers (exact).
        st = series_stats(col)
        out.append(_ascii.hist_ser(col, st["mean"], st["variance"]))
        a = _acf(col, nlags)
        out.append(_ascii.plot_cor(a, nlags, 1, nobs, freq))
        out.append(_ascii.plot_cor(_pacf(a), nlags, 0, nobs, freq))

    # cross-correlation functions between residual pairs (port of diagnose())
    if m > 1:
        out.append("\n--- Cross-correlation functions between residuals ---\n")
        clags = 3 * (1 + 2) if nobs >= 3 * (1 + 1) else nobs - 1 // 2
        clags = min(clags, nobs - 2)
        stats = [series_stats(res[:, k]) for k in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                ni, nj = s.names[i], s.names[j]
                out.append("\nCROSS CORRELATION a[%d] (%s) - a[%d] (%s)\n"
                           % (j + 1, nj, i + 1, ni))
                out.append("      %s --> %s IF k > 0\n" % (nj, ni))
                out.append("      %s --> %s IF k < 0\n" % (ni, nj))
                c1 = _ascii.ccf_corr(res[:, i], res[:, j], clags,
                                     stats[i]["mean"], stats[j]["mean"],
                                     stats[i]["std"], stats[j]["std"])
                c2 = _ascii.ccf_corr(res[:, j], res[:, i], clags,
                                     stats[j]["mean"], stats[i]["mean"],
                                     stats[j]["std"], stats[i]["std"])
                totcorr = np.concatenate([c1[::-1], c2[1:clags + 1]])
                out.append(_ascii.plot_ccf_ascii(totcorr, clags, nobs, freq=1))
                q2 = _ascii.chi_test_c(c2, clags + 1, nobs)
                q1 = _ascii.chi_test_c(c1, clags + 1, nobs)
                out.append("Q(%2d) = %6.3f     k >= 0      Q(%2d) = %6.3f      k > 0\n"
                           % (clags + 1, q2, clags, q2 - c2[0] ** 2 * (nobs + 2)))
                out.append("Q(%2d) = %6.3f     k <= 0      Q(%2d) = %6.3f      k < 0\n"
                           % (clags + 1, q1, clags, q1 - c1[0] ** 2 * (nobs + 2)))

    out.append("\n" + EQ + "\n")
    return "".join(out)
