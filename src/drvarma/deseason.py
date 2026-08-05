"""Harmonic seasonal adjustment (numpy port of deseason.c + seasonal_detection.c).

Delicate point (matches the C exactly):
  * The harmonic amplitudes are ESTIMATED on the d=1 DIFFERENCED basis: the
    regressand is the first difference of the raw *levels*, and each harmonic
    regressor cos(w t)/sin(w t) enters as its d-th difference.  The amplitude on
    the differenced regressor equals the level amplitude (differencing is applied
    to the regressors too), so estimating on d=1 does not bias the amplitudes.
  * The amplitudes are then mapped back to LEVEL seasonal dummies via the A0
    matrix (harmonics evaluated at periods 1..s-1) plus a sum-to-zero constraint,
    and subtracted from the raw *levels*.
"""

import numpy as np

try:
    from scipy.stats import f as _fdist
    def _f_crit(df1, df2, alpha=0.05):
        return float(_fdist.ppf(1.0 - alpha, df1, df2))
except Exception:  # pragma: no cover
    def _f_crit(df1, df2, alpha=0.05):
        return float("inf")


def _harmonic_design(n, d, s):
    """Design matrix (n x s): intercept + d-th differences of cos/sin harmonics.

    Row i corresponds to original time t = i + d + 1 (1-based), matching the C.
    """
    ncol = s                                   # 1 + (s-1) harmonics
    X = np.zeros((n, ncol))
    X[:, 0] = 1.0
    for i in range(n):
        t = i + d + 1
        col = 1
        for freq in range(1, s // 2 + 1):
            omega = 2.0 * np.pi * freq / s
            cvals = [np.cos(omega * (t - k)) for k in range(d + 1)]
            svals = [np.sin(omega * (t - k)) for k in range(d + 1)]
            if d == 0:
                dc, ds = cvals[0], svals[0]
            elif d == 1:
                dc, ds = cvals[0] - cvals[1], svals[0] - svals[1]
            else:
                dc, ds = (cvals[0] - 2 * cvals[1] + cvals[2],
                          svals[0] - 2 * svals[1] + svals[2])
            if freq < s / 2:
                X[i, col] = dc; col += 1
                X[i, col] = ds; col += 1
            elif s % 2 == 0 and freq == s // 2:
                X[i, col] = dc; col += 1
    return X


def harmonic_regression_differenced(y, d, s):
    """OLS harmonic regression on the differenced basis.

    Returns (coeffs (s-1,), f_stat, r_squared).
    """
    y = np.asarray(y, float)
    n = len(y)
    num_harm = s - 1
    total = num_harm + 1
    X = _harmonic_design(n, d, s)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    coeffs = c[1:]
    ypred = X @ c
    ymean = y.mean()
    sse = float(np.sum((y - ypred) ** 2))
    ssr = float(np.sum((ypred - ymean) ** 2))
    sst = float(np.sum((y - ymean) ** 2))
    r2 = ssr / sst if sst > 1e-12 else 0.0
    if n > total and num_harm > 0 and sse > 1e-12:
        f_stat = (ssr / num_harm) / (sse / (n - total))
    else:
        f_stat = 0.0
    return coeffs, f_stat, r2


def _A0_matrix(s):
    """Map harmonic amplitudes to the first s-1 level dummies (port of generate_A0_matrix)."""
    sz = s - 1
    A0 = np.zeros((sz, sz))
    for i in range(sz):
        t = i + 1
        col = 0
        for freq in range(1, s // 2 + 1):
            angle = 2.0 * np.pi * freq * t / s
            if freq < s / 2:
                A0[i, col] = np.cos(angle); col += 1
                A0[i, col] = np.sin(angle); col += 1
            elif s % 2 == 0 and freq == s // 2:
                A0[i, col] = np.cos(angle); col += 1
    return A0


def harmonics_to_dummies(coeffs, s):
    """Level seasonal dummies (length s, sum-to-zero) from harmonic amplitudes."""
    A0 = _A0_matrix(s)
    dummies = np.zeros(s)
    dummies[:s - 1] = A0 @ np.asarray(coeffs, float)
    dummies[s - 1] = -dummies[:s - 1].sum()
    return dummies


def _acf_at(x, k):
    """Autocorrelation at lag k alone (same estimator as `diagnostics.acf`)."""
    x = np.asarray(x, float).ravel()
    n = x.shape[0]
    if n <= k:
        return 0.0
    xc = x - x.mean()
    var = float((xc ** 2).mean())
    if var < 1e-300:
        return 0.0
    return float((xc[:n - k] * xc[k:]).sum()) / (n * var)


def deseasonalize_raw(raw, s, start_sub=1, mode="auto", d=1, alpha=0.05):
    """Harmonic seasonal adjustment of raw levels (port of deseasonalize_raw).

    Parameters
    ----------
    raw : (nobs, m) raw level data
    s : seasonal period (= frequency)
    start_sub : starting subperiod (1-based)
    mode : "auto" (adjust only series with significant seasonality) or "force"

    Returns
    -------
    (adjusted, dummies, info)
      adjusted : (nobs, m) deseasonalised levels
      dummies  : (m, s) level seasonal dummies (zeros for series left unadjusted)
      info     : list of per-series dicts {f_stat, r2, seasonal, adjusted}
    """
    raw = np.atleast_2d(np.asarray(raw, float))
    nobs, m = raw.shape
    num_harm = s - 1
    n_diff = nobs - d
    force = (mode == "force") or (mode == 1)
    adjusted = raw.copy()
    dummies = np.zeros((m, s))
    info = []
    for j in range(m):
        if n_diff <= num_harm + 1:
            info.append({"f_stat": 0.0, "r2": 0.0, "seasonal": False,
                         "adjusted": False, "acf_s_before": 0.0,
                         "acf_s_after": 0.0, "improved": True})
            continue
        diff = raw[1:, j] - raw[:-1, j]
        coeffs, f_stat, r2 = harmonic_regression_differenced(diff, d, s)
        f_crit = _f_crit(num_harm, n_diff - num_harm - 1, alpha)
        is_seasonal = f_stat > f_crit
        do_des = True if force else is_seasonal
        if do_des:
            # `_harmonic_design` builds the harmonics from t = i + d + 1, i.e. in a
            # phase RELATIVE to the start of the series, and never sees `start_sub`.
            # So `level` is indexed by offset-from-start, not by absolute subperiod.
            # Rotate it into absolute-subperiod indexing before using it, otherwise
            # the pattern is estimated in one phase and subtracted in another — which
            # for start_sub != 1 ADDS seasonal variance instead of removing it.
            # With start_sub == 1 the rotation is the identity (back-compatible).
            level_rel = harmonics_to_dummies(coeffs, s)
            level = np.empty(s)
            level[(np.arange(s) + start_sub - 1) % s] = level_rel
            dummies[j] = level
            periods = (np.arange(nobs) + start_sub - 1) % s
            adjusted[:, j] = raw[:, j] - level[periods]
        # POST-CONDITION: did the adjustment actually remove seasonality?
        #
        # Nothing used to check. The adjustment was assumed to work because the
        # F test said the series WAS seasonal — but detecting seasonality and
        # removing it correctly are different claims, and the phase off-by-one
        # fixed in `start_sub` ADDED seasonal variance for two years without a
        # single warning. |ACF(s)| of the differenced series, before against
        # after, is a one-line check that would have caught it on the first run.
        #
        # Reported, never enforced: on a short or weakly seasonal series the
        # comparison is noisy, and the caller — not this function — decides what
        # to do about it.
        acf_before = _acf_at(diff, s)
        acf_after = _acf_at(adjusted[1:, j] - adjusted[:-1, j], s) if do_des \
            else acf_before
        # "|ACF(s)| went down" is necessary but NOT sufficient, and the real bug
        # proves it: with the phase off by one month the adjustment still took
        # |ACF(12)| from 0.516 to 0.249 — an improvement, so a before/after test
        # passes — while the correct phase reaches 0.012. A wrong phase is still
        # subtracting a seasonal pattern, just the wrong one.
        # So compare against the BEST of the s possible phases: the applied
        # pattern should sit at (or very near) the minimum. Costs s extra
        # autocorrelations, which is nothing, and it does catch the one-month
        # shift that a before/after test lets through.
        acf_best = acf_after
        if do_des:
            for ph in range(s):
                cand = raw[:, j] - level[(np.arange(nobs) + ph) % s]
                a = _acf_at(cand[1:] - cand[:-1], s)
                if abs(a) < abs(acf_best):
                    acf_best = a
        info.append({"f_stat": f_stat, "r2": r2,
                     "seasonal": bool(is_seasonal), "adjusted": bool(do_des),
                     "acf_s_before": acf_before, "acf_s_after": acf_after,
                     "acf_s_best": acf_best,
                     # None, not True, when nothing was adjusted: "nothing to
                     # check" and "checked and fine" are different answers, and
                     # collapsing them would hide a series that was left alone
                     # when it should not have been.
                     #
                     # `improved` is the weak test (did it go down at all);
                     # `phase_ok` is the one with teeth. The 0.02 slack is for
                     # sampling noise: a neighbouring phase can beat the right
                     # one by a hair on a short series without anything being
                     # wrong.
                     "improved": (abs(acf_after) < abs(acf_before)) if do_des
                                 else None,
                     "phase_ok": (abs(acf_after) <= abs(acf_best) + 0.02)
                                 if do_des else None})
    return adjusted, dummies, info
