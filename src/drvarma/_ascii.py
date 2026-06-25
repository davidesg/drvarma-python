"""drvarma's own ASCII correlogram and histogram renderers.

Faithful ports of ``PlotCor`` and ``File_HistSer`` from the C ``diagnose.c`` so the
residual ``.out`` section matches drvarma exactly (pyfug's migration uses fue's
slightly different layout — different histogram width, ``± `` band labels, etc.).
The standardized time-series plot is reused from pyfug (it matches); these two are
written here.  See ``docs/FUE_REUSE.md``.
"""

import math

import numpy as np

from .diagnostics import ljung_box


def round_local(num):
    """Round half away from zero (port of nlatools.c round_local)."""
    t1 = abs(num)
    t2 = math.floor(t1)
    if t1 - t2 >= 0.5:
        t2 = math.ceil(t1)
    itmp = int(t2)
    return itmp if num >= 0.0 else -itmp


_RULER = "-------------+-------------------------+-------------------------+--------------"


def plot_cor(corr, lags, isacf, nobs, freq, npar=0):
    """ASCII correlogram (port of diagnose.c PlotCor).

    `corr` is the ACF (isacf=1) or PACF (isacf=0), 0-indexed so ``corr[i-1]`` is
    lag i.  Returns the rendered text block.
    """
    corr = np.asarray(corr, float).ravel()
    band = 2.0 / math.sqrt(nobs)
    out = []
    if isacf:
        marcas = "            -1                         0                         1  L-B Q  DF"
        out.append("Autocorrelation function (acf ")
    else:
        marcas = "            -1                         0                         1"
        out.append("Partial autocorrelation function (pacf ")
    out.append("bands =  %5.3f):\n" % band)
    out.append("\n")
    out.append(marcas + "\n")
    out.append(_RULER + "\n")

    band_pos = round_local(band * 25.0)
    for i in range(1, lags + 1):
        ci = corr[i - 1]
        line = [" "] * 53
        seasonal = (freq != 1) and (i % freq == 0)
        symbol = "+" if seasonal else "*"
        line[51] = "+" if seasonal else "|"
        posi = abs(round_local(ci * 25.0))
        if ci * 25.0 <= 0.0:
            for j in range(25 - posi, 26):
                line[j] = symbol
        else:
            for j in range(25, 25 + posi + 1):
                line[j] = symbol
        line[25] = "|"
        if line[25 + band_pos] == " ":
            line[25 + band_pos] = ":"
        if line[25 - band_pos] == " ":
            line[25 - band_pos] = ":"
        border = "+" if seasonal else "|"
        row = "%4d %7.3f %s%s" % (i, ci, border, "".join(line))

        want_q = isacf and (i - npar >= 1) and (
            (freq != 1 and i % freq == 0)
            or (freq != 1 and i % freq != 0 and i == lags)
            or (freq == 1 and i == lags))
        if want_q:
            Q, _, _ = ljung_box(corr[:i], nobs)
            row += "%6.2f %3d" % (Q, i - npar)
        out.append(row + "\n")

    out.append(_RULER + "\n")
    out.append(marcas + "\n")
    out.append("\n")
    return "".join(out)


def hist_ser(data, mean=None, var=None):
    """ASCII standardized histogram (port of diagnose.c File_HistSer)."""
    data = np.asarray(data, float).ravel()
    n = data.shape[0]
    if mean is None:
        mean = float(data.mean())
    if var is None:
        var = float(((data - mean) ** 2).mean())
    sd = math.sqrt(var)
    if not (sd > 0.0):
        return "(residual series is constant; histogram skipped)\n\n"
    z = (data - mean) / sd
    xmax = float(np.max(np.abs(z)))
    if not (xmax <= 8.0):
        return "Warning: at least one observation above 8 sigmas\n"
    xmax = 4.0 if xmax <= 4.0 else 8.0

    if xmax == 4.0:
        nphor = 4
        no, yes = "    ", "...."
        base1 = "        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+"
        base2 = "       -4      -3      -2      -1       0      +1      +2      +3      +4"
    else:
        nphor = 2
        no, yes = "  ", ".."
        base1 = "        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
        base2 = "       -8  -7  -6  -5  -4  -3  -2  -1   0  +1  +2  +3  +4  +5  +6  +7  +8"

    NumFil, NumCol = 17, 64
    # breakpoints (1-indexed), bandwidth 0.5
    breakk = [0.0, -xmax + 0.5]
    while breakk[-1] < xmax:
        breakk.append(breakk[-1] + 0.5)
    NumCat = len(breakk) - 1

    freqs = [0] * (NumCat + 1)
    Atip1 = Atip2 = 0
    for v in z:
        if v <= breakk[1]:
            freqs[1] += 1
        else:
            for j in range(2, NumCat + 1):
                if breakk[j - 1] < v <= breakk[j]:
                    freqs[j] += 1
                    break
        if abs(v) >= 2.0:
            Atip2 += 1
        if abs(v) >= 1.0:
            Atip1 += 1

    fmax = max(freqs[1:NumCat + 1])
    obs_per_fil = fmax / 16.0

    # bar rows j=2..NumFil (shist index j-1, i.e. rows 1..16)
    shist = [None] * NumFil
    for j in range(2, NumFil + 1):
        row = []
        for i in range(1, NumCat + 1):
            row.append(yes if freqs[i] > obs_per_fil * (NumFil - j) else no)
        shist[j - 1] = list("".join(row) + "|")

    # count-label rows (aux), placed at the top of each bar
    chk = [0] * (NumCat + 1)
    aux = [""] * NumFil
    for j in range(2, NumFil + 1):
        for i in range(1, NumCat + 1):
            if freqs[i] > obs_per_fil * (NumFil - j) and chk[i] == 0:
                s1 = str(freqs[i])
                if nphor == 2:
                    if len(s1) == 1:
                        s1 = s1 + " "
                else:
                    if len(s1) == 2:
                        s1 = " " + s1
                    elif len(s1) == 1:
                        s1 = "  " + s1
                    elif len(s1) == 3:
                        s1 = s1 + " "
                aux[j - 2] += s1
                chk[i] = 1
            else:
                aux[j - 2] += no

    # top row = aux[0] with a right border at index NumCol-1.  C quirk: the
    # border is written via shist[0][NumCol-1]='|' AFTER strcpy(aux[0]); if
    # aux[0] is shorter than NumCol-1 the '|' lands past the string's NUL and is
    # never printed (no border).  Replicate that exactly.
    top = list(aux[0])
    if len(top) >= NumCol:
        top = top[:NumCol]
        top[NumCol - 1] = "|"
    elif len(top) == NumCol - 1:
        top.append("|")
    shist[0] = top
    for j in range(2, NumFil):
        for k in range(len(aux[j - 1])):
            if k < len(shist[j - 1]) and aux[j - 1][k] != " ":
                shist[j - 1][k] = aux[j - 1][k]

    out = []
    out.append("Standardized time series histogram:\n\n")
    out.append(base2 + "\n")
    out.append(base1 + "\n")
    for i in range(NumFil):
        out.append("        |" + "".join(shist[i]) + "\n")
    out.append(base1 + "\n")
    out.append(base2 + "\n")
    out.append("\n")
    out.append("%16d values outside (-1,+1): %5.2f %% (31.74 %% expected)\n"
               % (Atip1, Atip1 * 100.0 / n))
    out.append("%16d values outside (-2,+2): %5.2f %% ( 4.56 %% expected)\n"
               % (Atip2, Atip2 * 100.0 / n))
    out.append("\n")
    return "".join(out)


def ccf_corr(data1, data2, lags, mean1, mean2, sd1, sd2):
    """Cross-covariances corr[0..lags] (lag k pairs data1_t with data2_{t+k}).

    Port of diagnose.c Ccf (divided by n·sd1·sd2).
    """
    x1 = np.asarray(data1, float).ravel() - mean1
    x2 = np.asarray(data2, float).ravel() - mean2
    n = x1.shape[0]
    den = n * sd1 * sd2
    corr = np.zeros(lags + 1)
    for k in range(lags + 1):
        corr[k] = float((x1[:n - k] * x2[k:]).sum()) / den
    return corr


def chi_test_c(corr, lags, nobs):
    """Cross-correlation portmanteau (port of diagnose.c ChiTestC).

    Q = n(n+2) Σ_{k=1}^{lags} corr[k]²/(n-k+1); `corr` is 0-indexed (corr[0]=lag0).
    """
    corr = np.asarray(corr, float).ravel()
    s = sum(corr[i] ** 2 / (nobs - i) for i in range(lags))
    return float(nobs * (nobs + 2) * s)


def plot_ccf_ascii(corr, lags, nobs, freq=1):
    """Two-sided cross-correlation correlogram (port of diagnose.c PlotCCF).

    `corr` is 0-indexed length 2·lags+1; position k shows lag k-lags.
    """
    corr = np.asarray(corr, float).ravel()
    band = 2.0 / math.sqrt(nobs)
    marcas = "            -1                         0                         1"
    out = ["CCF BANDS  2.0/SQRT(N) =  %2.5f:\n" % band]
    out.append("\n" + marcas + "\n" + _RULER + "\n")
    band_pos = int(band * 25.0 + 0.5)
    for i in range(1, 2 * lags + 2):
        lag = i - lags - 1
        ci = corr[i - 1]
        line = [" "] * 80
        seasonal = (freq != 1) and (lag % freq == 0)
        line[51] = "+" if seasonal else "|"
        pos = ci * 25.0
        posi = int(pos + (0.5 if pos >= 0 else -0.5))
        if pos <= 0.0:
            for j in range(25 - abs(posi), 26):
                line[j] = "*"
        else:
            for j in range(25, 25 + posi + 1):
                line[j] = "*"
        line[25] = "|"
        if line[25 + band_pos] == " ":
            line[25 + band_pos] = ":"
        if line[25 - band_pos] == " ":
            line[25 - band_pos] = ":"
        border = "+" if seasonal else "|"
        out.append("%4d %7.3f %s%s\n" % (lag, ci, border, "".join(line)))
    out.append(_RULER + "\n" + marcas + "\n")
    return "".join(out)
