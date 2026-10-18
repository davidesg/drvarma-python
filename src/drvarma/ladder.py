"""The ladder: fue's univariate models as the input of a VARMA.

The Python port of drvarma 5.0's ladder mode (``src/escalera.c`` in the atsw-gui
monorepo, design in ``engines/drvarma/docs/DESIGN-v5-ladder.md``). The C is the
oracle: ``tests/test_ladder.py`` pins its numbers.

Each series brings its full univariate model from a fue file, a ``.pre`` (an
optimum) or an ``.inp`` (a specification): Box-Cox, deterministic terms,
non-stationary operator, mean and ARMA factors. On the stationary series ``w``
the model is

    Phi(B) (w_t - mu) = Theta(B) a_t,        a_t ~ N(0, sigma2 Q)

with the DIAGONAL of each row taken from its file, ``phi_i(B) Phi_i(B^s)`` and
``theta_i(B) Theta_i(B^s)``, and the off-diagonal free up to the cross orders
``p`` and ``q``. With ``p = q = 0`` and a diagonal Q the system splits into the
m univariate models, and that is THE GATE: the joint likelihood evaluated at the
univariate optima must be the sum of the univariate ones, or nothing built on
top of it can be trusted.

Nothing about a univariate model is re-implemented here. The file is read by
``fue.load`` (the reference parser), and each series goes to its stationary
series and its ARMA polynomials through ``fue.cast_us.cast_us_py``, the same
code fue estimates with. Back to the level, in the forecast, it goes through
the pieces of ``fue.forecast``. drtran-python uses the same functions.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import _qnewt
from ._engine import elf_c, marma_c

_LOG2PI = 1.837877066                       # the constant of the C's est()


def _lik_label(lik):
    return {"shea": "exact, Shea (1989), AS 242",
            "both": "exact, Mauricio (1997), AS 311; checked against Shea (1989), AS 242"
            }.get(lik, "exact, Mauricio (1997), AS 311")


# The MA invertibility wall, one tolerance for both sides (the C's MA_WALL_TOL,
# atsw-gui lib/lik/lik.h): chekma refuses a root at modulus >= 1 + 5e-5, and a
# stop with a root at modulus >= 1 - 5e-5 is on the wall. Before, only >= 1
# counted, so a fit that reached the wall from inside said "converged".
MA_WALL_TOL = 5e-5


def _ma_boundary(theta):
    """How many MA inverse roots sit on the wall, and of how many.

    The eigenvalues of the MA companion matrix, as the C's chekma computes them
    (it refuses beyond 1 + MA_WALL_TOL; a root within MA_WALL_TOL of the unit
    circle is on the wall).
    theta is (q, m, m), Theta(B) = I - sum theta_k B^k.
    """
    theta = np.asarray(theta, float)
    if theta.ndim != 3 or theta.shape[0] == 0 or not np.any(theta):
        return 0, 0
    q, m, _ = theta.shape
    A = np.zeros((m * q, m * q))
    for k in range(q):
        A[:m, k * m:(k + 1) * m] = theta[k]
    for k in range(q - 1):
        A[(k + 1) * m:(k + 2) * m, k * m:(k + 1) * m] = np.eye(m)
    mod = np.abs(np.linalg.eigvals(A))
    return int(np.sum(mod >= 1.0 - MA_WALL_TOL)), m * q


_CRITERION = {1: "norm of scaled gradient <= gradtol",
              2: "scaled distance between last two steps <= steptol",
              3: "last global step failed to locate a lower point",
              4: "iteration limit reached",
              5: "five consecutive steps of maximum length taken"}


def _stop_block(r):
    """The C's report_stop, line for line: a stop on the MA wall says so."""
    if r.ma_boundary:
        head = f"  OPTIMIZER STOPPED at the MA invertibility boundary after {r.nit} iterations"
    else:
        head = (f"  OPTIMIZER {'CONVERGED' if r.termcode in (1, 2) else 'STOPPED'} "
                f"after {r.nit} iterations")
    out = ["=" * 61, head]
    if r.fk is not None:
        out.append(f"  Objective function = {r.fk:.12f}")
    if r.termcode in _CRITERION:
        out.append(f"  Convergence criterion: {_CRITERION[r.termcode]}")
    if r.ma_boundary:
        out.append(f"  MA boundary: {r.ma_boundary} of {r.ma_nroots} inverse roots within 5e-5 of the unit circle")
    out.append("=" * 61)
    return out


def _concentrated(m, n, f1, f2):
    """The concentrated log-likelihood from (f1, f2), as the C's est() writes it."""
    return (-0.5 * m * n * (_LOG2PI - math.log(m) - math.log(n) + 1.0)
            - 0.5 * n * (m * math.log(f1) + math.log(f2)))
GATE_TOL = 1e-6                             # |logL_diag - SUM logL_i|, relative
PRE_MOVE = 1e-3                             # a .pre that moves more is no optimum


def _warn_if_no_engine():
    """The ladder evaluates the likelihood thousands of times per fit. Without
    the compiled engine (the binary wheels) elf_c falls back to the pure-Python
    AS 311, ~250x slower: say so instead of degrading in silence. The same
    once-per-process warning as every other fallback (BUG-0005)."""
    from . import _engine
    if _engine._load_c() is None:
        _engine._warn_python_fallback(
            "the ladder will run the pure-Python likelihood", stacklevel=4)


# log(Q_ii/Q_11) beyond this overflows exp(): the point is inadmissible
_LVAR_MAX = 700.0


class LadderError(ValueError):
    """The input cannot be crossed, or the gate failed."""


class GateError(LadderError):
    """The joint cast does not reproduce the univariate models."""


# --------------------------------------------------------------------------- #
#  One series                                                                 #
# --------------------------------------------------------------------------- #

def _truncated(model, n):
    """A shallow copy of a fue model whose series ends at observation n."""
    import fue
    ts = model.series
    m2 = copy.copy(model)
    t2 = fue.TimeSeries(ts.data[:n], freq=ts.freq, start=ts.start, name=ts.name)
    t2._sin_fecha = getattr(ts, "_sin_fecha", False)
    m2.series = t2
    return m2


class LadderSeries:
    """A fue model and the state the ladder keeps for it.

    ``x`` is fue's free-parameter vector, in fue's order (deterministic terms,
    ARMA factors, fixed-frequency factors, mean): ``cast_us_py`` reads it. The
    ladder decides which of those entries it lets move (``free``).
    """

    def __init__(self, path, ts, model):
        from fue.cast_us import _build_initial_x
        self.path = str(path)
        self.ts = ts
        self.model = model
        self.name = ts.name
        self.nobs_full = int(ts.nobs)
        self.nobs = self.nobs_full
        self.x = np.asarray(_build_initial_x(model), float)
        self.x_file = self.x.copy()
        self.n_det = sum(int(bool(f)) for itv in model.interventions for f in itv.omega_free) \
            + sum(int(bool(f)) for itv in model.interventions for f in itv.delta_free)
        self.has_mu = bool(model.estimate_mu)
        self._spec_cache = {}
        self._w_cache = None                 # (nobs, det values) -> w
        self._poly_spec = self._make_poly_spec()

    # -- fue's casts ------------------------------------------------------- #
    def _make_poly_spec(self):
        """The model WITHOUT its deterministic terms, on a minimal series.

        Only the ARMA polynomials and the mean are wanted from it, and
        ``cast_us_py`` computes them from the same ``x`` minus the deterministic
        entries. Asking the full model would rebuild w, the expensive part, at
        every evaluation.
        """
        from fue.cast_us import build_est_spec
        m = _truncated(self.model, min(self.nobs_full, self.ornsop + 2))
        m.interventions = []
        return build_est_spec(m)

    def est_spec(self, n=None):
        from fue.cast_us import build_est_spec
        n = self.nobs if n is None else n
        if n not in self._spec_cache:
            self._spec_cache[n] = build_est_spec(_truncated(self.model, n))
        return self._spec_cache[n]

    @property
    def ornsop(self):
        from fue.forecast import _nonsop_coefs
        return len(_nonsop_coefs(self.model.d, self.model.D, self.freq,
                                 ifadf=self.model.ifadf or []))

    @property
    def freq(self):
        return self.ts.freq if self.ts.freq > 0 else 1

    def stationary(self):
        """w of the current sample, with the current deterministic terms."""
        from fue.cast_us import cast_us_py
        key = (self.nobs, tuple(self.x[:self.n_det]))
        if self._w_cache is None or self._w_cache[0] != key:
            *_, w, ifault = cast_us_py(self.x, self.est_spec())
            if ifault:
                return None
            self._w_cache = (key, np.asarray(w, float))
        return self._w_cache[1]

    def levels(self):
        """The transformed LEVELS of the current sample: the model's Box-Cox
        and its seasonal differences, but NOT its regular ones — what Box and
        Tiao's (1977) canonical analysis reads to ask whether the series need
        their regular differences jointly (§4.4). Deterministic terms are not
        removed (their effect stays in the levels)."""
        from fue.forecast import _boxcox
        m = self.model
        y = np.array([_boxcox(float(v), m.boxlam, m.refactor)
                      for v in self.ts.data[:self.nobs]])
        s = self.freq
        for _ in range(int(m.D or 0)):
            y = y[s:] - y[:-s]
        return y

    def polynomials(self):
        """(phi, theta, mu, ifault): the expanded ARMA of the series."""
        from fue.cast_us import cast_us_py
        p, q, phi, theta, mu, _w, ifault = cast_us_py(self.x[self.n_det:],
                                                      self._poly_spec)
        return np.asarray(phi, float), np.asarray(theta, float), float(mu), int(ifault)

    def orders(self):
        """(p, q) of the expanded ARMA. A side whose factors are all fixed at
        zero has no order: it is fue's placeholder, the AR(1) fixed at 0.0
        that a model without ARMA carries (sima BUG-0001), not an estimated
        operator."""
        phi, theta, _mu, _if = self.polynomials()
        m = self.model

        def inert(factors, frees):
            # a flag list of None, or a None factor, means free (as _split_specs)
            free = lambda i, j: frees is None or frees[i] is None or frees[i][j]
            return all(not free(i, j) and float(c) == 0.0
                       for i, f in enumerate(factors) for j, c in enumerate(f))

        p = 0 if inert(m.ar, m.ar_free) and inert(m.ar_s, m.ar_s_free) \
            and not m.ar_f else len(phi)
        q = 0 if inert(m.ma, m.ma_free) and inert(m.ma_s, m.ma_s_free) \
            and not m.ma_f else len(theta)
        return p, q

    # -- the echelon form's row factor (docs/ECHELON_PLAN.md, stage 4) ------ #
    def has_row_factor(self):
        """Does the model carry seasonal or fixed-frequency factors? In the
        echelon form they multiply the series' whole row."""
        m = self.model
        return bool(m.ar_s or m.ma_s or m.ar_f or m.ma_f)

    def _split_specs(self):
        """Two poly specs of the model, one with its regular factors only and
        one with the rest (the row factor), and the masks that pick each
        one's entries from x[n_det:] (fue's order: ar, ar_s, ma, ma_s, ar_f,
        ma_f, mu)."""
        if getattr(self, "_split", None) is not None:
            return self._split
        from fue.cast_us import build_est_spec
        m = self.model

        def nfree(factors, frees):
            return sum(sum(1 for j in range(len(f))
                           if frees is None or frees[i] is None or frees[i][j])
                       for i, f in enumerate(factors))
        sizes = [("reg", nfree(m.ar, m.ar_free)), ("fac", nfree(m.ar_s, m.ar_s_free)),
                 ("reg", nfree(m.ma, m.ma_free)), ("fac", nfree(m.ma_s, m.ma_s_free)),
                 ("fac", sum(1 for f in m.ar_f if f.free)),
                 ("fac", sum(1 for f in m.ma_f if f.free)),
                 ("both", int(self.has_mu))]
        masks = {"reg": [], "fac": []}
        for kind, k in sizes:
            for part in ("reg", "fac"):
                masks[part] += [kind in (part, "both")] * k
        specs = {}
        for part, drop in (("reg", ("ar_s", "ma_s", "ar_f", "ma_f")), ("fac", ("ar", "ma"))):
            mm = _truncated(self.model, min(self.nobs_full, self.ornsop + 2))
            mm.interventions = []
            for a in drop:
                setattr(mm, a, [])
                if a in ("ar", "ma", "ar_s", "ma_s"):
                    setattr(mm, a + "_free", None)
            specs[part] = build_est_spec(mm)
        self._split = (specs, {k: np.array(v, bool) for k, v in masks.items()})
        return self._split

    def split_polynomials(self):
        """(phi, theta, fphi, ftheta, mu, ifault): the regular ARMA, the row
        factor (the seasonal and fixed-frequency factors, expanded) and the
        mean. phi * fphi and theta * ftheta are ``polynomials()``'s."""
        from fue.cast_us import cast_us_py
        if not self.has_row_factor():
            phi, theta, mu, ifault = self.polynomials()
            return phi, theta, np.zeros(0), np.zeros(0), mu, ifault
        specs, masks = self._split_specs()
        x = self.x[self.n_det:]
        out = []
        for part in ("reg", "fac"):
            _p, _q, phi, theta, mu, _w, ifault = cast_us_py(x[masks[part]], specs[part])
            if ifault:
                return None, None, None, None, None, int(ifault)
            out.append((np.asarray(phi, float), np.asarray(theta, float), float(mu)))
        (phi, theta, mu), (fphi, ftheta, _mu) = out
        return phi, theta, fphi, ftheta, mu, 0

    def with_regular(self, k, zeros=None):
        """A copy whose regular ARMA is a free AR(k)/MA(k), the echelon row's
        own part, the rest of the model (the row factor, the deterministic
        terms, the differences, the mean) as the file has it. It starts at
        the file's regular polynomials when they fit in k, at zero if not.

        ``zeros``: ``{"ar": {lags}, "ma": {lags}}``, own lags (1..k) held at
        zero -- Tiao and Box's simplification on the row's own part, which
        on route U is no longer the univariate model's."""
        phi, theta, _f, _g, _mu, _if = self.split_polynomials()
        m2 = copy.copy(self.model)
        zeros = zeros or {}
        for kind in ("ar", "ma"):
            bad = [lag for lag in zeros.get(kind, ()) if not 1 <= int(lag) <= k]
            if bad:
                raise LadderError(f"{self.name}: own {kind.upper()} lag {bad[0]} is "
                                  f"outside 1..{k}")

        def seed(c, kind):
            c = list(map(float, c)) if len(c) <= k else []
            z = {int(lag) for lag in zeros.get(kind, ())}
            c = [0.0 if lag in z else v
                 for lag, v in enumerate(c + [0.0] * (k - len(c)), 1)]
            free = [[lag not in z for lag in range(1, k + 1)]] if (k and z) else None
            return ([c] if k else []), free
        m2.ar, m2.ar_free = seed(phi, "ar")
        m2.ma, m2.ma_free = seed(theta, "ma")
        new = LadderSeries(self.path, self.ts, m2)
        # the rest in its current state: the deterministic terms, the row
        # factor and the mean
        new.x[:self.n_det] = self.x[:self.n_det]
        if self.has_row_factor():
            new.x[new.n_det:][new._split_specs()[1]["fac"]] = \
                self.x[self.n_det:][self._split_specs()[1]["fac"]]
        elif self.has_mu:
            new.x[-1] = self.x[-1]
        new.nobs = self.nobs
        return new

    def factor_support(self):
        """(AR lags, MA lags) where the row factor can be non-zero, lag 0
        included: its structure, read by expanding it at arbitrary values."""
        from fue.cast_us import cast_us_py
        if not self.has_row_factor():
            return {0}, {0}
        specs, masks = self._split_specs()
        x0 = self.x[self.n_det:][masks["fac"]].copy()
        k = x0.size - int(self.has_mu)
        rng = np.random.default_rng(0)
        ar, ma, good = {0}, {0}, 0
        # admissible draws of either sign (a fixed-frequency MA needs c < 0);
        # the union of three: a lag that is zero by structure stays out
        for _ in range(200):
            x = x0.copy()
            x[:k] = rng.uniform(0.15, 0.45, k) * rng.choice([-1.0, 1.0], k)
            _p, _q, fphi, ftheta, _mu, _w, ifault = cast_us_py(x, specs["fac"])
            if ifault:
                continue
            ar |= {j + 1 for j, c in enumerate(fphi) if abs(c) > 1e-12}
            ma |= {j + 1 for j, c in enumerate(ftheta) if abs(c) > 1e-12}
            good += 1
            if good == 3:
                return ar, ma
        raise LadderError(f"{self.name}: no admissible values to read its row factor's "
                          "structure")

    def copy(self):
        """An independent copy, in the current state."""
        new = LadderSeries(self.path, self.ts, copy.copy(self.model))
        new.x, new.nobs = self.x.copy(), self.nobs
        return new

    def without_row_factor(self, ar_lags, ma_lags):
        """A copy whose ARMA is ONE regular AR and ONE regular MA, free at
        ``ar_lags`` and ``ma_lags`` and zero elsewhere, at the current
        expanded values (``polynomials()``): the row factor's restriction
        dropped, the alternative of ``Ladder.factor_test``."""
        phi, theta, _mu, _if = self.polynomials()
        m2 = copy.copy(self.model)

        def op(c, lags):
            D = max(lags, default=0)
            c = list(map(float, c)) + [0.0] * max(0, D - len(c))
            vals = [c[k] if k + 1 in lags else 0.0 for k in range(D)]
            return ([vals], [[k + 1 in lags for k in range(D)]]) if D else ([], None)
        m2.ar, m2.ar_free = op(phi, set(ar_lags))
        m2.ma, m2.ma_free = op(theta, set(ma_lags))
        m2.ar_s, m2.ma_s, m2.ar_s_free, m2.ma_s_free = [], [], None, None
        m2.ar_f, m2.ma_f = [], []
        new = LadderSeries(self.path, self.ts, m2)
        new.x[:self.n_det] = self.x[:self.n_det]
        if self.has_mu:
            new.x[-1] = self.x[-1]
        new.nobs = self.nobs
        return new

    def regular_orders(self):
        """The degrees of the regular ARMA: in the echelon form, the series'
        own part of its row, which the index bounds."""
        phi, theta, _f, _g, _mu, _if = self.split_polynomials()
        return len(phi), len(theta)

    # -- dates -------------------------------------------------------------- #
    def date_of(self, obs):
        """(year, period) of observation obs (1-based) of this series."""
        y0, p0 = self.ts.start
        f = self.freq
        k = (p0 - 1) + (obs - 1)
        return y0 + k // f, k % f + 1

    def canonicalize(self):
        """Put ``x`` in the INVERTIBLE form, with fue's own rule (BUG-0007).

        ``cast_us_py`` flips a regular MA(1) factor with |theta| > 1 to
        1/theta, and a fixed-frequency MA with c2 < -1 to 1/c2, before it
        builds the polynomial. The likelihood is therefore the same at theta
        and at 1/theta, and the optimiser may stop on either side. The model is
        the flipped one, so that is the value to report. Returns True if an
        entry changed.
        """
        m = self.model
        k = sum(int(bool(f)) for itv in m.interventions for f in itv.omega_free) \
            + sum(int(bool(f)) for itv in m.interventions for f in itv.delta_free)

        def nfree(factors, frees):
            n = 0
            for i, factor in enumerate(factors):
                free = frees[i] if frees is not None else None
                n += sum(1 for j in range(len(factor)) if free is None or free[j])
            return n
        k += nfree(m.ar, m.ar_free) + nfree(m.ar_s, m.ar_s_free)
        changed = False
        for i, factor in enumerate(m.ma):                 # regular MA factors
            free = m.ma_free[i] if m.ma_free is not None else None
            for j in range(len(factor)):
                if free is None or free[j]:
                    if len(factor) == 1 and abs(self.x[k]) > 1.0:
                        self.x[k] = 1.0 / self.x[k]; changed = True
                    k += 1
        k += nfree(m.ma_s, m.ma_s_free)
        k += sum(1 for ff in m.ar_f if ff.free)
        for ff in m.ma_f:                                 # fixed-frequency MA
            if ff.free:
                if self.x[k] < -1.0:
                    self.x[k] = 1.0 / self.x[k]; changed = True
                k += 1
        return changed

    def n_arma(self):
        """The series' own ARMA coefficients, those its .pre estimated: the
        block of x between the deterministic terms and the mean."""
        return int(self.x.size - self.n_det - int(self.has_mu))

    def free_mask(self, redet, fixarma):
        mask = np.zeros(self.x.size, bool)
        mask[:self.n_det] = bool(redet)
        mask[self.n_det:self.x.size - int(self.has_mu)] = not fixarma
        if self.has_mu:
            mask[-1] = True
        return mask

    def param_names(self):
        """Names in fue's order, one per entry of x."""
        m = self.model
        s = self.name
        out = []

        def label(itv):
            """The deterministic term as the .pre writes it: a harmonic by its
            frequency, an intervention by its date."""
            if itv.type in ("cos", "sin"):
                return f"{itv.type} {getattr(itv, 'harmonic', 0):g}"
            if itv.type in ("alter", "easter", "trend"):
                return itv.type
            y, p = self.date_of(int(itv.at) + 1)
            return f"{itv.type} {p}/{y}" if self.freq > 1 else f"{itv.type} {y}"

        for itv in m.interventions:
            for j, f in enumerate(itv.omega_free):
                if f:
                    out.append(f"w{j}_{s}[{label(itv)}]")
        for itv in m.interventions:
            for j, f in enumerate(itv.delta_free):
                if f:
                    out.append(f"d{j + 1}_{s}[{label(itv)}]")

        def fac(factors, frees, lo, lag):
            for k, factor in enumerate(factors):
                free = frees[k] if frees is not None else None
                for j, _v in enumerate(factor):
                    if free is None or free[j]:
                        tag = lo if len(factors) == 1 else f"{lo}{k + 1}"
                        out.append(f"{tag}_{s}[B^{(j + 1) * lag}]")
        fac(m.ar, m.ar_free, "phi", 1)
        fac(m.ar_s, m.ar_s_free, "Phi", self.freq)
        fac(m.ma, m.ma_free, "theta", 1)
        fac(m.ma_s, m.ma_s_free, "Theta", self.freq)
        for ff in m.ar_f:
            if ff.free:
                out.append(f"phif_{s}[f={ff.freq}]")
        for ff in m.ma_f:
            if ff.free:
                out.append(f"thetaf_{s}[f={ff.freq}]")
        if self.has_mu:
            out.append(f"mu_{s}")
        return out


def load(paths):
    """Read fue files (``.pre`` or ``.inp``) with fue's own parser.

    A file that fue does not read as a univariate model is refused, with fue's
    reason. What a file is, is decided by its content, not by its name.
    """
    import fue
    out = []
    for p in paths:
        try:
            ts, model = fue.load(str(p))
        except Exception as e:                          # fue says why
            hint = ""
            if str(p).endswith(".inp"):
                hint = ("\nIf it is a multivariate drvarma .inp (deprecated), "
                        f"convert it: drvarma -split {p}  (or drvarma.ladder.split)")
            raise LadderError(f"{p} is not a univariate model file of fue: {e}{hint}")
        out.append(LadderSeries(p, ts, model))
    return out


def _end_date(s):
    return s.date_of(s.nobs_full)


def check_alignment(series):
    """BUG-2: the same frequency and the same LAST date, or a refusal.

    The same rule, and the same words, as ``fuepre_check_alignment`` in the C
    and ``drtran.cast.check_alignment``.
    """
    ref = series[0]
    for s in series[1:]:
        if s.ts.freq != ref.ts.freq:
            raise LadderError(
                f"'{s.name}' is {s.ts.freq}-per-year and '{ref.name}' is "
                f"{ref.ts.freq}: they cannot be modelled jointly. Rebuild them at "
                f"the same frequency in art.")
    y0, p0 = _end_date(ref)
    for s in series[1:]:
        y, p = _end_date(s)
        if (y, p) != (y0, p0):
            raise LadderError(
                f"the series do NOT end on the same date: {ref.name} ends "
                f"{p0:02d}/{y0} and {s.name} ends {p:02d}/{y}. The joint cast "
                f"aligns at the END and trims to the shortest, which assumes a "
                f"common last observation; with different windows it would pair "
                f"observations that are years apart and say nothing. Rebuild both "
                f".pre in art over the window you mean to model.")


# --------------------------------------------------------------------------- #
#  The system                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Fit:
    """One estimation (or evaluation) of the ladder VARMA."""
    names: list
    x: np.ndarray
    std_errors: np.ndarray
    logL: float
    sigma2: float
    ifault: int
    nit: int = 0
    termcode: int = 0
    mu: np.ndarray = None
    phi: np.ndarray = None
    theta: np.ndarray = None
    qq: np.ndarray = None
    w: np.ndarray = None
    residuals: np.ndarray = None
    se_method: str = None      # "fdhess", "bfgs", or the fallback, said
    fk: float = None           # the optimiser's objective at the stop
    ma_boundary: int = 0       # MA inverse roots on the wall at the stop
    ma_nroots: int = 0         # (the C's est(): a stop on the invertibility wall)
    cov: np.ndarray = None     # covariance of the estimates (se_method's Hessian)

    @property
    def npar(self):
        return int(self.x.size)

    @property
    def sigma(self):
        return self.sigma2 * self.qq


class Ladder:
    """drvarma's ladder VARMA over fue models.

    Parameters
    ----------
    series : list of LadderSeries (``load``) or of paths
    p, q : cross orders (the off-diagonal dynamics)
    diagcov : diagonal innovation covariance
    redet : re-estimate the deterministic terms (default: fixed at the file)
    fixarma : keep the univariate ARMA factors fixed at the file
    method : 1 exact likelihood, xi truncated at 1e-3 (the C's default);
        2 exact, untruncated
    lik : "elf" (Mauricio, AS 311; default), "shea" (Shea, AS 242, the
        independent benchmark: the objective and the reported likelihood;
        residuals stay elf's), or "both" (elf is the objective; Shea is
        evaluated at every point and ``lik_check`` reports the largest
        |dlogL|). Needs the compiled engine. As the C's -lik.
    estwin : estimate on the first ``estwin`` observations of the FIRST series
        (the others are cut at the same date); ``recursive`` then forecasts
        from every later origin with the parameters fixed
    cross : how the cross MA enters: "additive" (default, as the C: a cross
        polynomial next to the univariate operators) or "residual" — Jenkins
        and Alavi's (1981) form (3.22), w - c = u(B) u*(B) alpha: the cross MA
        is a model for the univariate models' residuals, so row i of the MA is
        the univariate MA of series i TIMES row i of U*(B) = I - SUM U_k B^k.
        The two coincide when the univariate models have no MA factors. The
        cross AR is additive in both.
    start : where the cross terms start: "zero" (the default, as the C) or
        "preliminary" — Jenkins and Alavi's (1981) preliminary estimates: the
        cross MA from the cross covariances of the diagonal system's residuals
        (their prewhitened residuals), the cross AR from the multivariate
        Yule-Walker equations on the stationary series, the innovation
        covariance from the residuals'. Or, with ``echelon``, "echelon" --
        Lutkepohl and Poskitt's (1996) stage II least-squares estimates
        (drvarma.echelon) for the own and cross terms, A_0 included, and Q:
        the default with ``echelon`` (docs/STUDY-echelon-search.md: from
        zero, exact ML stops below the true parameters in 20 % of a
        5-dimensional system's fits; from here, in none). If that start is
        not admissible, zero is used and ``start_used`` says so.
    links : the cross dynamics only on these pairs, as the C's -links:
        ``"A<-B,C<-A"`` or a list of ``("A", "B")`` (B enters the equation of
        A: AR and MA, every lag up to p and q), with the series' names.
        None (default) links every pair.
    zeros : cross coefficients fixed at zero, by the names the fit prints:
        ``"AR3[CO2<-GAS], MA1[A<-B]"`` or a list of them — Tiao and Box's
        simplification by coefficient (1981, §4): the restricted model is
        refitted and compared by an LR test. Finer than ``links``, which
        removes a whole pair.
    rows : cross orders per equation, ``{"A": (p_A, q_A), ...}`` or
        ``"A(1,0), B(2,1)"``: equation A's cross AR runs up to lag p_A and
        its cross MA up to q_A, on every linked pair. A row not named takes
        (p, q); ``p`` and ``q`` then become the largest orders. The
        unequal row degrees of the echelon form, without its restriction
        table (docs/ECHELON_PLAN.md, stage 2): not identified on its own.
        ``links``, ``rows`` and ``zeros`` compose.
    echelon : Kronecker indices ``(n_1, ..., n_m)``, in the series' order:
        the reversed echelon form of Lutkepohl and Poskitt (1996, eq. 2.3)
        (docs/ECHELON_PLAN.md, stage 3). Row r carries its own AR/MA (the
        series' spec, of degree <= n_r), the cross AR at lags
        n_r - n_rc + 1 .. n_r, the cross MA at lags 1..n_r and, for c < r
        with n_c > n_r, the contemporaneous ``AR0[r<-c]``:
        A_0 = M_0 = I - (the AR0 terms), lower triangular. p and q are
        max n_r. ``links`` and ``zeros`` remove further coefficients (LP's
        stage IV); ``rows`` and ``cross="residual"`` do not apply.
        A model's seasonal and fixed-frequency factors are its row's
        FACTOR (stage 4): row r is F_r(B) [A(B) w]_r = G_r(B) [M(B) a]_r,
        F_r and G_r the AR and MA factors, estimated with the rest. Only the
        regular ARMA is the row's own part, of degree <= n_r.
    own : with ``echelon``, the row's own part: "spec" (default), the
        regular ARMA of the file, which must fit in n_r; or "free", a free
        AR(n_r)/MA(n_r) in its place (``LadderSeries.with_regular``) --
        Lutkepohl and Poskitt's row on a route-U model, whose univariate
        orders are not the index.
    own_zeros : with ``own="free"``, own lags held at zero, by series:
        ``{"A": {"ar": {2}, "ma": {1}}}`` -- the simplification of the free
        own part (``LadderSeries.with_regular``).
    """

    def __init__(self, series, p=0, q=0, diagcov=False, redet=False,
                 fixarma=False, method=1, estwin=None, hessian="fd", lik="elf",
                 links=None, start=None, cross="additive", zeros=None,
                 rows=None, echelon=None, own="spec", support=None,
                 own_zeros=None):
        _warn_if_no_engine()
        if series and not isinstance(series[0], LadderSeries):
            series = load(series)
        if own not in ("spec", "free"):
            raise ValueError("own must be 'spec' or 'free'")
        if own == "free":
            if echelon is None or len(echelon) != len(series):
                raise LadderError("own='free' is the echelon form's: give echelon=, "
                                  "one index per series")
            oz = dict(own_zeros or {})
            unknown = set(oz) - {s.name for s in series}
            if unknown:
                raise LadderError(f"own_zeros: no series {sorted(unknown)[0]}")
            series = [s.with_regular(int(k), oz.get(s.name))
                      for s, k in zip(series, echelon)]
        elif own_zeros:
            raise LadderError("own_zeros is the free own part's: give own='free' "
                              "(with own='spec' the zeros go in the files)")
        self.own = own
        self.series = list(series)
        check_alignment(self.series)
        self.p, self.q = int(p), int(q)
        self.diagcov = bool(diagcov)
        self.redet, self.fixarma = bool(redet), bool(fixarma)
        self.xitol = -1e-3 if method == 2 else 1e-3
        if hessian not in ("fd", "bfgs"):
            raise ValueError("hessian must be 'fd' or 'bfgs'")
        self.hessian = hessian
        if lik not in ("elf", "shea", "both"):
            raise ValueError("lik must be 'elf', 'shea' or 'both'")
        self.lik = lik
        self.lik_check = {"points": 0, "max": 0.0, "at_optimum": 0.0,
                          "one_only": 0}
        m = len(self.series)
        self.links = self._parse_links(links)
        if start is None:
            start = "echelon" if echelon is not None else "zero"
        if start not in ("zero", "preliminary", "echelon"):
            raise ValueError("start must be 'zero', 'preliminary' or 'echelon'")
        self.start = start
        self.start_used = None
        if cross not in ("additive", "residual"):
            raise ValueError("cross must be 'additive' or 'residual'")
        self.cross = cross
        # factor_test's alternative: the echelon form's free positions given
        # one by one, (kind, k, i, j) with k = -1 for AR0
        self._support = None if support is None else frozenset(support)
        self.echelon = self._parse_echelon(echelon, p, q, rows)
        if start == "echelon" and self.echelon is None:
            raise LadderError("start='echelon' is the echelon form's: give echelon=")
        if self.echelon is not None:
            self.p = self.q = max(self.echelon)
            rows = {s.name: (k, k) for s, k in zip(self.series, self.echelon)}
            if self._support is not None:
                rows = {s.name: tuple(max([k + 1 for kd, k, i, _j in self._support
                                           if kd == kind and i == r] + [0])
                                      for kind in ("AR", "MA"))
                        for r, s in enumerate(self.series)}
        self.rows = self._parse_rows(rows)
        if self.rows is not None:
            self.p = int(max(self.rows[:, 0]))
            self.q = int(max(self.rows[:, 1]))
        if self.links is not None and self.p == 0 and self.q == 0:
            raise LadderError("links needs cross dynamics: give p or q > 0")
        self.zeros = self._parse_zeros(zeros)
        # the cross part and Q, indexed by SERIES: they survive a change of
        # the active set, and the diagonal system seeds the full one as is.
        self._cAR = np.zeros((max(self.p, 1), m, m))
        self._cMA = np.zeros((max(self.q, 1), m, m))
        self._cA0 = np.zeros((m, m))      # the echelon form's contemporaneous terms
        self._lvar = np.zeros(m)
        self._qcov = np.zeros((m, m))
        self._set_structure(list(range(m)), self.p, self.q, self.diagcov)
        self.gate = None
        self.logL_diag = None
        self.result = None
        self.x_start = None      # the start of the requested fit (refit, sima)
        self.estwin = None
        if estwin:
            full = self.series[0].nobs_full
            if not 2 <= int(estwin) <= full:
                raise LadderError(f"estwin {estwin} out of range (2..{full})")
            self.estwin = int(estwin)
            self.set_origin(full - self.estwin)

    # -- structure -------------------------------------------------------- #
    def _parse_links(self, links):
        """None, or the m x m matrix of linked (equation i <- series j) pairs."""
        if links is None:
            return None
        names = [s.name for s in self.series]
        if isinstance(links, str):
            pairs = []
            for tok in links.split(","):
                if "<-" not in tok:
                    raise LadderError(f"links: '{tok.strip()}' is not of the form A<-B")
                a, b = tok.split("<-", 1)
                pairs.append((a.strip(), b.strip()))
        else:
            pairs = [tuple(x) for x in links]
        L = np.zeros((len(names), len(names)), bool)
        for a, b in pairs:
            if a not in names or b not in names or a == b:
                raise LadderError(f"links: '{a}<-{b}' does not name two different series")
            L[names.index(a), names.index(b)] = True
        return L

    _ROW = re.compile(r"\s*([^(,]+?)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*")

    def _parse_rows(self, rows):
        """None, or the m x 2 array of (p_i, q_i), the cross orders of each
        equation."""
        if rows is None:
            return None
        names = [s.name for s in self.series]
        if isinstance(rows, str):
            items = []
            for tok in re.findall(r"[^,()]+\([^)]*\)|[^,\s]+", rows):
                mm = self._ROW.fullmatch(tok)
                if not mm:
                    raise LadderError(f"rows: '{tok.strip()}' is not of the form A(p,q)")
                items.append((mm.group(1), (int(mm.group(2)), int(mm.group(3)))))
        else:
            items = list(dict(rows).items())
        R = np.tile([self.p, self.q], (len(names), 1))
        seen = set()
        for a, pq in items:
            if a not in names:
                raise LadderError(f"rows: '{a}' is not a series")
            if a in seen:
                raise LadderError(f"rows: '{a}' is given twice")
            seen.add(a)
            pr, qr = (int(x) for x in pq)
            if pr < 0 or qr < 0:
                raise LadderError(f"rows: the orders of '{a}' must be >= 0")
            R[names.index(a)] = (pr, qr)
        return R

    def _parse_echelon(self, echelon, p, q, rows):
        """None, or the tuple of Kronecker indices, checked against the
        specs and the other options."""
        if echelon is None:
            return None
        n = tuple(int(k) for k in echelon)
        m = len(self.series)
        if len(n) != m or min(n) < 0:
            raise LadderError(f"echelon: give {m} indices >= 0, one per series")
        if rows is not None:
            raise LadderError("echelon: the indices give the rows; do not give rows")
        if self.cross != "additive":
            raise LadderError("echelon: the residual cross MA (Jenkins-Alavi 3.22) is "
                              "not an echelon form; use cross='additive'")
        if (int(p), int(q)) not in ((0, 0), (max(n), max(n))):
            raise LadderError(f"echelon: p and q are max(n) = {max(n)}; give 0, 0")
        for s, k in zip(self.series, n):
            ar, ma = s.regular_orders()
            if self._support is None and (ar > k or ma > k):
                raise LadderError(
                    f"echelon: {s.name}'s regular ARMA has degrees ({ar}, {ma}) and its "
                    f"index is {k}. The regular part is the row's own, of degree <= "
                    f"n_r: give {s.name} an index of at least {max(ar, ma)}, or a "
                    f"model whose regular part fits in {k}. Seasonal and "
                    f"fixed-frequency factors are the row's factor, not its degree")
        return n

    def _index_table(self):
        n = self.echelon
        return [[min(n[r] + 1, n[c]) if c < r else min(n[r], n[c])
                 for c in range(len(n))] for r in range(len(n))]

    def _allowed(self, kind, k, i, j):
        """Is the cross slot (kind, lag k + 1, i <- j) in the model's
        structure (before zeros)? k = -1 is the contemporaneous AR0."""
        if self.echelon is None:
            return 0 <= k < self._row_order(kind, i)
        if self._support is not None:
            return (kind, k, i, j) in self._support
        n, nt = self.echelon, self._index_table()
        if kind == "MA":
            return 0 <= k < n[i]
        if k < 0:
            return j < i and nt[i][j] == n[i] + 1
        return n[i] - nt[i][j] + 1 <= k + 1 <= n[i]

    def _get(self, kind, k, i, j):
        if kind == "AR" and k < 0:
            return self._cA0[i, j]
        return (self._cAR if kind == "AR" else self._cMA)[k, i, j]

    def _put(self, kind, k, i, j, v):
        if kind == "AR" and k < 0:
            self._cA0[i, j] = v
        else:
            (self._cAR if kind == "AR" else self._cMA)[k, i, j] = v

    def _row_order(self, kind, i):
        """The cross order of equation i (AR or MA), before any change of
        the active structure."""
        if self.rows is None:
            return self.p if kind == "AR" else self.q
        return int(self.rows[i, 0 if kind == "AR" else 1])

    _ZERO = re.compile(r"\s*(AR|MA)(\d+)\[([^<\]]+)<-([^\]]+)\]\s*")

    def _parse_zeros(self, zeros):
        """frozenset of (kind, k, i, j), k 0-based, from the printed names."""
        if not zeros:
            return frozenset()
        toks = zeros.split(",") if isinstance(zeros, str) else list(zeros)
        names = [s.name for s in self.series]
        out = set()
        for tok in toks:
            if not str(tok).strip():
                continue
            mm = self._ZERO.fullmatch(str(tok))
            if not mm:
                raise LadderError(f"zeros: '{str(tok).strip()}' is not of the form AR3[A<-B]")
            kind, k, a, b = mm.group(1), int(mm.group(2)), mm.group(3).strip(), mm.group(4).strip()
            if a not in names or b not in names or a == b:
                raise LadderError(f"zeros: '{str(tok).strip()}' does not name two different series")
            i, j = names.index(a), names.index(b)
            if self.echelon is not None:
                if not self._allowed(kind, k - 1, i, j):
                    raise LadderError(f"zeros: '{str(tok).strip()}' is not a free coefficient "
                                      f"of the echelon form {self.echelon}")
            else:
                top = self._row_order(kind, i)
                if not 1 <= k <= top:
                    where = "" if self.rows is None else f" of {a}"
                    raise LadderError(f"zeros: '{str(tok).strip()}': the cross {kind} order{where} is {top}")
            if not self._linked(i, j):
                raise LadderError(f"zeros: '{str(tok).strip()}' is not a linked pair")
            out.add((kind, k - 1, i, j))
        return frozenset(out)

    def _slots(self):
        """The free cross coefficients, in the C's order (AR lag by lag, then
        MA), without those fixed at zero or outside the structure (rows, the
        echelon table). The echelon form's AR0 come first."""
        z = getattr(self, "zeros", frozenset())
        a0 = []
        if self.echelon is not None and (self._cp or self._cq):
            a0 = [("AR", -1, i, j) for i, j in self._pairs()
                  if self._allowed("AR", -1, i, j) and ("AR", -1, i, j) not in z]
        ar = [("AR", k, i, j) for k in range(self._cp) for i, j in self._pairs()
              if self._allowed("AR", k, i, j) and ("AR", k, i, j) not in z]
        ma = [("MA", k, i, j) for k in range(self._cq) for i, j in self._pairs()
              if self._allowed("MA", k, i, j) and ("MA", k, i, j) not in z]
        return a0 + ar + ma

    def _linked(self, i, j):
        return i != j and (self.links is None or bool(self.links[i, j]))

    def _pairs(self):
        """The linked (i, j) among the active series, in the C's order."""
        act = self._act
        return [(i, j) for i in act for j in act if self._linked(i, j)]

    def _set_structure(self, act, cp, cq, cdiag):
        self._act, self._cp, self._cq, self._cdiag = list(act), cp, cq, cdiag

    def _masks(self):
        return [self.series[i].free_mask(self.redet, self.fixarma) for i in self._act]

    def n_arma(self):
        """The ARMA coefficients behind the residuals, for Hosking's Q
        (BUG-0015): each active series' own (estimated by its .pre, even when
        the ladder holds them fixed) plus the free cross ones."""
        return (sum(self.series[i].n_arma() for i in self._act)
                + len(self._slots()))

    def npar(self):
        a = len(self._act)
        n = sum(int(mk.sum()) for mk in self._masks())
        n += len(self._slots())
        n += a - 1
        if not self._cdiag:
            n += a * (a - 1) // 2
        return n

    def pack(self):
        v = []
        for i, mk in zip(self._act, self._masks()):
            v.extend(self.series[i].x[mk])
        act = self._act
        for kind, k, i, j in self._slots():
            v.append(self._get(kind, k, i, j))
        v.extend(self._lvar[i] for i in act[1:])
        if not self._cdiag:
            v.extend(self._qcov[act[a], act[b]] for a in range(1, len(act)) for b in range(a))
        return np.asarray(v, float)

    def unpack(self, v):
        v = np.asarray(v, float)
        idx = 0
        for i, mk in zip(self._act, self._masks()):
            k = int(mk.sum())
            self.series[i].x[mk] = v[idx:idx + k]
            idx += k
        act = self._act
        for kind, k, i, j in self._slots():
            self._put(kind, k, i, j, v[idx]); idx += 1
        for i in act[1:]:
            self._lvar[i] = v[idx]; idx += 1
        if not self._cdiag:
            for a in range(1, len(act)):
                for b in range(a):
                    self._qcov[act[a], act[b]] = v[idx]; idx += 1

    def names(self):
        out = []
        for i, mk in zip(self._act, self._masks()):
            nm = self.series[i].param_names()
            out.extend(n for n, f in zip(nm, mk) if f)
        act = self._act
        S = [s.name for s in self.series]
        out.extend(f"{kind}{k + 1}[{S[i]}<-{S[j]}]" for kind, k, i, j in self._slots())
        out.extend(f"log(Q[{S[i]}]/Q[{S[act[0]]}])" for i in act[1:])
        if not self._cdiag:
            out.extend(f"Q[{S[act[a]]},{S[act[b]]}]" for a in range(1, len(act)) for b in range(a))
        return out

    # -- the cast ---------------------------------------------------------- #
    def cast(self, v):
        """vector -> (mu, phi, theta, qq, w, ifault) for ``elf``."""
        self.unpack(v)
        act, m = self._act, len(self._act)
        polys, ws, facs = [], [], []
        ech = self.echelon is not None and bool(self._cp or self._cq)
        for i in act:
            s = self.series[i]
            if ech:     # the regular part on the diagonal, the rest a row factor
                phi, theta, fphi, ftheta, mu, ifault = s.split_polynomials()
                facs.append((fphi, ftheta))
            else:
                phi, theta, mu, ifault = s.polynomials()
            if ifault:
                return None, None, None, None, None, 6
            w = s.stationary()
            if w is None:
                return None, None, None, None, None, 6
            polys.append((phi, theta, mu))
            ws.append(w)
        # The common window of ALL the series, not only of the active ones: a
        # series fitted alone for the gate has to see the same sample as in
        # the joint fit, or the identity fails (the C trims them all).
        others = [self.series[i].stationary() for i in range(len(self.series))
                  if i not in act]
        if any(o is None for o in others):
            return None, None, None, None, None, 6
        n = min(len(w) for w in ws + others)
        P = max([self._cp] + [len(ph) for ph, _t, _m in polys] + [1])
        Q = max([self._cq] + [len(th) for _p, th, _m in polys])
        if self.cross == "residual" and self._cq:
            Q = max(Q, self._cq + max(len(th) for _p, th, _m in polys))
        PHI = np.zeros((P, m, m))
        THETA = np.zeros((Q, m, m))
        for a, (ph, th, _mu) in enumerate(polys):
            PHI[:len(ph), a, a] = ph
            THETA[:len(th), a, a] = th
        for k in range(self._cp):
            for a, i in enumerate(act):
                for b, j in enumerate(act):
                    if self._linked(i, j):
                        PHI[k, a, b] = self._cAR[k, i, j]
        for k in range(self._cq):
            for a, i in enumerate(act):
                th_i = polys[a][1]
                for b, j in enumerate(act):
                    if self._linked(i, j):
                        U = self._cMA[k, i, j]
                        THETA[k, a, b] += U
                        if self.cross == "residual":
                            # theta_ii(B) (-U B^(k+1)): with theta_ii(B) =
                            # 1 - SUM t_l B^l, the term t_l U lands at
                            # lag l + k + 1 with the sign of +B, i.e. -t_l U
                            # in the I - SUM Theta_m B^m convention (3.22)
                            for l, t in enumerate(th_i, 1):
                                THETA[k + l, a, b] -= t * U
        if ech:
            # A_0 w_t = SUM PHI~_k w_t-k + A_0 a_t - SUM THETA~_k a_t-k, with
            # A_0 = I - C_0 (C_0 the AR0 terms, strictly lower): elf wants the
            # standard form, Phi_k = A_0^-1 PHI~_k and Theta_k = A_0^-1 THETA~_k
            C0 = self._cA0[np.ix_(act, act)]
            A0 = np.eye(m) - C0
            if any(f.size or g.size for f, g in facs):
                # stage 4: row a times its factor F_a(B) (AR) or G_a(B) (MA),
                # the series' seasonal and fixed-frequency factors
                from .echelon import times_row_factors
                # (LP's C(B) = A_0 - SUM PHI~_k B^k is -[-A_0, PHI~...])
                PHI = times_row_factors(np.concatenate([-A0[None], PHI]),
                                        [f for f, _g in facs])[1:]
                THETA = times_row_factors(np.concatenate([-A0[None], THETA]),
                                          [g for _f, g in facs])[1:]
            self._rows = (C0, PHI, THETA)    # LP's rows, for factor_test
            if np.any(C0):
                Ainv = np.linalg.inv(A0)                 # unit triangular
                PHI = np.einsum("ab,kbc->kac", Ainv, PHI)
                THETA = np.einsum("ab,kbc->kac", Ainv, THETA)
        QQ = np.zeros((m, m))
        QQ[0, 0] = 1.0
        for a in range(1, m):
            lv = self._lvar[act[a]]
            if not lv < _LVAR_MAX:      # exp overflows: the C gets inf and
                return None, None, None, None, None, 6   # rejects it (BUG-0018)
            QQ[a, a] = math.exp(lv)
        if not self._cdiag:
            for a in range(1, m):
                for b in range(a):
                    QQ[a, b] = QQ[b, a] = self._qcov[act[a], act[b]]
        MU = np.array([mu for _p, _t, mu in polys])
        W = np.column_stack([w[len(w) - n:] for w in ws])
        return MU, PHI, THETA, QQ, W, 0

    def _elf(self, mu, phi, theta, qq, w, atf=False):
        """The likelihood ``lik`` asks for, with elf's (logL, f1, f2, a, ifault).

        Shea gives f1, f2 (the objective and the reported likelihood); the
        residuals `a` are always elf's (Shea's are innovations, and the
        forecasts need exact residuals: drtran BUG-55)."""
        n, m = w.shape
        p, q = phi.shape[0], theta.shape[0]
        if self.lik == "elf":
            return elf_c(m, n, p, q, mu, phi, theta, qq, w, 1.0, self.xitol, atf)
        if self.lik == "shea":
            a = np.zeros((n + 1, m + 1))
            if atf:
                _l, _f1, _f2, a, ifa = elf_c(m, n, p, q, mu, phi, theta, qq, w,
                                             1.0, self.xitol, True)
                if ifa:
                    return _l, _f1, _f2, a, ifa
            lg, f1, f2, ifa = marma_c(m, n, p, q, mu, phi, theta, qq, w)
            return lg, f1, f2, a, ifa
        # both: elf is the objective, Shea is checked at the same point
        out = elf_c(m, n, p, q, mu, phi, theta, qq, w, 1.0, self.xitol, atf)
        _lg, g1, g2, gfa = marma_c(m, n, p, q, mu, phi, theta, qq, w)
        c = self.lik_check
        if bool(out[4]) != bool(gfa):
            c["one_only"] += 1
        elif not out[4] and out[1] > 0 and out[2] > 0 and g1 > 0 and g2 > 0:
            d = abs(_concentrated(m, n, out[1], out[2]) - _concentrated(m, n, g1, g2))
            c["points"] += 1
            c["max"] = max(c["max"], d)
            if atf:
                c["at_optimum"] = d
        return out

        # -- one fit ----------------------------------------------------------- #
    def _fit(self, optimize=True, maxits=500, grtol=1e-7, sptol=1e-7):
        x0 = self.pack()
        npar = x0.size
        mu, phi, theta, qq, w, ifault = self.cast(x0)
        names = self.names()
        if ifault:
            return Fit(names, x0, np.zeros(npar), float("nan"), float("nan"), ifault)
        _ll, f10, f20, _a, if0 = self._elf(mu, phi, theta, qq, w)
        if if0:
            return Fit(names, x0, np.zeros(npar), float("nan"), float("nan"), int(if0))
        m = w.shape[1]

        def objective(vec):
            mu, phi, theta, qq, w, ifa = self.cast(vec)
            if ifa:
                return 1.0
            _ll, f1, f2, _a, ifa = self._elf(mu, phi, theta, qq, w)
            if ifa or not np.isfinite(f1) or f1 <= 0.0 or f2 <= 0.0:
                return 1.0
            f = (f1 / f10) ** m * (f2 / f20)
            return f if np.isfinite(f) else 1.0

        nit, termcode, se, se_method, fk, cov = 0, 0, np.zeros(npar), None, None, None
        xhat = x0
        if optimize and npar:
            xk = np.zeros(npar + 1); xk[1:] = x0
            fk, bfac, nit, termcode = _qnewt.raxopt(
                lambda z: objective(z[1:npar + 1]), npar, xk, maxits, grtol, sptol)
            xhat = xk[1:npar + 1].copy()
            # the invertible form, where the model actually is (BUG-0007)
            self.unpack(xhat)
            if any(self.series[i].canonicalize() for i in self._act):
                xhat = self.pack()
                fk = objective(xhat)
            se, se_method, cov = self._std_errors(objective, xhat, fk, w.shape[0], bfac, nit)
        mu, phi, theta, qq, w, ifault = self.cast(xhat)
        nb, nr = _ma_boundary(theta)
        _ll, f1, f2, a, ifa = self._elf(mu, phi, theta, qq, w, atf=True)
        n, m = w.shape
        logL = (-0.5 * m * n * (_LOG2PI - np.log(m) - np.log(n) + 1.0)
                - 0.5 * n * (m * np.log(f1) + np.log(f2)))
        return Fit(names, xhat, se, float(logL), float(f1 / (n * m)), int(ifa),
                   int(nit), int(termcode), mu, phi, theta, qq, w,
                   np.asarray(a)[1:, 1:].copy(),      # elf_c: 1-based out
                   se_method, None if fk is None else float(fk), nb, nr, cov)

    def _std_errors(self, objective, xhat, fk, n, bfac, nit=1):
        """Mauricio's fdhess at the optimum (drvarma.stderr), or the BFGS
        Hessian: ``hessian`` chooses, and a Hessian that is not positive
        definite falls back to BFGS SAYING so. Q is normalised (Q11 = 1), so
        there is no flat direction to hold."""
        from .estimate_py import _covariance
        from .stderr import fd_covariance
        if self.hessian == "fd":
            cov, std, info = fd_covariance(objective, xhat, n)
            if cov is not None:
                return std, "fdhess", cov
            why = ("the optimum is on the boundary of the admissible region"
                   if info.get("boundary") else
                   "the Hessian is singular: some coefficients are not identified"
                   if info.get("singular") else "the Hessian is not positive definite")
            if not nit:
                # raxopt starts b at the identity: no iteration, no BFGS Hessian.
                return (np.full(xhat.size, np.nan),
                        f"none (fdhess: {why}; the search did not move, so it built no BFGS Hessian)",
                        None)
            c, std = _covariance(bfac, fk, n, xhat.size)
            return std, f"bfgs (fdhess: {why})", c
        c, std = _covariance(bfac, fk, n, xhat.size)
        return std, "bfgs", c

    # -- the gate and the estimation -------------------------------------- #
    def run_gate(self):
        """Each series alone, then the diagonal system EVALUATED at those optima.

        Returns a dict and raises ``GateError`` if the identity fails.
        """
        # The gate is an IDENTITY: evaluated with the UNTRUNCATED likelihood
        # whatever `method` says (as the C, 2026-09-28). elf's xi truncation is
        # not the same for one series as for m, and m6's pairs failed by up to
        # 0.0018 with the cast blamed. The estimation keeps `method`.
        xitol_asked, self.xitol = self.xitol, -1e-3
        try:
            return self._run_gate()
        finally:
            self.xitol = xitol_asked

    def _run_gate(self):
        m = len(self.series)
        n_all = [len(s.stationary()) for s in self.series]
        n_common = min(n_all)
        rows = []
        for i, s in enumerate(self.series):
            before = s.x.copy()
            self._set_structure([i], 0, 0, True)
            fit = self._fit()
            if fit.ifault:
                raise LadderError(f"univariate fit of {s.name} failed (ifault {fit.ifault})")
            mk = s.free_mask(self.redet, self.fixarma)
            move = float(np.max(np.abs(s.x[mk] - before[mk]))) if mk.any() else 0.0
            rows.append({"series": s.name, "logL": fit.logL, "sigma2": fit.sigma2,
                         "move": move, "trimmed": n_all[i] - n_common})
        s2 = np.array([r["sigma2"] for r in rows])
        self._lvar[:] = np.log(s2 / s2[0])
        self._set_structure(list(range(m)), 0, 0, True)
        joint = self._fit(optimize=False)
        total = float(sum(r["logL"] for r in rows))
        diff = joint.logL - total
        passed = (joint.ifault == 0 and abs(diff) <= GATE_TOL * (1.0 + abs(total)))
        self.gate = {"rows": rows, "sum": total, "joint": joint.logL,
                     "difference": diff, "passed": bool(passed)}
        if not passed:
            raise GateError(f"the diagonal gate failed (difference {diff:.3g}): the "
                            f"joint cast does not reproduce the univariate models")
        return self.gate

    def fit(self):
        """The gate, the diagonal system (the base of the LR test and the
        seed), then the requested model."""
        m = len(self.series)
        self.run_gate()
        self._set_structure(list(range(m)), 0, 0, True)
        diag = self._fit()
        if diag.ifault:
            raise LadderError(f"the diagonal system failed (ifault {diag.ifault})")
        self.logL_diag = diag.logL
        if self.p == 0 and self.q == 0 and self.diagcov:
            self.result = diag
        else:
            self._cAR[:] = 0.0; self._cMA[:] = 0.0; self._cA0[:] = 0.0; self._qcov[:] = 0.0
            self._set_structure(list(range(m)), self.p, self.q, self.diagcov)
            self.start_used = "zero"
            if self.start == "preliminary":
                self._preliminary(diag)
            elif self.start == "echelon":
                self._echelon_start()
            self.x_start = self.pack()    # where the requested fit starts
            self.result = self._fit()
        return self.result

    def _preliminary(self, diag):
        """Jenkins and Alavi's preliminary estimates for the cross terms and Q
        (see ``start``), from the fitted diagonal system."""
        from .identification_mv import residual_ma_preliminary, yule_walker
        m = len(self.series)
        res = np.asarray(diag.residuals, float)
        _mu, _phi, _th, _qq, W, _ifa = self.cast(self.pack())   # w_t: no cross terms in it
        seeds = {}
        if self.q:
            seeds["ma"] = residual_ma_preliminary(res, self._cq)
        if self.p:
            seeds["ar"] = yule_walker(W, self._cp)
        S = np.cov(res.T, bias=True)
        x0 = self.pack()
        for kind, k, i, j in self._slots():
            if k >= 0:                    # AR0 has no preliminary estimate: 0
                self._put(kind, k, i, j, seeds["ar" if kind == "AR" else "ma"][k, i, j])
        if not self._cdiag:
            for a in range(1, m):
                for b in range(a):
                    self._qcov[a, b] = S[a, b] / S[0, 0]
        self._take_start(x0, "preliminary")

    def _admissible(self):
        """Does elf accept the current state as a point of the likelihood?"""
        mu, phi, theta, qq, w, ifa = self.cast(self.pack())
        if ifa:
            return False
        _l, f1, f2, _a, ifa2 = self._elf(mu, phi, theta, qq, w)
        return not ifa2 and bool(np.isfinite(f1)) and f1 > 0 and f2 > 0

    def _take_start(self, x0, name):
        """Keep the start just set, or go back to ``x0`` with the cross terms
        and Q's covariances at zero if elf does not accept it."""
        if self._admissible():
            self.start_used = name
        else:
            self.unpack(x0)
            self._cAR[:] = 0.0; self._cMA[:] = 0.0; self._cA0[:] = 0.0; self._qcov[:] = 0.0
            self.start_used = f"zero (the {name} start was not admissible)"

    def _echelon_start(self):
        """Lutkepohl and Poskitt's stage II estimates (see ``start``): a long
        VAR on the stationary series, then least squares on the echelon
        form's free positions -- those the Ladder holds free: links, zeros
        and an own ARMA of degree below n_r stay at zero. The own AR/MA are
        set where the spec writes them as one regular factor (route V's);
        the mean stays the diagonal fit's. A row with a factor (stage 4: the
        series' seasonal and fixed-frequency factors, at the file's values)
        is fitted on the data filtered by it (``echelon.row_filter``). Least squares can leave an MA
        root outside the unit circle: the roots are pulled inside
        (``echelon.pull_inside``) and ``start_used`` says so."""
        from . import echelon as E
        m, n = len(self.series), self.echelon
        _mu, _phi, _th, _qq, W, _ifa = self.cast(self.pack())   # no cross terms in it
        h, e, _aic = E.long_var(W)
        ex = set()
        filters = []
        for r, s in enumerate(self.series):
            ar, ma = s.regular_orders()
            _p, _t, fphi, ftheta, _mu, _if = s.split_polynomials()
            filters.append((fphi, ftheta))
            ex |= {("A", k, r, r) for k in range(ar + 1, n[r] + 1)}
            ex |= {("M", k, r, r) for k in range(ma + 1, n[r] + 1)}
        if self.own == "free":
            # own lags held at zero (own_zeros) stay out of the least squares
            nm = set(self.names())
            for r, s in enumerate(self.series):
                for k in range(1, n[r] + 1):
                    if f"phi_{s.name}[B^{k}]" not in nm:
                        ex.add(("A", k, r, r))
                    if f"theta_{s.name}[B^{k}]" not in nm:
                        ex.add(("M", k, r, r))
        free = set(self._slots())
        for r in range(m):
            for c in range(m):
                if r == c:
                    continue
                for k in range(0, n[r] + 1):
                    if ("AR", k - 1, r, c) not in free:
                        ex.add(("A", k, r, c))
                    if k and ("MA", k - 1, r, c) not in free:
                        ex.add(("M", k, r, c))
        fit = E.stage2(W, e, h, n, exclude=ex, filters=filters)
        self.echelon_stage2 = fit
        A, M, lam_a, lam_m = E.pull_inside(fit.A, fit.M)
        x0 = self.pack()
        val = {}
        for r, s in enumerate(self.series):
            for k in range(1, n[r] + 1):
                val[f"phi_{s.name}[B^{k}]"] = -A[k, r, r]
                val[f"theta_{s.name}[B^{k}]"] = -M[k, r, r]
        names = self.names()
        v = self.pack()
        for a, nm in enumerate(names):
            if nm in val:
                v[a] = val[nm]
        self.unpack(v)
        for kind, k, i, j in self._slots():
            self._put(kind, k, i, j, -(A if kind == "AR" else M)[k + 1, i, j])
        S = fit.sigma
        act = self._act
        for i in act[1:]:
            self._lvar[i] = math.log(S[i, i] / S[act[0], act[0]])
        if not self._cdiag:
            for a in range(1, len(act)):
                for b in range(a):
                    self._qcov[act[a], act[b]] = S[act[a], act[b]] / S[act[0], act[0]]
        self._take_start(x0, "echelon")
        if self.start_used == "echelon" and (lam_a < 1.0 or lam_m < 1.0):
            pulled = [f"{w} x {lam:.3f}^k" for w, lam in (("AR", lam_a), ("MA", lam_m))
                      if lam < 1.0]
            self.start_used = f"echelon ({', '.join(pulled)}: inside the unit circle)"

    def irf_fevd_bands(self, horizon, ndraws=800, alpha=0.05, seed=0):
        """Monte-Carlo bands for the orthogonalised IRF and the FEVD of the fit.

        As `drvarma.irf.irf_fevd_bands` for the `.inp` path: draw the parameters
        from N(x_hat, cov), rebuild the model through the ladder's cast (the
        univariate factors and the cross terms together), recompute, and take
        percentiles. Draws that leave the stationary / invertible region, or
        give a Sigma that is not positive definite, are DISCARDED, not clipped,
        and counted: a large rejected share says the fit sits near a boundary.
        sigma2 is held at its estimate (it is concentrated out). The responses
        are those of the stationary series, as the point IRF.

        ``fevd_lo``/``fevd_hi`` are the band at the last horizon (m x m), as
        before; ``fevd_lo_h``/``fevd_hi_h`` are the band at every horizon
        (H x m x m, like `fevd`), for a figure that draws them all.
        """
        from .irf import oirf, fevd, _stable, _bands_from_draws
        r = self.result
        if r is None or r.cov is None or not np.all(np.isfinite(r.cov)):
            raise ValueError("the fit carries no usable covariance of the "
                             "estimates (se_method: %s); no bands" % (r.se_method if r else None))
        cov = 0.5 * (r.cov + r.cov.T)
        ev, V = np.linalg.eigh(cov)
        Lc = V @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))
        rng = np.random.default_rng(seed)
        od, fd, fh, rejected = [], [], [], 0
        try:
            for _ in range(int(ndraws)):
                vec = r.x + Lc @ rng.standard_normal(r.x.size)
                try:
                    _mu, phi, theta, qq, _w, ifa = self.cast(vec)
                    if ifa:
                        rejected += 1; continue
                    sig = r.sigma2 * np.asarray(qq, float)
                    np.linalg.cholesky(sig)
                    if phi.shape[0] and not _stable(phi):
                        rejected += 1; continue
                    if theta.shape[0] and not _stable(-np.asarray(theta, float)):
                        rejected += 1; continue
                    o = oirf(phi, theta, sig, horizon)
                    f = fevd(phi, theta, sig, horizon)
                except Exception:                     # noqa: BLE001
                    rejected += 1; continue
                if not (np.all(np.isfinite(o)) and np.all(np.isfinite(f))):
                    rejected += 1; continue
                od.append(o); fd.append(f[-1]); fh.append(f)
        finally:
            self.cast(r.x)                            # the state back at x_hat
        if len(od) < 20:
            raise ValueError(f"only {len(od)} of {ndraws} draws were admissible; "
                             "the fit is too close to the stationarity/invertibility "
                             "boundary for a meaningful band")
        o_lo, o_hi = _bands_from_draws(np.stack(od), alpha)
        f_lo, f_hi = _bands_from_draws(np.stack(fd), alpha)
        fh_lo, fh_hi = _bands_from_draws(np.stack(fh), alpha)
        return {"oirf_lo": o_lo, "oirf_hi": o_hi, "fevd_lo": f_lo, "fevd_hi": f_hi,
                "fevd_lo_h": fh_lo, "fevd_hi_h": fh_hi,
                "ndraws_used": len(od), "ndraws_rejected": rejected, "alpha": alpha}

    def refit(self, x=None):
        """Optimise the requested model again, from ``x`` (a packed vector,
        ``result.x`` shaped) or from the current state; no gate, no diagonal.

        What a study of an ill-defined estimation needs: restart from where a
        fit stopped -- e.g. with the MA roots pulled inside the invertibility
        wall -- and see whether the likelihood keeps rising (drvarma C BUGS.md,
        bench case c2: 66.21 -> 71.22 along the wall). Needs a previous fit().
        """
        if self.result is None:
            raise LadderError("refit needs a previous fit()")
        if x is not None:
            x = np.asarray(x, float)
            if x.size != self.result.x.size:
                raise ValueError(f"x has {x.size} parameters, the model {self.result.x.size}")
            self.unpack(x)
        self.result = self._fit()
        return self.result

    def lr_test(self):
        """LR of the cross dynamics (and covariances) against the diagonal."""
        from scipy.stats import chi2
        m = len(self.series)
        # the free cross coefficients of the requested model: rows and zeros
        # count (BUG-0017), not (p + q) for every linked pair
        saved = (self._act, self._cp, self._cq, self._cdiag)
        self._set_structure(list(range(m)), self.p, self.q, self.diagcov)
        nx = len(self._slots())          # rows, the echelon table and zeros
        self._set_structure(*saved)
        df = nx + (0 if self.diagcov else m * (m - 1) // 2)
        lr = 2.0 * (self.result.logL - self.logL_diag)
        return lr, df, (float(chi2.sf(max(lr, 0.0), df)) if df else float("nan"))

    def factor_test(self):
        """The check of the row factor (docs/ECHELON_SEEDED_DESIGN.md §5.4,
        §7 stage 4): the LR of the fitted form against the same rows with
        the factor's restriction dropped -- every position the product
        F_r(B) [A(B)]_r (and G_r(B) [M(B)]_r) can reach, own and cross, free.
        The alternative starts at the expansion of this optimum, where its
        likelihood is the same, so the LR is >= 0 by construction.

        Returns a dict: lr, df, p, logL (this fit's), logL_free,
        logL_start (the alternative at the start: equal to logL), and the
        alternative Ladder (``free``)."""
        from scipy.stats import chi2
        if self.result is None:
            raise LadderError("factor_test needs a previous fit()")
        if self.echelon is None or not any(s.has_row_factor() for s in self.series):
            raise LadderError("factor_test: the echelon form has no row factor "
                              "(seasonal or fixed-frequency factors) to test")
        if self.p == 0 and self.q == 0:
            raise LadderError("factor_test: indices all 0, no cross terms: each factor "
                              "is its univariate model's, to check there (art)")
        m = len(self.series)
        self._set_structure(list(range(m)), self.p, self.q, self.diagcov)
        self.unpack(self.result.x)
        self.cast(self.result.x)
        C0, PHI, THETA = self._rows
        sup = [s.factor_support() for s in self.series]
        support = set()
        for kind, k, i, j in self._slots():
            fa, fm = sup[i]
            base = k + 1                               # AR0: lag 0
            for jj in (fa if kind == "AR" else fm):
                lag = base + jj
                support.add((kind, -1, i, j) if lag == 0 else (kind, lag - 1, i, j))
            if kind == "AR" and k < 0:                 # M_0 = A_0, times G_i
                support |= {("MA", jj - 1, i, j) for jj in fm if jj}
        series = []
        for s, (fa, fm) in zip(self.series, sup):
            if not s.has_row_factor():
                series.append(s.copy())
                continue
            ar, ma = s.regular_orders()
            series.append(s.without_row_factor(
                {l + jj for l in range(ar + 1) for jj in fa} - {0},
                {l + jj for l in range(ma + 1) for jj in fm} - {0}))
        free = Ladder(series, echelon=self.echelon, support=support,
                      diagcov=self.diagcov, redet=self.redet, fixarma=self.fixarma,
                      method=2 if self.xitol < 0 else 1, hessian=self.hessian,
                      lik=self.lik, start="zero")
        free._set_structure(list(range(m)), free.p, free.q, free.diagcov)
        for kind, k, i, j in free._slots():
            v = C0[i, j] if k < 0 else (PHI if kind == "AR" else THETA)[k, i, j]
            free._put(kind, k, i, j, v)
        free._lvar[:] = self._lvar
        free._qcov[:] = self._qcov
        free.x_start = free.pack()
        start = free._fit(optimize=False)
        free.result = free._fit()
        free.logL_diag = self.logL_diag
        df = free.npar() - self.npar()
        lr = 2.0 * (free.result.logL - self.result.logL)
        return {"lr": lr, "df": df,
                "p": float(chi2.sf(max(lr, 0.0), df)) if df > 0 else float("nan"),
                "logL": self.result.logL, "logL_free": free.result.logL,
                "logL_start": start.logL, "free": free}

    # -- forecasting ------------------------------------------------------- #
    def set_origin(self, k):
        """Every series ends k periods before its full end: the same date for
        all of them, since they all end on the same date."""
        for s in self.series:
            if s.nobs_full - k < 2:
                raise LadderError(f"{s.name} has no data {k} periods before its end")
            s.nobs = s.nobs_full - k

    def _level(self, s, wf):
        """From a forecast of w (mean included) back to the level, with the
        .pre model in reverse: deterministic terms, the operator, Box-Cox.
        The pieces are fue's own (``fue.forecast``)."""
        from fue.forecast import (_boxcox, _build_xi, _nonsop_coefs,
                                  _reconstruct_params)
        m = s.model
        n, L = s.nobs, len(wf)
        itv_omega, itv_delta, *_rest = _reconstruct_params(m, s.x)
        xi = _build_xi(m, n, s.freq, L, itv_omega, itv_delta)       # 1-based
        r = _nonsop_coefs(m.d, m.D, s.freq, ifadf=m.ifadf or [])
        b = np.zeros(n + L + 1)
        for t in range(1, n + 1):
            b[t] = _boxcox(float(s.ts.data[t - 1]), m.boxlam, m.refactor) - xi[t]
        for l in range(1, L + 1):
            tt = n + l
            acc = float(wf[l - 1])
            for i, ri in enumerate(r):
                acc += ri * b[tt - 1 - i]
            b[tt] = acc
        ystar = b[1:] + xi[1:]
        return ystar, r

    @staticmethod
    def _inv(s, z):
        from fue.forecast import _inv_boxcox
        return _inv_boxcox(float(z), s.model.boxlam, s.model.refactor)

    def forecast(self, L):
        """L-step forecasts from the current end of every series, with the
        fitted parameters held fixed. Returns one dict per series."""
        from .forecast import forecast_w, lead_sd as _lead_sd, series_psi
        from .irf import psi_weights
        if self.result is None:
            raise LadderError("fit() first")
        r = self.result
        mu, phi, theta, qq, w, ifault = self.cast(r.x)
        if ifault:
            raise LadderError(f"cannot build the model at this origin (ifault {ifault})")
        # the EXACT residuals (atf = True): the MA part forecasts with them
        _ll, _f1, _f2, a, ifa = self._elf(mu, phi, theta, qq, w, atf=True)
        if ifa:
            raise LadderError(f"cannot compute the residuals (ifault {ifa})")
        a = np.asarray(a)[1:, 1:]
        sigma = r.sigma2 * qq
        f = forecast_w(phi, theta, mu, w, a, L)
        psi = psi_weights(phi, theta, L)
        out = []
        for c, i in enumerate(self._act):
            s = self.series[i]
            ystar, rn = self._level(s, f[:, c])
            P = series_psi(psi, c, rn, L)

            def lead_sd(l, d):
                return _lead_sd(P, sigma, l, d)

            n = s.nobs
            sd = np.array([lead_sd(l, 0) for l in range(1, L + 1)])
            lvl = np.array([self._inv(s, ystar[n + l - 1]) for l in range(1, L + 1)])
            lo = np.array([self._inv(s, ystar[n + l - 1] - 1.96 * sd[l - 1]) for l in range(1, L + 1)])
            hi = np.array([self._inv(s, ystar[n + l - 1] + 1.96 * sd[l - 1]) for l in range(1, L + 1)])
            out.append({"series": s.name, "origin": s.date_of(n),
                        "dates": [s.date_of(n + l) for l in range(1, L + 1)],
                        "level": lvl, "low95": lo, "high95": hi, "sd": sd,
                        "sd_period": np.array([lead_sd(l, 1) for l in range(1, L + 1)]),
                        "sd_annual": np.array([lead_sd(l, s.freq) for l in range(1, L + 1)]),
                        "ystar": ystar})
        return out

    def recursive(self, H):
        """Fixed-parameter forecasts from every origin, from the end of the
        estimation window (``estwin``) to the end of the data.

        Returns ``(rows, summary)``: rows ``(origin, series, horizon, level,
        actual or None)`` and, per series and horizon, n/MAE/RMSE/MAPE.
        """
        if self.estwin is None:
            raise LadderError("recursive() needs estwin")
        k0 = self.series[0].nobs_full - self.estwin
        rows, acc = [], {}
        try:
            for k in range(k0, -1, -1):
                self.set_origin(k)
                for fc in self.forecast(H):
                    s = next(z for z in self.series if z.name == fc["series"])
                    for l in range(1, H + 1):
                        t = s.nobs + l
                        act = float(s.ts.data[t - 1]) if t <= s.nobs_full else None
                        rows.append((fc["origin"], s.name, l, float(fc["level"][l - 1]), act))
                        if act is not None:
                            e = act - fc["level"][l - 1]
                            a = acc.setdefault((s.name, l), [0, 0.0, 0.0, 0.0])
                            a[0] += 1; a[1] += abs(e); a[2] += e * e
                            a[3] += abs(e / act) if abs(act) > 1e-12 else 0.0
        finally:
            self.set_origin(k0)
        summary = {key: {"n": v[0], "MAE": v[1] / v[0], "RMSE": math.sqrt(v[2] / v[0]),
                         "MAPE": 100.0 * v[3] / v[0]} for key, v in acc.items()}
        return rows, summary



# --------------------------------------------------------------------------- #
#  split: out of the multivariate .inp                                         #
# --------------------------------------------------------------------------- #

def search_row_factors(series, n_max=3, criterion="SC", passes=3):
    """The echelon index search on route U's models (stage 4): each row
    filtered by its factor (the model's seasonal and fixed-frequency
    factors), in passes. The first takes the factors of the univariate
    fits (the gate's); each next one, those of the exact-ML fit of the
    previous choice (``own="free"``), where the cross terms no longer leak
    into them. It stops when the choice repeats, or after ``passes``.
    With no factor it is one plain search.

    The univariate factors mislead the search when a series' marginal
    mixes the other rows' factors: on simulated quarterly (2, 1) systems
    with an annual AR on one row and an annual MA on the other, it found
    the indices in 28 % of the replications with them, and with the second
    pass's factors in most (docs/STUDY-echelon-rowfactor.md).

    Returns (the last ``echelon.Search``, the passes: a list of
    (indices, "univariate models" | "joint fit of (n...)"))."""
    from . import echelon as E
    if series and not isinstance(series[0], LadderSeries):
        series = load(series)
    m = len(series)
    L = Ladder([s.copy() for s in series])
    L.run_gate()
    L._set_structure(list(range(m)), 0, 0, True)
    W = L.cast(L.pack())[4]
    filt = [tuple(s.split_polynomials()[2:4]) for s in L.series]
    if not any(s.has_row_factor() for s in series):
        return E.search(W, n_max=n_max, criterion=criterion), []
    sr = E.search(W, n_max=n_max, criterion=criterion, filters=filt)
    hist = [(sr.indices, "univariate models")]
    for _ in range(int(passes) - 1):
        n = sr.indices
        F = Ladder([s.copy() for s in series], echelon=n, own="free")
        F.fit()
        filt = [tuple(s.split_polynomials()[2:4]) for s in F.series]
        sr = E.search(W, n_max=n_max, criterion=criterion, filters=filt)
        hist.append((sr.indices, f"joint fit of {n}"))
        if sr.indices == n:
            break
    return sr, hist


def split(path, mean=False, harmonics=False, ar=0, ma=0, scale=100.0, out_dir="."):
    """Convert a multivariate drvarma ``.inp`` (deprecated since 5.0) into one
    univariate ``.inp`` of fue per series, ``out_dir/<name>.inp``.

    The same files as ``drvarma -split`` in C. What was a run option of the
    ``.inp`` path becomes part of each specification: ``mean`` (an estimated
    mean), ``scale`` (the rescaling factor), ``harmonics`` (the seasonal
    harmonics as estimated deterministic terms, what ``-deseason`` computed
    outside the model), ``ar``/``ma`` (a free regular AR(P)/MA(Q): with them
    and cross orders p = P, q = Q the ladder is the old full VARMA(P,Q)).

    Existing files are never overwritten. Returns the paths written.
    """
    import os
    from . import __version__
    from .inp import load as load_mv

    src = path if os.path.exists(path) else path + ".inp"
    series, spec = load_mv(src)
    data = np.asarray(series.data, float)
    s = int(series.freq)
    year, sub = series.start
    names = list(series.names)
    specs = []
    if harmonics and s > 1:
        for k in range(1, (s + 1) // 2):
            specs += [f"cos {k}", f"sin {k}"]
        if s % 2 == 0:
            specs.append("alter")
    nd = len(specs)

    def coefs(title, order):
        if order <= 0:
            return [f"** {title}:", "0"]
        return [f"** {title}:", f"1 {order}", "**"] + ["0.000000  1"] * order

    written = []
    for j, name in enumerate(names):
        dst = os.path.join(out_dir, f"{name}.inp")
        if os.path.exists(dst):
            raise LadderError(f"{dst} exists; it is not overwritten")
        L = ["************************************************",
             "* Input file for program FUE                   *",
             "************************************************",
             f"* Split from {src} by drvarma-python {__version__}", "",
             "** Frequency of time series: either 1(A), 4(Q) or 12(M):", f" {s}",
             "** Number of observations and starting date of time series:",
             f" {data.shape[0]}  {sub if s > 1 else 1} {year} {name}",
             "** Number of deterministic variables (including seasonal components):",
             f"{nd}"]
        if nd:
            L += ["**"] + specs + ["**", " ".join(["0"] * nd)]
            for _k in range(nd):
                L += ["**", "0.000000  1"]
            L += ["**", " ".join(["0"] * nd)]
        L += coefs("Number and orders of regular AR operators", ar)
        L += coefs("Number and orders of annual AR operators", 0)
        L += coefs("Number and orders of regular MA operators", ma)
        L += coefs("Number and orders of anual MA operators", 0)
        L += ["** Number and frequencies of regular AR(2) operators with fixed frequency:", "0",
              "** Number and frequencies of regular MA(2) operators with fixed frequency:", "0",
              "** Mean parameter (mu):", "0.000000 1" if mean else "0",
              "** Box-Cox lambda, regular differences and complete annual differences:",
              f" {spec.lam:g}  {spec.d}  {spec.D}",
              "** Individual factors of the annual difference (starting at freq 0.0):",
              " ".join(["0"] * (s // 2 + 1)) if s > 1 else " 0",
              "** ACF/PACF bands (0 Automatic) and reescaling factor:", f" 0 {scale:g}",
              "** Time series (stochastic and non-standard deterministic variables):"]
        L += [f"{v:.15g}" for v in data[:, j]]
        with open(dst, "w") as fh:
            fh.write("\n".join(L) + "\n")
        written.append(dst)
    return written


# --------------------------------------------------------------------------- #
#  Reports (the sections of the C's ladder .out)                               #
# --------------------------------------------------------------------------- #

def write_report(L, path, forecasts=None, recursive=None):
    """The ladder's .out: the gate, the model, the LR test, Sigma, and the
    forecasts and their out-of-sample evaluation when given."""
    from . import __version__
    r = L.result
    S = L.series
    out = [f"Program          : drvarma-python {__version__} (ladder mode: fue files)",
           f"Model            : VARMA with univariate diagonals; cross orders p={L.p} q={L.q}"
           + ("; cross MA in the residual-model form (Jenkins-Alavi 3.22)"
              if getattr(L, "cross", "additive") == "residual" else ""),
           *([f"Cross links      : " + ", ".join(
               f"{L.series[i].name}<-{L.series[j].name}"
               for i in range(len(L.series)) for j in range(len(L.series))
               if i != j and L.links[i, j])] if L.links is not None else []),
           *([f"Echelon form     : Kronecker indices ("
               + ", ".join(f"{s.name} {k}" for s, k in zip(L.series, L.echelon))
               + "), A0 lower triangular (Lutkepohl-Poskitt 1996)"]
             if getattr(L, "echelon", None) is not None else []),
           *([f"Cross rows       : " + ", ".join(
               f"{s.name}({L.rows[i, 0]},{L.rows[i, 1]})" for i, s in enumerate(L.series))]
             if getattr(L, "rows", None) is not None else []),
           *([f"Fixed at zero    : " + ", ".join(sorted(
               f"{kind}{k + 1}[{L.series[i].name}<-{L.series[j].name}]"
               for kind, k, i, j in L.zeros))] if getattr(L, "zeros", None) else []),
           f"Innovation cov.  : {'diagonal' if L.diagcov else 'full'}",
           f"Deterministics   : {'re-estimated' if L.redet else 'fixed at the file'}",
           f"Univariate ARMA  : {'fixed at the file' if L.fixarma else 're-estimated jointly'}",
           f"Likelihood       : {_lik_label(L.lik)}",
           f"Standard errors  : {r.se_method or L.hessian}",
           ""]
    out.append("Univariate models:")
    for i, s in enumerate(S, 1):
        y0, p0 = s.ts.start
        y1, p1 = s.date_of(s.nobs_full)
        out.append(f"  [{i}] {s.name:<14} {s.nobs_full:5d} obs  {p0}/{y0} - {p1}/{y1}"
                   f"  lambda {s.model.boxlam:g}  d={s.model.d} D={s.model.D}"
                   f"  det={len(s.model.interventions)}  {s.path}")
    g = L.gate
    out += ["", "=" * 61, "  THE DIAGONAL GATE", "=" * 61,
            f"  {'series':<14} {'logL':>16} {'sigma2':>14} {'max|move|':>12}"]
    for row in g["rows"]:
        flag = ("" if row["move"] <= PRE_MOVE else
                "  <- trimmed sample" if row["trimmed"] else "  <- not an optimum")
        out.append(f"  {row['series']:<14} {row['logL']:16.6f} {row['sigma2']:14.6g} "
                   f"{row['move']:12.3g}{flag}")
    out += [f"  {'SUM':<14} {g['sum']:16.6f}", f"  {'joint diagonal':<14} {g['joint']:16.6f}",
            f"  {'difference':<14} {g['difference']:16.3g}",
            f"  GATE: {'PASSED' if g['passed'] else 'FAILED'}", ""]
    if r.termcode:                       # the optimiser ran: how it stopped
        out += [""] + _stop_block(r) + [""]
    out += ["=" * 61, "  ESTIMATED MODEL", "=" * 61, f"Number of parameters: {r.npar}",
            f"  {'Parameter':<34} {'Estimate':>12} {'Std.Error':>12} {'t-stat':>9}"]
    for n, v, se in zip(r.names, r.x, r.std_errors):
        if not np.isfinite(se):           # no standard errors: se_method says why
            out.append(f"  {n:<34} {v:12.6f}")
            continue
        t = v / se if se > 0 else 0.0
        out.append(f"  {n:<34} {v:12.6f} {se:12.6f} {t:9.3f}")
    out += ["", f"Exact log-likelihood: {r.logL:.6f}"]
    if L.lik == "both":
        c = L.lik_check
        out += [f"Shea check       : {c['points']} points; max |dlogL| = "
                f"{c['max']:.3e}, at the optimum {c['at_optimum']:.3e}"
                + (f"; {c['one_only']} points admissible for one algorithm only"
                   if c['one_only'] else "")]
    if not (L.p == 0 and L.q == 0 and L.diagcov):
        lr, df, pv = L.lr_test()
        out += ["", "Cross dynamics against the diagonal system (the univariates):",
                f"  logL diagonal {L.logL_diag:.6f}, logL full {r.logL:.6f}",
                f"  LR = {lr:.4f}   df = {df}   p-value = {pv:.4f}"]
    out += ["", f"Innovation covariance Sigma = sigma2 * Q  (sigma2 = {r.sigma2:.8g}):"]
    for a, row in enumerate(r.sigma):
        out.append(f"  {S[a].name:<14}" + "".join(f" {v:14.6g}" for v in row))
    if forecasts:
        out += ["", "=" * 61, "  FORECASTS (each series back to its level with its model)", "=" * 61]
        out += _forecast_lines(forecasts, L)
    if recursive:
        _rows, summary = recursive
        out += ["", "=" * 61, "  RECURSIVE FORECAST EVALUATION (out of sample)", "=" * 61,
                "  Parameters estimated ONCE on the estimation window and held FIXED.",
                "  The univariate models (the diagonal system) are the yardstick."]
        for s in S:
            out += ["", f"  {s.name}", "    h      n        MAE         RMSE        MAPE(%)"]
            for (nm, h), v in sorted(summary.items(), key=lambda kv: kv[0][1]):
                if nm == s.name:
                    out.append(f"  {h:3d}  {v['n']:5d}  {v['MAE']:11.6f}  {v['RMSE']:11.6f}"
                               f"  {v['MAPE']:10.4f}")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def _forecast_lines(forecasts, L):
    out = []
    for fc in forecasts:
        s = next(z for z in L.series if z.name == fc["series"])
        y, p = fc["origin"]
        sc = 100.0 / s.model.refactor
        n = s.nobs
        ys = fc["ystar"]
        out += [f"Series {fc['series']} (from {p}/{y}):",
                f"  date   {'Level':>10} {'Low95':>10} {'High95':>10} {'mon%':>8} {'std':>7}"
                f" {'ann%':>8} {'std':>7}"]
        for l, (yy, pp) in enumerate(fc["dates"], 1):
            g2 = ys[n + l - 1] - ys[n + l - 2]
            g3 = ys[n + l - 1] - ys[n + l - 1 - s.freq] if n + l - 1 - s.freq >= 0 else 0.0
            out.append(f"{pp:3d}/{yy:4d} {fc['level'][l-1]:10.4f} {fc['low95'][l-1]:10.4f}"
                       f" {fc['high95'][l-1]:10.4f} {sc*g2:8.4f} {sc*fc['sd_period'][l-1]:7.4f}"
                       f" {sc*g3:8.4f} {sc*fc['sd_annual'][l-1]:7.4f}")
        out.append("")
    return out


def write_forecast(L, forecasts, path):
    lines = [f"Forecasts from the ladder VARMA (cross orders p={L.p} q={L.q})",
             "Level/Low95/High95 in original units. mon%/ann% = period/annual "
             "variation and std (100*delta/refactor; % for lambda=0).", ""]
    lines += _forecast_lines(forecasts, L)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_recursive(L, rows, H, path):
    with open(path, "w") as fh:
        fh.write(f"Recursive fixed-parameter forecasts, ladder VARMA (p={L.p} q={L.q})\n")
        fh.write(f"estwin={L.estwin} obs of {L.series[0].name}, horizon={H}\n")
        fh.write("origin series horizon level\n")
        for (y, p), s, h, v, _a in rows:
            fh.write(f"{p}/{y} {s} {h} {v:.6f}\n")


def main(argv):
    """drvarma F1 F2 [...] p q [-diagcov] [-redet] [-fixarma] [-m M] [-o NAME]
                            [-forecast H [-estwin N]]"""
    import argparse
    import os
    ap = argparse.ArgumentParser(prog="drvarma", description=
                                 "The ladder: fue's univariate models as a VARMA.")
    ap.add_argument("items", nargs="+", help="fue files (.pre/.inp), then p and q")
    ap.add_argument("-diagcov", action="store_true")
    ap.add_argument("-redet", action="store_true")
    ap.add_argument("-fixarma", action="store_true")
    ap.add_argument("-m", type=int, default=1, dest="method")
    ap.add_argument("-o", default=None, dest="name")
    ap.add_argument("-forecast", type=int, default=None)
    ap.add_argument("-estwin", type=int, default=None)
    ap.add_argument("-lik", choices=("elf", "shea", "both"), default="elf",
                    help="exact likelihood: elf (AS 311, default), shea (AS 242), "
                         "or both (elf checked against Shea at every point)")
    ap.add_argument("-links", default=None,
                    help='cross dynamics only on these pairs, "A<-B,C<-A" (B enters '
                         "the equation of A); default: every pair")
    ap.add_argument("-hessian", choices=("fd", "bfgs"), default="fd",
                    help="standard errors: fdhess at the optimum (default) or the "
                         "BFGS Hessian of the search (docs/STUDY-standard-errors.md)")
    a = ap.parse_args(argv)
    if len(a.items) < 3:
        ap.error("give the fue files, then p and q")
    files, p, q = a.items[:-2], int(a.items[-2]), int(a.items[-1])
    if a.estwin and not a.forecast:
        raise SystemExit("ERROR: -estwin needs a horizon: give -forecast H")
    name = a.name or "_".join(os.path.splitext(os.path.basename(f))[0] for f in files)
    try:
        L = Ladder(files, p, q, diagcov=a.diagcov, redet=a.redet, fixarma=a.fixarma,
                   method=a.method, estwin=a.estwin, hessian=a.hessian,
                   lik=a.lik, links=a.links)
        L.fit()
    except GateError as e:
        print(f"ERROR: {e}"); return 5
    except LadderError as e:
        print(f"ERROR: {e}"); return 4 if "same date" in str(e) or "per-year" in str(e) else 2
    fc = rec = None
    if a.forecast:
        fc = L.forecast(a.forecast)
        write_forecast(L, fc, name + ".forecast")
        if a.estwin:
            rec = L.recursive(a.forecast)
            write_recursive(L, rec[0], a.forecast, name + ".recursive")
    write_report(L, name + ".out", fc, rec)
    print(f"Exact log-likelihood: {L.result.logL:.6f}")
    print(f"Full results written to {name}.out")
    return 0


def split_main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="drvarma -split", description=
                                 "One univariate .inp of fue per series (the multivariate "
                                 ".inp is deprecated).")
    ap.add_argument("file")
    ap.add_argument("-mean", action="store_true")
    ap.add_argument("-harmonics", action="store_true")
    ap.add_argument("-ar", type=int, default=0)
    ap.add_argument("-ma", type=int, default=0)
    ap.add_argument("-scale", type=float, default=100.0)
    ap.add_argument("-dir", default=".")
    a = ap.parse_args(argv)
    try:
        for f in split(a.file, a.mean, a.harmonics, a.ar, a.ma, a.scale, a.dir):
            print(f)
    except LadderError as e:
        print(f"ERROR: {e}"); return 2
    return 0


def is_fue_file(path):
    """True if fue reads ``path`` as a univariate model (content, not name)."""
    import os
    if not os.path.isfile(path):
        return False
    try:
        import fue
        fue.load(path)
        return True
    except Exception:
        return False
