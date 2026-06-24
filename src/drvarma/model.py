"""High-level Model API: specification, fit (via the C engine), results."""

import numpy as np
from . import transform


class Model:
    """A VARMA(p,q) model specification with Box-Cox/differencing transform.

    Parameters
    ----------
    series : MultiSeries
    lam, d, D : Box-Cox lambda and regular/seasonal differencing orders
    scale : rescale factor after Box-Cox (default 100, as the C engine)
    p, q : AR, MA orders
    include_mean : estimate a mean/drift term
    diag_ar, diag_ma, diag_cov : diagonal restrictions
    method : 1 = exact ML, 2 = approximate
    twostep : Hannan-Rissanen initialisation (q>0 only)
    """

    def __init__(self, series, lam=0.0, d=1, D=0, scale=transform.DEFAULT_SCALE,
                 p=0, q=0, include_mean=False,
                 diag_ar=False, diag_ma=False, diag_cov=False,
                 method=1, twostep=False):
        self.series = series
        self.lam, self.d, self.D, self.scale = lam, d, D, scale
        self.p, self.q = p, q
        self.include_mean = include_mean
        self.diag_ar, self.diag_ma, self.diag_cov = diag_ar, diag_ma, diag_cov
        self.method, self.twostep = method, twostep
        self.result = None

    def fit(self):
        """Transform the series and estimate by exact ML (C engine)."""
        from ._engine import estimate_w
        w, bc = transform.transform(self.series.data, lam=self.lam, d=self.d,
                                    D=self.D, s=self.series.freq, scale=self.scale)
        self._w, self._bc = w, bc
        self.result = estimate_w(
            w, p=self.p, q=self.q, include_mean=self.include_mean,
            diag_ar=self.diag_ar, diag_ma=self.diag_ma, diag_cov=self.diag_cov,
            method=self.method, twostep=self.twostep,
        )
        return self

    def forecast(self, L, b=0):
        """Forecast L steps ahead (origin = last obs minus b); returns levels (L, m)."""
        if self.result is None:
            raise RuntimeError("call fit() before forecast()")
        from .forecast import forecast_levels
        levels, _ = forecast_levels(self.result, self._w, self._bc,
                                    lam=self.lam, scale=self.scale,
                                    d=self.d, D=self.D, s=self.series.freq, L=L, b=b)
        return levels

    def diagnostics(self, lag=None):
        """Multivariate residual diagnostics: Hosking Q and Jarque-Bera.

        `lag` defaults to freq + 2 (14 for monthly), matching the C engine.
        """
        if self.result is None:
            raise RuntimeError("call fit() before diagnostics()")
        from .diagnostics import hosking_q, jarque_bera_mv
        res = self.result["residuals"]
        if lag is None:
            lag = (self.series.freq + 2) if self.series.freq > 1 else 10
        Q, qdf, qp = hosking_q(res, lag)
        JB, jdf, jp = jarque_bera_mv(res)
        return {"hosking_Q": Q, "hosking_df": qdf, "hosking_p": qp, "hosking_lag": lag,
                "JB": JB, "JB_df": jdf, "JB_p": jp}

    # convenience accessors -------------------------------------------------
    @property
    def params(self):
        return None if self.result is None else self.result["params"]

    @property
    def sigma2(self):
        return None if self.result is None else self.result["sigma2"]

    @property
    def loglik(self):
        return None if self.result is None else self.result["logelf"]

    @property
    def ifault(self):
        return None if self.result is None else self.result["ifault"]
