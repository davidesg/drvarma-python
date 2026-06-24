"""MultiSeries: a multivariate time series with date metadata.

Mirrors the lightweight style of fue's TimeSeries but holds `m` series in an
(nobs x m) array, matching drvarma's multivariate `.inp` data layout.
"""

import numpy as np


class MultiSeries:
    """A multivariate time series.

    Parameters
    ----------
    data : array-like, shape (nobs, m)
        Observations in chronological order; one column per series.
    freq : int
        Observations per year: 1 (annual), 4 (quarterly), 12 (monthly).
    start : tuple (year, period)
        First observation; period is 1-based (Jan = 1 for monthly).
    names : sequence of str, optional
        Series labels (default y1..ym).
    """

    def __init__(self, data, freq=12, start=(2000, 1), names=None):
        self.data = np.atleast_2d(np.asarray(data, dtype=float))
        if self.data.shape[0] == 1 and self.data.shape[1] != 1 and (names is None or len(names) != 1):
            # a single row passed as 1-D -> treat as one observation of m series
            pass
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        self.freq = int(freq)
        self.start = tuple(start)
        m = self.data.shape[1]
        self.names = list(names) if names is not None else [f"y{j+1}" for j in range(m)]
        if len(self.names) != m:
            raise ValueError(f"names has {len(self.names)} entries but data has {m} columns")

    @property
    def nobs(self):
        return self.data.shape[0]

    @property
    def m(self):
        return self.data.shape[1]

    def column(self, j):
        """Series j (0-based) as a 1-D array."""
        return self.data[:, j]

    def __repr__(self):
        return (f"MultiSeries(m={self.m}, nobs={self.nobs}, freq={self.freq}, "
                f"start={self.start}, names={self.names})")
