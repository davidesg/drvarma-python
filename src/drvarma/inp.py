"""Reader/writer for drvarma multivariate `.inp` files.

Format (lines beginning with '*' or '**' are comments/labels and are skipped;
the actual values are read in order from the remaining lines):

    * optional comment
    ** Frequency (1=A, 4=Q, 12=M):
     12
    ** Series, observations, start (subperiod year):
     3 216 1 2002
    ** Series names:
     IPC_ES IPC_FR IPC_DE
    ** Box-Cox lambda, regular differences, annual differences:
     0.0 1 0
    ** Data:
     69.53 81.94 77.70
     ...

This token-stream parser mirrors the C reader (inp_int/inp_real/inp_token),
which skips comment/label lines and reads values positionally.
"""

from dataclasses import dataclass
from .series import MultiSeries


@dataclass
class InpSpec:
    """Transformation header of a drvarma .inp file."""
    lam: float          # Box-Cox lambda (0 = log, 1 = identity)
    d: int              # regular differences
    D: int              # seasonal differences


def _tokens(path):
    """Yield whitespace-separated tokens from non-comment lines."""
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("*"):     # '' , '*' and '**' are skipped
                continue
            for tok in s.split():
                yield tok


def load(path):
    """Load a drvarma `.inp` file.

    Returns
    -------
    (series, spec) : (MultiSeries, InpSpec)
    """
    it = _tokens(path)
    nxt = lambda: next(it)

    freq = int(nxt())
    nser = int(nxt()); nobs = int(nxt())
    start_sub = int(nxt()); start_year = int(nxt())
    names = [nxt() for _ in range(nser)]
    lam = float(nxt()); d = int(nxt()); D = int(nxt())

    data = [[float(nxt()) for _ in range(nser)] for _ in range(nobs)]
    series = MultiSeries(data, freq=freq, start=(start_year, start_sub), names=names)
    return series, InpSpec(lam=lam, d=d, D=D)


def save(path, series, spec, comment="DRVARMA input"):
    """Write a drvarma `.inp` file from a MultiSeries and an InpSpec."""
    yr, sub = series.start
    with open(path, "w", encoding="latin-1") as f:
        f.write(f"* {comment}\n")
        f.write("** Frequency (1=A, 4=Q, 12=M):\n")
        f.write(f" {series.freq}\n")
        f.write("** Series, observations, start (subperiod year):\n")
        f.write(f" {series.m} {series.nobs} {sub} {yr}\n")
        f.write("** Series names:\n")
        f.write(" " + " ".join(series.names) + "\n")
        f.write("** Box-Cox lambda, regular differences, annual differences:\n")
        f.write(f" {spec.lam:g} {spec.d} {spec.D}\n")
        f.write("** Data:\n")
        for i in range(series.nobs):
            f.write(" ".join(f"{v:.4f}" for v in series.data[i]) + "\n")
