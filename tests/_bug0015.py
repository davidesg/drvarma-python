"""BUG-0015's declared divergence from the C binary, for the byte-exact .out
tests: until the C (diagnose.c, drvarma-v5) subtracts the ARMA coefficients,
its Hosking line says Q(m^2 s) and the port's Q(m^2 s - k). The Q value must
be the same; the port's df must be the C's minus k; the verdict line under it
follows the p-value and may differ. Everything else stays byte-exact."""
import re

_Q = re.compile(r"^\s*Q\((\d+)\) = ([\d.]+), p-value = ")


def without_hosking(c_sec, py_sec, k):
    """Both sections with the Hosking line and its verdict checked and taken
    out, so the rest can be compared byte for byte."""
    def split(sec):
        lines = sec.splitlines(keepends=True)
        at = [i for i, ln in enumerate(lines) if _Q.match(ln)]
        if not at:
            return sec, None
        i = at[0]
        return "".join(lines[:i] + lines[i + 2:]), _Q.match(lines[i]).groups()
    c_rest, cq = split(c_sec)
    p_rest, pq = split(py_sec)
    assert (cq is None) == (pq is None)
    if cq is not None:
        assert pq[1] == cq[1], "the same Q"
        assert int(pq[0]) == int(cq[0]) - k, "df = the C's m^2 s minus k"
    return c_rest, p_rest
