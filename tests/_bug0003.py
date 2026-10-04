"""BUG-0003's declared divergence from the C binary.

The port deseasonalizes after the log (log -> d=1 -> deseasonalization, as
art); the C (drvarma.c / deseason.c, and the binary these tests run) still
adjusts the raw levels before the log. With `-deseason` every number moves,
so the parity tests that use it are expected to fail until drvarma-v5 changes
the C. `strict=True`: the day they pass again, they fail, as the reminder to
remove this mark.
"""
import pytest

c_deseason_in_levels = pytest.mark.xfail(
    strict=True,
    reason="BUG-0003: the C binary still deseasonalizes the levels before the "
           "log; the port does log -> d=1 -> deseasonalization (drvarma-v5)")
