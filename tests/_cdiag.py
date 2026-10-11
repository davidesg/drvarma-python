"""The C binary the residual diagnosis is checked against.

Since drvec BUG-42 (2026-10-11) the multivariate block (Hosking with
df = m^2 s - k and s = freq + 2, Lutkepohl's Jarque-Bera) and the per-series
Ljung-Box df live in atsw-gui's lib/diagmv, and only the drvarma built there
prints them; drvarma_v.04.1 keeps the old ones. DRVARMA_C_BIN overrides."""
import os

C_DIAG_BIN = os.environ.get("DRVARMA_C_BIN") or os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "atsw-gui", "engines",
    "drvarma", "bin", "drvarma")
