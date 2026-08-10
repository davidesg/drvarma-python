"""drvarma — multivariate VARMA modelling (Python port).

Free software under the GNU General Public License v2 or later (see COPYING).
Python port of the drvarma C engine; see docs/MIGRATION_PLAN.md.
"""

# Read from the installed metadata rather than repeated here: a hand-written
# constant drifts, and the copy that drifts is always the one nobody builds
# from. This one said "0.1.1" while the distribution was already further on.
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    try:
        __version__ = _pkg_version("drvarma")
    except PackageNotFoundError:            # running from a source tree
        __version__ = "0.0.0.dev0"
except ImportError:                         # pragma: no cover
    __version__ = "0.0.0.dev0"
from .series import MultiSeries
from .inp import load, save, InpSpec
from .model import Model
from . import (transform, forecast, diagnostics, irf, deseason, datasets,
               report, report_forecast, elfvarma_py, estimate_py, plots,
               volatility)

__all__ = ["MultiSeries", "load", "save", "InpSpec", "Model",
           "transform", "forecast", "diagnostics", "irf", "deseason",
           "datasets", "report", "report_forecast", "elfvarma_py",
           "estimate_py", "plots", "volatility", "__version__"]
