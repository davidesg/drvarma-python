"""drvarma — multivariate VARMA modelling (Python port).

Free software under the GNU General Public License v2 or later (see COPYING).
Python port of the drvarma C engine; see docs/MIGRATION_PLAN.md.
"""

__version__ = "0.1.0"

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
