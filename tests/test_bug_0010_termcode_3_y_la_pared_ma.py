"""BUG-0010: termcode 3 read one way, and the MA wall said by Model too.

The external review's case (§2.4, §4.5): VARMA(1,1) on Spain's CPI and WTI,
lambda = 0, d = 1, with mean. It ends with Theta_1's inverse roots at modulus
1.00005, termcode 3 and ifault 0. The package said "AT the optimum, not a
fault" in estimate_py and "not a maximum" in the report; the ladder and the C
already reported a stop on the MA wall, Model did not.
"""
import os

import numpy as np
import pandas as pd
import pytest

from drvarma import Model, MultiSeries
from drvarma.report import _convergence_block

D = os.path.join(os.path.dirname(__file__), "data", "bug_0010")


@pytest.fixture(scope="module")
def fit():
    a = pd.read_csv(os.path.join(D, "IPC_ES.csv"))
    b = pd.read_csv(os.path.join(D, "WTI.csv"))
    ms = MultiSeries(np.column_stack([a.value, b.value]), freq=12, start=(2002, 1),
                     names=["IPC", "WTI"])
    return Model(ms, lam=0.0, d=1, p=1, q=1, include_mean=True).fit()


def test_the_review_case_is_on_the_wall(fit):
    assert fit.termcode == 3 and fit.ifault == 0
    assert np.all(np.abs(np.abs(np.linalg.eigvals(fit.theta[0])) - 1.0) < 5e-5)
    assert (fit.ma_boundary, fit.ma_nroots) == (2, 2)
    assert fit.converged is False


def test_the_report_says_the_fact_as_the_c(fit):
    txt = _convergence_block(fit)
    assert "OPTIMIZER STOPPED at the MA invertibility boundary" in txt
    assert "MA boundary: 2 of 2 inverse roots within 5e-5 of the unit circle" in txt
    assert "not a maximum" not in txt and "AT the optimum" not in txt
    assert "re-estimate from these values" in txt
