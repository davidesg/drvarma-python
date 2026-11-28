"""BUG-0020 (the C's BUG-0009): when the diagonal system cannot be evaluated
at the univariate optima, the gate says so and names an own MA factor on the
invertibility wall, instead of "difference nan" and blaming the cast."""
import glob
import os

import pytest

from drvarma.ladder import GateError, Ladder, _fault_msg, _own_ma_list

D = os.path.join(os.path.dirname(__file__), "data", "bug0020")


def test_la_puerta_nombra_el_ma_en_la_pared():
    L = Ladder(sorted(glob.glob(os.path.join(D, "y*.inp"))), 0, 0)
    with pytest.raises(GateError) as e:
        L.run_gate()
    msg = str(e.value)
    assert "could not be evaluated at the univariate optima (AR nonstationary)" in msg
    assert "invertibility wall: y2 (1.0000)" in msg
    assert "joint cast does not reproduce" not in msg
    g = L.gate
    assert g["ifault"] == 3 and not g["passed"]
    assert [w[0] for w in g["own_ma_wall"]] == ["y2"]


def test_las_piezas_del_mensaje_son_las_del_c():
    assert _fault_msg(3) == "AR nonstationary"
    assert _fault_msg(99) == "unknown"
    assert _own_ma_list([("y2", "regular MA", 2, "", 0.99990),
                         ("y3", "regular MA", 1, "theta = 1.0", 1.0)]) == "y2 (0.9999), y3 (1.0000)"
