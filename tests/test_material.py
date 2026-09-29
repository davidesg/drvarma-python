"""The packaged copy of the bug register is the register (tools/sync_material.py)."""
import filecmp
import os

from drvarma.register import bugs_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_register_is_found():
    d = bugs_dir()
    assert d and any(f.startswith("BUG-0010") for f in os.listdir(d))


def test_the_packaged_copy_is_in_sync():
    src = os.path.join(ROOT, "bugs")
    dst = os.path.join(ROOT, "src", "drvarma", "material", "bugs")
    if not os.path.isdir(dst):
        return                                  # not synced here: bugs_dir() uses bugs/
    names = sorted(f for f in os.listdir(src) if f.endswith(".md"))
    assert sorted(os.listdir(dst)) == names, "run tools/sync_material.py"
    _m, mismatch, errors = filecmp.cmpfiles(src, dst, names, shallow=False)
    assert not mismatch and not errors, f"stale: {mismatch} -- run tools/sync_material.py"
