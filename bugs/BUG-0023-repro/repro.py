"""BUG-0023: collecting tests/test_regression_bugs.py sets DRVARMA_NO_ENGINE
for the whole pytest session.

The module runs `os.environ.setdefault("DRVARMA_NO_ENGINE", "1")` at import,
and pytest imports every test module before running any test: from then on
every test of a full run computes with the pure-Python port, whatever it was
written to test. This script imports the module the way collection does and
checks the variable afterwards. Exits 1 while the import leaks it.

    python3 bugs/BUG-0023-repro/repro.py      (from the repository root)
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.pop("DRVARMA_NO_ENGINE", None)

path = os.path.join(ROOT, "tests", "test_regression_bugs.py")
spec = importlib.util.spec_from_file_location("test_regression_bugs", path)
spec.loader.exec_module(importlib.util.module_from_spec(spec))

leaked = os.environ.get("DRVARMA_NO_ENGINE")
print("DRVARMA_NO_ENGINE after importing test_regression_bugs:", repr(leaked))
sys.exit(1 if leaked else 0)
