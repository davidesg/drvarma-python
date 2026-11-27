"""BUG-0031: the binary wheels stop at cp314, and a pure-Python wheel is always
there to catch any other interpreter in silence.

[tool.cibuildwheel] builds "cp310-* ... cp314-*" and skips "*-win32" and the
free-threaded "cp31?t-*"; py3-none-any is published next to them. CPython 3.15.0
is final since 2026-10-09, and `uv tool install` (how the course installs the
suite) takes the newest Python it finds or downloads. On 3.15, a free-threaded
build or a 32-bit Windows Python, pip and uv install the py3-none-any wheel
without a word and every system estimate runs on the port (hours where the engine takes
seconds: TODO.md, the speed study).

The repro reads pyproject.toml offline and exits 1 while the build list leaves
out a CPython that is final, or while the pure wheel is published with no
binary for it. FINAL lists the released minor versions (update it).

    python3 bugs/BUG-0031-repro/repro.py   # exits 1 while open
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"]     # 3.15.0: 2026-10-09
txt = open(os.path.join(HERE, "..", "..", "pyproject.toml"), encoding="utf-8").read()
block = txt.split("[tool.cibuildwheel]", 1)[1].split("\n[", 1)[0]
build = re.search(r'build\s*=\s*"([^"]*)"', block).group(1).split()
skip = re.search(r'skip\s*=\s*"([^"]*)"', block)
skip = skip.group(1).split() if skip else []
built = {f"3.{mm.group(1)}" for b in build if (mm := re.match(r"cp3(\d+)-", b))}
missing = [v for v in FINAL if v not in built]
print("cibuildwheel build:", " ".join(build))
print("cibuildwheel skip :", " ".join(skip))
print("final CPython without a binary wheel:", ", ".join(missing) or "none")
for s in skip:
    if s in ("*-win32",) or "t-*" in s:
        print(f"  also no binary for: {s}  -> py3-none-any, the port (it warns: BUG-0005)")
# The fix is an abi3 wheel (setup.py): one binary per platform for every
# CPython >= 3.10. Without it a version in the list is still per-version only.
setup = open(os.path.join(HERE, "..", "..", "setup.py"), encoding="utf-8").read()
abi3 = "py_limited_api" in setup and '"cp310"' in setup
print("abi3 wheel (one binary for every CPython >= 3.10):", "yes" if abi3 else "NO")
sys.exit(1 if missing or not abi3 else 0)
