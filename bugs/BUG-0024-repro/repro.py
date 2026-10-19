"""BUG-0024: built from a long directory (pip's temp clone of a git URL or of
an sdist), the Windows engine's object path passes 260 characters and the
optional extension is dropped in silence.

setup.py hands cffi an ABSOLUTE build dir (os.path.join(_HERE, "build")), so
the generated source _drvarma_engine.c has an absolute path, and setuptools
nests that whole path under build_temp for its object file:

    <project>\\build\\temp.win-amd64-cpython-3XX\\Release\\<project without drive>\\build\\drvarma\\_drvarma_engine.obj

The project directory appears TWICE. Under pip's
C:\\Users\\runneradmin\\AppData\\Local\\Temp\\pip-install-*\\drvarma_<32 hex>
that is ~270 characters, MSVC stops with "C1083: Cannot open compiler
generated file: '': Invalid argument", and build_ext only warns (drvec-python
CI run 39054783492, windows-latest, both Pythons).

This repro builds the extension on THIS machine (it needs GSL and a C
compiler) from a copy of the checkout, finds the object file, and exits 1
while its path under build/ contains the project directory again, printing
the length that path would have under the CI's pip directory on Windows.

    python3 bugs/BUG-0024-repro/repro.py
"""
import glob
import ntpath
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pip's clone of the git URL in that CI run (its length is what matters)
PIP_DIR = (r"C:\Users\runneradmin\AppData\Local\Temp\pip-install-h2zpo_y4"
           r"\drvarma_394e887b76224e0380324e54e85107a1")
MAX_PATH = 260

with tempfile.TemporaryDirectory() as tmp:
    proj = os.path.join(tmp, "drvarma_project")
    shutil.copytree(ROOT, proj, ignore=shutil.ignore_patterns(
        ".git", "build", "*.so", "*.o", "*.pyd", "__pycache__", "*.egg-info",
        "_drvarma_engine.c"))
    # pip builds a local directory in place (as it builds its own clone of a
    # git URL or an unpacked sdist), with the build backend pyproject.toml asks for
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps",
                        "-w", os.path.join(tmp, "wheel"), proj],
                       cwd=proj, capture_output=True, text=True)
    objs = glob.glob(os.path.join(proj, "build", "temp*", "**", "_drvarma_engine.o*"),
                     recursive=True)
    if not objs:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit("no object file: the engine did not compile here (GSL? a compiler?)")
    rel = os.path.relpath(objs[0], proj)
    nested = proj.lstrip(os.sep) in rel
    print("object file, under the project:", rel)
    # the same layout on Windows, under pip's directory
    parts = rel.split(os.sep)
    i = parts.index("Release") + 1 if "Release" in parts else 2
    tail = parts[i:]
    if nested:                                 # the project path, repeated
        k = len(proj.lstrip(os.sep).split(os.sep))
        tail = ntpath.splitdrive(PIP_DIR)[1].lstrip("\\").split("\\") + tail[k:]
    win = ntpath.join(PIP_DIR, "build", "temp.win-amd64-cpython-310", "Release",
                      *tail).replace(".o", ".obj")
    print("on Windows, under pip's clone: %d characters (MAX_PATH %d)" % (len(win), MAX_PATH))
    print("  " + win)
    print("the project directory is nested in the object path:", nested)
sys.exit(1 if nested else 0)
