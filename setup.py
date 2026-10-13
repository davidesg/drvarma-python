"""Build hook for the optional CFFI C engine.

Project metadata lives in ``pyproject.toml``; this file only wires up the
``drvarma._drvarma_engine`` extension so ``pip install`` compiles it when the
GSL development headers are available.  The extension is marked *optional*: if
it cannot be built (e.g. GSL missing), the install still succeeds and the
package runs in pure-Python mode (the C-engine tests skip themselves).
"""

import importlib.util
import os

from setuptools import setup

_HERE = os.path.dirname(os.path.abspath(__file__))


def _cffi_extension():
    """Load src/drvarma/_build_cffi.py standalone and return its Extension.

    Loaded by file path (not ``import drvarma._build_cffi``) so it does not pull
    in the package's runtime imports (numpy, scipy, ...) at build time.
    """
    path = os.path.join(_HERE, "src", "drvarma", "_build_cffi.py")
    spec = importlib.util.spec_from_file_location("_drvarma_build_cffi", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ext = mod.ffi.distutils_extension(os.path.join(_HERE, "build"))
    ext.optional = True          # a build failure must not abort the install
    return ext


def _sync_material():
    """Copy bugs/ into the package (tools/sync_material.py) when building from
    a checkout. The copy is generated and ignored by git, so without this an
    install from git, or a wheel built without running the tool first, carries
    no register (BUG-0021). An sdist carries the copy already and no bugs/."""
    if not os.path.isdir(os.path.join(_HERE, "bugs")):
        return
    path = os.path.join(_HERE, "tools", "sync_material.py")
    spec = importlib.util.spec_from_file_location("_drvarma_sync_material", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n, c = mod.sync()
    print("drvarma: bug register, %d files (%d copied) into the package." % (n, c))


_sync_material()

ext_modules = []
if os.environ.get("DRVARMA_NO_ENGINE"):
    # Force a pure-Python build (e.g. to produce a py3-none-any wheel for PyPI).
    print("drvarma: DRVARMA_NO_ENGINE set; building pure-Python (no C engine).")
else:
    try:
        ext_modules = [_cffi_extension()]
    except Exception as exc:  # pragma: no cover - depends on the build environment
        print("drvarma: skipping C engine extension (%s); pure-Python install." % exc)

setup(ext_modules=ext_modules)
