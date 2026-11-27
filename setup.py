"""Build hook for the optional CFFI C engine.

Project metadata lives in ``pyproject.toml``; this file only wires up the
``drvarma._drvarma_engine`` extension so ``pip install`` compiles it when the
GSL development headers are available.  The extension is marked *optional*: if
it cannot be built (e.g. GSL missing), the install still succeeds and the
package runs in pure-Python mode (the C-engine tests skip themselves).
With ``DRVARMA_REQUIRE_ENGINE`` set (the binary wheels) it is required: a
build that cannot compile it fails.
"""

import importlib.util
import os
import sysconfig

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
    # RELATIVE, as cffi's own default and fue's cffi_modules (BUG-0024): with an
    # absolute directory the generated source's path is absolute, setuptools
    # nests all of it under build_temp for the object file, and from pip's
    # temp directory on Windows that path passes MAX_PATH (cl.exe C1083).
    # setuptools runs the build with the project as the working directory.
    ext = mod.ffi.distutils_extension(os.path.relpath(os.path.join(_HERE, "build")))
    # A build failure must not abort the install, unless the build is meant to
    # carry the engine (the binary wheels, a CI job): then it fails loudly
    # instead of dropping the engine with a warning (BUG-0022, BUG-0024).
    ext.optional = not os.environ.get("DRVARMA_REQUIRE_ENGINE")
    # abi3 (BUG-0031): cffi already compiles against the limited API (the .pyd
    # links python3.dll), so one wheel per platform serves every CPython from
    # 3.10 on, the ones released after the wheel included. The wheels were
    # per minor version and stopped at cp314; on 3.15 pip took the
    # py3-none-any wheel and every fit ran on the port. A free-threaded
    # build has no limited API: it keeps a version-specific extension.
    ext.py_limited_api = _ABI3
    return ext


# The limited API does not exist on a free-threaded CPython (3.13t, 3.14t...).
_ABI3 = not sysconfig.get_config_var("Py_GIL_DISABLED")


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
        if os.environ.get("DRVARMA_REQUIRE_ENGINE"):
            raise
        print("drvarma: skipping C engine extension (%s); pure-Python install." % exc)

# The abi3 tag goes on the wheel only when it carries the extension: the pure
# wheel stays py3-none-any, the port for every platform without a binary.
options = {"bdist_wheel": {"py_limited_api": "cp310"}} if ext_modules and _ABI3 else {}
setup(ext_modules=ext_modules, options=options)
