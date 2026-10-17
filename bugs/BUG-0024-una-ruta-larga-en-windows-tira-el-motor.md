---
id: BUG-0024
title: Built on Windows from pip's temp directory (a git URL or an sdist), the engine's object path passes 260 characters and the optional extension is dropped in silence
status: open
severity: high
component: packaging
found_in: 0.3.0
fixed_in:
reported: 2026-10-17
reporter: David / Claude — drvec-python's off-Linux CI (run 39054125424, then 39054783492 with pip -v)
tags: [packaging, windows, engine, build]
references:
  - setup.py _cffi_extension (ffi.distutils_extension(os.path.join(_HERE, "build")))
  - bugs/BUG-0024-repro/repro.py
  - BUG-0022 (the same silence, another cause)
---

## Summary

`pip install "drvarma @ git+https://github.com/davidesg/drvarma-python@<ref>"`
on windows-latest, with GSL from vcpkg where `_build_cffi.py` looks for it,
installs drvarma WITHOUT `_drvarma_engine.pyd`. MSVC stops on the generated
source:

    _drvarma_engine.c : fatal error C1083: Cannot open compiler generated file: '': Invalid argument

and setuptools, the extension being `optional`, only warns
(`build_ext: building extension "drvarma._drvarma_engine" failed`). drvarma
then runs on the pure-Python port. In the same job fue's engine, built the
same way from its git URL, compiles and loads.

## Impact

Every Windows build from pip's temp directory: an install from git, and an
install from the SDIST (pip unpacks it under the same
`...\AppData\Local\Temp\pip-install-*\drvarma_<hex>`), which is what a Windows
user without a matching wheel gets (a Python with no cp3XX wheel yet). The
published wheels are not affected: cibuildwheel builds from the runner's
short checkout path (`D:\a\drvarma-python\drvarma-python`).

drvec 0.1.0 needs the engine for GSL's eigensolvers (drvec._gsl): without it
its Johansen seed and printed eigenvalues fall back to numpy/scipy.

## Reproduction

`python3 bugs/BUG-0024-repro/repro.py` — builds the extension from a copy of
the checkout with `pip wheel` (needs GSL and a C compiler; Linux is enough),
finds the object file and exits 1 while its path under `build/` contains the
project directory again. It prints the path that layout gives under the CI's
pip directory on Windows: 275 characters.

## Root cause

`setup.py` calls `ffi.distutils_extension(os.path.join(_HERE, "build"))`: an
ABSOLUTE directory, so the extension's generated source is
`<project>\build\drvarma\_drvarma_engine.c`. setuptools forms an object name
by appending the source's path, drive stripped, to `build_temp`; for an
absolute source that is the whole project path again:

    <project>\build\temp.win-amd64-cpython-310\Release\<project without drive>\build\drvarma\_drvarma_engine.obj

With pip's `<project>` (~100 characters) that is ~270, over MAX_PATH, and
`cl.exe` cannot create it. The other sources (`csrc\...`) are relative and
land at short paths. fue's build passes a relative directory, and its object
is `build\temp...\Release\build\temp...\Release\fue._fue_engine.obj`.

## Fix

Proposed: give cffi a directory RELATIVE to the project (setuptools runs the
backend with the project as the working directory), e.g.
`ffi.distutils_extension("build")`, or `os.path.relpath(...)` as
`_build_cffi.py` already does for `csrc`. The object then lands at
`build\temp...\Release\build\drvarma\_drvarma_engine.obj`.

And so that the next cause is not silent either: honour
`DRVARMA_REQUIRE_ENGINE` at BUILD time too (set `ext.optional = False` when
it is set), so that a build that is meant to carry the engine fails instead
of warning.

Workaround (drvec-python's tests.yml): clone to a short path
(`$RUNNER_TEMP/dv`) and `pip install` that directory.

## Validation

The repro exits 0 once the object file no longer nests the project path; on
Windows, `pip install` from git loads the engine
(`drvarma._engine.engine_backend() == "c"`), which drvec-python's manual CI
run checks.
