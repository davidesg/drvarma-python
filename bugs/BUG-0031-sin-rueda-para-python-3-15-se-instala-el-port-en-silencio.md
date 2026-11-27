---
id: BUG-0031
title: Binary wheels stop at cp314 (and skip win32 and free-threaded); on CPython 3.15, final since 2026-10-09, pip and uv install the py3-none-any wheel in silence and every fit runs on the port
status: fixed
severity: high
component: packaging
found_in: 0.3.4
fixed_in: unreleased (after 0.3.4)
reported: 2026-11-27
reporter: David / Claude — the P08 run of the course (UCM)
tags: [packaging, wheels, engine, python-3.15, windows, macos]
references:
  - pyproject.toml [tool.cibuildwheel] build / skip
  - .github/workflows/publish.yml (cp310-cp314)
  - bugs/BUG-0031-repro/repro.py
  - BUG-0005, BUG-0022, BUG-0024 (the same silence, other causes)
  - TODO.md «ESTUDIO — La velocidad del port en Python» (what the port costs)
---

## Summary

`[tool.cibuildwheel]` builds `cp310-* … cp314-*` and skips `*-win32` and
`cp31?t-*` (free-threaded). A `py3-none-any` wheel is published next to them.
Any interpreter outside that list gets the pure wheel without a word:
- CPython 3.15, final since 2026-10-09;
- a free-threaded build;
- a 32-bit Python on Windows. `sys.platform` and `platform.machine()` still
  print "win32 AMD64", so the provenance does not show it.

drvarma then computes on the port, which is a faithful port of the C and on sima's orders costs hours where the engine takes seconds (TODO.md, the speed study). fue publishes the same set of
wheels and falls back the same way.

The course installs the suite with `uv tool install atsw==…`, and uv takes the
newest Python it finds or downloads. In October and November 2026 that is 3.15.

## Impact

In the P08 the sima scripts record `drvarma engine python` on Windows AMD64
(drvarma 0.3.1, 0.3.3 and 0.3.4) and on macOS arm64 (0.3.1 and 0.3.2), all
versions that publish a binary for that platform up to cp314. The Windows
wheels do carry the engine: GSL is static, and the only imports are the CRT,
`VCRUNTIME140.dll` and `python3.dll`, checked on 0.3.1 and 0.3.4 cp312.

So an interpreter without a wheel is the most likely cause, and 3.15 the most
likely interpreter. This is not confirmed: the scripts do not record the
Python version (sima BUG-0020). The same groups' art scripts show fue on
«motor Python» too. On macOS 0.3.1, which only had `macosx_14_0` wheels,
macOS < 14 also fell back.

## Reproduction

`python3 bugs/BUG-0031-repro/repro.py` reads pyproject.toml offline and exits
1 while a final CPython has no binary wheel (the list FINAL is kept by hand):

```
cibuildwheel build: cp310-* cp311-* cp312-* cp313-* cp314-*
cibuildwheel skip : *-win32 *-manylinux_i686 pp* cp31?t-*
final CPython without a binary wheel: 3.15
  also no binary for: *-win32  -> py3-none-any, the port, in silence
  also no binary for: cp31?t-*  -> py3-none-any, the port, in silence
```

To confirm on an affected machine:
- Windows: `%APPDATA%\uv\tools\atsw\Scripts\python.exe -c "import sys, drvarma; print(sys.version); print(drvarma.engine_backend(), drvarma.engine_load_error())"`
- macOS: the same with `~/.local/share/uv/tools/atsw/bin/python`.

## Root cause

The build list is a fixed set of minor versions, and the pure wheel turns
"no binary for this Python" into a successful install instead of an error or
a build from the sdist.

## Fix

- Build cp315, and add a release-time check that the newest final CPython is
  in the list. Consider an abi3 wheel: the extension already links
  `python3.dll`, which would cover future versions.
- Stop publishing `py3-none-any`, or publish it under an extra or another
  name, so that an unsupported interpreter builds from the sdist (and fails
  loudly without a compiler) instead of degrading.
- On the user side, the suite's installation guide should pin the Python:
  `uv tool install --python 3.13 atsw==…`.

## Validation

The repro passes (cp315 built). On a 3.15 interpreter `pip install drvarma`
brings a binary wheel and `engine_backend()` says "c".

## Resolution (2026-11-27)

- The extension is built for the limited API (`py_limited_api`, setup.py) and
  the wheel is tagged `cp310-abi3`: one binary per platform serves every
  CPython from 3.10 on, 3.15 and the ones after it included. cffi already
  compiled against the limited API; abi3audit finds 0 violations.
- cibuildwheel 4.3.5 (CPython 3.15, abi3audit after the repair step) builds
  it once on cp310 and smoke-tests it on cp310–cp315.
- The `py3-none-any` wheel is NOT dropped: the pure-Python port stays the
  drvarma of every platform without a binary (David, 27-nov-2026), and it
  warns when it computes (BUG-0005). Making that warning reach sima's user is
  sima BUG-0020.
- Still without a binary: free-threaded builds (no limited API) and 32-bit
  Windows. They get the port, with its warning.
- The repro now also checks the abi3 configuration.
