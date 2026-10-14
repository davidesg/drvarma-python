---
id: BUG-0022
title: The Windows wheels carry no compiled engine — GSL is never installed for the Windows build, the optional extension fails in silence, and the smoke test skips
status: open
severity: high
component: packaging
found_in: 0.1.7
fixed_in:
reported: 2026-10-14
reporter: David / Claude — atsw 2.0.0 install preflight on every wheel platform
tags: [packaging, windows, wheels, engine]
references:
  - pyproject.toml [tool.cibuildwheel.windows]
  - src/drvarma/_build_cffi.py (GSL dirs)
  - tests/test_smoke.py test_a_binary_wheel_can_actually_load_its_engine
---

## Summary

Every `win_amd64` wheel of drvarma ships without `drvarma/_drvarma_engine*.pyd`.
On Windows drvarma computes with the pure-Python port, 20–250x slower (the
ladder ~250x), and says so only in a RuntimeWarning. Measured on the published
0.1.7 and on 0.2.0 built from `ladder`.

## Impact

High for Windows users: the engine the wheel exists to deliver is not there,
and sima's ladder runs on the port. Results are the same; time is not.
Linux (manylinux, musllinux) and macOS wheels load the engine.

## Reproduction

`python3 bugs/BUG-0022-repro/repro.py` — downloads the latest `win_amd64`
wheel from PyPI and looks for the extension; exits 1 while it has none.

## Root cause

Three things together:
- `[tool.cibuildwheel.windows]` installs no GSL (fue's does, with vcpkg);
- `_build_cffi.py` has no Windows search path for GSL (fue's reads `GSL_ROOT`
  and the vcpkg default);
- the extension is `optional`, so its compile error does not fail the build,
  and the smoke test SKIPS when the extension is missing ("pure-Python build").
  The CI log said «5 passed, 1 skipped».

## Fix

- `[tool.cibuildwheel.windows]`: `before-all = "vcpkg install
  gsl:x64-windows-static-md"` and `GSL_ROOT`, as fue.
- `_build_cffi.py`: on win32 the GSL dirs come from `GSL_ROOT` or the vcpkg
  default.
- `test-environment = { DRVARMA_REQUIRE_ENGINE = "1" }` for cibuildwheel, and
  the smoke test FAILS instead of skipping when it is set and the extension is
  missing. A binary wheel without its engine cannot pass again.

## Validation

The atsw 2.0.0 preflight (art-python, branch `preflight-atsw-2.0.0`) on
Windows, Python 3.10–3.13: `drvarma: motor C` OK. The repro exits 0 once a
fixed wheel is on PyPI.
