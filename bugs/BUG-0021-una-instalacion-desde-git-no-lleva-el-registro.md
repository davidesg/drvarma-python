---
id: BUG-0021
title: An install from git carries no bug register; sima's engine-defects resource says it is missing
status: fixed
severity: low
component: packaging
found_in: 0.2.0.dev
fixed_in: 0.2.0
reported: 2026-10-13
reporter: David / Claude — sima's CI against drvarma@ladder
tags: [packaging, register, sima]
references:
  - setup.py (_sync_material)
  - tools/sync_material.py
  - src/drvarma/register.py
  - sima: tests/test_resources.py
---

## Summary

The wheel carries the register as a copy in `src/drvarma/material/bugs/`,
made by `tools/sync_material.py` and ignored by git. `pip install
"drvarma @ git+...@ladder"` builds from a fresh clone where nobody ran the
tool, so the installed package has no `material/bugs/` and `bugs_dir()`
returns None. sima then serves "this installation carries no the drvarma
register", and its `test_resources` fails in CI (two tests).

## Impact

Every install from the repository (sima's CI, any user installing from git),
and any wheel built without running the tool first. Local editable installs
did not show it: `bugs_dir()` falls back to the checkout's `bugs/`.

## Root cause

The copy depended on a manual step before the build.

## Fix

`setup.py` runs the sync whenever `bugs/` is there, that is, on any build
from a checkout. An sdist already carries the copy (package data) and no
`bugs/`, so it is left alone.

## Validation

`pip install` of a fresh clone into a clean venv: `bugs_dir()` points into
site-packages, with the register's files. sima's CI, `test_resources`.
