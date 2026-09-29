#!/usr/bin/env python3
"""Copy `bugs/` to `src/drvarma/material/bugs/`, which the wheel distributes.

The bug register is the engine's memory of why things are as they are, and
sima serves it as the MCP resource `sima://engine-defects` (the defects of the
ladder live here, not in sima). A wheel only carries `src/`, so without this
copy an installation would have no register to serve: the lesson of art's
BUG-0125, where the resources existed, could be asked for, and answered with
nothing. `bugs/` stays the only original; the copy is generated and ignored by
git. Run it before building; `tests/test_material.py` fails if it is stale.
"""
from __future__ import annotations

import filecmp
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "src", "drvarma", "material", "bugs")


def sync():
    src = os.path.join(ROOT, "bugs")
    os.makedirs(DEST, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.endswith(".md"))
    copied = 0
    for f in names:
        o, d = os.path.join(src, f), os.path.join(DEST, f)
        if not (os.path.exists(d) and filecmp.cmp(o, d, shallow=False)):
            shutil.copy2(o, d)
            copied += 1
    for f in os.listdir(DEST):
        if f not in names:
            os.remove(os.path.join(DEST, f))
    return len(names), copied


if __name__ == "__main__":
    n, c = sync()
    print(f"bugs: {n} files, {c} copied -> {os.path.relpath(DEST, ROOT)}")
