"""Where the engine's bug register is, installed or in the working tree.

The installed wheel carries a copy in `drvarma/material/bugs/`
(tools/sync_material.py); in the repository the original is `bugs/`. The copy
is looked for first, then the original, and None says that neither is there,
which a caller must report as such, not as "no defects" (art BUG-0125).
"""
from __future__ import annotations

import os


def bugs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "material", "bugs"),
              os.path.join(os.path.dirname(os.path.dirname(here)), "bugs")):
        if os.path.isdir(d) and any(f.startswith("BUG-") for f in os.listdir(d)):
            return d
    return None
