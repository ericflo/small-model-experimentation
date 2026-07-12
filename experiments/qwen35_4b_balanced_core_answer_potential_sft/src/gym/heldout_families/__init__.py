"""Evaluation-only held-family registry."""

from __future__ import annotations

import importlib

HELDOUT_FAMILIES = ("brinework", "spindle")


def load(name: str):
    if name not in HELDOUT_FAMILIES:
        raise KeyError(f"unknown held-out family: {name!r}")
    return importlib.import_module(f"{__name__}.{name}")
