"""Training-family registry; this module never imports held-out families."""

from __future__ import annotations

import importlib

TRAIN_FAMILIES = (
    "caravan",
    "foundry_ledger",
    "stallwright",
    "runeward",
    "kilnrite",
    "glyphgate",
    "loomfix",
    "ferrier",
    "burrowmaze",
    "gatepost",
)


def load(name: str):
    if name not in TRAIN_FAMILIES:
        raise KeyError(f"unknown training family: {name!r}")
    return importlib.import_module(f"{__name__}.{name}")
