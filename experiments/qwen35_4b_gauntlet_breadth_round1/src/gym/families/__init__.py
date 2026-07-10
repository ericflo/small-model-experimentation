"""Gym family registry.

Import modules lazily so a broken family fails loudly at selftest time
without taking down the whole registry.
"""

from __future__ import annotations

import importlib

TRAINED_FAMILIES = (
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
HELDOUT_FAMILIES = (
    "brinework",
    "spindle",
)
ALL_FAMILIES = TRAINED_FAMILIES + HELDOUT_FAMILIES


def load(name: str):
    if name not in ALL_FAMILIES:
        raise KeyError(f"unknown gym family: {name!r}")
    return importlib.import_module(f"{__name__}.{name}")
