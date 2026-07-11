"""mazeferry — partial-map navigation composed with typed tool chaining."""

from __future__ import annotations

import sys

from . import compound_core as core

FAMILY = "mazeferry"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True


class Episode(core.MazeToolEpisode):
    def __init__(self, seed: int, level: int):
        super().__init__(FAMILY, seed, level)


OraclePolicy = core.OraclePolicy


def selftest() -> dict:
    return core.selftest_module(sys.modules[__name__], (core.NoNavigationPolicy, core.NoToolsPolicy))
