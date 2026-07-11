"""tripleforge — held-out code induction, dependency control, and typed tools."""

from __future__ import annotations

import sys

from . import compound_core as core

FAMILY = "tripleforge"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True


class Episode(core.CodedToolEpisode):
    def __init__(self, seed: int, level: int):
        super().__init__(FAMILY, seed, level)


OraclePolicy = core.OraclePolicy


def selftest() -> dict:
    return core.selftest_module(
        sys.modules[__name__],
        (core.NoDiscoveryPolicy, core.NoControlPolicy, core.NoToolsPolicy),
    )
