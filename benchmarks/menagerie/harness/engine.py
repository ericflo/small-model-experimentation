"""Lockstep execution core for Menagerie benchmark families."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Callable

from . import parsing


_REQUIRED_ATTRS = ("META", "generate", "Env", "score", "oracle_policy", "random_policy")
_REQUIRED_META_KEYS = ("name", "capability", "paradigm", "action_format")


def discover_families(families_dir) -> dict[str, ModuleType]:
    """Load all contract-conformant family modules from a families directory."""

    root = Path(families_dir)
    found: dict[str, ModuleType] = {}
    if not root.exists():
        return found

    for family_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = family_dir.name
        if name.startswith(".") or name.startswith("_"):
            continue
        family_py = family_dir / "family.py"
        if not family_py.exists():
            continue
        module_name = f"menagerie_family_{name}"
        spec = importlib.util.spec_from_file_location(module_name, family_py)
        if spec is None or spec.loader is None:
            raise ValueError(f"family {name}: could not load {family_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for attr in _REQUIRED_ATTRS:
            if not hasattr(module, attr):
                raise ValueError(f"family {name}: missing required attribute {attr}")
        if not isinstance(module.META, dict):
            raise ValueError(f"family {name}: META must be a dict")
        for key in _REQUIRED_META_KEYS:
            if key not in module.META:
                raise ValueError(f"family {name}: META missing key {key}")
        found[name] = module
    return found


@dataclass
class EpisodeState:
    family: str
    module: ModuleType
    item: dict
    env: object
    pending_obs: str
    history: list[dict]
    turns_used: int
    max_turns: int
    done: bool
    score: dict | None


def build_episodes(
    family_name: str,
    module: ModuleType,
    items: list[dict],
    episode_max_turns: int | None = None,
) -> list[EpisodeState]:
    """Create reset episode states for generated items."""

    episodes: list[EpisodeState] = []
    for item in items:
        env = module.Env(item)
        pending_obs = env.reset()
        if item["mode"] == "atom":
            max_turns = 1
        elif episode_max_turns:
            max_turns = min(item["max_turns"], episode_max_turns)
        else:
            max_turns = item["max_turns"]
        episodes.append(
            EpisodeState(
                family=family_name,
                module=module,
                item=item,
                env=env,
                pending_obs=pending_obs,
                history=[],
                turns_used=0,
                max_turns=max_turns,
                done=False,
                score=None,
            )
        )
    return episodes


def run_lockstep(episodes: list[EpisodeState], batch_act: Callable[[list[dict]], list[str]]) -> dict:
    """Run all episodes in batched lockstep until completion."""

    rounds = 0
    wall_by_family: dict[str, float] = {}

    while True:
        live = [episode for episode in episodes if not episode.done]
        if not live:
            break

        contexts = [
            {
                "family": episode.family,
                "module": episode.module,
                "item": episode.item,
                "item_id": episode.item["id"],
                "mode": episode.item["mode"],
                "meta": episode.module.META,
                "history": episode.history,
                "obs": episode.pending_obs,
                "turn_index": episode.turns_used,
                "max_turns": episode.max_turns,
            }
            for episode in live
        ]

        t0 = perf_counter()
        actions = batch_act(contexts)
        dt = perf_counter() - t0
        if len(actions) != len(live):
            raise ValueError(f"batch_act returned {len(actions)} actions for {len(live)} contexts")

        if live:
            per_context = dt / len(live)
            for episode in live:
                wall_by_family[episode.family] = wall_by_family.get(episode.family, 0.0) + per_context

        for episode, raw_action in zip(live, actions):
            action = parsing.canonical_action(raw_action, episode.item["mode"])
            next_obs, done = episode.env.step(action)
            episode.history.append({"obs": episode.pending_obs, "action": action})
            episode.turns_used += 1
            episode.pending_obs = next_obs
            if done or episode.turns_used >= episode.max_turns:
                episode.done = True
                episode.score = episode.module.score(episode.item, episode.history)
        rounds += 1

    per_item = []
    for episode in episodes:
        if episode.score is None:
            episode.score = episode.module.score(episode.item, episode.history)
        per_item.append(
            {
                "family": episode.family,
                "id": episode.item["id"],
                "level": episode.item["level"],
                "mode": episode.item["mode"],
                "turns": episode.turns_used,
                "max_turns": episode.max_turns,
                "score": float(episode.score["score"]),
                "score_detail": episode.score,
                "transcript": episode.history,
            }
        )

    return {"per_item": per_item, "per_family_wall_s": wall_by_family, "rounds": rounds}
