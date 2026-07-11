"""Shared gym plumbing: answer extraction, deterministic RNG, selftest harness.

Every family module in ``gym/families/`` builds against this file and nothing
else. Keep it dependency-free (stdlib only) so generators, verifiers, and
selftests run on CPU under any python3.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Callable, Iterable

ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", re.IGNORECASE)

# The repo's generation protocol stops on <|endoftext|> (the pinned HF model
# EOS) and deliberately generates THROUGH <|im_end|>, so decoded text carries
# literal terminal markers. Every scorer and every SFT target must cut the
# answer region at the first marker (precedent: verified_macro_invention
# model_harness._TERMINAL_MARKERS).
TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")


def strip_terminal_markers(text: str) -> str:
    for marker in TERMINAL_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    return text

# Public menagerie family names must never appear in gym content (firewall
# hygiene: gym vocabulary is disjoint from the instrument's public names).
FORBIDDEN_WORDS = (
    "chronicle",
    "lockpick",
    "menders",
    "mirage",
    "rites",
    "siftstack",
    "sirens",
    "stockade",
    "toolsmith",
    "warren",
    "menagerie",
)

ATOM_PROMPT_CHAR_LIMIT = 1400
# Frontier levels (L5+) deliberately train harder-than-test: more entities
# and longer streams need more prompt room than the deployed-atom shape.
ATOM_PROMPT_CHAR_LIMIT_FRONTIER = 2300
EPISODE_OBS_CHAR_LIMIT = 800


def atom_prompt_limit(level: int) -> int:
    return ATOM_PROMPT_CHAR_LIMIT if level <= 4 else ATOM_PROMPT_CHAR_LIMIT_FRONTIER


ATOM_ANSWER_INSTRUCTION = (
    "End your reply with exactly one final line of the form:\n"
    "ANSWER: <value>"
)
EPISODE_ACTION_INSTRUCTION = (
    "Reply with exactly one action line and nothing else."
)


def rng_for(*parts: Any) -> random.Random:
    """A deterministic RNG namespaced by arbitrary hashable parts.

    Families derive ALL randomness from this; no global RNG, no wall clock.
    """
    key = json.dumps(parts, sort_keys=True, default=str)
    return random.Random(key)


def split_think(text: str) -> tuple[str, str]:
    """Split a completed generation into (thinking, answer_region).

    The runner emits ``<thinking...></think>\\n\\nanswer``; when thinking is
    disabled the whole text is the answer region. The answer region is cut at
    the first terminal marker so scorers and SFT targets never see them.
    """
    marker = "</think>"
    if marker in text:
        head, _, tail = text.rpartition(marker)
        return head, strip_terminal_markers(tail)
    return "", strip_terminal_markers(text)


def extract_answer(reply_text: str) -> str | None:
    """Return the value of the LAST ``ANSWER: <value>`` line, else None.

    Operates on the answer region only (thinking is ignored), case-insensitive
    and whitespace-tolerant, mirroring the terse-atom protocol.
    """
    _, region = split_think(reply_text)
    value = None
    for line in region.splitlines():
        match = ANSWER_RE.match(line)
        if match:
            value = match.group(1).strip()
    return value


def extract_action(reply_text: str) -> str:
    """Return the first non-empty line of the answer region, stripped.

    Episode protocol: the model must reply with exactly one action line; we
    take the first non-empty line and let the family judge its grammar.
    """
    _, region = split_think(reply_text)
    for line in region.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def canon_int(value: str) -> int | None:
    """Canonicalize an integer answer ('12', ' 12.', '12 crates' -> 12)."""
    match = re.search(r"-?\d+", value.replace(",", ""))
    return int(match.group(0)) if match else None


def canon_word(value: str) -> str:
    """Canonicalize a single-word answer: lowercase, strip punctuation."""
    return re.sub(r"[^a-z0-9_\- ]", "", value.strip().lower()).strip()


def canon_list(value: str) -> list[str]:
    """Canonicalize a comma/space separated list of tokens (order kept)."""
    parts = re.split(r"[,\s]+", value.strip())
    return [canon_word(part) for part in parts if canon_word(part)]


def score_exact_int(gold: int, reply_text: str) -> float:
    answer = extract_answer(reply_text)
    if answer is None:
        return 0.0
    return 1.0 if canon_int(answer) == gold else 0.0


def score_exact_word(gold: str, reply_text: str) -> float:
    answer = extract_answer(reply_text)
    if answer is None:
        return 0.0
    return 1.0 if canon_word(answer) == canon_word(gold) else 0.0


# ---------------------------------------------------------------------------
# Selftest harness helpers
# ---------------------------------------------------------------------------


class SelftestError(AssertionError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SelftestError(message)


def selftest_atoms(
    module: Any,
    *,
    n_per_level: int = 40,
    degenerate_replies: Iterable[str] = ("", "ANSWER: 0", "ANSWER: yes"),
    oracle_min: float = 0.95,
    degenerate_max: float = 0.15,
) -> dict[str, Any]:
    """Standard atom-family selftest; returns a stats dict on success."""
    stats: dict[str, Any] = {"family": module.FAMILY, "levels": {}}
    for level in module.LEVELS:
        items_a = module.gen_atoms(7, level, n_per_level)
        items_b = module.gen_atoms(7, level, n_per_level)
        _check(
            json.dumps(items_a, sort_keys=True) == json.dumps(items_b, sort_keys=True),
            f"{module.FAMILY} L{level}: generator is not deterministic",
        )
        items_c = module.gen_atoms(8, level, n_per_level)
        _check(
            json.dumps(items_a, sort_keys=True) != json.dumps(items_c, sort_keys=True),
            f"{module.FAMILY} L{level}: seeds 7 and 8 generated identical items",
        )
        ids = [item["id"] for item in items_a]
        _check(len(ids) == len(set(ids)), f"{module.FAMILY} L{level}: duplicate item ids")
        json.dumps(items_a)  # JSON-safety

        oracle_scores = []
        for item in items_a:
            _check(
                len(item["prompt"]) <= atom_prompt_limit(level),
                f"{module.FAMILY} L{level} {item['id']}: prompt exceeds "
                f"{atom_prompt_limit(level)} chars ({len(item['prompt'])})",
            )
            lowered = item["prompt"].lower()
            for word in FORBIDDEN_WORDS:
                _check(word not in lowered, f"{module.FAMILY}: forbidden word {word!r} in prompt")
            oracle_reply = module.oracle_atom(item)
            oracle_scores.append(module.score_atom(item, oracle_reply))
            # Marker tolerance: the runner's decoded text carries terminal
            # markers; a polluted-but-correct reply must still score.
            polluted = module.score_atom(item, oracle_reply + "<|im_end|>\njunk after")
            _check(
                polluted >= module.score_atom(item, oracle_reply),
                f"{module.FAMILY} L{level} {item['id']}: terminal-marker pollution changes score",
            )
        oracle_mean = sum(oracle_scores) / len(oracle_scores)
        _check(
            oracle_mean >= oracle_min,
            f"{module.FAMILY} L{level}: oracle mean {oracle_mean:.3f} < {oracle_min}",
        )

        degenerate_means = []
        for reply in degenerate_replies:
            scores = [module.score_atom(item, reply) for item in items_a]
            degenerate_means.append(sum(scores) / len(scores))
        worst = max(degenerate_means)
        _check(
            worst <= degenerate_max,
            f"{module.FAMILY} L{level}: degenerate reply scores {worst:.3f} > {degenerate_max}",
        )
        stats["levels"][level] = {
            "oracle_mean": round(oracle_mean, 4),
            "degenerate_max": round(worst, 4),
        }
    return stats


def selftest_episodes(
    module: Any,
    *,
    n_per_level: int = 12,
    oracle_min: float = 0.95,
    random_max: float = 0.15,
) -> dict[str, Any]:
    """Standard episode-family selftest; returns a stats dict on success."""
    stats: dict[str, Any] = {"family": module.FAMILY, "levels": {}}
    for level in module.LEVELS:
        oracle_scores: list[float] = []
        random_scores: list[float] = []
        for index in range(n_per_level):
            seed = 1000 + index
            episode = module.Episode(seed, level)
            twin = module.Episode(seed, level)
            _check(
                json.dumps(episode.spec, sort_keys=True) == json.dumps(twin.spec, sort_keys=True),
                f"{module.FAMILY} L{level}: episode spec not deterministic",
            )
            json.dumps(episode.spec)
            _check(
                len(episode.initial_observation()) <= EPISODE_OBS_CHAR_LIMIT,
                f"{module.FAMILY} L{level}: initial observation too long",
            )
            lowered = (episode.system_prompt() + episode.initial_observation()).lower()
            for word in FORBIDDEN_WORDS:
                _check(word not in lowered, f"{module.FAMILY}: forbidden word {word!r} in episode")

            # Action-validity flag semantics: garbage must set last_action_ok
            # False; the oracle's first action must set it True.
            flag_probe = module.Episode(seed, level)
            flag_probe.step("XYZZY 42 GARBAGE")
            _check(
                getattr(flag_probe, "last_action_ok", None) is False,
                f"{module.FAMILY} L{level}: last_action_ok not False after garbage action",
            )
            flag_probe2 = module.Episode(seed, level)
            oracle_first = module.OraclePolicy(flag_probe2).act(
                [flag_probe2.initial_observation()]
            )
            flag_probe2.step(oracle_first)
            _check(
                getattr(flag_probe2, "last_action_ok", None) is True,
                f"{module.FAMILY} L{level}: last_action_ok not True after oracle action",
            )

            oracle_scores.append(
                _run_episode(module.Episode(seed, level), module.OraclePolicy)
            )
            random_scores.append(
                _run_episode(module.Episode(seed, level), _RandomActionPolicy)
            )
        oracle_mean = sum(oracle_scores) / len(oracle_scores)
        random_mean = sum(random_scores) / len(random_scores)
        _check(
            oracle_mean >= oracle_min,
            f"{module.FAMILY} L{level}: oracle episode mean {oracle_mean:.3f} < {oracle_min}",
        )
        _check(
            random_mean <= random_max,
            f"{module.FAMILY} L{level}: random episode mean {random_mean:.3f} > {random_max}",
        )
        stats["levels"][level] = {
            "oracle_mean": round(oracle_mean, 4),
            "random_mean": round(random_mean, 4),
        }
    return stats


class _RandomActionPolicy:
    """Emits short garbage actions; the floor every episode family must beat."""

    def __init__(self, episode: Any):
        self._rng = rng_for("random-policy", episode.spec)

    def act(self, observation_history: list[str]) -> str:
        verbs = ["DO", "GO", "CALL", "PROBE", "PATCH", "READ", "SET", "OPEN", "RUN"]
        return f"{self._rng.choice(verbs)} {self._rng.randint(0, 9)}"


def _run_episode(episode: Any, policy_class: Callable[[Any], Any]) -> float:
    policy = policy_class(episode)
    history = [episode.initial_observation()]
    for _ in range(episode.max_turns):
        action = policy.act(history)
        observation, done = episode.step(action)
        _check(
            len(observation) <= EPISODE_OBS_CHAR_LIMIT,
            f"{episode.__class__.__module__}: observation exceeds "
            f"{EPISODE_OBS_CHAR_LIMIT} chars",
        )
        lowered = observation.lower()
        for word in FORBIDDEN_WORDS:
            _check(
                word not in lowered,
                f"{episode.__class__.__module__}: forbidden word {word!r} in observation",
            )
        history.append(observation)
        if done:
            break
    score = episode.score()
    _check(0.0 <= score <= 1.0, "episode score outside [0,1]")
    return score
