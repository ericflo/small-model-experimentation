import ast
import json
import random
import sys
import time
from pathlib import Path

from . import family


sys.dont_write_bytecode = True


_MODES = ("atom", "episode")
_LEVELS = (1, 2, 3, 4)
_EPISODE_MAX = {1: 4, 2: 4, 3: 10, 4: 14}


def main():
    _gate_1_determinism()
    _gate_2_seed_disjointness()
    _gate_3_oracle_perfection()
    random_means = _gate_4_random_floor()
    _gate_5_degenerate_resistance()
    noisy_means = _gate_6_monotone_difficulty()
    ms_per_item = _gate_7_budgets()
    _gate_8_purity()
    print(_format_means("random_means", random_means))
    print(_format_means("noisy_means", noisy_means))
    print("budget_ms_per_item %.3f" % ms_per_item)
    print("SELFTEST PASS rites")


def _all_items(seed, n):
    out = []
    for level in _LEVELS:
        for mode in _MODES:
            out.extend(family.generate(seed, level, n, mode))
    return out


def _jsonable(items):
    return json.dumps(items, sort_keys=True, separators=(",", ":"))


def _gate_1_determinism():
    for level in _LEVELS:
        for mode in _MODES:
            a = family.generate(7, level, 6, mode)
            b = family.generate(7, level, 6, mode)
            assert _jsonable(a) == _jsonable(
                b
            ), "Gate 1 Determinism: generate(7, %s, 6, %s) changed" % (level, mode)


def _gate_2_seed_disjointness():
    left = _all_items(7, 6)
    right = _all_items(8, 6)
    left_ids = {item["id"] for item in left}
    right_ids = {item["id"] for item in right}
    assert not (
        left_ids & right_ids
    ), "Gate 2 Seed disjointness: seeds 7 and 8 share item ids"
    left_obs = {family.Env(item).reset() for item in left}
    right_obs = {family.Env(item).reset() for item in right}
    assert not (
        left_obs & right_obs
    ), "Gate 2 Seed disjointness: seeds 7 and 8 share reset observations"


def _gate_3_oracle_perfection():
    for item in _all_items(7, 6):
        transcript, _seen = _rollout(item, lambda it, hist, obs: family.oracle_policy(it, hist))
        result = family.score(item, transcript)
        assert (
            result["score"] == 1.0
        ), "Gate 3 Oracle perfection: %s scored %.6f" % (item["id"], result["score"])


def _gate_4_random_floor():
    rng = random.Random(123)
    means = {}
    all_scores = []
    for level in _LEVELS:
        scores = []
        for mode in _MODES:
            for item in family.generate(23, level, 80, mode):
                transcript, _seen = _rollout(
                    item, lambda it, hist, obs: family.random_policy(it, hist, rng)
                )
                scores.append(family.score(item, transcript)["score"])
        means[level] = sum(scores) / len(scores)
        all_scores.extend(scores)
    overall = sum(all_scores) / len(all_scores)
    assert overall <= 0.05, (
        "Gate 4 Random floor: overall %.6f > 0.05; %s"
        % (overall, _format_means("per_level", means))
    )
    return means


def _gate_5_degenerate_resistance():
    policies = [
        (
            "empty-string policy",
            lambda it, hist, obs: "",
        ),
        (
            "constant most-plausible-answer policy",
            lambda it, hist, obs: (
                "ANSWER: " + it["_actions"][0]
                if it["mode"] == "atom"
                else "ENACT " + it["_actions"][0]
            ),
        ),
        (
            "echo-the-observation policy",
            lambda it, hist, obs: obs,
        ),
    ]
    items = _all_items(29, 32)
    for name, policy in policies:
        scores = []
        for item in items:
            transcript, _seen = _rollout(item, policy)
            scores.append(family.score(item, transcript)["score"])
        mean = sum(scores) / len(scores)
        assert mean <= 0.1, (
            "Gate 5 Degenerate resistance: %s mean %.6f > 0.1" % (name, mean)
        )


def _gate_6_monotone_difficulty():
    rng = random.Random(456)
    means = {}
    for level in _LEVELS:
        scores = []
        for item in family.generate(31, level, 96, "episode"):
            def choose(it, hist, obs):
                if rng.random() < 0.5:
                    return family.oracle_policy(it, hist)
                return family.random_policy(it, hist, rng)

            transcript, _seen = _rollout(item, choose)
            scores.append(family.score(item, transcript)["score"])
        means[level] = sum(scores) / len(scores)
    for earlier, later in zip(_LEVELS, _LEVELS[1:]):
        assert means[later] <= means[earlier] + 0.05, (
            "Gate 6 Monotone difficulty: %s"
            % _format_means("noisy_means", means)
        )
    return means


def _gate_7_budgets():
    rng = random.Random(789)
    for item in _all_items(37, 10):
        initial = family.Env(item).reset()
        if item["mode"] == "atom":
            assert item["max_turns"] == 1, (
                "Gate 7 Budgets: atom max_turns for %s is %s" % (item["id"], item["max_turns"])
            )
            assert len(initial) <= 1200, (
                "Gate 7 Budgets: atom prompt for %s is %s chars" % (item["id"], len(initial))
            )
        else:
            assert item["max_turns"] == _EPISODE_MAX[item["level"]], (
                "Gate 7 Budgets: episode max_turns for %s is %s"
                % (item["id"], item["max_turns"])
            )
            for policy_name, policy in (
                ("oracle", lambda it, hist, obs: family.oracle_policy(it, hist)),
                ("random", lambda it, hist, obs: family.random_policy(it, hist, rng)),
            ):
                _transcript, seen = _rollout(item, policy)
                for obs in seen:
                    assert len(obs) <= 800, (
                        "Gate 7 Budgets: %s observation for %s is %s chars"
                        % (policy_name, item["id"], len(obs))
                    )

    start = time.perf_counter()
    count = 0
    for level in _LEVELS:
        for mode in _MODES:
            for item in family.generate(41, level, 12, mode):
                transcript, _seen = _rollout(
                    item, lambda it, hist, obs: family.oracle_policy(it, hist)
                )
                family.score(item, transcript)
                count += 1
    ms_per_item = ((time.perf_counter() - start) * 1000.0) / float(count)
    assert ms_per_item < 50.0, (
        "Gate 7 Budgets: generation+oracle+scoring %.3f ms/item >= 50" % ms_per_item
    )
    return ms_per_item


def _gate_8_purity():
    path = Path(family.__file__)
    tree = ast.parse(path.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {"collections", "random", "re"}
    assert imports <= allowed, (
        "Gate 8 Purity: non-stdlib or unapproved imports in family.py: %s"
        % sorted(imports - allowed)
    )


def _rollout(item, choose_action):
    env = family.Env(item)
    obs = env.reset()
    history = []
    seen = [obs]
    for _turn in range(item["max_turns"]):
        action = choose_action(item, history, obs)
        next_obs, done = env.step(action)
        history.append({"obs": obs, "action": action})
        seen.append(next_obs)
        obs = next_obs
        if done:
            break
    return history, seen


def _format_means(label, means):
    return "%s " % label + " ".join("L%s=%.6f" % (level, means[level]) for level in _LEVELS)


if __name__ == "__main__":
    main()
