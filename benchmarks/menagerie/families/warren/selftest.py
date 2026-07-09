import ast
import json
import random
import re
import sys
import time
from pathlib import Path

from . import family


LEVELS = (1, 2, 3, 4)
MODES = ("episode", "atom")


def main():
    gate_1_determinism()
    gate_2_seed_disjointness()
    gate_3_oracle_perfection()
    gate_4_random_floor()
    gate_5_degenerate_resistance()
    gate_6_monotone_difficulty()
    gate_7_budgets()
    gate_8_purity()
    gate_9_bare_answer_fallback()
    print("warren selftest: ALL GATES PASSED")


def gate_1_determinism():
    for level in LEVELS:
        for mode in MODES:
            left = json.dumps(family.generate(7, level, 6, mode), sort_keys=True)
            right = json.dumps(family.generate(7, level, 6, mode), sort_keys=True)
            assert left == right, f"gate 1 determinism failed for L{level} {mode}"
    print("gate 1 determinism: PASS")


def gate_2_seed_disjointness():
    tokens_7 = _tokens_for_seed(7)
    tokens_8 = _tokens_for_seed(8)
    overlap = tokens_7 & tokens_8
    assert not overlap, f"gate 2 seed disjointness failed; overlap sample: {sorted(overlap)[:5]}"
    print(f"gate 2 seed disjointness: PASS ({len(tokens_7)} vs {len(tokens_8)} tokens)")


def gate_3_oracle_perfection():
    for level in LEVELS:
        for mode in MODES:
            for item in family.generate(7, level, 8, mode):
                if mode == "episode":
                    result = _run_episode(item, "oracle", random.Random(3000 + level))
                else:
                    obs = family.Env(item).reset()
                    action = family.oracle_policy(item, [])
                    result = family.score(item, [{"obs": obs, "action": action}])
                assert result["score"] == 1.0, f"gate 3 oracle failed for {item['id']}: {result}"
    print("gate 3 oracle perfection: PASS")


def gate_4_random_floor():
    breakdown = {}
    scores = []
    for level in LEVELS:
        for mode in MODES:
            local = []
            items = family.generate(11, level, 24, mode)
            for idx, item in enumerate(items):
                rng = random.Random(4100 + level * 1000 + idx * 17 + (0 if mode == "episode" else 500))
                if mode == "episode":
                    result = _run_episode(item, "random", rng)
                else:
                    obs = family.Env(item).reset()
                    action = family.random_policy(item, [], rng)
                    result = family.score(item, [{"obs": obs, "action": action}])
                local.append(result["score"])
                scores.append(result["score"])
            breakdown[(level, mode)] = _mean(local)
    pooled = _mean(scores)
    assert pooled <= 0.15, f"gate 4 random floor failed: pooled mean {pooled:.4f} > 0.15"
    print(f"gate 4 random floor: PASS pooled_mean={pooled:.4f}")
    for level in LEVELS:
        print(
            f"  L{level}: episode={breakdown[(level, 'episode')]:.4f}, "
            f"atom={breakdown[(level, 'atom')]:.4f}"
        )


def gate_5_degenerate_resistance():
    means = {}
    for policy_name in ("empty", "constant", "echo"):
        scores = []
        for level in LEVELS:
            for mode in MODES:
                for item in family.generate(13, level, 18, mode):
                    if mode == "episode":
                        result = _run_episode(item, policy_name, random.Random(0))
                    else:
                        obs = family.Env(item).reset()
                        if policy_name == "empty":
                            action = ""
                        elif policy_name == "constant":
                            action = "ANSWER: 2"
                        else:
                            action = obs
                        result = family.score(item, [{"obs": obs, "action": action}])
                    scores.append(result["score"])
        means[policy_name] = _mean(scores)
        assert means[policy_name] <= 0.1, (
            f"gate 5 degenerate resistance failed for {policy_name}: "
            f"{means[policy_name]:.4f} > 0.1"
        )
    print(
        "gate 5 degenerate resistance: PASS "
        f"empty={means['empty']:.4f}, constant={means['constant']:.4f}, echo={means['echo']:.4f}"
    )


def gate_6_monotone_difficulty():
    level_means = []
    for level in LEVELS:
        scores = []
        for mode in MODES:
            items = family.generate(17, level, 36, mode)
            for rep in range(2):
                for idx, item in enumerate(items):
                    rng = random.Random(6100 + level * 10000 + rep * 1000 + idx * 29 + (0 if mode == "episode" else 300))
                    if mode == "episode":
                        result = _run_episode(item, "noisy", rng)
                    else:
                        obs = family.Env(item).reset()
                        if rng.random() < 0.5:
                            action = family.oracle_policy(item, [])
                        else:
                            action = family.random_policy(item, [], rng)
                        result = family.score(item, [{"obs": obs, "action": action}])
                    scores.append(result["score"])
        level_means.append(_mean(scores))
    for idx in range(len(level_means) - 1):
        assert level_means[idx] + 0.05 >= level_means[idx + 1], (
            "gate 6 monotone difficulty failed: "
            f"L{idx + 1}={level_means[idx]:.4f}, L{idx + 2}={level_means[idx + 1]:.4f}"
        )
    print(
        "gate 6 monotone difficulty: PASS "
        + ", ".join(f"L{level}={level_means[level - 1]:.4f}" for level in LEVELS)
    )


def gate_7_budgets():
    total_items = 0
    start_time = time.perf_counter()
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(19, level, 24, mode)
            total_items += len(items)
            for item in items:
                if mode == "atom":
                    assert item["max_turns"] == 1, f"gate 7 atom max_turns failed for {item['id']}"
                    prompt = family.Env(item).reset()
                    assert len(prompt) <= 1200, f"gate 7 atom prompt too long for {item['id']}: {len(prompt)}"
                    family.score(item, [{"obs": prompt, "action": family.oracle_policy(item, [])}])
                else:
                    cap = 4 if level in (1, 2) else 10 if level == 3 else 14
                    assert item["max_turns"] <= cap, f"gate 7 max_turns cap failed for {item['id']}"
                    env = family.Env(item)
                    obs = env.reset()
                    assert len(obs) <= 800, f"gate 7 episode observation too long for {item['id']}: {len(obs)}"
                    obs, done = env.step("NOPE")
                    assert len(obs) <= 800, f"gate 7 corrective observation too long for {item['id']}: {len(obs)}"
                    history = [{"obs": env.reset(), "action": family.oracle_policy(item, [])}]
                    family.score(item, history)
    for seed in (24**10 - 5, -(24**10 - 5)):
        atoms = family.generate(seed, 4, 2, "atom")
        episodes = family.generate(seed, 4, 2, "episode")
        total_items += len(atoms) + len(episodes)
        for item in atoms:
            prompt = family.Env(item).reset()
            assert len(prompt) <= 1200, f"gate 7 large-seed atom prompt too long for {item['id']}: {len(prompt)}"
        for item in episodes:
            for chamber in item["graph"]:
                obs = family._render_episode_obs(item, chamber, item["max_turns"], "No such tunnel here. ")
                assert len(obs) <= 800, f"gate 7 large-seed episode observation too long for {item['id']}: {len(obs)}"
    elapsed = time.perf_counter() - start_time
    ms_per_item = elapsed * 1000.0 / total_items
    assert ms_per_item < 50.0, f"gate 7 speed failed: {ms_per_item:.3f} ms/item >= 50"
    print(f"gate 7 budgets: PASS generation+scoring={ms_per_item:.3f} ms/item")


def gate_8_purity():
    path = Path(family.__file__)
    tree = ast.parse(path.read_text())
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in sys.stdlib_module_names:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            root = (node.module or "").split(".", 1)[0]
            if root not in sys.stdlib_module_names:
                bad.append(node.module or "")
    assert not bad, f"gate 8 purity failed; non-stdlib imports: {bad}"
    print("gate 8 purity: PASS")


def gate_9_bare_answer_fallback():
    scores = {"bare": [], "empty": [], "constant": [], "echo": []}
    for level in LEVELS:
        for item in family.generate(23, level, 18, "atom"):
            obs = family.Env(item).reset()
            oracle_out = family.oracle_policy(item, [])
            bare = re.sub(r"(?i)^\s*answer\s*:\s*", "", oracle_out)
            result = family.score(item, [{"obs": obs, "action": bare}])
            assert result["score"] == 1.0, f"gate 9 bare oracle failed for {item['id']}: {result}"
            scores["bare"].append(result["score"])

            for policy_name, action in (
                ("empty", ""),
                ("constant", "ANSWER: 2"),
                ("echo", obs),
            ):
                result = family.score(item, [{"obs": obs, "action": action}])
                scores[policy_name].append(result["score"])

    means = {policy_name: _mean(values) for policy_name, values in scores.items()}
    for policy_name in ("empty", "constant", "echo"):
        assert means[policy_name] <= 0.1, (
            f"gate 9 fallback resistance failed for {policy_name}: "
            f"{means[policy_name]:.4f} > 0.1"
        )
    print(
        "gate 9 bare answer fallback: PASS "
        f"bare={means['bare']:.4f}, empty={means['empty']:.4f}, "
        f"constant={means['constant']:.4f}, echo={means['echo']:.4f}"
    )


def _tokens_for_seed(seed):
    tokens = set()
    for level in LEVELS:
        for mode in MODES:
            for item in family.generate(seed, level, 6, mode):
                tokens.update(item["graph"])
                for tunnels in item["graph"].values():
                    tokens.update(tunnels)
    return tokens


def _run_episode(item, policy_name, rng):
    env = family.Env(item)
    obs = env.reset()
    assert len(obs) <= 800, f"episode observation too long for {item['id']}: {len(obs)}"
    history = []
    constant = None
    if policy_name == "constant":
        constant = "MOVE " + _first_tunnel_from_obs(obs)
    done = False
    while not done and len(history) < item["max_turns"]:
        if policy_name == "oracle":
            action = family.oracle_policy(item, history)
        elif policy_name == "random":
            action = family.random_policy(item, history, rng)
        elif policy_name == "noisy":
            if rng.random() < 0.5:
                action = family.oracle_policy(item, history)
            else:
                action = family.random_policy(item, history, rng)
        elif policy_name == "empty":
            action = ""
        elif policy_name == "constant":
            action = constant
        elif policy_name == "echo":
            action = obs
        else:
            raise AssertionError(f"unknown policy {policy_name}")
        history.append({"obs": obs, "action": action})
        obs, done = env.step(action)
        assert len(obs) <= 800, f"episode observation too long for {item['id']}: {len(obs)}"
    return family.score(item, history)


def _first_tunnel_from_obs(obs):
    for line in obs.splitlines():
        if line.startswith("- "):
            return line.split()[1]
    raise AssertionError("no tunnel token found in observation")


def _mean(values):
    assert values, "cannot average empty values"
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
