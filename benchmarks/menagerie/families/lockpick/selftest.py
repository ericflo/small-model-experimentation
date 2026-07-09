import ast
import json
import random
import sys
import time
from pathlib import Path

from . import family


def main():
    gate_determinism()
    gate_seed_disjointness()
    gate_oracle_perfection()
    gate_random_floor()
    gate_degenerate_resistance()
    gate_bare_answer_fallback()
    noisy_means = gate_monotone_difficulty()
    gate_budgets()
    gate_purity()
    print("lockpick selftest: ALL 9 GATES PASSED")
    print(
        "lockpick noisy-oracle means: "
        + ", ".join(f"L{level}={noisy_means[level]:.4f}" for level in range(1, 5))
    )


def gate_determinism():
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            first = json.dumps(family.generate(7, level, 6, mode), sort_keys=True, separators=(",", ":"))
            second = json.dumps(family.generate(7, level, 6, mode), sort_keys=True, separators=(",", ":"))
            assert first == second, (
                "Determinism: generate(7, L, 6, mode) twice -> identical JSON, all levels/modes. "
                f"level={level} mode={mode} item=aggregate observed=mismatch"
            )
    print("gate 1 Determinism: passed")


def gate_seed_disjointness():
    sigs = {}
    ids = {}
    for seed in (7, 8):
        sigs[seed] = set()
        ids[seed] = set()
        for level in range(1, 5):
            for mode in ("atom", "episode"):
                for item in family.generate(seed, level, 6, mode):
                    sigs[seed].add(_content_signature(item))
                    ids[seed].add(item["id"])
    sig_overlap = sigs[7] & sigs[8]
    id_overlap = ids[7] & ids[8]
    assert not sig_overlap, (
        "Seed disjointness: seeds 7 vs 8 share no item content. "
        f"level=all mode=all item=aggregate observed={len(sig_overlap)} content overlaps"
    )
    assert not id_overlap, (
        "Seed disjointness: seeds 7 vs 8 share no item content. "
        f"level=all mode=all item=aggregate observed={len(id_overlap)} id overlaps"
    )
    print("gate 2 Seed disjointness: passed")


def gate_oracle_perfection():
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in family.generate(7, level, 6, mode):
                if mode == "atom":
                    obs = family.Env(item).reset()
                    action = family.oracle_policy(item, [])
                    transcript = [{"obs": obs, "action": action}]
                else:
                    transcript = _drive_episode(item, lambda current_item, history: family.oracle_policy(current_item, history))
                result = family.score(item, transcript)
                assert result["score"] == 1.0, (
                    "Oracle perfection: oracle_policy scores 1.0 on every item (all levels, both modes). "
                    f"level={level} mode={mode} item={item['id']} observed={result}"
                )
    print("gate 3 Oracle perfection: passed")


def gate_random_floor():
    rng = random.Random(4404)
    scores = []
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in family.generate(17, level, 12, mode):
                if mode == "atom":
                    obs = family.Env(item).reset()
                    action = family.random_policy(item, [], rng)
                    transcript = [{"obs": obs, "action": action}]
                else:
                    transcript = _drive_episode(
                        item,
                        lambda current_item, history, local_rng=rng: family.random_policy(
                            current_item, history, local_rng
                        ),
                    )
                scores.append(family.score(item, transcript)["score"])
    mean = sum(scores) / len(scores)
    assert mean <= 0.05, (
        "Random floor: mean random_policy score <= 0.05 (or <= 0.15 with justification in README for ratio-scored families). "
        f"level=all mode=all item=aggregate observed={mean:.4f}"
    )
    print(f"gate 4 Random floor: passed mean={mean:.4f}")


def gate_degenerate_resistance():
    policies = {
        "empty-string": _empty_action,
        "constant-target": _constant_target_action,
        "echo-observation": None,
    }
    for name, policy in policies.items():
        scores = []
        for level in range(1, 5):
            for mode in ("atom", "episode"):
                for item in family.generate(23, level, 6, mode):
                    if name == "echo-observation":
                        transcript = _drive_echo(item)
                    elif mode == "atom":
                        obs = family.Env(item).reset()
                        action = policy(item, [])
                        transcript = [{"obs": obs, "action": action}]
                    else:
                        transcript = _drive_episode(item, policy)
                    scores.append(family.score(item, transcript)["score"])
        mean = sum(scores) / len(scores)
        assert mean <= 0.1, (
            "Degenerate resistance: empty-string policy, constant most-plausible-answer policy, and echo-the-observation policy each score <= 0.1 mean. "
            f"level=all mode=all item={name} observed={mean:.4f}"
        )
    print("gate 5 Degenerate resistance: passed")


def gate_bare_answer_fallback():
    scores = []
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in family.generate(29, level, 6, mode):
                if mode == "atom":
                    obs = family.Env(item).reset()
                    action = _bare_answer_action(item, [])
                    transcript = [{"obs": obs, "action": action}]
                else:
                    transcript = _drive_episode(item, _bare_answer_action)
                result = family.score(item, transcript)
                assert result["score"] == 1.0, (
                    "Bare-answer fallback: bare-answer oracle (no ANSWER: prefix) scores 1.0 on every item. "
                    f"level={level} mode={mode} item={item['id']} observed={result}"
                )
                scores.append(result["score"])
    degenerates = {
        "empty-string": _empty_action,
        "constant-target": _constant_target_action,
        "echo-observation": None,
    }
    for name, policy in degenerates.items():
        dscores = []
        for level in range(1, 5):
            for mode in ("atom", "episode"):
                for item in family.generate(29, level, 6, mode):
                    if name == "echo-observation":
                        transcript = _drive_echo(item)
                    elif mode == "atom":
                        obs = family.Env(item).reset()
                        transcript = [{"obs": obs, "action": policy(item, [])}]
                    else:
                        transcript = _drive_episode(item, policy)
                    dscores.append(family.score(item, transcript)["score"])
        dmean = sum(dscores) / len(dscores)
        assert dmean <= 0.1, (
            "Bare-answer fallback: adding the bare fallback must not lift empty/echo/constant policies above 0.1. "
            f"level=all mode=all item={name} observed={dmean:.4f}"
        )
    print(f"gate 9 Bare-answer fallback: passed bare_oracle_mean={sum(scores) / len(scores):.4f}")


def gate_monotone_difficulty():
    means = {}
    rollouts = 4
    items_per_mode = 48
    for level in range(1, 5):
        scores = []
        for mode in ("atom", "episode"):
            items = family.generate(31, level, items_per_mode, mode)
            for item_index, item in enumerate(items):
                for rep in range(rollouts):
                    rng = random.Random(600000 + level * 10000 + item_index * 97 + rep * 13 + (0 if mode == "atom" else 5000))
                    if mode == "atom":
                        obs = family.Env(item).reset()
                        action = _noisy_action(item, [], rng)
                        transcript = [{"obs": obs, "action": action}]
                    else:
                        transcript = _drive_episode(
                            item,
                            lambda current_item, history, local_rng=rng: _noisy_action(
                                current_item, history, local_rng
                            ),
                        )
                    scores.append(family.score(item, transcript)["score"])
        assert len(scores) >= 96, (
            "Monotone difficulty: noisy-oracle (eps=0.5: each turn, 50% oracle action / 50% random) mean score is non-increasing L1->L4 (tolerance 0.05). "
            f"level={level} mode=all item=aggregate observed={len(scores)} rollouts"
        )
        means[level] = sum(scores) / len(scores)
    for level in range(1, 4):
        assert means[level + 1] <= means[level] + 0.05, (
            "Monotone difficulty: noisy-oracle (eps=0.5: each turn, 50% oracle action / 50% random) mean score is non-increasing L1->L4 (tolerance 0.05). "
            f"level={level}->{level + 1} mode=all item=aggregate observed={means[level]:.4f}->{means[level + 1]:.4f}"
        )
    print(
        "gate 6 Monotone difficulty: passed "
        + ", ".join(f"L{level}={means[level]:.4f}" for level in range(1, 5))
    )
    return means


def gate_budgets():
    rng = random.Random(7707)
    for level in range(1, 5):
        cap = 4 if level in (1, 2) else 10 if level == 3 else 14
        for mode in ("atom", "episode"):
            for item_index, item in enumerate(family.generate(37, level, 6, mode)):
                assert item["max_turns"] <= cap, (
                    "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
                    f"level={level} mode={mode} item={item['id']} observed=max_turns {item['max_turns']} > {cap}"
                )
                reset_obs = family.Env(item).reset()
                assert _seq_text(item["solution"]) not in reset_obs, (
                    "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
                    f"level={level} mode={mode} item={item['id']} observed=reset leaks solution"
                )
                if mode == "atom":
                    prompt = reset_obs
                    assert len(prompt) <= 1200, (
                        "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
                        f"level={level} mode={mode} item={item['id']} observed=prompt length {len(prompt)}"
                    )
                    assert item["max_turns"] == 1, (
                        "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
                        f"level={level} mode={mode} item={item['id']} observed=max_turns {item['max_turns']}"
                    )
                else:
                    if item_index == 0:
                        _assert_adversarial_observation_limits(item)
                    _assert_episode_observation_limits(
                        item,
                        lambda current_item, history: family.oracle_policy(current_item, history),
                    )
                    _assert_episode_observation_limits(
                        item,
                        lambda current_item, history, local_rng=rng: family.random_policy(
                            current_item, history, local_rng
                        ),
                    )

    start = time.perf_counter()
    count = 0
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in family.generate(41, level, 6, mode):
                if mode == "atom":
                    transcript = [{"obs": "", "action": f"ANSWER: {_seq_text(item['solution'])}"}]
                else:
                    transcript = [{"obs": "", "action": f"OPEN {_seq_text(item['solution'])}"}]
                family.score(item, transcript)
                count += 1
    elapsed = time.perf_counter() - start
    per_item = elapsed / count
    assert count >= 24 and per_item < 0.05, (
        "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
        f"level=all mode=all item=aggregate observed={per_item * 1000:.3f} ms/item over {count} items"
    )
    print(f"gate 7 Budgets: passed avg_generate_score={per_item * 1000:.3f}ms/item")


def gate_purity():
    path = Path(__file__).with_name("family.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".", 1)[0])
    nonstdlib = sorted(name for name in imported if name not in sys.stdlib_module_names)
    assert not nonstdlib, (
        "Purity: module imports nothing beyond the stdlib. "
        f"level=all mode=all item=family.py observed={nonstdlib}"
    )
    print("gate 8 Purity: passed")


def _drive_episode(item, policy):
    env = family.Env(item)
    obs = env.reset()
    history = []
    for turn in range(item["max_turns"]):
        action = policy(item, history)
        next_obs, done = env.step(action)
        history.append({"obs": obs, "action": action})
        obs = next_obs
        if done:
            break
    return history


def _drive_echo(item):
    env = family.Env(item)
    obs = env.reset()
    if item["mode"] == "atom":
        return [{"obs": obs, "action": obs}]
    history = []
    for turn in range(item["max_turns"]):
        action = obs
        next_obs, done = env.step(action)
        history.append({"obs": obs, "action": action})
        obs = next_obs
        if done:
            break
    return history


def _assert_episode_observation_limits(item, policy):
    env = family.Env(item)
    obs = env.reset()
    assert len(obs) <= 800, (
        "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
        f"level={item['level']} mode={item['mode']} item={item['id']} observed=reset length {len(obs)}"
    )
    history = []
    for turn in range(item["max_turns"]):
        action = policy(item, history)
        next_obs, done = env.step(action)
        assert len(next_obs) <= 800, (
        "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
        f"level={item['level']} mode={item['mode']} item={item['id']} observed=turn {turn} length {len(next_obs)}"
        )
        history.append({"obs": obs, "action": action})
        obs = next_obs
        if done:
            break


def _assert_adversarial_observation_limits(item):
    env = family.Env(item)
    env.reset()
    valid_probe = "PROBE " + _seq_text(item["alphabet"][: item["seq_len"]])
    actions = [
        "PROBE " + "q" * 2000 + (" " + item["alphabet"][0]) * (item["seq_len"] - 1),
        "OPEN " + "z" * 3000 + (" " + item["alphabet"][0]) * (item["seq_len"] - 1),
        "x" * 3000,
        "PROBE " + " ".join([item["alphabet"][0]] * 200),
    ]
    actions.extend([valid_probe] * item["probe_budget"])
    actions.append(valid_probe)
    for turn, action in enumerate(actions):
        obs, _done = env.step(action)
        assert len(obs) <= 800, (
            "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item. "
            f"level={item['level']} mode={item['mode']} item={item['id']} observed=adversarial turn {turn} length {len(obs)}"
        )


def _empty_action(item, history):
    return ""


def _constant_target_action(item, history):
    if item["mode"] == "atom":
        return f"ANSWER: {_seq_text(item['target'])}"
    return f"OPEN {_seq_text(item['target'])}"


def _bare_answer_action(item, history):
    if item["mode"] == "atom":
        return _seq_text(item["solution"])
    return family.oracle_policy(item, history)


def _noisy_action(item, history, rng):
    if rng.random() < 0.5:
        return family.oracle_policy(item, history)
    return family.random_policy(item, history, rng)


def _content_signature(item):
    return (
        tuple(item["alphabet"]),
        json.dumps(item["rule_spec"], sort_keys=True, separators=(",", ":")),
        tuple(item["solution"]),
        tuple(item["target"]),
    )


def _seq_text(seq):
    return " ".join(seq)


if __name__ == "__main__":
    main()
