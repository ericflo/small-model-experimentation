import ast
import importlib
import json
import pathlib
import random
import sys
import time
from collections import Counter

from . import family


GATES = {
    1: "Determinism: generate(7, L, 6, mode) twice -> identical JSON, all levels/modes.",
    2: "Seed disjointness: seeds 7 vs 8 share no item content.",
    3: "Oracle perfection: oracle_policy scores 1.0 on every item (all levels, both modes).",
    4: "Random floor: mean random_policy score <= 0.05 (or <= 0.15 with justification in README for ratio-scored families).",
    5: "Degenerate resistance: empty-string policy, constant most-plausible-answer policy, and echo-the-observation policy each score <= 0.1 mean.",
    6: "Monotone difficulty: noisy-oracle (eps=0.5: each turn, 50% oracle action / 50% random) mean score is non-increasing L1->L4 (tolerance 0.05).",
    7: "Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item.",
    8: "Purity: module imports nothing beyond the stdlib.",
    9: "Bare-answer fallback: bare-answer-oracle scores 1.0; echo/constant/empty stay <= 0.1.",
}


def main():
    gate_1()
    gate_2()
    gate_3()
    gate_4()
    gate_5()
    gate_6()
    gate_7()
    gate_8()
    gate_9()


def _dump(items):
    return json.dumps(items, sort_keys=True, separators=(",", ":"))


def _items(seed, level, mode, n=40):
    return family.generate(seed, level, n, mode)


def _drive(item, action):
    env = family.Env(item)
    obs = env.reset()
    env.step(action)
    return [{"obs": obs, "action": action}]


def _score_action(item, action):
    return family.score(item, _drive(item, action))["score"]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _assert(condition, gate, detail):
    assert condition, f"Gate {gate} failed: {GATES[gate]} {detail}"


def gate_1():
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            first = _dump(family.generate(7, level, 6, mode))
            second = _dump(family.generate(7, level, 6, mode))
            _assert(first == second, 1, f"Mismatch at L{level} {mode}.")
    print(f"PASS gate 1: {GATES[1]}")


def gate_2():
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            left = {item["document"] + "\n" + item["query"] for item in _items(7, level, mode, 40)}
            right = {item["document"] + "\n" + item["query"] for item in _items(8, level, mode, 40)}
            _assert(left.isdisjoint(right), 2, f"Overlap at L{level} {mode}.")
    print(f"PASS gate 2: {GATES[2]}")


def gate_3():
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in _items(7, level, mode, 40):
                env = family.Env(item)
                obs = env.reset()
                action = family.oracle_policy(item, [{"obs": obs}])
                env.step(action)
                result = family.score(item, [{"obs": obs, "action": action}])
                _assert(result["score"] == 1.0, 3, f"Oracle missed {item['id']}.")
    print(f"PASS gate 3: {GATES[3]}")


def gate_4():
    rng = random.Random(4104)
    scores = []
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            for item in _items(11, level, mode, 40):
                obs = family.Env(item).reset()
                action = family.random_policy(item, [{"obs": obs}], rng)
                scores.append(_score_action(item, action))
    _assert(_mean(scores) <= 0.05, 4, f"Mean random score was {_mean(scores):.4f}.")
    print(f"PASS gate 4: {GATES[4]}")


def gate_5():
    items = []
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            items.extend(_items(13, level, mode, 40))
    most_common_gold, _count = Counter(item["gold"] for item in items).most_common(1)[0]

    empty_scores = [_score_action(item, "") for item in items]
    constant_scores = [_score_action(item, "ANSWER: " + most_common_gold) for item in items]
    echo_scores = []
    for item in items:
        obs = family.Env(item).reset()
        echo_scores.append(_score_action(item, obs))

    _assert(_mean(empty_scores) <= 0.1, 5, f"Empty mean was {_mean(empty_scores):.4f}.")
    _assert(_mean(constant_scores) <= 0.1, 5, f"Constant mean was {_mean(constant_scores):.4f}.")
    _assert(_mean(echo_scores) <= 0.1, 5, f"Echo mean was {_mean(echo_scores):.4f}.")
    print(f"PASS gate 5: {GATES[5]}")


def gate_6():
    means = []
    for level in range(1, 5):
        coin = random.Random(6060)
        noise = random.Random(9090)
        scores = []
        for mode in ("atom", "episode"):
            for item in _items(17, level, mode, 60):
                obs = family.Env(item).reset()
                if coin.random() < 0.5:
                    action = family.oracle_policy(item, [{"obs": obs}])
                else:
                    action = family.random_policy(item, [{"obs": obs}], noise)
                scores.append(_score_action(item, action))
        means.append(_mean(scores))
    for earlier, later in zip(means, means[1:]):
        _assert(later <= earlier + 0.05, 6, f"Means were {means}.")
    print(f"PASS gate 6: {GATES[6]}")


def gate_7():
    all_items = []
    start = time.perf_counter()
    for seed in (7, 8, 9, 125, 127, 137):
        for level in range(1, 5):
            for mode in ("atom", "episode"):
                batch = _items(seed, level, mode, 40)
                all_items.extend(batch)
                for item in batch:
                    obs = family.Env(item).reset()
                    _assert(item["max_turns"] == 1, 7, f"max_turns mismatch for {item['id']}.")
                    if mode == "atom":
                        _assert(len(obs) <= 1200, 7, f"Atom prompt too long for {item['id']}: {len(obs)}.")
                    else:
                        _assert(len(obs) <= 800, 7, f"Episode observation too long for {item['id']}: {len(obs)}.")
                    answer_line = "ANSWER: " + item["gold"]
                    _assert(answer_line not in obs.splitlines(), 7, f"Gold ANSWER line leaked in {item['id']}.")
                    action = family.oracle_policy(item, [{"obs": obs}])
                    family.score(item, [{"obs": obs, "action": action}])
    elapsed = time.perf_counter() - start
    _assert(elapsed / len(all_items) < 0.05, 7, "Generation+scoring exceeded 50 ms/item.")
    _assert_histograms()
    print(f"PASS gate 7: {GATES[7]}")


def _assert_histograms():
    for seed in (7, 8, 9, 125, 127, 137):
        for level in range(1, 5):
            for mode in ("atom", "episode"):
                batch = _items(seed, level, mode, 40)
                counts = Counter(item["gold"] for item in batch)
                worst = max(counts.values())
                assert worst <= 4, (
                    "Anti-degeneracy failed: no gold may exceed 10% of a 40-item "
                    f"batch; saw {worst} for seed {seed} L{level} {mode}."
                )


def gate_8():
    path = pathlib.Path(family.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    bad = sorted(name for name in imports if name not in stdlib)
    _assert(not bad, 8, f"Non-stdlib imports: {bad}.")
    importlib.invalidate_caches()
    print(f"PASS gate 8: {GATES[8]}")


def gate_9():
    items = []
    for level in range(1, 5):
        for mode in ("atom", "episode"):
            items.extend(_items(13, level, mode, 40))
    most_common_gold, _count = Counter(item["gold"] for item in items).most_common(1)[0]

    bare_scores = [_score_action(item, str(item["gold"])) for item in items]
    empty_scores = [_score_action(item, "") for item in items]
    constant_scores = [_score_action(item, "ANSWER: " + most_common_gold) for item in items]
    echo_scores = []
    for item in items:
        obs = family.Env(item).reset()
        echo_scores.append(_score_action(item, obs))

    _assert(_mean(bare_scores) == 1.0, 9, f"Bare-answer mean was {_mean(bare_scores):.4f}.")
    _assert(_mean(empty_scores) <= 0.1, 9, f"Empty mean was {_mean(empty_scores):.4f}.")
    _assert(_mean(constant_scores) <= 0.1, 9, f"Constant mean was {_mean(constant_scores):.4f}.")
    _assert(_mean(echo_scores) <= 0.1, 9, f"Echo mean was {_mean(echo_scores):.4f}.")
    print(f"PASS gate 9: {GATES[9]}")


if __name__ == "__main__":
    main()
