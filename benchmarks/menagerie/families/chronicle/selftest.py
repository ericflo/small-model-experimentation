import ast
import json
import random
import sys
import time

from . import family


BATCH = 40
LEVELS = (1, 2, 3, 4)
MODES = ("atom", "episode")


def main():
    gate_determinism()
    gate_seed_disjointness()
    gate_oracle_perfection()
    gate_random_floor()
    gate_degenerate_resistance()
    gate_monotone_difficulty()
    gate_budgets()
    gate_purity()
    gate_bare_answer_fallback()
    gate_answer_format_last_line()
    print("PASS chronicle selftest: 10 contract gates passed")


def all_items(seed=7, n=BATCH):
    batches = {}
    for mode in MODES:
        for level in LEVELS:
            batches[(level, mode)] = family.generate(seed, level, n, mode)
    return batches


def score_action(item, action):
    return family.score(item, [{"obs": item["prompt"], "action": action}])["score"]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def last_event_destination_policy(item):
    lines = item["prompt"].splitlines()
    try:
        start = lines.index("Events:") + 1
    except ValueError:
        return "ANSWER: "
    event_lines = lines[start:-2]
    for line in reversed(event_lines):
        if ">" in line:
            return "ANSWER: " + line.rsplit(">", 1)[1].strip()
    return "ANSWER: "


def gate_determinism():
    for mode in MODES:
        for level in LEVELS:
            a = family.generate(7, level, 6, mode)
            b = family.generate(7, level, 6, mode)
            ja = json.dumps(a, sort_keys=True, separators=(",", ":"))
            jb = json.dumps(b, sort_keys=True, separators=(",", ":"))
            assert ja == jb, f"Gate 1 determinism failed for L{level} {mode}"


def gate_seed_disjointness():
    prompts_7 = set()
    prompts_8 = set()
    for mode in MODES:
        for level in LEVELS:
            prompts_7.update(item["prompt"] for item in family.generate(7, level, BATCH, mode))
            prompts_8.update(item["prompt"] for item in family.generate(8, level, BATCH, mode))
    overlap = prompts_7 & prompts_8
    assert not overlap, f"Gate 2 seed disjointness failed: {len(overlap)} shared prompts"


def gate_oracle_perfection():
    for (level, mode), items in all_items().items():
        scores = [score_action(item, family.oracle_policy(item, [])) for item in items]
        assert all(score == 1.0 for score in scores), f"Gate 3 oracle failed for L{level} {mode}"


def gate_bare_answer_fallback():
    for (level, mode), items in all_items().items():
        scores = [score_action(item, item["gold"]) for item in items]
        assert all(score == 1.0 for score in scores), f"Gate 9 bare answer failed for L{level} {mode}"


def gate_random_floor():
    for (level, mode), items in all_items().items():
        scores = [
            score_action(item, family.random_policy(item, [], random.Random(1234 + i)))
            for i, item in enumerate(items)
        ]
        value = mean(scores)
        assert value <= 0.05, f"Gate 4 random floor failed for L{level} {mode}: {value:.3f}"


def gate_degenerate_resistance():
    for (level, mode), items in all_items().items():
        counts = {}
        for item in items:
            counts[item["gold"]] = counts.get(item["gold"], 0) + 1
        modal = max(counts, key=counts.get)
        policies = {
            "empty-string": lambda item: "",
            "constant most-plausible-answer": lambda item: "ANSWER: " + modal,
            "echo-the-observation": lambda item: item["prompt"],
            "last-event-destination": last_event_destination_policy,
        }
        for name, policy in policies.items():
            value = mean([score_action(item, policy(item)) for item in items])
            assert value <= 0.1, f"Gate 5 {name} failed for L{level} {mode}: {value:.3f}"


def gate_monotone_difficulty():
    for mode in MODES:
        means = []
        for level in LEVELS:
            items = family.generate(7, level, BATCH, mode)
            scores = []
            for i, item in enumerate(items):
                if i % 2 == 0:
                    action = family.oracle_policy(item, [])
                else:
                    action = family.random_policy(item, [], random.Random(4321 + i))
                scores.append(score_action(item, action))
            means.append(mean(scores))
        for earlier, later in zip(means, means[1:]):
            assert later <= earlier + 0.05, f"Gate 6 monotone difficulty failed for {mode}: {means}"


def gate_budgets():
    start = time.perf_counter()
    total = 0
    for mode in MODES:
        for level in LEVELS:
            items = family.generate(7, level, BATCH, mode)
            total += len(items)
            limit = 1200 if mode == "atom" else 800
            for item in items:
                assert len(item["prompt"]) <= limit, f"Gate 7 prompt budget failed for {item['id']}"
                assert item["max_turns"] == 1, f"Gate 7 max_turns failed for {item['id']}"
                action = family.oracle_policy(item, [])
                scored = family.score(item, [{"obs": item["prompt"], "action": action}])
                assert scored["score"] == 1.0, f"Gate 7 scoring sanity failed for {item['id']}"
    elapsed = time.perf_counter() - start
    per_item_ms = (elapsed / total) * 1000.0
    assert per_item_ms < 50.0, f"Gate 7 speed failed: {per_item_ms:.3f} ms/item"


def gate_purity():
    with open("families/chronicle/family.py", "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    non_stdlib = sorted(name for name in imported if name not in sys.stdlib_module_names)
    assert not non_stdlib, f"Gate 8 purity failed: non-stdlib imports {non_stdlib}"


def gate_answer_format_last_line():
    for level in LEVELS:
        items = family.generate(7, level, BATCH, "atom")
        for item in items:
            last_line = item["prompt"].splitlines()[-1]
            assert "answer:" in last_line.lower(), f"Gate 10 answer format failed for {item['id']}"


if __name__ == "__main__":
    main()
