import ast
import inspect
import json
import random
import sys
import time

from . import family


MODES = ("atom", "episode")
LEVELS = (1, 2, 3, 4)


def main():
    gates = [
        ("1. Determinism", gate_determinism),
        ("2. Seed disjointness", gate_seed_disjointness),
        ("3. Oracle perfection", gate_oracle_perfection),
        ("4. Random floor", gate_random_floor),
        ("5. Degenerate resistance", gate_degenerate_resistance),
        ("6. Monotone difficulty", gate_monotone_difficulty),
        ("7. Budgets", gate_budgets),
        ("8. Purity", gate_purity),
    ]
    for label, gate in gates:
        try:
            gate()
        except Exception as exc:
            print(f"FAIL {label}: {exc}")
            raise SystemExit(1)
        print(f"OK {label}")


def gate_determinism():
    for level in LEVELS:
        for mode in MODES:
            first = family.generate(7, level, 6, mode)
            second = family.generate(7, level, 6, mode)
            left = json.dumps(first, sort_keys=True)
            right = json.dumps(second, sort_keys=True)
            if left != right:
                raise AssertionError(f"generate differs for L{level} {mode}")


def gate_seed_disjointness():
    left = _content_set(7)
    right = _content_set(8)
    overlap = left & right
    if overlap:
        sample = sorted(overlap)[:5]
        raise AssertionError(f"seeds 7 and 8 share content: {sample}")


def gate_oracle_perfection():
    for item in _batch(seed=7, n=8):
        result = _run_policy(item, lambda it, hist, rng: family.oracle_policy(it, hist))
        if result["score"] != 1.0:
            raise AssertionError(f"oracle failed {item['id']}: {result}")


def gate_random_floor():
    rng = random.Random("sirens:selftest:random-floor")
    scores = [
        _run_policy(item, family.random_policy, rng)["score"]
        for item in _batch(seed=17, n=12)
    ]
    mean = sum(scores) / len(scores)
    if mean > 0.05:
        raise AssertionError(f"random mean {mean:.4f} exceeds 0.05")


def gate_degenerate_resistance():
    pool = _batch(seed=23, n=8)
    if len(pool) < 48:
        raise AssertionError("degenerate pool has fewer than 48 items")
    modal = _modal([item["target"] for item in pool])

    def empty_policy(item, history, rng):
        return ""

    def constant_policy(item, history, rng):
        if item["mode"] == "atom":
            return f"ANSWER: {modal}"
        return f"SUBMIT {modal}"

    def echo_policy(item, history, rng):
        return history[-1]["obs"] if history else ""

    for name, policy in (
        ("empty-string", empty_policy),
        ("constant most-plausible-answer", constant_policy),
        ("echo-the-observation", echo_policy),
    ):
        scores = [_run_policy(item, policy)["score"] for item in pool]
        mean = sum(scores) / len(scores)
        if mean > 0.1:
            raise AssertionError(f"{name} mean {mean:.4f} exceeds 0.1")


def gate_monotone_difficulty():
    for mode in MODES:
        means = []
        for level in LEVELS:
            rng = random.Random(f"sirens:selftest:noisy:{mode}:{level}")
            scores = []
            for item in family.generate(31, level, 12, mode):
                for _ in range(1024):
                    scores.append(_run_noisy_policy(item, rng)["score"])
            means.append(sum(scores) / len(scores))
        for idx in range(3):
            if means[idx + 1] > means[idx] + 0.05:
                raise AssertionError(
                    f"{mode} noisy means not monotone: "
                    + ", ".join(f"L{i + 1}={value:.4f}" for i, value in enumerate(means))
                )


def gate_budgets():
    for item in _batch(seed=41, n=10):
        if item["mode"] == "atom":
            prompt = family.Env(item).reset()
            if len(prompt) > 1200:
                raise AssertionError(f"atom prompt too long for {item['id']}")
            if item["max_turns"] != 1:
                raise AssertionError(f"atom max_turns not 1 for {item['id']}")
        else:
            env = family.Env(item)
            initial = env.reset()
            if len(initial) > 800:
                raise AssertionError(f"initial observation too long for {item['id']}")
            for doc_id in item["doc_ids"]:
                obs, _done = family.Env(item).step(f"READ {doc_id}")
                if len(obs) > 800:
                    raise AssertionError(f"READ observation too long for {item['id']}")
            cap = 4 if item["level"] in (1, 2) else 10 if item["level"] == 3 else 14
            if item["max_turns"] > cap:
                raise AssertionError(f"max_turns cap exceeded for {item['id']}")
        for doc_id, text in item["docs"].items():
            if len(text) > 600:
                raise AssertionError(f"document {doc_id} too long for {item['id']}")

    start = time.perf_counter()
    items = _batch(seed=43, n=40)
    for item in items:
        transcript = _oracle_transcript(item)
        family.score(item, transcript)
    per_item = (time.perf_counter() - start) / len(items)
    if per_item >= 0.05:
        raise AssertionError(f"generation+scoring {per_item:.5f}s/item exceeds 0.05")


def gate_purity():
    source = inspect.getsource(family)
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    stdlib = getattr(sys, "stdlib_module_names", set(sys.builtin_module_names))
    nonstdlib = sorted(name for name in imports if name not in stdlib)
    if nonstdlib:
        raise AssertionError(f"non-stdlib imports in family.py: {nonstdlib}")


def _batch(seed, n):
    items = []
    for level in LEVELS:
        for mode in MODES:
            items.extend(family.generate(seed, level, n, mode))
    return items


def _content_set(seed):
    content = set()
    for item in _batch(seed, n=6):
        content.update(item.get("entities", []))
        content.update(item.get("fields", []))
        content.update(item.get("doc_ids", []))
        content.add(item["target"])
        content.add(item["session_key"])
        content.update(item.get("decoy_values", []))
        content.update(item.get("alternate_values", []))
        content.update(item["docs"].values())
    return content


def _run_policy(item, policy, rng=None):
    if rng is None:
        rng = random.Random(f"sirens:selftest:policy:{item['id']}")
    env = family.Env(item)
    obs = env.reset()
    history = []
    for _ in range(item["max_turns"]):
        action = policy(item, history + [{"obs": obs, "action": ""}], rng)
        history.append({"obs": obs, "action": action})
        obs, done = env.step(action)
        if done:
            break
    return family.score(item, history)


def _run_noisy_policy(item, rng):
    def policy(it, hist, local_rng):
        if local_rng.random() < 0.5:
            return family.oracle_policy(it, hist)
        return family.random_policy(it, hist, local_rng)

    return _run_policy(item, policy, rng)


def _oracle_transcript(item):
    env = family.Env(item)
    obs = env.reset()
    history = []
    for _ in range(item["max_turns"]):
        action = family.oracle_policy(item, history)
        history.append({"obs": obs, "action": action})
        obs, done = env.step(action)
        if done:
            break
    return history


def _modal(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(values, key=lambda value: (counts[value], value))


if __name__ == "__main__":
    main()
