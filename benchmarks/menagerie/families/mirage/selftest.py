import ast
import inspect
import json
import random
import sys
import time

from . import family


LEVELS = (1, 2, 3, 4)
MODES = ("atom", "episode")


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
        ("9. Fallback bare-answer", gate_fallback_bare_answer),
    ]
    results = {}
    for label, gate in gates:
        try:
            result = gate()
        except Exception as exc:
            print(f"FAIL {label}: {exc}")
            raise SystemExit(1)
        results[label] = result
        if result is None:
            print(f"OK {label}")
        else:
            print(f"OK {label}: {result}")
    print("mirage selftest: ALL 9 GATES PASSED")


def gate_determinism():
    meta = family.META
    if meta.get("name") != "mirage":
        raise AssertionError("META name is not mirage")
    if meta.get("paradigm") != "single-turn":
        raise AssertionError("META paradigm is not single-turn")
    if "calibrated abstention" not in meta.get("capability", ""):
        raise AssertionError("META capability does not name calibrated abstention")
    if "ANSWER:" not in meta.get("action_format", "") or "IMPOSSIBLE" not in meta.get("action_format", ""):
        raise AssertionError("META action_format is incomplete")

    for level in LEVELS:
        for mode in MODES:
            first = family.generate(7, level, 6, mode)
            second = family.generate(7, level, 6, mode)
            if _json(first) != _json(second):
                raise AssertionError(f"generate differs for L{level} {mode}")
            _assert_even_balance_and_pairs(first, level, mode)


def gate_seed_disjointness():
    left = _content_set(7)
    right = _content_set(8)
    overlap = left & right
    if overlap:
        raise AssertionError("seeds 7 and 8 share content: " + repr(sorted(overlap)[:5]))


def gate_oracle_perfection():
    for item in _batch(seed=7, n=10):
        result = _run_action(item, family.oracle_policy(item, []))
        if result["score"] != 1.0:
            raise AssertionError(f"oracle failed {item['id']}: {result}")
        if item["label"] == "solvable":
            upper_result = _run_action(item, "ANSWER: " + item["forced_answer"].upper())
            if upper_result["score"] != 1.0:
                raise AssertionError(f"uppercase answer failed {item['id']}: {upper_result}")


def gate_random_floor():
    rng = random.Random("mirage:selftest:random-floor")
    scores = []
    for item in _batch(seed=17, n=60):
        for _roll in range(16):
            scores.append(_run_action(item, family.random_policy(item, [], rng))["score"])
    mean = _mean(scores)
    if mean > 0.05:
        raise AssertionError(f"random mean {mean:.4f} exceeds 0.05")
    return "mean=" + _fmt(mean)


def gate_degenerate_resistance():
    pool = _batch(seed=23, n=60)
    modal_token = _modal([item["forced_answer"] for item in pool if item["label"] == "solvable"])
    policies = (
        ("empty-string", lambda item, obs: "", 0.10),
        ("constant most-common value-token", lambda item, obs: "ANSWER: " + modal_token, 0.10),
        ("echo-the-observation", lambda item, obs: obs, 0.10),
        ("constant IMPOSSIBLE", lambda item, obs: "ANSWER: IMPOSSIBLE", 0.55),
    )
    means = []
    for name, policy, cap in policies:
        scores = []
        for item in pool:
            obs = family.Env(item).reset()
            scores.append(_run_action(item, policy(item, obs), obs)["score"])
        mean = _mean(scores)
        if mean > cap:
            raise AssertionError(f"{name} mean {mean:.4f} exceeds {cap:.2f}")
        means.append(name + "=" + _fmt(mean))
    return " ".join(means)


def gate_fallback_bare_answer():
    bare_scores = []
    for item in _batch(seed=29, n=10):
        result = _run_action(item, item["forced_answer"])
        if result["score"] != 1.0:
            raise AssertionError(f"bare-answer oracle failed {item['id']}: {result}")
        bare_scores.append(result["score"])

    pool = _batch(seed=23, n=60)
    modal_token = _modal([item["forced_answer"] for item in pool if item["label"] == "solvable"])
    policies = (
        ("empty-string", lambda item, obs: "", 0.10),
        ("echo-the-observation", lambda item, obs: obs, 0.10),
        ("constant most-common value-token", lambda item, obs: "ANSWER: " + modal_token, 0.10),
    )
    means = []
    for name, policy, cap in policies:
        scores = []
        for item in pool:
            obs = family.Env(item).reset()
            scores.append(_run_action(item, policy(item, obs), obs)["score"])
        mean = _mean(scores)
        if mean > cap:
            raise AssertionError(f"{name} mean {mean:.4f} exceeds {cap:.2f}")
        means.append(name + "=" + _fmt(mean))
    return "bare_answer_mean=" + _fmt(_mean(bare_scores)) + " " + " ".join(means)


def gate_monotone_difficulty():
    means = {}
    rng = random.Random("mirage:selftest:noisy")
    for level in LEVELS:
        scores = []
        for mode in MODES:
            for item in family.generate(31, level, 60, mode):
                for _roll in range(16):
                    if rng.random() < 0.5:
                        action = family.oracle_policy(item, [])
                    else:
                        action = family.random_policy(item, [], rng)
                    scores.append(_run_action(item, action)["score"])
        means[level] = _mean(scores)
    for level in (1, 2, 3):
        if means[level + 1] > means[level] + 0.05:
            raise AssertionError(
                "noisy means not monotone: "
                + " ".join(f"L{idx}={means[idx]:.4f}" for idx in LEVELS)
            )
    return " ".join(f"L{level}={_fmt(means[level])}" for level in LEVELS)


def gate_budgets():
    for item in _batch(seed=37, n=24):
        obs = family.Env(item).reset()
        if item["mode"] == "atom" and len(obs) > 1200:
            raise AssertionError(f"atom prompt too long for {item['id']}: {len(obs)}")
        if item["mode"] == "episode" and len(obs) > 800:
            raise AssertionError(f"episode observation too long for {item['id']}: {len(obs)}")
        if len(obs) > 800:
            raise AssertionError(f"prompt exceeds compact 800-char mirage budget for {item['id']}: {len(obs)}")
        if item["max_turns"] != 1:
            raise AssertionError(f"max_turns is not 1 for {item['id']}")
        if item["label"] in obs:
            raise AssertionError(f"label leaks in prompt for {item['id']}")
        corrective, done = family.Env(item).step("")
        if not done or len(corrective) > 800:
            raise AssertionError(f"malformed step handling failed for {item['id']}")

    start = time.perf_counter()
    count = 0
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(41, level, 120, mode)
            for item in items:
                obs = family.Env(item).reset()
                action = family.oracle_policy(item, [])
                result = family.score(item, [{"obs": obs, "action": action}])
                if result["score"] != 1.0:
                    raise AssertionError(f"oracle scoring failed during timing for {item['id']}")
                count += 1
    elapsed = time.perf_counter() - start
    ms_per_item = (elapsed / count) * 1000.0
    if ms_per_item >= 50.0:
        raise AssertionError(f"generation+scoring {ms_per_item:.3f} ms/item exceeds 50")
    return "generation+scoring=" + _fmt(ms_per_item) + "ms/item"


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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            raise AssertionError("family.py calls print")
    stdlib = getattr(sys, "stdlib_module_names", set(sys.builtin_module_names))
    nonstdlib = sorted(name for name in imports if name not in stdlib)
    if nonstdlib:
        raise AssertionError("non-stdlib imports in family.py: " + repr(nonstdlib))


def _batch(seed, n):
    items = []
    for level in LEVELS:
        for mode in MODES:
            items.extend(family.generate(seed, level, n, mode))
    return items


def _run_action(item, action, obs=None):
    if obs is None:
        obs = family.Env(item).reset()
    env = family.Env(item)
    env.reset()
    next_obs, done = env.step(action)
    if not done:
        raise AssertionError(f"environment did not finish for {item['id']}: {next_obs}")
    return family.score(item, [{"obs": obs, "action": action}])


def _assert_even_balance_and_pairs(items, level, mode):
    labels = [item["label"] for item in items]
    if labels.count("solvable") != len(items) // 2 or labels.count("unsolvable") != len(items) // 2:
        raise AssertionError(f"label balance failed for L{level} {mode}: {labels}")
    by_pair = {}
    for item in items:
        if item["level"] != level or item["mode"] != mode or item["max_turns"] != 1:
            raise AssertionError(f"basic item fields wrong for {item['id']}")
        by_pair.setdefault(item["pair_id"], []).append(item)
    for pair_id, pair_items in by_pair.items():
        if len(pair_items) != 2:
            raise AssertionError(f"pair {pair_id} does not contain two items")
        left, right = sorted(pair_items, key=lambda item: item["label"])
        if left["skeleton"] != right["skeleton"]:
            raise AssertionError(f"paired skeleton mismatch for {pair_id}")
        if _type_counts(left["constraints"]) != _type_counts(right["constraints"]):
            raise AssertionError(f"constraint type multiset mismatch for {pair_id}")


def _content_set(seed):
    content = set()
    for item in _batch(seed, n=6):
        content.add(item["id"])
        content.add(item["prompt"])
        content.update(item["cycle"])
        content.update(item["entities"])
        content.update(item["offcycle"])
        content.add(json.dumps(item["constraints"], sort_keys=True, separators=(",", ":")))
    return content


def _type_counts(constraints):
    counts = {}
    for constraint in constraints:
        counts[constraint["kind"]] = counts.get(constraint["kind"], 0) + 1
    return counts


def _modal(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(values, key=lambda value: (counts[value], value))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fmt(value):
    return "%.6f" % value


if __name__ == "__main__":
    main()
