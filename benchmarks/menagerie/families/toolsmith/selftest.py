import ast
import json
import random
import sys
import time

from . import family


LEVELS = (1, 2, 3, 4)
MODES = ("atom", "episode")


def main():
    metrics = {}
    gate_1_determinism()
    gate_2_seed_disjointness()
    gate_3_oracle_perfection()
    metrics["random_floor"] = gate_4_random_floor()
    gate_5_degenerate_resistance()
    metrics["noisy"] = gate_6_monotone_difficulty()
    metrics["mean_ms"] = gate_7_budgets()
    gate_8_purity()
    print("toolsmith selftest: all gates passed")
    return metrics


def gate_1_determinism():
    for level in LEVELS:
        for mode in MODES:
            a = _canonical(family.generate(7, level, 6, mode))
            b = _canonical(family.generate(7, level, 6, mode))
            _require(a == b, "gate 1 determinism failed for L%s %s" % (level, mode))
    print("gate 1 determinism: ok")


def gate_2_seed_disjointness():
    salts = [family._seed_salt(seed) for seed in range(7140)]
    _require(len(salts) == len(set(salts)), "gate 2 salt injectivity failed")
    for left, right in ((7, 8), (0, 63), (419, 420)):
        tokens_left = _tokens_for_seed(left)
        tokens_right = _tokens_for_seed(right)
        shared = sorted(tokens_left & tokens_right)
        _require(
            not shared,
            "gate 2 seed disjointness failed for %s vs %s, shared tokens: %s" % (left, right, shared[:10]),
        )
    print("gate 2 seed disjointness: ok")


def gate_3_oracle_perfection():
    for item in _items(7, 10):
        transcript = _run_policy(item, lambda it, hist, obs: family.oracle_policy(it, hist))
        result = family.score(item, transcript)
        _require(result.get("score") == 1.0, "gate 3 oracle failed on %s: %r" % (item["id"], result))
    print("gate 3 oracle perfection: ok")


def gate_4_random_floor():
    rng = random.Random(4004)
    scores = []
    for item in _items(17, 48):
        transcript = _run_policy(item, lambda it, hist, obs: family.random_policy(it, hist, rng))
        scores.append(family.score(item, transcript).get("score", 0.0))
    mean = _mean(scores)
    _require(mean <= 0.05, "gate 4 random floor failed: mean %.4f > 0.05" % mean)
    print("gate 4 random floor: ok mean=%.4f" % mean)
    return mean


def gate_5_degenerate_resistance():
    policies = [
        ("empty", lambda item, hist, obs: ""),
        ("constant-submit", lambda item, hist, obs: "ANSWER: tool1 -> tool2" if item["mode"] == "atom" else "SUBMIT value"),
        ("constant-call", lambda item, hist, obs: "ANSWER: tool1 -> tool2" if item["mode"] == "atom" else "CALL tool(value)"),
        ("echo", lambda item, hist, obs: obs),
    ]
    means = {}
    sample = list(_items(23, 24))
    for name, policy in policies:
        scores = []
        for item in sample:
            transcript = _run_policy(item, policy)
            scores.append(family.score(item, transcript).get("score", 0.0))
        means[name] = _mean(scores)
        _require(means[name] <= 0.1, "gate 5 %s failed: mean %.4f > 0.1" % (name, means[name]))
    detail = " ".join("%s=%.4f" % (name, means[name]) for name in sorted(means))
    print("gate 5 degenerate resistance: ok " + detail)


def gate_6_monotone_difficulty():
    means = {mode: [] for mode in MODES}
    for mode in MODES:
        for level in LEVELS:
            scores = []
            items = family.generate(31, level, 40, mode)
            for item_index, item in enumerate(items):
                for rep in range(5):
                    seed = 600000 + item_index * 97 + rep * 1009 + (0 if mode == "atom" else 200000)
                    rng = random.Random(seed)
                    transcript = _run_policy(item, lambda it, hist, obs, r=rng: _noisy_action(it, hist, r))
                    scores.append(family.score(item, transcript).get("score", 0.0))
            means[mode].append(_mean(scores))

    for mode in MODES:
        for idx in range(3):
            prev = means[mode][idx]
            nxt = means[mode][idx + 1]
            margin = prev + 0.05 - nxt
            _require(
                margin >= 0.02,
                "gate 6 monotone failed for %s L%s->L%s: %.4f then %.4f, margin %.4f"
                % (mode, idx + 1, idx + 2, prev, nxt, margin),
            )
    print("gate 6 monotone difficulty: ok " + _format_means(means))
    return means


def gate_7_budgets():
    started = time.perf_counter()
    items = []
    for level in LEVELS:
        for mode in MODES:
            items.extend(family.generate(41, level, 8, mode))
    score_total = 0.0
    for item in items:
        prompt = family.Env(item).reset()
        if item["mode"] == "atom":
            _require(len(prompt) <= 1200, "gate 7 atom prompt too long for %s" % item["id"])
            _require(item["max_turns"] == 1, "gate 7 atom max_turns wrong for %s" % item["id"])
        else:
            _require(len(prompt) <= 800, "gate 7 episode observation too long for %s" % item["id"])
            cap = 4 if item["level"] in (1, 2) else 10 if item["level"] == 3 else 14
            _require(item["max_turns"] <= cap, "gate 7 episode max_turns cap failed for %s" % item["id"])
        transcript = _run_policy(item, lambda it, hist, obs: family.oracle_policy(it, hist))
        for entry in transcript:
            _require(len(str(entry.get("obs", ""))) <= 1200, "gate 7 obs too long for %s" % item["id"])
            _require(len(str(entry.get("next_obs", ""))) <= 800, "gate 7 next obs too long for %s" % item["id"])
        score_total += family.score(item, transcript).get("score", 0.0)
    elapsed = time.perf_counter() - started
    mean_ms = 1000.0 * elapsed / len(items)
    _require(len(items) >= 50, "gate 7 timing sample too small")
    _require(mean_ms < 50.0, "gate 7 timing failed: %.3f ms/item" % mean_ms)
    _require(score_total > 0, "gate 7 scoring sanity failed")
    print("gate 7 budgets: ok mean_ms=%.3f items=%s" % (mean_ms, len(items)))
    return mean_ms


def gate_8_purity():
    with open(family.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    stdlib = getattr(sys, "stdlib_module_names", set())
    bad = sorted(name for name in imports if name != "__future__" and stdlib and name not in stdlib)
    allowed = {"hashlib", "random", "re"}
    unexpected = sorted(imports - allowed)
    _require(not bad, "gate 8 non-stdlib imports: %s" % bad)
    _require(not unexpected, "gate 8 unexpected imports in family.py: %s" % unexpected)
    print("gate 8 purity: ok imports=%s" % ",".join(sorted(imports)))


def _items(seed, n):
    for level in LEVELS:
        for mode in MODES:
            for item in family.generate(seed, level, n, mode):
                yield item


def _run_policy(item, policy):
    env = family.Env(item)
    obs = env.reset()
    transcript = []
    if item.get("mode") == "atom":
        action = policy(item, transcript, obs)
        return [{"obs": obs, "action": action}]

    for _ in range(item.get("max_turns", 1)):
        action = policy(item, transcript, obs)
        next_obs, done = env.step(action)
        transcript.append({"obs": obs, "action": action, "next_obs": next_obs})
        obs = next_obs
        if done:
            break
    return transcript


def _noisy_action(item, history, rng):
    if rng.random() < 0.5:
        return family.oracle_policy(item, history)
    return family.random_policy(item, history, rng)


def _tokens_for_seed(seed):
    tokens = set()
    for item in _items(seed, 6):
        for tool in item["registry"]:
            tokens.add(tool["name"])
            tokens.update(tool["args"])
            tokens.add(tool["out"])
        for start in item["starts"]:
            tokens.add(start["value"])
            tokens.add(start["type"])
        for step in item["chain"]:
            tokens.add(step["tool"])
            tokens.update(step["args"])
            tokens.add(step["out"])
        tokens.add(item["target"])
        tokens.add(item["goal_type"])
    return tokens


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _format_means(means):
    chunks = []
    for mode in MODES:
        parts = ["L%s=%.4f" % (level, means[mode][level - 1]) for level in LEVELS]
        chunks.append("%s[%s]" % (mode, " ".join(parts)))
    return " ".join(chunks)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
