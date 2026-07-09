import ast
import json
import random
import time

from . import family


LEVELS = (1, 2, 3, 4)
MODES = ("atom", "episode")


def main():
    gate_determinism()
    gate_seed_disjointness()
    gate_oracle_perfection()
    random_mean = gate_random_floor()
    gate_degenerate_resistance()
    noisy_means = gate_monotone_difficulty()
    ms_per_item = gate_budgets()
    gate_purity()
    print("random-floor mean: " + _fmt(random_mean))
    print(
        "noisy-oracle means: "
        + " ".join("L" + str(level) + "=" + _fmt(noisy_means[level]) for level in LEVELS)
    )
    print("budget ms/item: " + _fmt(ms_per_item))
    print("stockade selftest: PASS")


def gate_determinism():
    for level in LEVELS:
        for mode in MODES:
            first = family.generate(7, level, 6, mode)
            second = family.generate(7, level, 6, mode)
            if _json(first) != _json(second):
                _fail(1, "Determinism", "generate(7, L%s, 6, %s) differed" % (level, mode))


def gate_seed_disjointness():
    by_seed = {}
    for seed in (7, 8):
        item_jsons = set()
        names = set()
        for level in LEVELS:
            for mode in MODES:
                for item in family.generate(seed, level, 6, mode):
                    item_jsons.add(_json(item))
                    names.update(res["name"] for res in item["resources"])
        by_seed[seed] = (item_jsons, names)

    shared_items = by_seed[7][0] & by_seed[8][0]
    if shared_items:
        _fail(2, "Seed disjointness", "seeds 7 and 8 shared full item JSON")
    shared_names = by_seed[7][1] & by_seed[8][1]
    if shared_names:
        sample = sorted(shared_names)[:5]
        _fail(2, "Seed disjointness", "seeds 7 and 8 shared resource names " + repr(sample))


def gate_oracle_perfection():
    for item in _items(7, 6):
        transcript, _observations = _run_policy(item, "oracle", None)
        result = family.score(item, transcript)
        if result["score"] != 1.0:
            _fail(3, "Oracle perfection", item["id"] + " scored " + repr(result))


def gate_random_floor():
    rng = random.Random(404)
    scores = []
    for item in _items(7, 16):
        transcript, _observations = _run_policy(item, "random", rng)
        scores.append(family.score(item, transcript)["score"])
    mean = _mean(scores)
    if mean > 0.15:
        _fail(4, "Random floor", "mean random_policy score was " + _fmt(mean))
    return mean


def gate_degenerate_resistance():
    policies = ("empty", "constant", "echo")
    for policy in policies:
        scores = []
        for item in _items(7, 6):
            transcript, _observations = _run_policy(item, policy, None)
            scores.append(family.score(item, transcript)["score"])
        mean = _mean(scores)
        if mean > 0.10:
            _fail(
                5,
                "Degenerate resistance",
                policy + " policy mean score was " + _fmt(mean),
            )


def gate_monotone_difficulty():
    rng = random.Random(606)
    means = {}
    for level in LEVELS:
        scores = []
        for mode in MODES:
            for item in family.generate(7, level, 12, mode):
                for _rollout in range(3):
                    transcript, _observations = _run_policy(item, "noisy", rng)
                    scores.append(family.score(item, transcript)["score"])
        means[level] = _mean(scores)

    for level in (1, 2, 3):
        if means[level + 1] > means[level] + 0.05:
            _fail(
                6,
                "Monotone difficulty",
                "L%s=%s, L%s=%s"
                % (level, _fmt(means[level]), level + 1, _fmt(means[level + 1])),
            )
    return means


def gate_budgets():
    start = time.perf_counter()
    items = list(_items(9, 8))
    scores = []
    observations_by_item = []
    for item in items:
        transcript, observations = _run_policy(item, "oracle", None)
        scores.append(family.score(item, transcript)["score"])
        observations_by_item.append((item, observations))
    elapsed = time.perf_counter() - start

    for item, observations in observations_by_item:
        if item["mode"] == "atom":
            if item["max_turns"] > 1:
                _fail(7, "Budgets", item["id"] + " atom max_turns exceeded 1")
            if len(observations[0]) > 1200:
                _fail(7, "Budgets", item["id"] + " atom prompt exceeded 1200 chars")
        else:
            rounds = len(item["rounds"])
            if item["max_turns"] != rounds:
                _fail(7, "Budgets", item["id"] + " episode max_turns did not equal rounds")
            cap = 4 if item["level"] <= 2 else 10 if item["level"] == 3 else 14
            if item["max_turns"] > rounds or item["max_turns"] > cap:
                _fail(7, "Budgets", item["id"] + " episode max_turns exceeded cap")

        for obs in observations:
            if len(obs) > 800:
                _fail(7, "Budgets", item["id"] + " observation exceeded 800 chars")

    for item, score in zip(items, scores):
        if score != 1.0:
            _fail(7, "Budgets", item["id"] + " oracle scoring failed during timing")

    ms_per_item = (elapsed / len(items)) * 1000.0
    if ms_per_item >= 50.0:
        _fail(7, "Budgets", "generation+scoring took " + _fmt(ms_per_item) + " ms/item")
    return ms_per_item


def gate_purity():
    allowed = {"random", "re"}
    with open(family.__file__, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=family.__file__)
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in allowed:
                    bad.append(root)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in allowed:
                bad.append("." * node.level + (node.module or ""))
    if bad:
        _fail(8, "Purity", "family.py imported non-allowlisted modules " + repr(sorted(bad)))


def _items(seed, n):
    for level in LEVELS:
        for mode in MODES:
            for item in family.generate(seed, level, n, mode):
                yield item


def _run_policy(item, policy, rng):
    env = family.Env(item)
    obs = env.reset()
    observations = [obs]
    history = []
    transcript = []
    done = False
    turns = 0
    while not done:
        if turns > item["max_turns"] + 1:
            _fail(7, "Budgets", item["id"] + " environment did not terminate")
        action = _action_for_policy(item, history, obs, policy, rng)
        transcript.append({"obs": obs, "action": action})
        next_obs, done = env.step(action)
        observations.append(next_obs)
        history.append({"obs": obs, "action": action})
        obs = next_obs
        turns += 1
    return transcript, observations


def _action_for_policy(item, history, obs, policy, rng):
    if policy == "oracle":
        return family.oracle_policy(item, history)
    if policy == "random":
        return family.random_policy(item, history, rng)
    if policy == "noisy":
        if rng.random() < 0.50:
            return family.oracle_policy(item, history)
        return family.random_policy(item, history, rng)
    if policy == "empty":
        return ""
    if policy == "constant":
        return "ANSWER: NONE" if item["mode"] == "atom" else "TAKE NONE"
    if policy == "echo":
        return obs
    raise ValueError("unknown policy " + policy)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fmt(value):
    return "%.6f" % value


def _fail(number, name, message):
    raise AssertionError("Gate %s %s: %s" % (number, name, message))


if __name__ == "__main__":
    main()
