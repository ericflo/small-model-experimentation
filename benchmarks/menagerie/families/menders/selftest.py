import ast
import json
import random
import statistics
import sys
import time
from pathlib import Path

from . import family


LEVELS = (1, 2, 3, 4)
MODES = ("atom", "episode")


def _json_blob(items):
    return json.dumps(items, sort_keys=True, separators=(",", ":"))


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _run_item(item, policy, rng=None, observations=None):
    env = family.Env(item)
    obs = env.reset()
    if observations is not None:
        observations.append((item["mode"], obs))
    history = []
    if item["mode"] == "atom":
        if rng is None:
            action = policy(item, history)
        else:
            action = policy(item, history, rng)
        history.append({"obs": obs, "action": action})
        return family.score(item, history)["score"]
    for _ in range(item["max_turns"]):
        if rng is None:
            action = policy(item, history)
        else:
            action = policy(item, history, rng)
        history.append({"obs": obs, "action": action})
        obs, done = env.step(action)
        if observations is not None:
            observations.append((item["mode"], obs))
        if done:
            break
    return family.score(item, history)["score"]


def _episode_policy_scores(items, policy, seed, observations=None):
    rng = random.Random(seed)
    return [_run_item(item, policy, rng, observations) for item in items]


def _constant_policy(item, history):
    if item["mode"] == "atom":
        return "ANSWER: 1: set a 0"
    return "MEND 1: set a 0"


def _empty_policy(item, history):
    return ""


def _echo_policy(item, history):
    return history[-1]["obs"] if history else ""


def _bare_oracle_policy(item, history):
    full = family.oracle_policy(item, history)
    parsed = family._extract_answer(full)
    if parsed is None:
        parsed = family._extract_mend(full)
    if parsed is None:
        return ""
    return "%d: %s" % (parsed[0], parsed[1])


def _noisy_policy(item, history, rng):
    if rng.random() < 0.5:
        return family.oracle_policy(item, history)
    return family.random_policy(item, history, rng)


def _content_set(items):
    content = set()
    for item in items:
        content.update(item["tokens"].values())
        content.update(item["variables"])
        content.update(item["program"])
        content.update(item["_correct_program"])
    return content


def gate_1_determinism():
    # 1. Determinism: generate(7, L, 6, mode) twice -> identical JSON, all levels/modes.
    for level in LEVELS:
        for mode in MODES:
            first = family.generate(7, level, 6, mode)
            second = family.generate(7, level, 6, mode)
            assert _json_blob(first) == _json_blob(second), (
                "Gate 1 Determinism: generate(7, L, 6, mode) twice -> "
                "identical JSON, all levels/modes; failed L%s %s" % (level, mode)
            )


def gate_2_seed_disjointness():
    # 2. Seed disjointness: seeds 7 vs 8 share no item content.
    for level in LEVELS:
        for mode in MODES:
            seed_7 = _content_set(family.generate(7, level, 6, mode))
            seed_8 = _content_set(family.generate(8, level, 6, mode))
            overlap = seed_7 & seed_8
            assert not overlap, (
                "Gate 2 Seed disjointness: seeds 7 vs 8 share no item "
                "content; failed L%s %s overlap=%r" % (level, mode, sorted(overlap)[:3])
            )


def gate_3_oracle_perfection(observations):
    # 3. Oracle perfection: oracle_policy scores 1.0 on every item (all levels, both modes).
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(17, level, 28, mode)
            for item in items:
                result = _run_item(item, family.oracle_policy, None, observations)
                assert result == 1.0, (
                    "Gate 3 Oracle perfection: oracle_policy scores 1.0 on "
                    "every item (all levels, both modes); failed %s score=%s"
                    % (item["id"], result)
                )


def gate_4_random_floor(observations):
    # 4. Random floor: mean random_policy score <= 0.05 (or <= 0.15 with justification in README for ratio-scored families).
    scores = []
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(23, level, 32, mode)
            scores.extend(_episode_policy_scores(items, family.random_policy, 4000 + level, observations))
    mean = _mean(scores)
    assert mean <= 0.05, (
        "Gate 4 Random floor: mean random_policy score <= 0.05 "
        "(or <= 0.15 with justification in README for ratio-scored families); "
        "mean=%.4f" % mean
    )


def gate_5_degenerate_resistance(observations):
    # 5. Degenerate resistance: empty-string policy, constant most-plausible-answer policy, and echo-the-observation policy each score <= 0.1 mean.
    items = []
    for level in LEVELS:
        for mode in MODES:
            items.extend(family.generate(29, level, 28, mode))
    policies = (
        ("empty-string policy", _empty_policy),
        ("constant most-plausible-answer policy", _constant_policy),
        ("echo-the-observation policy", _echo_policy),
    )
    for name, policy in policies:
        scores = [_run_item(item, policy, None, observations) for item in items]
        mean = _mean(scores)
        assert mean <= 0.1, (
            "Gate 5 Degenerate resistance: empty-string policy, constant "
            "most-plausible-answer policy, and echo-the-observation policy "
            "each score <= 0.1 mean; %s mean=%.4f" % (name, mean)
        )


def gate_9_bare_fallback(observations):
    # 9. Bare fallback: bare valid repairs work, but degenerate bare/echo/empty actions still fail.
    for level in LEVELS:
        atom_items = family.generate(43, level, 24, "atom")
        for item in atom_items:
            obs = family.Env(item).reset()
            observations.append((item["mode"], obs))

            action = _bare_oracle_policy(item, [])
            score = family.score(item, [{"obs": obs, "action": action}])["score"]
            assert score == 1.0, (
                "Gate 9 Bare fallback: atom bare-answer oracle failed %s score=%s"
                % (item["id"], score)
            )

            score = family.score(item, [{"obs": obs, "action": obs}])["score"]
            assert score <= 0.1, (
                "Gate 9 Bare fallback: atom echo observation degenerate failed %s score=%s"
                % (item["id"], score)
            )

            score = family.score(item, [{"obs": obs, "action": "1: set a 0"}])["score"]
            assert score <= 0.1, (
                "Gate 9 Bare fallback: atom bare constant degenerate failed %s score=%s"
                % (item["id"], score)
            )

            score = family.score(item, [{"obs": obs, "action": ""}])["score"]
            assert score <= 0.1, (
                "Gate 9 Bare fallback: atom empty degenerate failed %s score=%s"
                % (item["id"], score)
            )

        episode_items = family.generate(43, level, 24, "episode")
        for item in episode_items:
            score = _run_item(item, _bare_oracle_policy, None, observations)
            assert score == 1.0, (
                "Gate 9 Bare fallback: episode bare-answer oracle failed %s score=%s"
                % (item["id"], score)
            )

            score = _run_item(item, _echo_policy, None, observations)
            assert score <= 0.1, (
                "Gate 9 Bare fallback: episode echo observation degenerate failed %s score=%s"
                % (item["id"], score)
            )

            score = _run_item(item, _constant_policy, None, observations)
            assert score <= 0.1, (
                "Gate 9 Bare fallback: episode constant degenerate failed %s score=%s"
                % (item["id"], score)
            )

            score = _run_item(item, _empty_policy, None, observations)
            assert score <= 0.1, (
                "Gate 9 Bare fallback: episode empty degenerate failed %s score=%s"
                % (item["id"], score)
            )


def gate_6_monotone_difficulty(observations):
    # 6. Monotone difficulty: noisy-oracle (eps=0.5: each turn, 50% oracle action / 50% random) mean score is non-increasing L1->L4 (tolerance 0.05).
    means = []
    for level in LEVELS:
        items = family.generate(31, level, 64, "episode")
        rng = random.Random(6000 + level)
        scores = [_run_item(item, _noisy_policy, rng, observations) for item in items]
        means.append(_mean(scores))
    for prev, curr, level in zip(means, means[1:], (2, 3, 4)):
        assert curr <= prev + 0.05, (
            "Gate 6 Monotone difficulty: noisy-oracle (eps=0.5: each turn, "
            "50% oracle action / 50% random) mean score is non-increasing "
            "L1->L4 (tolerance 0.05); means=%r failed at L%s" % (means, level)
        )
    return means


def gate_7_budgets(observations):
    # 7. Budgets: prompt/observation char limits, max_turns caps, generation+scoring < 50 ms/item.
    for mode, obs in observations:
        assert len(obs) <= 800, (
            "Gate 7 Budgets: prompt/observation char limits, max_turns caps, "
            "generation+scoring < 50 ms/item; observation length=%d" % len(obs)
        )
        if mode == "atom":
            assert len(obs) <= 1200, (
                "Gate 7 Budgets: prompt/observation char limits, max_turns "
                "caps, generation+scoring < 50 ms/item; atom prompt length=%d"
                % len(obs)
            )
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(37, level, 8, mode)
            for item in items:
                if mode == "atom":
                    assert item["max_turns"] == 1, (
                        "Gate 7 Budgets: prompt/observation char limits, "
                        "max_turns caps, generation+scoring < 50 ms/item; "
                        "atom max_turns=%s" % item["max_turns"]
                    )
                elif level in (1, 2):
                    assert item["max_turns"] <= 4, (
                        "Gate 7 Budgets: prompt/observation char limits, "
                        "max_turns caps, generation+scoring < 50 ms/item; "
                        "L%s max_turns=%s" % (level, item["max_turns"])
                    )
                elif level == 3:
                    assert item["max_turns"] <= 10, (
                        "Gate 7 Budgets: prompt/observation char limits, "
                        "max_turns caps, generation+scoring < 50 ms/item; "
                        "L3 max_turns=%s" % item["max_turns"]
                    )
                else:
                    assert item["max_turns"] <= 14, (
                        "Gate 7 Budgets: prompt/observation char limits, "
                        "max_turns caps, generation+scoring < 50 ms/item; "
                        "L4 max_turns=%s" % item["max_turns"]
                    )
    start = time.perf_counter()
    count = 0
    for level in LEVELS:
        for mode in MODES:
            items = family.generate(41, level, 32, mode)
            for item in items:
                family.score(item, [])
                count += 1
    elapsed = time.perf_counter() - start
    avg = elapsed / count
    assert avg < 0.05, (
        "Gate 7 Budgets: prompt/observation char limits, max_turns caps, "
        "generation+scoring < 50 ms/item; avg=%.6fs" % avg
    )


def gate_8_purity():
    # 8. Purity: module imports nothing beyond the stdlib.
    source_path = Path(family.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    stdlib.update({"__future__"})
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    nonstdlib = sorted(name for name in imported if name not in stdlib)
    assert not nonstdlib, (
        "Gate 8 Purity: module imports nothing beyond the stdlib; nonstdlib=%r"
        % nonstdlib
    )


def main():
    observations = []
    gate_1_determinism()
    gate_2_seed_disjointness()
    gate_3_oracle_perfection(observations)
    gate_4_random_floor(observations)
    gate_5_degenerate_resistance(observations)
    gate_9_bare_fallback(observations)
    means = gate_6_monotone_difficulty(observations)
    gate_7_budgets(observations)
    gate_8_purity()
    print(
        "menders selftest PASS; gate9 bare fallback PASS; noisy means=%s"
        % ",".join("%.3f" % x for x in means)
    )


if __name__ == "__main__":
    main()
