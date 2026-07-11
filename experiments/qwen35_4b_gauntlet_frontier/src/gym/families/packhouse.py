"""packhouse — bounded optimization: crate-to-cart assignment (atoms only).

A warehouse loads freight carts. Each crate has a weight and a value; each
named cart has a weight capacity. The item asks for the value-maximizing
ASSIGNMENT of crates to carts under (a) per-cart weight capacities and
(b, from L3) one-kind-of-goods-per-cart exclusivity. The verifier
brute-forces the optimum at generation time (depth-first branch and bound
with a fractional-knapsack upper bound); scoring parses the
"cart: ids; cart: ids" plan case- and space-tolerantly and returns
achieved_value / optimal_value for feasible plans, 0 for any constraint
violation, unknown name/id, or double assignment. Instances where a
greedy-by-value baseline reaches >= 0.95 of the optimum are rejected at
generation, so the task rewards actual search. All content is invented.

Levels scale the crate count (8/10/12/13/14/15), the cart count (2 below
L4, 3 at L4+), the kind palette (none / 2 / 3), and tighten the combined
capacity fraction. At the frontier levels (L5/L6) a second gate rejects any
instance that a greedy-by-density baseline solves exactly, so no single
sorting heuristic cracks frontier items.
"""

from __future__ import annotations

from .. import base

FAMILY = "packhouse"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = False

CART_NAMES = ("Farrow", "Osselt", "Tinbar", "Quenby", "Marlock", "Veldane")
KINDS = ("wet", "dry", "loose")

_LEVEL_SHAPE = {
    # level: (n_crates, n_carts, n_kinds, cap_frac_range)
    1: (8, 2, 0, (0.52, 0.66)),
    2: (10, 2, 0, (0.48, 0.62)),
    3: (12, 2, 2, (0.48, 0.62)),
    4: (13, 3, 2, (0.46, 0.60)),
    5: (14, 3, 3, (0.44, 0.58)),
    6: (15, 3, 3, (0.40, 0.54)),
}

# Reject-and-regenerate gate: greedy-by-value must land strictly below this
# fraction of the optimum, so search (not sorting) is what scores.
GREEDY_GATE = 0.95


def _solve(
    crates: list[dict], caps: list[int], kinds_active: bool
) -> tuple[int, list[int]]:
    """Optimal loaded value and per-crate cart index (-1 = left behind).

    Depth-first branch and bound. Crates are visited in value-density order
    so the fractional-knapsack relaxation over the combined remaining
    capacity (which ignores cart splits and kind locks, hence is a valid
    upper bound) prunes aggressively. Carts with identical remaining room
    and kind lock are interchangeable, so only the first is branched.
    """
    n = len(crates)
    k = len(caps)
    order = sorted(
        range(n), key=lambda i: (-crates[i]["value"] / crates[i]["weight"], i)
    )
    values = [crates[i]["value"] for i in order]
    weights = [crates[i]["weight"] for i in order]
    kinds = [crates[i]["kind"] for i in order]

    best_value = -1
    best_assign: list[int] = [-1] * n
    assign = [-1] * n
    load = [0] * k
    lock: list[str | None] = [None] * k

    def dfs(i: int, cur: int) -> None:
        nonlocal best_value, best_assign
        if cur > best_value:
            best_value = cur
            best_assign = assign.copy()
        if i == n:
            return
        room = float(sum(caps[c] - load[c] for c in range(k)))
        upper = float(cur)
        for j in range(i, n):
            if weights[j] <= room:
                upper += values[j]
                room -= weights[j]
            else:
                upper += values[j] * room / weights[j]
                break
        # Values are integers: no improvement is possible unless the
        # relaxation clears best_value + 1 (0.5 absorbs float error).
        if upper <= best_value + 0.5:
            return
        w, v = weights[i], values[i]
        seen: set[tuple[int, str | None]] = set()
        for c in range(k):
            room_c = caps[c] - load[c]
            if room_c < w:
                continue
            key = (room_c, lock[c])
            if key in seen:
                continue
            seen.add(key)
            if kinds_active and lock[c] is not None and lock[c] != kinds[i]:
                continue
            prev = lock[c]
            if kinds_active:
                lock[c] = kinds[i]
            load[c] += w
            assign[i] = c
            dfs(i + 1, cur + v)
            assign[i] = -1
            load[c] -= w
            lock[c] = prev
        dfs(i + 1, cur)

    dfs(0, 0)
    result = [-1] * n
    for pos, orig in enumerate(order):
        result[orig] = best_assign[pos]
    return best_value, result


def _greedy_value(
    crates: list[dict], caps: list[int], kinds_active: bool, *, by_density: bool = False
) -> int:
    """Greedy baseline: best crate first (value, or value/weight), first cart
    that fits."""
    if by_density:
        key = lambda i: (-crates[i]["value"] / crates[i]["weight"], i)  # noqa: E731
    else:
        key = lambda i: (-crates[i]["value"], i)  # noqa: E731
    k = len(caps)
    load = [0] * k
    lock: list[str | None] = [None] * k
    total = 0
    for i in sorted(range(len(crates)), key=key):
        w = crates[i]["weight"]
        kind = crates[i]["kind"]
        for c in range(k):
            if load[c] + w > caps[c]:
                continue
            if kinds_active and lock[c] is not None and lock[c] != kind:
                continue
            load[c] += w
            if kinds_active:
                lock[c] = kind
            total += crates[i]["value"]
            break
    return total


def _kinds_phrase(palette: list[str]) -> str:
    if len(palette) == 2:
        return f"{palette[0]} and {palette[1]}"
    return f"{', '.join(palette[:-1])}, or {palette[-1]}"


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict | None:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    n, k, n_kinds, frac_range = _LEVEL_SHAPE[level]
    kinds_active = n_kinds > 0

    cart_names = rng.sample(CART_NAMES, k)
    palette = rng.sample(KINDS, n_kinds) if kinds_active else []
    # Distinct values keep the optimum unique-ish (few value-sum collisions).
    values = rng.sample(range(8, 80), n)
    crates = [
        {
            "id": f"c{i + 1}",
            "weight": rng.randint(3, 11),
            "value": values[i],
            "kind": rng.choice(palette) if kinds_active else None,
        }
        for i in range(n)
    ]
    if kinds_active:
        for kind in palette:
            if sum(1 for crate in crates if crate["kind"] == kind) < 2:
                return None

    total_weight = sum(crate["weight"] for crate in crates)
    total_cap = round(total_weight * rng.uniform(*frac_range))
    if k == 2:
        first = round(total_cap * rng.uniform(0.38, 0.62))
        caps = [first, total_cap - first]
    else:
        c1 = round(total_cap * rng.uniform(0.26, 0.40))
        c2 = round(total_cap * rng.uniform(0.26, 0.40))
        caps = [c1, c2, total_cap - c1 - c2]
    if min(caps) < 6:
        return None

    optimal_value, assign = _solve(crates, caps, kinds_active)
    total_value = sum(values)
    # The capacity constraint must bind: load-everything must be infeasible
    # and something must be loadable.
    if not (0 < optimal_value < total_value):
        return None
    greedy = _greedy_value(crates, caps, kinds_active)
    if greedy >= GREEDY_GATE * optimal_value:
        return None
    if level >= 5:
        # Frontier gate: greedy-by-density must also fall short of optimal.
        if _greedy_value(crates, caps, kinds_active, by_density=True) >= optimal_value:
            return None

    plan: dict[str, list[str]] = {name: [] for name in cart_names}
    for i, cart_index in enumerate(assign):
        if cart_index >= 0:
            plan[cart_names[cart_index]].append(crates[i]["id"])

    # Format example built from the three LOWEST-value crates so literally
    # copying the example line stays a floor, not a strategy.
    example_ids = [
        crate["id"] for crate in sorted(crates, key=lambda c: (c["value"], c["id"]))[:3]
    ]
    lines = [
        "The Verrowfield packhouse is loading freight carts for the morning",
        "run. Each crate below goes onto exactly one cart or stays on the",
        "dock. Load crates so the total value loaded is as large as possible",
        "while every rule holds.",
        "",
        "Carts:",
    ]
    for name, cap in zip(cart_names, caps):
        lines.append(f"- {name}: holds up to {cap} stone")
    lines += ["", "Crates (weight in stone, value in marks):"]
    for crate in crates:
        tag = f" ({crate['kind']})" if kinds_active else ""
        lines.append(
            f"{crate['id']}{tag}: weight {crate['weight']}, value {crate['value']}"
        )
    lines += [
        "",
        "Rules:",
        "- The crates on a cart must not weigh more than the cart holds.",
    ]
    if kinds_active:
        lines.append(
            f"- One kind of goods per cart: never mix {_kinds_phrase(palette)} "
            "crates on the same cart."
        )
    lines += [
        "",
        "Reply with one line assigning crates to carts, for example:",
        f"{cart_names[0]}: {example_ids[0]}, {example_ids[1]}; "
        f"{cart_names[1]}: {example_ids[2]}",
        "List each loaded crate once; leave out empty carts and crates left",
        "on the dock. Do NOT answer with the total value; answer with the",
        "assignment itself.",
        "",
        "End your reply with exactly one final line of the form:",
        f"ANSWER: {cart_names[0]}: <crate ids>; {cart_names[1]}: <crate ids>",
    ]
    prompt = "\n".join(lines)

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": {
            "carts": [
                {"name": name, "cap": cap} for name, cap in zip(cart_names, caps)
            ],
            "crates": crates,
            "kinds_active": kinds_active,
            "optimal_value": optimal_value,
            "optimal_plan": plan,
            "greedy_value": greedy,
        },
    }


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        item = None
        for attempt in range(200):
            candidate = _gen_one(seed, level, index, attempt)
            if candidate is not None and len(candidate["prompt"]) <= base.atom_prompt_limit(level):
                item = candidate
                break
        if item is None:  # pragma: no cover - generator bug
            raise RuntimeError(f"{FAMILY}: no valid instance for L{level} index {index}")
        items.append(item)
    return items


def score_atom(item: dict, reply_text: str) -> float:
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    gold = item["gold"]
    carts = {cart["name"].lower(): cart for cart in gold["carts"]}
    crates = {crate["id"]: crate for crate in gold["crates"]}

    assigned: dict[str, str] = {}  # crate id -> cart key
    for segment in answer.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            return 0.0
        cart_part, ids_part = segment.split(":", 1)
        cart_key = base.canon_word(cart_part)
        if cart_key not in carts:
            return 0.0
        for token in base.canon_list(ids_part):
            if token not in crates:
                return 0.0
            if token in assigned and assigned[token] != cart_key:
                return 0.0  # double assignment across carts
            assigned[token] = cart_key
    if not assigned:
        return 0.0

    groups: dict[str, list[str]] = {}
    for crate_id, cart_key in assigned.items():
        groups.setdefault(cart_key, []).append(crate_id)
    for cart_key, ids in groups.items():
        if sum(crates[i]["weight"] for i in ids) > carts[cart_key]["cap"]:
            return 0.0
        if gold["kinds_active"]:
            if len({crates[i]["kind"] for i in ids}) > 1:
                return 0.0
    achieved = sum(crates[i]["value"] for i in assigned)
    return achieved / gold["optimal_value"]


def oracle_atom(item: dict) -> str:
    plan = item["gold"]["optimal_plan"]
    parts = [f"{name}: {', '.join(ids)}" for name, ids in plan.items() if ids]
    return "ANSWER: " + "; ".join(parts)


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    stats = base.selftest_atoms(
        module,
        degenerate_replies=(
            "",
            "ANSWER: 0",
            "ANSWER: yes",
            "ANSWER: none",
            "ANSWER: c1, c2",
            "ANSWER: cart: c1",
        ),
    )
    # Family-specific exactness checks.
    for level in LEVELS:
        for item in gen_atoms(11, level, 12):
            gold = item["gold"]
            crates = {crate["id"]: crate for crate in gold["crates"]}
            cart0 = gold["carts"][0]["name"]
            cart1 = gold["carts"][1]["name"]
            total_value = sum(crate["value"] for crate in gold["crates"])
            assert 0 < gold["optimal_value"] < total_value, "constraint does not bind"
            assert (
                gold["greedy_value"] < GREEDY_GATE * gold["optimal_value"]
            ), "greedy gate not enforced"
            if level >= 5:
                caps = [cart["cap"] for cart in gold["carts"]]
                density = _greedy_value(
                    gold["crates"], caps, gold["kinds_active"], by_density=True
                )
                assert density < gold["optimal_value"], "density gate not enforced"
            assert score_atom(item, oracle_atom(item)) == 1.0, "oracle not optimal"

            all_ids = ", ".join(crate["id"] for crate in gold["crates"])
            assert (
                score_atom(item, f"ANSWER: {cart0}: {all_ids}") == 0.0
            ), "overload not rejected"
            assert score_atom(item, f"ANSWER: {cart0}: z9") == 0.0, "unknown id kept"
            assert score_atom(item, "ANSWER: Grimble: c1") == 0.0, "unknown cart kept"
            assert score_atom(item, "ANSWER:") == 0.0, "empty answer not rejected"

            plan = gold["optimal_plan"]
            loaded = [i for ids in plan.values() for i in ids]
            some_id = loaded[0]
            assert (
                score_atom(item, f"ANSWER: {cart0}: {some_id}; {cart1}: {some_id}")
                == 0.0
            ), "double assignment not rejected"

            # Objective partial credit: drop the highest-value loaded crate.
            if len(loaded) >= 2:
                drop = max(loaded, key=lambda i: crates[i]["value"])
                parts = []
                for name, ids in plan.items():
                    kept = [i for i in ids if i != drop]
                    if kept:
                        parts.append(f"{name}: {', '.join(kept)}")
                partial = score_atom(item, "ANSWER: " + "; ".join(parts))
                assert 0.0 < partial < 1.0, "objective partial credit broken"

            # Case/space tolerance: a mangled oracle plan still scores 1.0.
            mangled = " ; ".join(
                f" {name.upper()} : {' , '.join(i.upper() for i in ids)}"
                for name, ids in plan.items()
                if ids
            )
            assert score_atom(item, f"answer: {mangled}") == 1.0, "case tolerance broken"

            if gold["kinds_active"]:
                cap0 = gold["carts"][0]["cap"]
                mixed = None
                crate_list = gold["crates"]
                for a in range(len(crate_list)):
                    for b in range(a + 1, len(crate_list)):
                        ca, cb = crate_list[a], crate_list[b]
                        if (
                            ca["kind"] != cb["kind"]
                            and ca["weight"] + cb["weight"] <= cap0
                        ):
                            mixed = (ca["id"], cb["id"])
                            break
                    if mixed:
                        break
                if mixed:
                    assert (
                        score_atom(item, f"ANSWER: {cart0}: {mixed[0]}, {mixed[1]}")
                        == 0.0
                    ), "kind exclusivity not enforced"
    stats["extra_checks"] = (
        "binding, greedy-gate, frontier-density-gate, overload=0, unknown-id=0, "
        "unknown-cart=0, double-assign=0, partial-credit, case-tolerance, "
        "kind-exclusivity"
    )
    return stats
