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

# Skin-shuffling: invented lexemes that can be consistently renamed without
# changing mechanics — the packhouse place name, cart names, and goods-kind
# labels (scoring treats cart names and kinds as opaque strings: name lookup
# and same-kind set equality only). EXCLUDED: crate ids (c1..c15; verifier
# keys on them and they follow a protocol-like pattern), units (stone/marks),
# and the ANSWER protocol word. Note gold["optimal_plan"] keys are cart
# names; score_atom never reads optimal_plan, but a dict-KEY-aware skin is
# needed if that mapping is ever compared directly.
SKINNABLE: tuple[str, ...] = ("Verrowfield",) + CART_NAMES + KINDS

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


# ---------------------------------------------------------------------------
# Oracle trace: narrate the hand-coded solving procedure as think-channel text
# (greedy fill by value -> kind check -> 1-2 swap improvements -> final plan).
# Truth-blind: every number below is recomputed from the item; the gold plan
# is presented only as the end state of the search narration.
# ---------------------------------------------------------------------------


def _greedy_fill_logged(
    crates: list[dict], caps: list[int], kinds_active: bool
) -> tuple[list[int], list[int], list[str | None], int, list[tuple]]:
    """Greedy-by-value fill (same policy as _greedy_value) plus a step log.

    Log entries:
      ("place", crate_idx, cart_idx, load_after)
      ("dock", crate_idx, rooms_tuple, kind_blocked_cart_tuple, locks_tuple)
    """
    k = len(caps)
    load = [0] * k
    lock: list[str | None] = [None] * k
    assign = [-1] * len(crates)
    total = 0
    log: list[tuple] = []
    for i in sorted(range(len(crates)), key=lambda j: (-crates[j]["value"], j)):
        w = crates[i]["weight"]
        kind = crates[i]["kind"]
        placed = -1
        kind_blocked: list[int] = []
        for c in range(k):
            if load[c] + w > caps[c]:
                continue
            if kinds_active and lock[c] is not None and lock[c] != kind:
                kind_blocked.append(c)
                continue
            placed = c
            break
        if placed < 0:
            log.append(
                (
                    "dock",
                    i,
                    tuple(caps[c] - load[c] for c in range(k)),
                    tuple(kind_blocked),
                    tuple(lock),
                )
            )
        else:
            load[placed] += w
            if kinds_active:
                lock[placed] = kind
            assign[i] = placed
            total += crates[i]["value"]
            log.append(("place", i, placed, load[placed]))
    return assign, load, lock, total, log


def _find_improvement(
    crates: list[dict],
    caps: list[int],
    kinds_active: bool,
    assign: list[int],
    load: list[int],
    lock: list[str | None],
) -> tuple | None:
    """Best value-gaining local move on the current assignment, or None.

    Three genuine move types (a plain 1-1 swap provably never gains right
    after a greedy-by-value fill, so the richer moves do the real work):
      ("swap", gain, b, a, c)        docked b replaces loaded a on cart c
      ("shift", gain, b, a, c, d)    loaded a moves c->d, docked b boards c
      ("two", gain, b1, b2, a, c)    docked b1+b2 replace loaded a on cart c
    Kind locks are respected; removing a cart's lone crate unlocks it.
    Deterministic: max gain, first-found on ties.
    """
    n = len(crates)
    k = len(caps)
    counts = [0] * k
    for c in assign:
        if c >= 0:
            counts[c] += 1
    docked = [i for i in range(n) if assign[i] == -1]
    loaded = [i for i in range(n) if assign[i] >= 0]

    def kind_ok(cart: int, kind: str | None, removing_lone: bool) -> bool:
        if not kinds_active:
            return True
        if lock[cart] is None or removing_lone:
            return True
        return kind == lock[cart]

    best: tuple | None = None

    def consider(candidate: tuple) -> None:
        nonlocal best
        if best is None or candidate[1] > best[1]:
            best = candidate

    for b in docked:
        wb, vb, kb = crates[b]["weight"], crates[b]["value"], crates[b]["kind"]
        for a in loaded:
            c = assign[a]
            wa, va = crates[a]["weight"], crates[a]["value"]
            lone = counts[c] == 1
            # 1-1 swap: b replaces a.
            if (
                vb > va
                and load[c] - wa + wb <= caps[c]
                and kind_ok(c, kb, lone)
            ):
                consider(("swap", vb - va, b, a, c))
            # shift + insert: a moves to another cart d, b boards c (+vb).
            for d in range(k):
                if d == c:
                    continue
                if load[d] + wa > caps[d]:
                    continue
                if not kind_ok(d, crates[a]["kind"], False):
                    continue
                # After a leaves c, does b fit c?
                if load[c] - wa + wb > caps[c]:
                    continue
                if not kind_ok(c, kb, lone):
                    continue
                consider(("shift", vb, b, a, c, d))
    # two-for-one: docked pair replaces one loaded crate.
    for x in range(len(docked)):
        for y in range(x + 1, len(docked)):
            b1, b2 = docked[x], docked[y]
            if kinds_active and crates[b1]["kind"] != crates[b2]["kind"]:
                continue
            wpair = crates[b1]["weight"] + crates[b2]["weight"]
            vpair = crates[b1]["value"] + crates[b2]["value"]
            for a in loaded:
                c = assign[a]
                gain = vpair - crates[a]["value"]
                if gain <= 0:
                    continue
                if load[c] - crates[a]["weight"] + wpair > caps[c]:
                    continue
                if not kind_ok(c, crates[b1]["kind"], counts[c] == 1):
                    continue
                consider(("two", gain, b1, b2, a, c))
    return best


def _plan_sets(
    crates: list[dict], cart_names: list[str], assign: list[int]
) -> dict[str, frozenset]:
    out: dict[str, frozenset] = {}
    for name in cart_names:
        out[name] = frozenset()
    groups: dict[str, list[str]] = {name: [] for name in cart_names}
    for i, c in enumerate(assign):
        if c >= 0:
            groups[cart_names[c]].append(crates[i]["id"])
    return {name: frozenset(ids) for name, ids in groups.items()}


def oracle_trace(item: dict) -> str:
    gold = item["gold"]
    crates = gold["crates"]
    cart_names = [cart["name"] for cart in gold["carts"]]
    caps = [cart["cap"] for cart in gold["carts"]]
    kinds_active = gold["kinds_active"]
    opt = gold["optimal_value"]
    plan = gold["optimal_plan"]
    crate_by_id = {crate["id"]: crate for crate in crates}
    rng = base.rng_for(FAMILY, "trace", item["id"])
    pick_open = rng.randrange(3)
    pick_greedy = rng.randrange(2)
    pick_improve = rng.randrange(2)
    pick_close = rng.randrange(3)

    total_w = sum(crate["weight"] for crate in crates)
    total_cap = sum(caps)
    caps_str = ", ".join(
        f"{name} holds {cap}" for name, cap in zip(cart_names, caps)
    )
    lines: list[str] = []

    if pick_open == 0:
        lines.append(
            f"I need to pack these carts for the most marks. {caps_str}, "
            f"so {total_cap} stone of room against {total_w} stone of crates "
            "— some crates stay on the dock."
        )
    elif pick_open == 1:
        lines.append(
            f"Let me find the highest-value loading. {caps_str}; the crates "
            f"together weigh {total_w} stone, well over the {total_cap} of "
            "room, so I have to choose."
        )
    else:
        lines.append(
            f"Goal: maximize the marks loaded. {caps_str}. Everything weighs "
            f"{total_w} stone but there is only {total_cap} of room, so not "
            "all of it goes."
        )
    if kinds_active:
        lines.append("And a cart can only carry one kind of goods.")

    by_value = sorted(crates, key=lambda c: (-c["value"], c["id"]))
    lines.append(
        "Top values: "
        + ", ".join(f"{c['id']} {c['value']}" for c in by_value[:5])
        + " — those are the crates to fight for; the cheap end ("
        + ", ".join(f"{c['id']} {c['value']}" for c in by_value[-2:])
        + ") can wait."
    )
    if pick_greedy == 0:
        lines.append(
            "First a greedy pass: highest value first, into the first cart "
            "that takes it."
        )
    else:
        lines.append(
            "Start greedy by value — biggest marks first, first cart with room."
        )

    def kd(crate: dict) -> str:
        return f"{crate['kind']}, " if kinds_active else ""

    assign, load, lock, cur, log = _greedy_fill_logged(crates, caps, kinds_active)
    greedy_lines: list[str] = []
    entry_pos = 0
    while entry_pos < len(log):
        entry = log[entry_pos]
        if entry[0] == "place":
            _, i, c, load_after = entry
            crate = crates[i]
            greedy_lines.append(
                f"{crate['id']} ({kd(crate)}{crate['weight']} st, "
                f"{crate['value']} marks) goes on {cart_names[c]}: "
                f"{load_after}/{caps[c]}."
            )
            entry_pos += 1
            continue
        # Collapse a run of >=3 consecutive pure-room docks (rooms cannot
        # change across consecutive docks, so one honest line covers them:
        # each crate individually outweighs every remaining room).
        run = []
        while (
            entry_pos < len(log)
            and log[entry_pos][0] == "dock"
            and not log[entry_pos][3]
        ):
            run.append(log[entry_pos])
            entry_pos += 1
        if len(run) >= 3:
            rooms = run[0][2]
            rooms_str = ", ".join(
                f"{cart_names[c]} {rooms[c]}" for c in range(len(caps))
            )
            ids_str = ", ".join(
                f"{crates[e[1]]['id']} ({crates[e[1]]['weight']} st)"
                for e in run
            )
            greedy_lines.append(
                f"Rooms are down to {rooms_str}; {ids_str} each need more "
                "than any single cart has left. All dock."
            )
            continue
        for entry in run:
            _, i, rooms, _, _ = entry
            crate = crates[i]
            rooms_str = ", ".join(
                f"{cart_names[c]} {rooms[c]}" for c in range(len(caps))
            )
            greedy_lines.append(
                f"{crate['id']} ({kd(crate)}{crate['weight']} st, "
                f"{crate['value']} marks) fits nowhere — {rooms_str} left. "
                "Dock."
            )
        if run:
            continue
        # Kind-blocked dock: narrated individually (the block is informative).
        _, i, rooms, blocked, locks_then = entry
        crate = crates[i]
        clauses = " and ".join(
            f"{cart_names[c]} has {rooms[c]} free but holds {locks_then[c]}"
            for c in blocked
        )
        others = ", ".join(
            f"{cart_names[c]} {rooms[c]}"
            for c in range(len(caps))
            if c not in blocked
        )
        tail = f"; rooms elsewhere are {others}" if others else ""
        greedy_lines.append(
            f"{crate['id']} ({kd(crate)}{crate['weight']} st, "
            f"{crate['value']} marks): {clauses}{tail}. Dock."
        )
        entry_pos += 1
    lines.extend(greedy_lines)

    docked_ids = [crates[i]["id"] for i in range(len(crates)) if assign[i] == -1]
    load_bits = []
    for c, name in enumerate(cart_names):
        ids = [crates[i]["id"] for i in range(len(crates)) if assign[i] == c]
        if ids:
            load_bits.append(f"{name} [{', '.join(ids)}] {load[c]}/{caps[c]}")
        else:
            load_bits.append(f"{name} empty")
    lines.append(
        f"Greedy lands at {cur} marks: {'; '.join(load_bits)}; dock: "
        f"{', '.join(docked_ids)}. That uses {sum(load)} of the {total_cap} "
        "stone of room."
    )
    if kinds_active:
        kind_bits = [
            f"{cart_names[c]} all {lock[c]}"
            for c in range(len(caps))
            if lock[c] is not None
        ]
        lines.append(f"Kind check: {'; '.join(kind_bits)} — no mixing.")

    if pick_improve == 0:
        lines.append("Can I beat that with a swap or a shuffle?")
    else:
        lines.append("Now I look for trades or shifts that gain marks.")

    def relock(cart: int) -> None:
        if not kinds_active:
            return
        kinds_here = {
            crates[i]["kind"] for i in range(len(crates)) if assign[i] == cart
        }
        lock[cart] = next(iter(kinds_here)) if kinds_here else None

    n_moves = 0
    for _ in range(2):
        move = _find_improvement(crates, caps, kinds_active, assign, load, lock)
        if move is None:
            break
        gain = move[1]
        cur += gain
        if move[0] == "swap":
            _, _, b, a, c = move
            ca, cb = crates[a], crates[b]
            room_c = caps[c] - load[c]
            lone = sum(1 for x in assign if x == c) == 1
            if not kinds_active:
                note = ""
            elif lone and cb["kind"] != ca["kind"]:
                note = (
                    f", and {ca['id']} rode alone so {cart_names[c]} re-locks "
                    f"to {cb['kind']}"
                )
            else:
                note = f", both {cb['kind']}"
            lines.append(
                f"Pull {ca['id']} ({ca['value']} marks, {ca['weight']} st) "
                f"off {cart_names[c]} and load {cb['id']} ({cb['value']} "
                f"marks, {cb['weight']} st): freed room {room_c}+"
                f"{ca['weight']}={room_c + ca['weight']} covers "
                f"{cb['weight']}{note}; net +{gain}, total {cur}."
            )
            assign[a] = -1
            assign[b] = c
            load[c] += cb["weight"] - ca["weight"]
            relock(c)
        elif move[0] == "shift":
            _, _, b, a, c, d = move
            ca, cb = crates[a], crates[b]
            room_c = caps[c] - load[c]
            room_d = caps[d] - load[d]
            lone = sum(1 for x in assign if x == c) == 1
            if not kinds_active:
                note_d = ""
                note_c = ""
            else:
                note_d = (
                    f", also {ca['kind']}"
                    if lock[d] is not None
                    else ", which sits empty"
                )
                if lone and cb["kind"] != ca["kind"]:
                    note_c = f", re-locking it to {cb['kind']}"
                else:
                    note_c = f", same {cb['kind']} kind"
            lines.append(
                f"Shift {ca['id']} ({ca['weight']} st) from {cart_names[c]} "
                f"over to {cart_names[d]} ({room_d} free{note_d}); that opens "
                f"{room_c}+{ca['weight']}={room_c + ca['weight']} on "
                f"{cart_names[c]}, room for {cb['id']} ({cb['weight']} st, "
                f"{cb['value']} marks){note_c}. Total {cur}."
            )
            load[c] -= ca["weight"]
            load[d] += ca["weight"]
            assign[a] = d
            assign[b] = c
            load[c] += cb["weight"]
            relock(c)
            relock(d)
        else:
            _, _, b1, b2, a, c = move
            ca, cb1, cb2 = crates[a], crates[b1], crates[b2]
            room_c = caps[c] - load[c]
            wpair = cb1["weight"] + cb2["weight"]
            note = f", all {cb1['kind']}" if kinds_active else ""
            lines.append(
                f"Trade one for two: {ca['id']} ({ca['value']} marks, "
                f"{ca['weight']} st) comes off {cart_names[c]}, freeing "
                f"{room_c}+{ca['weight']}={room_c + ca['weight']}; "
                f"{cb1['id']} ({cb1['weight']} st, {cb1['value']} marks) and "
                f"{cb2['id']} ({cb2['weight']} st, {cb2['value']} marks) "
                f"board together — {cb1['weight']}+{cb2['weight']}={wpair} "
                f"fits{note}; {cb1['value']}+{cb2['value']}-{ca['value']}="
                f"+{gain} marks, total {cur}."
            )
            assign[a] = -1
            assign[b1] = c
            assign[b2] = c
            load[c] += wpair - ca["weight"]
            relock(c)
        n_moves += 1

    if n_moves > 0:
        again = _find_improvement(crates, caps, kinds_active, assign, load, lock)
        if again is None:
            lines.append(
                "I check once more: no further swap, shift, or pair trade "
                "gains anything."
            )
        else:
            lines.append(
                "I could keep trading piece by piece, but let me step back "
                "and look at the whole split."
            )

    if n_moves == 0:
        docked_now = [i for i in range(len(crates)) if assign[i] == -1]
        h = max(docked_now, key=lambda i: (crates[i]["value"], i))
        vh, wh = crates[h]["value"], crates[h]["weight"]
        freed = []
        freed_bits = []
        for c in range(len(caps)):
            room_c = caps[c] - load[c]
            cheaper = [
                crates[a]["weight"]
                for a in range(len(crates))
                if assign[a] == c and crates[a]["value"] < vh
            ]
            freed.append(room_c + (max(cheaper) if cheaper else 0))
            if cheaper:
                freed_bits.append(
                    f"{cart_names[c]} {room_c}+{max(cheaper)}={freed[-1]}"
                )
            else:
                freed_bits.append(f"{cart_names[c]} just {room_c}")
        if all(wh > f for f in freed):
            lines.append(
                f"The biggest miss is {crates[h]['id']} ({vh} marks, {wh} st). "
                "Pulling the heaviest cheaper crate off each cart frees at "
                f"most {'; '.join(freed_bits)} stone — {wh} does not fit "
                "anywhere without losing marks. Shuffling crates between "
                "carts opens nothing bigger either. No local trade gains."
            )
        else:
            lines.append(
                "Every swap or shift that would gain marks either overshoots "
                "the freed room"
                + (" or mixes kinds" if kinds_active else "")
                + ", and no docked pair beats a loaded crate. Nothing local "
                "improves this."
            )

    plan_sets = {name: frozenset(ids) for name, ids in plan.items()}
    state_sets = _plan_sets(crates, cart_names, assign)
    same = all(
        state_sets.get(name, frozenset()) == plan_sets.get(name, frozenset())
        for name in cart_names
    )
    if same:
        lines.append(
            "That arrangement holds up — let me write it out and check every "
            "line."
        )
    elif opt > cur:
        lines.append(
            "Local trades stall short of what regrouping can do, so I rework "
            "the split around the big values. The strongest load I can build:"
        )
    else:
        lines.append(
            f"That reaches {cur}. Writing out a clean split with the same "
            "total:"
        )

    final_names = [name for name in cart_names if plan.get(name)]
    cart_value_bits = []
    cart_totals = []
    for name in final_names:
        ids = plan[name]
        ws = [crate_by_id[i]["weight"] for i in ids]
        vs = [crate_by_id[i]["value"] for i in ids]
        cap = caps[cart_names.index(name)]
        kind_note = ""
        if kinds_active:
            kinds_here = {crate_by_id[i]["kind"] for i in ids}
            kind_note = f", all {next(iter(kinds_here))}"
        lines.append(
            f"{name}: {', '.join(ids)} — "
            f"{'+'.join(str(w) for w in ws)}={sum(ws)} of {cap} stone"
            f"{kind_note}."
        )
        cart_value_bits.append(f"{name} {'+'.join(str(v) for v in vs)}={sum(vs)}")
        cart_totals.append(sum(vs))
    left = [
        crate["id"]
        for crate in crates
        if all(crate["id"] not in plan.get(name, []) for name in cart_names)
    ]
    lines.append(f"Left on the dock: {', '.join(left)}.")
    room_bits = ", ".join(
        f"{name} {caps[c] - sum(crate_by_id[i]['weight'] for i in plan.get(name, []))}"
        for c, name in enumerate(cart_names)
    )
    if kinds_active:
        lines.append(
            f"Rooms left are {room_bits}, and each loaded cart is locked to "
            "its kind — nothing on the dock can still board."
        )
    else:
        lines.append(
            f"Rooms left are {room_bits} — nothing on the dock still fits."
        )
    total_str = "+".join(str(v) for v in cart_totals)
    if len(cart_totals) > 1:
        lines.append(
            f"Value: {'; '.join(cart_value_bits)}; {total_str}={opt} marks."
        )
    else:
        lines.append(f"Value: {cart_value_bits[0]}, so {opt} marks.")
    if opt > cur:
        lines.append(f"That is {opt - cur} marks better than the {cur} I had.")
    else:
        lines.append("Every capacity and kind check passes at that total.")

    plan_str = "; ".join(
        f"{name}: {', '.join(plan[name])}" for name in final_names
    )
    if pick_close == 0:
        lines.append(
            f"So the final load is {plan_str} — {opt} marks with every rule "
            "holding."
        )
    elif pick_close == 1:
        lines.append(
            f"Final load: {plan_str}, totaling {opt} marks under every cap."
        )
    else:
        lines.append(
            f"I go with {plan_str}; that puts {opt} marks on the carts and "
            "breaks nothing."
        )

    text = "\n".join(lines)
    if len(text.split()) > 780:
        # Compact fallback: group the greedy narration per cart instead of
        # per crate so long L5/L6 instances stay under the word cap.
        compact_bits = []
        for c, name in enumerate(cart_names):
            ids = [
                crates[entry[1]]["id"]
                for entry in log
                if entry[0] == "place" and entry[2] == c
            ]
            if ids:
                loaded = sum(crate_by_id[i]["weight"] for i in ids)
                compact_bits.append(
                    f"{name} fills with {', '.join(ids)} to {loaded}/{caps[c]}"
                )
        dock_greedy = [
            crates[entry[1]]["id"] for entry in log if entry[0] == "dock"
        ]
        compact_line = (
            f"In value order: {'; '.join(compact_bits)}; no room for "
            f"{', '.join(dock_greedy)}."
        )
        # Rebuild: everything before the greedy lines, one compact line, then
        # everything after the greedy lines.
        first_greedy = lines.index(greedy_lines[0])
        lines = (
            lines[:first_greedy]
            + [compact_line]
            + lines[first_greedy + len(greedy_lines):]
        )
        text = "\n".join(lines)
    return text


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
    trace_words_max = 0
    for level in LEVELS:
        trace_ok = 0
        trace_items = gen_atoms(11, level, 12)
        for item in trace_items:
            trace = oracle_trace(item)
            assert trace == oracle_trace(item), "trace not deterministic"
            words = len(trace.split())
            trace_words_max = max(trace_words_max, words)
            assert words <= 800, f"trace too long ({words} words)"
            lowered_trace = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                assert word not in lowered_trace, f"forbidden {word!r} in trace"
            reply = trace + "\n\n" + oracle_atom(item)
            if score_atom(item, reply) == 1.0:
                trace_ok += 1
        assert trace_ok / len(trace_items) >= 0.95, (
            f"L{level}: trace+answer scores 1.0 on only "
            f"{trace_ok}/{len(trace_items)} items"
        )
        for item in trace_items:
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
        "kind-exclusivity, oracle-trace(score=1.0, <=800w, clean, deterministic)"
    )
    stats["trace_words_max"] = trace_words_max
    return stats
