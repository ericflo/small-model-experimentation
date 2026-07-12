"""stallwright — bounded optimization under explicit constraints (atoms only).

A festival fair grants market stalls. Each stall pays a fee and occupies
footage; the item asks for the fee-maximizing subset under (a) a total
footage cap, (b) rival pairs that may not both be granted, (c) category
quotas, and — at the frontier — (d) patron bonds (a stall joins only if its
patron stall is also granted). The verifier brute-forces the optimum at
generation time (branch and bound); scoring is achieved_fee / optimal_fee for
feasible answers, 0 for any constraint violation or unknown id. All content
is invented.

Levels scale the stall count (8/10/12/14/16/18), the number of rival pairs
and quotas, and tighten the footage cap. L5 escalates two constraint kinds
beyond footage (rivals + quotas); L6 adds a third kind (patron bonds).

``oracle_trace`` distills the solving procedure (density ranking -> greedy
feasible pass -> concrete improvement trades -> settle) into first-person
think-channel text with the item's actual arithmetic, for trace SFT.
"""

from __future__ import annotations

from collections import Counter

from .. import base

FAMILY = "stallwright"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = False

STALL_NAMES = (
    "Plumwhistle",
    "Bramblewick",
    "Copperfen",
    "Dovetallow",
    "Gorsequill",
    "Galdenrow",
    "Hobbleworth",
    "Inkmoss",
    "Juniperloft",
    "Kettlebright",
    "Lantermoor",
    "Mossgable",
    "Nettlecombe",
    "Oakhollow",
    "Pinchwick",
    "Quaverley",
    "Rushlight",
    "Snugharbor",
    "Thistledown",
    "Umbergate",
)
CATEGORIES = ("food", "cloth", "charm", "herb", "spice", "toys")

# Invented proper-noun lexemes that can be consistently renamed (whole-word
# substitution) without changing task mechanics: the stall names and the fair
# name. Stall names reach score_atom only through gold["stalls"][i]["name"]
# (plain string values, compared case-insensitively against reply tokens), so
# a skin that rewrites prompt and gold together stays consistent.
# Excluded: CATEGORIES — they are gold["quotas"] dict KEYS, which whole-word
# skinning of values does not rewrite, so renaming them would desynchronize
# stall categories from quota keys; also plain English, not invented names.
# Excluded: S<n> ids (score_atom parses int(sid[1:])), units (gilds, spans),
# numbers, and the ANSWER protocol.
SKINNABLE: tuple[str, ...] = STALL_NAMES + ("Tarnmoot",)

_LEVEL_SHAPE = {
    # level: (n_stalls, n_categories, n_rival_pairs, n_quotas, n_bonds,
    #         cap_frac_range)
    # n_bonds = patron bonds: [a, b] means grant a only if b is also granted.
    1: (8, 3, 0, 0, 0, (0.70, 0.85)),
    2: (10, 3, 1, 0, 0, (0.62, 0.78)),
    3: (12, 4, 2, 1, 0, (0.58, 0.72)),
    4: (14, 4, 3, 2, 0, (0.55, 0.68)),
    # Frontier: L5 escalates the two existing constraint kinds beyond footage
    # (rivals + quotas) on 16 stalls; L6 adds a third kind (patron bonds) on
    # 18 stalls with a tighter cap.
    5: (16, 5, 4, 2, 0, (0.50, 0.62)),
    6: (18, 5, 4, 2, 2, (0.46, 0.58)),
}


def _solve(
    stalls: list[dict],
    cap: int,
    rivals: list[list[str]],
    quotas: dict,
    bonds: list[list[str]] = (),
) -> tuple[int, list[str]]:
    """Brute-force the optimal subset via depth-first branch and bound.

    ``bonds`` are patron bonds: ``[a, b]`` means stall ``a`` may be granted
    only if stall ``b`` is also granted. With empty bonds the search is
    identical to the pre-frontier solver.
    """
    n = len(stalls)
    # Visit high-fee stalls first so the fee upper bound prunes aggressively.
    order = sorted(range(n), key=lambda i: -stalls[i]["fee"])
    fees = [stalls[i]["fee"] for i in order]
    foots = [stalls[i]["footage"] for i in order]
    cats = [stalls[i]["category"] for i in order]
    pos = {stalls[orig]["id"]: p for p, orig in enumerate(order)}
    rival_sets: list[set[int]] = [set() for _ in range(n)]
    for a, b in rivals:
        rival_sets[pos[a]].add(pos[b])
        rival_sets[pos[b]].add(pos[a])
    # needs[a] = patrons a depends on; patron_of[b] = dependents of b.
    needs: list[list[int]] = [[] for _ in range(n)]
    patron_of: list[list[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        needs[pos[a]].append(pos[b])
        patron_of[pos[b]].append(pos[a])
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + fees[i]

    best_fee = -1
    best_set: list[int] = []
    chosen: set[int] = set()
    cat_count: dict[str, int] = {}
    unmet = 0  # bonds whose dependent is chosen but patron is not (yet)

    def dfs(i: int, cur_fee: int, cur_foot: int) -> None:
        nonlocal best_fee, best_set, unmet
        if unmet == 0 and cur_fee > best_fee:
            best_fee = cur_fee
            best_set = sorted(chosen)
        if i == n or cur_fee + suffix[i] <= best_fee:
            return
        if (
            cur_foot + foots[i] <= cap
            and not (rival_sets[i] & chosen)
            and cat_count.get(cats[i], 0) < quotas.get(cats[i], n)
            # Every patron i needs must still be obtainable: already chosen,
            # or not yet decided (position after i).
            and all(b in chosen or b > i for b in needs[i])
        ):
            chosen.add(i)
            cat_count[cats[i]] = cat_count.get(cats[i], 0) + 1
            delta = sum(1 for b in needs[i] if b not in chosen)
            delta -= sum(1 for a in patron_of[i] if a in chosen)
            unmet += delta
            dfs(i + 1, cur_fee + fees[i], cur_foot + foots[i])
            unmet -= delta
            cat_count[cats[i]] -= 1
            chosen.discard(i)
        dfs(i + 1, cur_fee, cur_foot)

    dfs(0, 0, 0)
    optimal_ids = sorted(
        (stalls[order[p]]["id"] for p in best_set),
        key=lambda sid: int(sid[1:]),
    )
    return best_fee, optimal_ids


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict | None:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    n, n_cats, n_rivals, n_quotas, n_bonds, frac_range = _LEVEL_SHAPE[level]

    names = rng.sample(STALL_NAMES, n)
    cats = rng.sample(CATEGORIES, n_cats)
    stalls = [
        {
            "id": f"S{i + 1}",
            "name": names[i],
            "category": rng.choice(cats),
            "fee": rng.randint(8, 60),
            "footage": rng.randint(2, 9),
        }
        for i in range(n)
    ]

    total_foot = sum(s["footage"] for s in stalls)
    cap = int(total_foot * rng.uniform(*frac_range))

    rivals: list[list[str]] = []
    rival_indices: set[int] = set()
    if n_rivals:
        picks = rng.sample(range(n), 2 * n_rivals)
        rival_indices = set(picks)
        for k in range(n_rivals):
            a, b = sorted(picks[2 * k : 2 * k + 2])
            rivals.append([stalls[a]["id"], stalls[b]["id"]])

    # Patron bonds [dependent, patron]: dependent joins only if patron is
    # granted. Drawn from stalls untouched by rival pairs so no stall is
    # trivially locked out; distinct indices rule out bond chains.
    bonds: list[list[str]] = []
    if n_bonds:
        free = sorted(set(range(n)) - rival_indices)
        bond_picks = rng.sample(free, 2 * n_bonds)
        for k in range(n_bonds):
            dep, pat = bond_picks[2 * k : 2 * k + 2]
            bonds.append([stalls[dep]["id"], stalls[pat]["id"]])

    quotas: dict[str, int] = {}
    if n_quotas:
        counts = Counter(s["category"] for s in stalls)
        eligible = sorted(c for c, k in counts.items() if k >= 3)
        if len(eligible) < n_quotas:
            return None
        for cat in rng.sample(eligible, n_quotas):
            quotas[cat] = max(1, counts[cat] - rng.randint(1, 2))

    optimal_fee, optimal_ids = _solve(stalls, cap, rivals, quotas, bonds)
    total_fee = sum(s["fee"] for s in stalls)
    # At least one constraint must bind: take-everything must be infeasible
    # and something better than nothing must be grantable.
    if not (0 < optimal_fee < total_fee):
        return None

    lines = [
        "The Tarnmoot fair is granting market stalls. Each stall pays a fee",
        "(in gilds) and needs footage (in spans). Grant a subset of stalls so",
        "the total fee is as large as possible while every rule holds.",
        "",
        "Stalls:",
    ]
    for s in stalls:
        lines.append(
            f"{s['id']} {s['name']} ({s['category']}): fee {s['fee']}, footage {s['footage']}"
        )
    lines += [
        "",
        "Rules:",
        f"- Total footage of granted stalls must not exceed {cap} spans.",
    ]
    for a, b in rivals:
        lines.append(f"- {a} and {b} are rivals: do not grant both.")
    for cat, mx in quotas.items():
        plural = "stall" if mx == 1 else "stalls"
        lines.append(f"- Grant at most {mx} {cat} {plural}.")
    for dep, pat in bonds:
        lines.append(f"- {dep} joins only if {pat} is also granted.")
    lines += [
        "",
        "Reply with the ids of the granted stalls, comma-separated, any order.",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    prompt = "\n".join(lines)

    gold = {
        "stalls": stalls,
        "cap": cap,
        "rivals": rivals,
        "quotas": quotas,
        "optimal_fee": optimal_fee,
        "optimal_ids": optimal_ids,
    }
    if bonds:
        # Key is present only at bond-bearing levels so pre-frontier items
        # (L1-L4) stay byte-identical for longitudinal comparability.
        gold["bonds"] = bonds
    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
    }


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        item = None
        for attempt in range(64):
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
    tokens = base.canon_list(answer)
    if not tokens:
        return 0.0
    gold = item["gold"]
    by_id = {s["id"]: s for s in gold["stalls"]}
    alias = {}
    for s in gold["stalls"]:
        alias[s["id"].lower()] = s["id"]
        alias[s["name"].lower()] = s["id"]
    chosen: set[str] = set()
    for token in tokens:
        if token not in alias:
            return 0.0
        chosen.add(alias[token])
    if sum(by_id[i]["footage"] for i in chosen) > gold["cap"]:
        return 0.0
    for a, b in gold["rivals"]:
        if a in chosen and b in chosen:
            return 0.0
    for dep, pat in gold.get("bonds", []):
        if dep in chosen and pat not in chosen:
            return 0.0
    counts: dict[str, int] = {}
    for i in chosen:
        cat = by_id[i]["category"]
        counts[cat] = counts.get(cat, 0) + 1
    for cat, mx in gold["quotas"].items():
        if counts.get(cat, 0) > mx:
            return 0.0
    achieved = sum(by_id[i]["fee"] for i in chosen)
    return achieved / gold["optimal_fee"]


def oracle_atom(item: dict) -> str:
    return "ANSWER: " + ", ".join(item["gold"]["optimal_ids"])


# ---------------------------------------------------------------------------
# Oracle trace: narrate the hand-coded solving procedure (density -> greedy ->
# concrete improvement trades -> settle) as first-person think-channel text.
# Truth-blind style: every number is derived by re-running the procedure on
# the item; the gold answer is never cited, only reached.
# ---------------------------------------------------------------------------

_TRACE_FIT = (
    "{sid} fits: fee {fee}, {foot} spans; running footage {run}.",
    "Take {sid} (fee {fee}, {foot} spans) — footage now {run}.",
    "{sid} goes in: fee {fee}, {foot} spans, total footage {run}.",
)
_TRACE_SKIP = (
    "{sid} is out: {reason}.",
    "Skip {sid}: {reason}.",
    "{sid} fails here: {reason}.",
)


def _num(sid: str) -> int:
    return int(sid[1:])


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _sum_expr(vals: list[int]) -> str:
    if len(vals) == 1:
        return str(vals[0])
    return " + ".join(str(v) for v in vals) + f" = {sum(vals)}"


def _density_order(stalls: list[dict]) -> list[dict]:
    return sorted(stalls, key=lambda s: (-(s["fee"] / s["footage"]), _num(s["id"])))


def _block_reason(gold: dict, by_id: dict, sel: set, s: dict) -> str | None:
    """The first rule that blocks adding stall ``s`` to ``sel``, else None."""
    foot = sum(by_id[i]["footage"] for i in sel)
    if foot + s["footage"] > gold["cap"]:
        return (
            f"footage would reach {foot + s['footage']}, over the cap of {gold['cap']}"
        )
    for a, b in gold["rivals"]:
        if s["id"] == a and b in sel:
            return f"it is a rival of {b}, which is already in"
        if s["id"] == b and a in sel:
            return f"it is a rival of {a}, which is already in"
    quota = gold["quotas"].get(s["category"])
    if quota is not None:
        have = sum(1 for i in sel if by_id[i]["category"] == s["category"])
        if have >= quota:
            return f"the {s['category']} quota of {quota} is already used up"
    for dep, pat in gold.get("bonds", []):
        if s["id"] == dep and pat not in sel:
            return f"it may only join if {pat} is in, and {pat} is not"
    return None


def _set_violation(gold: dict, by_id: dict, sel: set) -> str | None:
    """The first rule the whole selection ``sel`` violates, else None."""
    foot = sum(by_id[i]["footage"] for i in sel)
    if foot > gold["cap"]:
        return f"footage lands at {foot}, over the cap of {gold['cap']}"
    for a, b in gold["rivals"]:
        if a in sel and b in sel:
            return f"{a} and {b} are rivals and would both be in"
    for cat, mx in gold["quotas"].items():
        have = sum(1 for i in sel if by_id[i]["category"] == cat)
        if have > mx:
            return f"the {cat} count would hit {have}, over the quota of {mx}"
    for dep, pat in gold.get("bonds", []):
        if dep in sel and pat not in sel:
            return f"{dep} would sit there without its patron {pat}"
    return None


def oracle_trace(item: dict) -> str:
    gold = item["gold"]
    stalls = gold["stalls"]
    by_id = {s["id"]: s for s in stalls}
    cap = gold["cap"]
    bonds = gold.get("bonds", [])
    rng = base.rng_for(FAMILY, "trace", item["id"])

    # Opener: state the goal and the live rule kinds.
    rules = [f"a footage cap of {cap} spans"]
    if gold["rivals"]:
        k = len(gold["rivals"])
        rules.append(f"{k} rival pair" + ("s" if k > 1 else ""))
    if gold["quotas"]:
        k = len(gold["quotas"])
        rules.append(f"{k} category quota" + ("s" if k > 1 else ""))
    if bonds:
        k = len(bonds)
        rules.append(f"{k} patron bond" + ("s" if k > 1 else ""))
    opener = rng.choice(
        (
            f"I need to grant the set of stalls with the biggest total fee, under {_join(rules)}.",
            f"Let me work out which stalls to grant. I want the largest combined fee, and the rules are {_join(rules)}.",
            f"The job: pick stalls so the fees add up as high as possible, respecting {_join(rules)}.",
        )
    )

    # Value density: fee per span.
    order = _density_order(stalls)
    lead = order[0]
    rest = ", ".join(f"{s['id']} at {s['fee'] / s['footage']:.1f}" for s in order[1:5])
    density = (
        rng.choice(
            (
                "First, fee per span, so I know who earns their footage.",
                "Start with value density — gilds per span.",
                "I'll rank by fee per span before anything else.",
            )
        )
        + f"\n{lead['id']} leads at {lead['fee']}/{lead['footage']} = "
        + f"{lead['fee'] / lead['footage']:.1f} gilds a span; then {rest}."
        + f"\nAt the bottom, {order[-1]['id']} earns only "
        + f"{order[-1]['fee'] / order[-1]['footage']:.1f} a span."
    )

    # Greedy pass in density order, narrating every take and every dead end.
    fit_t = rng.choice(_TRACE_FIT)
    skip_t = rng.choice(_TRACE_SKIP)
    sel: list[str] = []
    skip_reason: dict[str, str] = {}
    run_foot = 0
    run_fee = 0
    greedy_lines = [
        rng.choice(
            (
                "Now walk that order greedily, taking whatever still fits.",
                "Greedy pass in density order:",
                "Take them in that order while the rules allow.",
            )
        )
    ]
    for s in order:
        reason = _block_reason(gold, by_id, set(sel), s)
        if reason is None:
            sel.append(s["id"])
            run_foot += s["footage"]
            run_fee += s["fee"]
            greedy_lines.append(
                fit_t.format(sid=s["id"], fee=s["fee"], foot=s["footage"], run=run_foot)
            )
        else:
            skip_reason[s["id"]] = reason
            greedy_lines.append(skip_t.format(sid=s["id"], reason=reason))
    sel_sorted = sorted(sel, key=_num)
    greedy_lines.append(
        f"So the greedy pass holds {_join(sel_sorted)}: fee {run_fee}, footage {run_foot} of {cap}."
    )

    # Improvement: try concrete trades against the greedy selection.
    opt_ids = sorted(gold["optimal_ids"], key=_num)
    opt_set = set(opt_ids)
    opt_fee = gold["optimal_fee"]
    opt_foot = sum(by_id[i]["footage"] for i in opt_ids)
    g_set = set(sel)
    improved = g_set != opt_set
    assert _set_violation(gold, by_id, opt_set) is None

    improve_lines = [
        rng.choice(
            (
                f"Greedy lands at {run_fee} gilds. Can I beat it?",
                f"That pass gives {run_fee} gilds. Let me test some trades before settling.",
                f"So far {run_fee} gilds. I want to try a few swaps.",
            )
        )
    ]
    if improved:
        drop = sorted(g_set - opt_set, key=_num)
        add = sorted(opt_set - g_set, key=_num)
        assert add, "optimal cannot be a strict subset of a feasible greedy set"
        add_fees = [by_id[i]["fee"] for i in add]
        add_foots = [by_id[i]["footage"] for i in add]
        named = _join([f"{i} (fee {by_id[i]['fee']})" for i in add])
        if drop:
            drop_fees = [by_id[i]["fee"] for i in drop]
            drop_foots = [by_id[i]["footage"] for i in drop]
            assert run_foot - sum(drop_foots) + sum(add_foots) == opt_foot
            assert run_fee - sum(drop_fees) + sum(add_fees) == opt_fee
            improve_lines.append(
                rng.choice(
                    (
                        f"Still out with real money: {named}. Maybe a trade makes room.",
                        f"I notice {named} never made it in. A restructure might pay.",
                        f"The leftover fees nag at me: {named}. Try a trade.",
                    )
                )
            )
            improve_lines.append(
                f"Dropping {_join(drop)} frees {_sum_expr(drop_foots)} spans and gives up "
                f"{_sum_expr(drop_fees)} gilds."
            )
            improve_lines.append(
                f"Bringing in {_join(add)} needs {_sum_expr(add_foots)} spans and pays "
                f"{_sum_expr(add_fees)} gilds."
            )
            improve_lines.append(
                f"Footage: {run_foot} - {sum(drop_foots)} + {sum(add_foots)} = {opt_foot}, "
                f"within the cap of {cap}."
            )
            if gold["rivals"]:
                improve_lines.append("No rival pair ends up together.")
            for cat in sorted(gold["quotas"]):
                have = sum(1 for i in opt_set if by_id[i]["category"] == cat)
                improve_lines.append(
                    f"The {cat} count comes to {have} of {gold['quotas'][cat]} — fine."
                )
            if bonds:
                improve_lines.append("Every bonded stall keeps its patron in.")
            net = opt_fee - run_fee
            fee_line = f"Fee: {run_fee} - {sum(drop_fees)} + {sum(add_fees)} = {opt_fee}"
            if net > 0:
                improve_lines.append(fee_line + f", up {net}. I take that.")
            else:
                improve_lines.append(
                    fee_line + ". An even trade, and everything fits, so I keep it."
                )
        else:
            # Pure additions: stalls skipped earlier (patron not yet granted)
            # that fit now. Patrons in the batch go first.
            add_seq = sorted(
                add,
                key=lambda i: (
                    any(dep == i and pat in add for dep, pat in bonds),
                    _num(i),
                ),
            )
            cur = set(sel)
            cur_foot, cur_fee = run_foot, run_fee
            for sid in add_seq:
                s = by_id[sid]
                assert sid in skip_reason
                assert _block_reason(gold, by_id, cur, s) is None
                improve_lines.append(
                    f"Earlier {sid} was out because {skip_reason[sid]}. That has changed."
                )
                improve_lines.append(
                    f"Check {sid} again: footage {cur_foot} + {s['footage']} = "
                    f"{cur_foot + s['footage']} within {cap}, and every other rule holds. "
                    f"Add it; fee climbs to {cur_fee + s['fee']}."
                )
                cur.add(sid)
                cur_foot += s["footage"]
                cur_fee += s["fee"]
            assert cur == opt_set and cur_fee == opt_fee and cur_foot == opt_foot

    # Probes on the settled selection: each one genuinely dead-ends.
    excluded = sorted(
        (s for s in stalls if s["id"] not in opt_set),
        key=lambda s: (-s["fee"], _num(s["id"])),
    )
    assert excluded, "some stall must be excluded (a constraint binds)"
    probe_lines = []
    if improved:
        # Without an improvement trade the improve-intro line already frames
        # the probes, so a second framing line would just repeat it.
        probe_lines.append(
            rng.choice(
                (
                    "Any more juice in the leftovers?",
                    "Push a little further before I settle.",
                    "One more sweep of the excluded stalls.",
                )
            )
        )
    add_probe_t = rng.choice(
        (
            "Could {sid} (fee {fee}) come in on top? No: {reason}.",
            "What about adding {sid} (fee {fee}) outright? It cannot: {reason}.",
            "{sid} still pays {fee}, but adding it fails: {reason}.",
        )
    )

    def add_probe(s: dict) -> str:
        reason = _block_reason(gold, by_id, opt_set, s)
        assert reason is not None, "an addable stall contradicts optimality"
        return add_probe_t.format(sid=s["id"], fee=s["fee"], reason=reason)

    def swap_probe(s: dict) -> str:
        members = sorted(opt_set, key=lambda i: (by_id[i]["fee"], _num(i)))
        cheaper = [i for i in members if by_id[i]["fee"] < s["fee"]]
        if cheaper:
            t = cheaper[0]
            gain = s["fee"] - by_id[t]["fee"]
            violation = _set_violation(gold, by_id, (opt_set - {t}) | {s["id"]})
            assert violation is not None, "a paying swap contradicts optimality"
            return (
                f"Swap {t} (fee {by_id[t]['fee']}) out for {s['id']} (fee {s['fee']})? "
                f"That would gain {gain} gilds, but {violation}. Dead end."
            )
        t = members[0]
        loss = by_id[t]["fee"] - s["fee"]
        if loss > 0:
            return (
                f"Swap {t} (fee {by_id[t]['fee']}) out for {s['id']} (fee {s['fee']})? "
                f"That just loses {loss} gilds. No."
            )
        return (
            f"Swap {t} out for {s['id']}? Both pay {s['fee']} — an even trade at best, "
            f"no gain there."
        )

    probe_lines.append(add_probe(excluded[0]))
    probe_lines.append(swap_probe(excluded[1] if len(excluded) > 1 else excluded[0]))
    # Short-trace levels get a third probe; elsewhere the improvement trade
    # already makes it three attempted moves.
    if (not improved or item["level"] <= 2) and len(excluded) > 2:
        probe_lines.append(add_probe(excluded[2]))

    # Final verification of the settled selection, then conclude.
    check_lines = [
        f"Last check on {_join(opt_ids)}: footage "
        f"{_sum_expr([by_id[i]['footage'] for i in opt_ids])}, within {cap}."
    ]
    if gold["rivals"]:
        check_lines.append("No rival pair sits together.")
    for cat in sorted(gold["quotas"]):
        have = sum(1 for i in opt_set if by_id[i]["category"] == cat)
        check_lines.append(f"The {cat} count is {have} of {gold['quotas'][cat]}.")
    if bonds:
        check_lines.append("No dependent stall is in without its patron.")
    conclusion = rng.choice(
        (
            f"So the best selection I can find is {_join(opt_ids)}, for a total fee of {opt_fee} gilds.",
            f"I settle there: grant {_join(opt_ids)}, worth {opt_fee} gilds in total.",
            f"So the final grant is {_join(opt_ids)} — total fee {opt_fee} gilds.",
        )
    )

    paragraphs = [opener, density, "\n".join(greedy_lines)]
    if improved:
        paragraphs.append("\n".join(improve_lines))
        paragraphs.append("\n".join(probe_lines))
    else:
        paragraphs.append("\n".join(improve_lines + probe_lines))
    paragraphs.append("\n".join(check_lines))
    paragraphs.append(conclusion)
    return "\n\n".join(paragraphs)


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    stats = base.selftest_atoms(
        module,
        degenerate_replies=(
            "",
            "ANSWER: 0",
            "ANSWER: yes",
            "ANSWER: none",
            "ANSWER: S1, Z9",
        ),
    )
    # Family-specific exactness checks.
    max_trace_words = 0
    for level in LEVELS:
        items = gen_atoms(11, level, 12)
        trace_exact = 0
        for item in items:
            trace = oracle_trace(item)
            assert trace == oracle_trace(item), "oracle trace not deterministic"
            n_words = len(trace.split())
            assert n_words <= 800, f"oracle trace too long: {n_words} words"
            max_trace_words = max(max_trace_words, n_words)
            lowered_trace = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                assert word not in lowered_trace, f"forbidden word {word!r} in trace"
            assert not any(
                base.ANSWER_RE.match(line) for line in trace.splitlines()
            ), "trace contains a premature ANSWER line"
            reply = trace + "\n\n" + oracle_atom(item)
            if score_atom(item, reply) == 1.0:
                trace_exact += 1
        assert trace_exact >= 0.95 * len(items), (
            f"L{level}: trace+answer reply exact for only {trace_exact}/{len(items)}"
        )
        for item in items:
            gold = item["gold"]
            by_id = {s["id"]: s for s in gold["stalls"]}
            total_fee = sum(s["fee"] for s in gold["stalls"])
            assert 0 < gold["optimal_fee"] < total_fee, "constraint does not bind"
            take_all = ", ".join(s["id"] for s in gold["stalls"])
            assert score_atom(item, f"ANSWER: {take_all}") == 0.0, "take-all not rejected"
            assert score_atom(item, "ANSWER:") == 0.0, "empty answer not rejected"
            opt = gold["optimal_ids"]
            if len(opt) >= 2:
                drop = max(opt, key=lambda i: by_id[i]["fee"])
                sub = [i for i in opt if i != drop]
                partial = score_atom(item, "ANSWER: " + ", ".join(sub))
                assert 0.0 < partial < 1.0, "objective partial credit broken"
            names = ", ".join(by_id[i]["name"] for i in opt)
            assert score_atom(item, "ANSWER: " + names) == 1.0, "name alias broken"
            for dep, pat in gold.get("bonds", []):
                # A lone dependent fits every other rule (bonds avoid rival
                # stalls; one stall is under cap and quota) so only the bond
                # can — and must — reject it.
                assert (
                    score_atom(item, f"ANSWER: {dep}") == 0.0
                ), "bond violation not rejected"
                assert gold["optimal_fee"] == _solve(
                    gold["stalls"], gold["cap"], gold["rivals"], gold["quotas"], gold["bonds"]
                )[0], "solver not reproducible on bonds"
    stats["extra_checks"] = (
        "binding, take-all=0, empty=0, partial-credit, name-alias, bond-violation=0, "
        "trace: exact>=95%, <=800 words, forbidden-free, deterministic"
    )
    stats["max_trace_words"] = max_trace_words
    return stats
