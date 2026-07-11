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
    for level in LEVELS:
        for item in gen_atoms(11, level, 12):
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
        "binding, take-all=0, empty=0, partial-credit, name-alias, bond-violation=0"
    )
    return stats
