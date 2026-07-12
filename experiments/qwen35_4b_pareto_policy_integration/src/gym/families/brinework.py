"""brinework — HELD-OUT near-transfer probe: state tracking + triage (atoms).

Never trained on. This family measures whether skills installed by the ten
trained families transfer to a genuinely fresh surface: a coastal
salt-evaporation works whose pans (Pan I..Pan VIII) hold brine counted in
lugs. The record style is deliberately different from caravan's numbered
prose events — a terse day-log ("Day 7 — Pan III filled, 6 lugs.") mixing
(a) state-changing entries (fills, drains, whole-pan tips, rain halving a
pan rounded down), (b) worker entries with aliases declared inline
("Maro — the tally calls him 'Old Salt' —"), and (c) superseding correction
lines that retroactively replace an earlier day's figure (latest wins).

Questions: final lugs in one pan / total lugs across pans / how many entries
mention a worker under any of that worker's names / the post-correction value
a day's entry records. Verifier: an exact simulator over the canonical
(post-correction) records, run at generation time. Levels scale day-entry
count (7/12/18/20, frontier 30/38), pan and worker counts, and
alias/correction density (L6 interleaves aliases and corrections densest).
"""

from __future__ import annotations

from .. import base

FAMILY = "brinework"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = False

_NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII")

_WORKERS = ("Maro", "Ilbet", "Sunna", "Kereth", "Dovan", "Pila", "Ostrec", "Venna")
_PRONOUN = {
    "Maro": "him",
    "Ilbet": "her",
    "Sunna": "her",
    "Kereth": "him",
    "Dovan": "him",
    "Pila": "her",
    "Ostrec": "him",
    "Venna": "her",
}
_POSSESSIVE = {
    "Maro": "his",
    "Ilbet": "her",
    "Sunna": "her",
    "Kereth": "his",
    "Dovan": "his",
    "Pila": "her",
    "Ostrec": "his",
    "Venna": "her",
}
_ALIASES = (
    "Old Salt",
    "Longstep",
    "Halfmoon",
    "Greywake",
    "Tidesong",
    "Dryboots",
    "Saltcrow",
    "Rakehand",
)

_CHORES = (
    "raked {pan}",
    "kept watch at {pan}",
    "skimmed the crust off {pan}",
    "patched the sluice at {pan}",
    "walked the dyke past {pan}",
)

_HEADER = "Merrowspit salt-works day-log. Lugs measure brine; every pan starts empty."

_LEVEL_SHAPE = {
    # level: (n_days, n_pans, n_workers, alias_prob, corr_lo, corr_hi,
    #         n_rain, n_tip, n_roster, allow_double_correction)
    1: (7, 3, 1, 0.60, 0, 1, 0, 0, 2, False),
    2: (12, 4, 2, 0.85, 1, 1, 1, 0, 3, False),
    3: (18, 5, 3, 1.00, 2, 2, 2, 1, 4, False),
    4: (20, 6, 3, 1.00, 3, 3, 2, 2, 4, True),
    # Frontier: longer logs, more pans/workers, denser alias + correction
    # interleaving (L6 densest). Prompts use the frontier char budget.
    5: (30, 7, 4, 1.00, 4, 5, 3, 2, 5, True),
    6: (38, 8, 5, 1.00, 6, 7, 4, 3, 4, True),
}

# Frontier levels attribute more fill/drain entries to (aliased) workers so
# alias bookkeeping interleaves with the count-tracking; L1-L4 keep the
# original (cap 3, p 0.25) so their RNG stream is untouched.
_ATTRIB_SHAPE = {5: (6, 0.40), 6: (8, 0.45)}


def _gen_days(rng, level: int) -> tuple[list[dict], list[str], list[str], dict]:
    (n_days, n_pans, n_workers, alias_p, _c_lo, _c_hi,
     n_rain, n_tip, n_roster, _dbl) = _LEVEL_SHAPE[level]
    pan_ids = sorted(rng.sample(range(len(_NUMERALS)), n_pans))
    pans = [f"Pan {_NUMERALS[i]}" for i in pan_ids]
    names = list(rng.sample(_WORKERS, n_workers))
    alias_pool = list(rng.sample(_ALIASES, n_workers))
    aliases = {
        name: alias_pool[k] for k, name in enumerate(names) if rng.random() < alias_p
    }

    n_fill = max(3, round(n_days * 0.36))
    n_drain = n_days - n_fill - n_roster - n_rain - n_tip
    body = (
        ["fill"] * (n_fill - 2)
        + ["drain"] * n_drain
        + ["rain"] * n_rain
        + ["tip"] * n_tip
        + ["roster"] * n_roster
    )
    rng.shuffle(body)
    kinds = ["fill", "fill"] + body

    state = {pan: 0 for pan in pans}
    declared: set[str] = set()
    entries: list[dict] = []
    attributions = 0
    for kind in kinds:
        wet = [pan for pan in pans if state[pan] > 0]
        if kind in ("drain", "tip") and not wet:
            kind = "fill"  # keep the canonical record clamp-free
        entry: dict = {"kind": kind}
        if kind == "fill":
            pan = rng.choice(pans)
            lugs = rng.randint(2, 9)
            state[pan] += lugs
            entry.update({"pan": pan, "lugs": lugs, "shown": lugs})
        elif kind == "drain":
            pan = rng.choice(wet)
            lugs = rng.randint(1, min(6, state[pan]))
            state[pan] -= lugs
            entry.update({"pan": pan, "lugs": lugs, "shown": lugs})
        elif kind == "rain":
            damp = [pan for pan in pans if state[pan] >= 2] or pans
            pan = rng.choice(damp)
            state[pan] //= 2
            entry.update({"pan": pan})
        elif kind == "tip":
            src = rng.choice(wet)
            dst = rng.choice([pan for pan in pans if pan != src])
            state[dst] += state[src]
            state[src] = 0
            entry.update({"src": src, "dst": dst})
        else:  # roster chatter; never changes a count
            worker = rng.choice(names)
            chore = rng.choice(_CHORES).format(pan=rng.choice(pans))
            entry.update({"worker": worker, "chore": chore})
            if worker in aliases and worker not in declared:
                entry["declare"] = True
                entry["alias"] = aliases[worker]
                entry["chore"] = "raked " + rng.choice(pans)
                declared.add(worker)
            else:
                use_alias = worker in declared and rng.random() < 0.5
                entry["ref"] = aliases[worker] if use_alias else worker
        attrib_cap, attrib_p = _ATTRIB_SHAPE.get(level, (3, 0.25))
        if kind in ("fill", "drain") and attributions < attrib_cap and rng.random() < attrib_p:
            worker = rng.choice(names)
            use_alias = worker in declared and rng.random() < 0.5
            entry["worker"] = worker
            entry["ref"] = aliases[worker] if use_alias else worker
            attributions += 1
        entries.append(entry)
    return entries, pans, names, aliases


def _pick_other(rng, excluded: tuple[int, ...]) -> int:
    while True:
        value = rng.randint(1, 12)
        if value not in excluded:
            return value


def _gen_corrections(rng, entries: list[dict], level: int) -> list[dict]:
    (n_days, _p, _w, _a, c_lo, c_hi, _r, _t, _ro, dbl_ok) = _LEVEL_SHAPE[level]
    fixable = [i for i, e in enumerate(entries) if e["kind"] in ("fill", "drain")]
    n_lines = rng.randint(c_lo, c_hi)
    double = dbl_ok and n_lines >= 2 and rng.random() < 0.5
    n_targets = min(n_lines - (1 if double else 0), len(fixable))
    if n_targets <= 0:
        return []
    targets = sorted(rng.sample(fixable, n_targets))
    double_at = rng.randrange(n_targets) if double else -1
    corrections: list[dict] = []
    for j, t in enumerate(targets):
        day = t + 1
        entry = entries[t]
        true_lugs = entry["lugs"]
        wrong = _pick_other(rng, (true_lugs,))
        entry["shown"] = wrong  # the day entry prints the superseded figure
        record = {"day": day, "pan": entry["pan"], "kind": entry["kind"]}
        if j == double_at:
            mid = _pick_other(rng, (true_lugs, wrong))
            first_after = rng.randint(day, n_days)
            second_after = rng.randint(first_after, n_days)
            corrections.append(
                dict(record, value=mid, prev=wrong, after=first_after)
            )
            corrections.append(
                dict(record, value=true_lugs, prev=mid, after=second_after)
            )
        else:
            corrections.append(
                dict(record, value=true_lugs, prev=wrong, after=rng.randint(day, n_days))
            )
    corrections.sort(key=lambda c: c["after"])  # stable: log order
    return corrections


def _simulate(entries: list[dict], corrections: list[dict], pans: list[str]):
    """Exact simulator over the canonical (post-correction) records."""
    ruled: dict[int, int] = {}
    for corr in corrections:  # log order; the latest correction wins
        ruled[corr["day"]] = corr["value"]
    state = {pan: 0 for pan in pans}
    for i, entry in enumerate(entries):
        kind = entry["kind"]
        if kind == "fill":
            state[entry["pan"]] += ruled.get(i + 1, entry["shown"])
        elif kind == "drain":
            state[entry["pan"]] -= ruled.get(i + 1, entry["shown"])
        elif kind == "rain":
            state[entry["pan"]] //= 2
        elif kind == "tip":
            state[entry["dst"]] += state[entry["src"]]
            state[entry["src"]] = 0
        # roster entries never change a count
    return state, ruled


def _lugs(count: int) -> str:
    return f"{count} lug" + ("" if count == 1 else "s")


def _entry_line(day: int, entry: dict) -> str:
    kind = entry["kind"]
    if kind in ("fill", "drain"):
        verb = "filled" if kind == "fill" else "drained"
        tail = f", by {entry['ref']}" if entry.get("ref") else ""
        return f"Day {day} — {entry['pan']} {verb}, {_lugs(entry['shown'])}{tail}."
    if kind == "rain":
        return f"Day {day} — rain over {entry['pan']}."
    if kind == "tip":
        return f"Day {day} — {entry['src']} tipped into {entry['dst']}."
    if entry.get("declare"):
        return (
            f"Day {day} — {entry['worker']} — the tally calls "
            f"{_PRONOUN[entry['worker']]} '{entry['alias']}' — {entry['chore']}."
        )
    return f"Day {day} — {entry['ref']} {entry['chore']}."


def _render_log(entries: list[dict], corrections: list[dict]) -> list[str]:
    after: dict[int, list[dict]] = {}
    for corr in corrections:
        after.setdefault(corr["after"], []).append(corr)
    lines: list[str] = []
    for i, entry in enumerate(entries):
        day = i + 1
        lines.append(_entry_line(day, entry))
        for corr in after.get(day, []):
            verb = "filled with" if corr["kind"] == "fill" else "drained of"
            lines.append(
                f"Correction to day {corr['day']}: {corr['pan']} was {verb} "
                f"{_lugs(corr['value'])}, not {corr['prev']}."
            )
    return lines


def _rules_text(level: int) -> str:
    (_d, _p, _w, _a, _cl, _ch, n_rain, n_tip, _ro, _dbl) = _LEVEL_SHAPE[level]
    parts = ["a fill adds lugs to a pan, a drain draws lugs off"]
    if n_rain:
        parts.append("rain halves a pan, round down")
    if n_tip:
        parts.append("a tip pours one pan wholly into another")
    parts.append("a correction replaces that day's figure (the latest correction wins)")
    return "Rules: " + "; ".join(parts) + "; nothing else changes a count."


def _make_question(rng, entries, pans, names, final, ruled):
    counts = {name: 0 for name in names}
    declared: set[str] = set()
    for entry in entries:
        worker = entry.get("worker")
        if worker:
            counts[worker] += 1
        if entry.get("declare"):
            declared.add(entry["worker"])
    mentioned = [name for name in names if counts[name] > 0]

    pool = ["pan", "pan", "pan", "total", "total"]
    if mentioned:
        pool += ["mention", "mention"]
    if ruled:
        pool += ["corrected", "corrected"]
    kind = rng.choice(pool)

    if kind == "pan":
        # Prefer pans with nonzero final counts so a constant "0" reply stays
        # a floor, not a strategy (caravan pattern).
        nonzero = [pan for pan in pans if final[pan] > 0]
        source = nonzero if nonzero and rng.random() > 0.10 else pans
        pan = rng.choice(source)
        return f"How many lugs does {pan} hold at the close of the log?", final[pan]
    if kind == "total":
        return (
            "How many lugs do all the pans hold together at the close of the log?",
            sum(final.values()),
        )
    if kind == "mention":
        strong = [n for n in mentioned if n in declared and counts[n] >= 2]
        name = (
            rng.choice(strong)
            if strong and rng.random() < 0.8
            else rng.choice(mentioned)
        )
        if name in declared:
            question = (
                f"How many log entries mention the worker {name}, "
                f"under any of {_POSSESSIVE[name]} names?"
            )
        else:
            question = f"How many log entries mention the worker {name}?"
        return question, counts[name]
    day = rng.choice(sorted(ruled))
    return (
        f"As finally corrected, how many lugs does the day {day} entry record?",
        ruled[day],
    )


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        for attempt in range(20):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.atom_prompt_limit(level):
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    entries, pans, names, _aliases = _gen_days(rng, level)
    corrections = _gen_corrections(rng, entries, level)
    final, ruled = _simulate(entries, corrections, pans)
    question, gold = _make_question(rng, entries, pans, names, final, ruled)

    lines = [_HEADER, _rules_text(level), ""]
    lines += _render_log(entries, corrections)
    lines += ["", question, "", base.ATOM_ANSWER_INSTRUCTION]
    prompt = "\n".join(lines)

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
    }


def score_atom(item: dict, reply_text: str) -> float:
    return base.score_exact_int(item["gold"], reply_text)


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


def selftest() -> dict:
    return base.selftest_atoms(__import__(__name__, fromlist=["x"]))
