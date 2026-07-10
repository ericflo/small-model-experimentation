"""foundry_ledger — information triage over a noisy commission ledger (atoms only).

A bell foundry's ledger mixes commission records from aliased patrons (a
patron signs under 1-3 declared written forms), near-miss distractor patrons
(confusable names that must NOT be counted), and amendment records that
supersede earlier values (latest day wins per field). The item asks one
aggregation or lookup question; gold is computed from a canonical structured
record set and the render order of ledger lines is shuffled deterministically.

Levels scale record count (~8/14/22/28 under the 1400-char prompt cap),
alias fan-out, amendment count (including chained amendments on one field),
and distractor density. All vocabulary is invented; the verifier is exact.
"""

from __future__ import annotations

from .. import base

FAMILY = "foundry_ledger"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = False

# Patron name stem -> near-miss distractor spellings. Every variant is
# distinct from every stem AND from every stem's plural alias form, so all
# written names in one ledger are distinct strings (confusable, never equal).
STEMS = {
    "Ostren": ("Ostrend", "Ostrem"),
    "Velmar": ("Velmarn", "Velmor"),
    "Torbek": ("Torbeck", "Torbex"),
    "Quarel": ("Quarell", "Quarelm"),
    "Dovett": ("Dovetta", "Dovest"),
    "Marnis": ("Marnist", "Marnix"),
    "Brenli": ("Brenlin", "Brenlo"),
    "Peldra": ("Peldran", "Peldrim"),
    "Sorvin": ("Sorvind", "Sorvane"),
    "Halvet": ("Halvett", "Halven"),
}
ORGS = ("Guild", "Lodge", "House", "Order")

_LEVEL_SHAPE = {
    # level: (n_commissions, n_amendments, n_real_patrons, n_distractors, max_chains)
    1: (8, 0, 3, 0, 0),
    2: (11, 3, 4, 1, 0),
    3: (17, 5, 4, 2, 1),
    4: (21, 7, 5, 3, 2),
}

# Query-kind draw pools (weights by repetition). L2+ leans on aggregation,
# which exercises aliases + distractors + supersession together.
_KINDS_L1 = ("count", "fee_total", "bells_total", "fee_of", "bells_of")
_KINDS_L2P = (
    "count", "count",
    "fee_total", "fee_total", "fee_total",
    "bells_total", "bells_total",
    "fee_of", "fee_of",
    "bells_of",
)


def _build_ledger(rng, level: int) -> dict:
    """Generate the canonical structured record set plus the chosen query."""
    n_c, n_a, n_real, n_d, max_chains = _LEVEL_SHAPE[level]

    # --- patrons -----------------------------------------------------------
    stems = rng.sample(sorted(STEMS), n_real)
    entities = []
    for stem in stems:
        org = rng.choice(ORGS)
        entities.append(
            {"stem": stem, "org": org, "canonical": f"{stem} {org}",
             "forms": [f"{stem} {org}"]}
        )
    qi = rng.randrange(n_real)
    kind = rng.choice(_KINDS_L1 if level == 1 else _KINDS_L2P)

    # --- aliases (L2+): query patron always aliased; some others too -------
    if level >= 2:
        def add_aliases(ent: dict, n: int) -> None:
            pool = [f"{ent['stem']}s", f"{ent['stem']} & Kin"]
            rng.shuffle(pool)
            ent["forms"].extend(pool[:n])

        add_aliases(entities[qi], 2 if level >= 3 else rng.randint(1, 2))
        others = [i for i in range(n_real) if i != qi]
        rng.shuffle(others)
        n_other = {2: rng.randint(0, 1), 3: 1, 4: 2}[level]
        for i in others[:n_other]:
            add_aliases(entities[i], rng.randint(1, 2))

    # --- distractors: first always mimics the query patron -----------------
    mimic = []
    if n_d:
        others = [i for i in range(n_real) if i != qi]
        rng.shuffle(others)
        mimic.append(qi)
        if n_d >= 3 and rng.random() < 0.5:
            mimic.append(qi)  # both near-miss spellings of the query stem
        for i in others:
            if len(mimic) == n_d:
                break
            mimic.append(i)
    used: dict[str, int] = {}
    distractors = []
    for t in mimic:
        stem = entities[t]["stem"]
        variant = STEMS[stem][used.get(stem, 0)]
        used[stem] = used.get(stem, 0) + 1
        org = entities[t]["org"] if rng.random() < 0.6 else rng.choice(ORGS)
        distractors.append({"canonical": f"{variant} {org}", "forms": [f"{variant} {org}"]})
    all_entities = entities + distractors

    # --- commission allocation: query patron 2-5, everyone else >=1 --------
    q_lo, q_hi = {1: (2, 3), 2: (2, 3), 3: (3, 4), 4: (3, 5)}[level]
    counts = [1] * len(all_entities)
    counts[qi] = rng.randint(q_lo, q_hi)
    pool = list(range(n_real)) * 2 + list(range(n_real, len(all_entities)))
    for _ in range(n_c - sum(counts)):
        counts[rng.choice(pool)] += 1

    slots = [ei for ei, c in enumerate(counts) for _ in range(c)]
    rng.shuffle(slots)
    cids = rng.sample(range(10, 99), n_c)
    days = rng.sample(range(1, 70), n_c)
    commissions = [
        {"cid": cids[i], "day": days[i], "entity": slots[i],
         "bells": rng.randint(1, 9), "fee": rng.randint(20, 199)}
        for i in range(n_c)
    ]
    # Written forms: cycle through each entity's declared forms so the query
    # patron provably signs under multiple names when aliased.
    for ei, ent in enumerate(all_entities):
        idxs = [i for i, c in enumerate(commissions) if c["entity"] == ei]
        forms = [ent["forms"][k % len(ent["forms"])] for k in range(len(idxs))]
        rng.shuffle(forms)
        for i, form in zip(idxs, forms):
            commissions[i]["form"] = form

    q_idxs = [i for i, c in enumerate(commissions) if c["entity"] == qi]

    k_idx = None
    if kind in ("fee_of", "bells_of"):
        k_idx = rng.choice(q_idxs) if level >= 2 else rng.randrange(n_c)

    # --- amendments: force relevance to the query, then random fill --------
    amend_pairs: list[tuple[int, str]] = []
    chains_used = 0
    if n_a:
        if kind in ("fee_of", "bells_of"):
            field = "fee" if kind == "fee_of" else "bells"
            amend_pairs.append((k_idx, field))
            if chains_used < max_chains and rng.random() < 0.6:
                amend_pairs.append((k_idx, field))  # chained: later day wins
                chains_used += 1
        elif kind in ("fee_total", "bells_total"):
            field = "fee" if kind == "fee_total" else "bells"
            amend_pairs.append((rng.choice(q_idxs), field))
        else:  # count: amendments on the query patron must NOT add commissions
            amend_pairs.append((rng.choice(q_idxs), rng.choice(("fee", "bells"))))
        guard = 0
        while len(amend_pairs) < n_a and guard < 400:
            guard += 1
            pair = (rng.randrange(n_c), rng.choice(("fee", "bells")))
            if pair in amend_pairs:
                if chains_used < max_chains and amend_pairs.count(pair) < 2:
                    amend_pairs.append(pair)
                    chains_used += 1
            else:
                amend_pairs.append(pair)

    a_days = rng.sample(range(70, 99), len(amend_pairs))
    amendments = []
    for (ci, field), day in zip(amend_pairs, a_days):
        taken = {commissions[ci][field]} | {
            a["value"] for a in amendments if a["ci"] == ci and a["field"] == field
        }
        while True:
            value = rng.randint(20, 199) if field == "fee" else rng.randint(1, 9)
            if value not in taken:
                break
        amendments.append(
            {"ci": ci, "cid": commissions[ci]["cid"], "field": field,
             "value": value, "day": day}
        )

    # --- effective values: apply amendments in day order (latest wins) -----
    eff = [{"fee": c["fee"], "bells": c["bells"]} for c in commissions]
    for a in sorted(amendments, key=lambda a: a["day"]):
        eff[a["ci"]][a["field"]] = a["value"]

    return {
        "entities": entities, "distractors": distractors,
        "commissions": commissions, "amendments": amendments, "eff": eff,
        "qi": qi, "q_idxs": q_idxs, "kind": kind, "k_idx": k_idx,
    }


def _question_and_gold(ledger: dict) -> tuple[str, int]:
    kind = ledger["kind"]
    eff = ledger["eff"]
    canon = ledger["entities"][ledger["qi"]]["canonical"]
    prefix = "After all amendments, " if ledger["amendments"] else ""
    if kind == "count":
        gold = len(ledger["q_idxs"])
        question = f"How many commissions does {canon} hold in this ledger?"
    elif kind == "fee_total":
        gold = sum(eff[i]["fee"] for i in ledger["q_idxs"])
        question = f"{prefix}what is the total fee in marks across all commissions of {canon}?"
    elif kind == "bells_total":
        gold = sum(eff[i]["bells"] for i in ledger["q_idxs"])
        question = f"{prefix}how many bells in total do the commissions of {canon} call for?"
    elif kind == "fee_of":
        cid = ledger["commissions"][ledger["k_idx"]]["cid"]
        gold = eff[ledger["k_idx"]]["fee"]
        question = f"{prefix}what is the fee in marks of commission {cid}?"
    else:  # bells_of
        cid = ledger["commissions"][ledger["k_idx"]]["cid"]
        gold = eff[ledger["k_idx"]]["bells"]
        question = f"{prefix}how many bells does commission {cid} call for?"
    return question[0].upper() + question[1:], gold


def _render_prompt(rng, ledger: dict, question: str) -> str:
    lines = [
        "Bell-foundry commission ledger, in clerk order (not day order).",
        "Line format: C<id> d<day>: <patron as written> b<bells> f<fee in marks>.",
    ]
    if ledger["amendments"]:
        lines.append(
            "Amend d<n>: C<id> f=<marks> or b=<bells> corrects that field; "
            "the latest day wins."
        )
    notes = [
        f"Note: {ent['canonical']} also signs as "
        + " or ".join(f"'{form}'" for form in ent["forms"][1:]) + "."
        for ent in ledger["entities"]
        if len(ent["forms"]) > 1
    ]
    if notes:
        lines.append("Names not listed in a Note belong to different patrons.")
        lines.extend(notes)

    records = [
        f"C{c['cid']} d{c['day']}: {c['form']} b{c['bells']} f{c['fee']}"
        for c in ledger["commissions"]
    ] + [
        f"Amend d{a['day']}: C{a['cid']} "
        f"{'f' if a['field'] == 'fee' else 'b'}={a['value']}"
        for a in ledger["amendments"]
    ]
    rng.shuffle(records)

    return "\n".join(lines + [""] + records + ["", question, "", base.ATOM_ANSWER_INSTRUCTION])


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    ledger = _build_ledger(rng, level)
    question, gold = _question_and_gold(ledger)
    prompt = _render_prompt(rng, ledger, question)
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
        for attempt in range(20):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
                break
        items.append(item)
    return items


def score_atom(item: dict, reply_text: str) -> float:
    return base.score_exact_int(item["gold"], reply_text)


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


def selftest() -> dict:
    return base.selftest_atoms(__import__(__name__, fromlist=["x"]))
