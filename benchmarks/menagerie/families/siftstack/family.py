import hashlib
import random
import re


META = {
    "name": "siftstack",
    "capability": "Resolve aliases, superseded ledger facts, and aggregate document-bound values.",
    "paradigm": "single-turn",
    "action_format": "End with a final line: ANSWER: <value>",
}


_SYLLABLES = (
    "qa", "qe", "qi", "qo",
    "xa", "xe", "xi", "xo",
    "za", "ze", "zi", "zo",
    "vu", "vo", "ve", "vi",
    "ju", "jo", "ki", "ku",
)
_BAD_PARTS = (
    "and", "the", "for", "sum", "count", "latest", "alias", "file", "field",
    "slot", "value", "one", "two", "six", "ten", "red", "blue", "green",
    "black", "white", "cat", "dog", "man", "run", "new", "old",
)
_ANSWER_RE = re.compile(r"^answer\s*:\s*(.*?)\s*$", re.IGNORECASE)


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if level not in {1, 2, 3, 4}:
        raise ValueError("level must be one of {1,2,3,4}")
    if mode not in {"atom", "episode"}:
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")

    items = []
    histogram = {}
    max_seen = n // 10 if n >= 20 else None
    for index in range(n):
        attempt = 0
        while True:
            item = _make_item(seed, level, mode, index, attempt)
            obs = _observation(item)
            limit = 800 if mode == "episode" else 1200
            if len(obs) > limit:
                attempt += 1
                if attempt > 200:
                    raise RuntimeError("could not satisfy observation budget")
                continue
            gold = item["gold"]
            if max_seen is None or histogram.get(gold, 0) < max_seen:
                break
            attempt += 1
            if attempt > 200:
                raise RuntimeError("could not satisfy answer histogram")
        histogram[gold] = histogram.get(gold, 0) + 1
        items.append(item)
    return items


class Env:
    def __init__(self, item: dict):
        self.item = item

    def reset(self) -> str:
        return _observation(self.item)

    def step(self, action: str) -> tuple[str, bool]:
        if _extract_answer(action) is None:
            return ("Reply with a final line: ANSWER: <value>", True)
        return ("", True)


def score(item: dict, transcript: list[dict]) -> dict:
    expected = str(item.get("gold", ""))
    got_text = None
    try:
        if transcript:
            last = transcript[-1]
            if isinstance(last, dict):
                action = last.get("action")
                got_text = _extract_answer(action)
                if got_text is None:
                    got_text = _fallback_bare_answer(action, item.get("gold_type"))
    except Exception:
        got_text = None

    if got_text is None:
        return {"score": 0.0, "expected": expected, "got": None}

    if item.get("gold_type") == "int":
        expected_int = int(expected)
        parsed = _parse_int(got_text)
        if parsed is None:
            return {"score": 0.0, "expected": str(expected_int), "got": got_text.strip().lower()}
        return {
            "score": 1.0 if parsed == expected_int else 0.0,
            "expected": str(expected_int),
            "got": str(parsed),
        }

    got = got_text.strip().lower()
    exp = expected.strip().lower()
    return {"score": 1.0 if got == exp else 0.0, "expected": exp, "got": got}


def oracle_policy(item: dict, history: list[dict]) -> str:
    return "ANSWER: " + str(item["gold"])


def random_policy(item: dict, history: list[dict], rng) -> str:
    if rng.randrange(2) == 0:
        return "ANSWER: " + str(rng.randrange(1000))
    return "ANSWER: " + _random_codeword(rng)


def _make_item(seed, level, mode, index, attempt):
    rng = _rng(seed, level, mode, index, attempt)
    used = set()
    names = _names(rng, used)
    query_type = _query_type(seed, level, mode, index)

    if mode == "episode":
        records, query = _build_episode(names, level, query_type, seed, index, attempt)
    elif level == 1:
        records, query = _build_l1_atom(names, query_type, seed, index, attempt)
    elif level == 2:
        records, query = _build_l2_atom(names, query_type, seed, index, attempt)
    elif level == 3:
        records, query = _build_l3_atom(names, query_type, seed, index, attempt)
    else:
        records, query = _build_l4_atom(names, query_type, seed, index, attempt)

    gold, gold_type = _compute_gold(records, query)
    document = _render_document(records, mode)
    query_text = _render_query(query, mode)
    return {
        "id": f"siftstack-{seed}-{level}-{mode}-{index}-{attempt}",
        "level": level,
        "mode": mode,
        "max_turns": 1,
        "records": records,
        "document": document,
        "query": query_text,
        "query_spec": query,
        "gold": str(gold),
        "gold_type": gold_type,
    }


def _rng(seed, level, mode, index, attempt):
    material = f"siftstack|{seed}|{level}|{mode}|{index}|{attempt}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _names(rng, used):
    target = _entity(rng, used)
    alias1 = _entity(rng, used)
    alias2 = _entity(rng, used)
    near = _near_miss(rng, target, used)
    distractors = [_entity(rng, used) for _ in range(4)]
    fields = [_token(rng, used) for _ in range(5)]
    slots = [_token(rng, used) for _ in range(18)]
    statuses = [_token(rng, used) for _ in range(10)]
    return {
        "target": target,
        "alias1": alias1,
        "alias2": alias2,
        "near": near,
        "distractors": distractors,
        "fields": fields,
        "slots": slots,
        "statuses": statuses,
    }


def _token(rng, used, fragments=None):
    for _ in range(5000):
        if fragments is None:
            count = 3 + rng.randrange(2)
            parts = [rng.choice(_SYLLABLES) for _ in range(count)]
        else:
            parts = list(fragments)
        raw = "".join(parts)
        if raw not in used and not _bad_token(raw):
            used.add(raw)
            return raw
        fragments = None
    raise RuntimeError("token stock exhausted")


def _entity(rng, used):
    return _token(rng, used).title()


def _near_miss(rng, entity, used):
    raw = entity.lower().split()[0]
    pieces = [raw[i:i + 2] for i in range(0, len(raw), 2)]
    for _ in range(100):
        pos = rng.randrange(len(pieces))
        replacement = rng.choice(_SYLLABLES)
        if replacement == pieces[pos]:
            continue
        mutated = pieces[:]
        mutated[pos] = replacement
        candidate = "".join(mutated)
        if candidate not in used and not _bad_token(candidate):
            used.add(candidate)
            return candidate.title()
    return _entity(rng, used)


def _bad_token(token):
    if len(token) < 4:
        return True
    return any(part in token for part in _BAD_PARTS)


def _random_codeword(rng):
    parts = [rng.choice(_SYLLABLES) for _ in range(3)]
    return "".join(parts)


def _query_type(seed, level, mode, index):
    if mode == "episode":
        choices = ("COUNT", "SUM", "LATEST")
    elif level == 1:
        choices = ("COUNT", "LATEST")
    elif level == 2:
        choices = ("COUNT", "SUM")
    else:
        choices = ("COUNT", "SUM", "LATEST")
    return choices[(seed + level * 5 + index) % len(choices)]


def _count_goal(seed, level, index, attempt):
    return 2 + ((seed * 3 + level * 5 + index + attempt) % 8)


def _numeric(seed, level, index, salt, attempt, low=20, high=180):
    span = high - low + 1
    return low + ((seed * 41 + level * 37 + index * 29 + salt * 17 + attempt * 13) % span)


def _record(kind, entity, field=None, slot=None, value=None, alias=None, update=False):
    record = {"kind": kind, "entity": entity}
    if kind == "alias":
        record["alias"] = alias
    else:
        record["field"] = field
        record["slot"] = slot
        record["value"] = str(value)
        record["value_kind"] = "int" if isinstance(value, int) else "token"
        record["update"] = bool(update)
    return record


def _build_l1_atom(names, query_type, seed, index, attempt):
    if query_type == "COUNT":
        return _records_for_count(
            names, seed, 1, index, attempt,
            aliases=0, updates=0, distractors=2, near=False,
        )
    return _records_for_latest(
        names, seed, 1, index, attempt,
        aliases=0, updates=0, distractors=2, near=False,
        latest_numeric=((index + seed) % 2 == 0),
    )


def _build_l2_atom(names, query_type, seed, index, attempt):
    if query_type == "COUNT":
        return _records_for_count(
            names, seed, 2, index, attempt,
            aliases=1, updates=0, distractors=3, near=False,
        )
    return _records_for_sum(
        names, seed, 2, index, attempt,
        aliases=1, updates=0, distractors=3, near=False,
        slots_used=4 + ((index + attempt) % 2),
    )


def _build_l3_atom(names, query_type, seed, index, attempt):
    if query_type == "COUNT":
        return _records_for_count(
            names, seed, 3, index, attempt,
            aliases=2, updates=2, distractors=3, near=True,
        )
    if query_type == "SUM":
        return _records_for_sum(
            names, seed, 3, index, attempt,
            aliases=2, updates=2, distractors=4, near=True,
            slots_used=5,
        )
    return _records_for_latest(
        names, seed, 3, index, attempt,
        aliases=2, updates=2, distractors=4, near=True,
        latest_numeric=((index + seed) % 2 == 0),
    )


def _build_l4_atom(names, query_type, seed, index, attempt):
    if query_type == "COUNT":
        return _records_for_count(
            names, seed, 4, index, attempt,
            aliases=2, updates=3, distractors=3, near=True,
        )
    if query_type == "SUM":
        return _records_for_sum(
            names, seed, 4, index, attempt,
            aliases=2, updates=3, distractors=4, near=True,
            slots_used=6,
        )
    return _records_for_latest(
        names, seed, 4, index, attempt,
        aliases=2, updates=3, distractors=4, near=True,
        latest_numeric=((index + seed) % 2 == 0),
    )


def _build_episode(names, level, query_type, seed, index, attempt):
    aliases = 1 if level <= 2 else 2
    updates = 1 if level <= 2 else 2
    distractors = 2
    if query_type == "COUNT":
        return _records_for_count(
            names, seed, level, index, attempt,
            aliases=aliases, updates=updates, distractors=distractors, near=True,
            compact=True,
        )
    if query_type == "SUM":
        return _records_for_sum(
            names, seed, level, index, attempt,
            aliases=aliases, updates=updates, distractors=distractors, near=True,
            slots_used=4 + (level >= 3),
            compact=True,
        )
    return _records_for_latest(
        names, seed, level, index, attempt,
        aliases=aliases, updates=updates, distractors=distractors, near=True,
        latest_numeric=((index + seed) % 2 == 0),
        compact=True,
    )


def _alias_records(names, count):
    if count <= 0:
        return []
    records = [_record("alias", names["target"], alias=names["alias1"])]
    if count >= 2:
        records.append(_record("alias", names["alias1"], alias=names["alias2"]))
    return records


def _entity_cycle(names, aliases):
    if aliases >= 2:
        return [names["target"], names["alias1"], names["alias2"]]
    if aliases == 1:
        return [names["target"], names["alias1"]]
    return [names["target"]]


def _records_for_count(names, seed, level, index, attempt, aliases, updates, distractors, near, compact=False):
    field = names["fields"][0]
    other_field = names["fields"][1]
    status_yes = names["statuses"][0]
    status_no = names["statuses"][1]
    goal = _count_goal(seed, level, index, attempt)
    entities = _entity_cycle(names, aliases)
    records = []
    records.extend(_alias_records(names, aliases)[:1])

    slots = names["slots"]
    change_slots = min(updates, 3, max(0, goal - 1))
    for pos in range(goal):
        entity = entities[pos % len(entities)]
        if pos < change_slots:
            records.append(_record("fact", entity, field, slots[pos], status_no))
            records.append(_record("fact", entities[(pos + 1) % len(entities)], field, slots[pos], status_yes, update=True))
        else:
            records.append(_record("fact", entity, field, slots[pos], status_yes))

    miss_count = 1 + (level >= 3 and not compact)
    for pos in range(miss_count):
        slot = slots[goal + pos]
        records.append(_record("fact", entities[(goal + pos) % len(entities)], field, slot, status_no))

    if aliases >= 2:
        insert_at = max(1, len(records) // 2)
        records.insert(insert_at, _alias_records(names, aliases)[1])

    if updates > change_slots:
        for extra in range(updates - change_slots):
            slot = slots[goal + miss_count + extra]
            entity = entities[(extra + 1) % len(entities)]
            records.append(_record("fact", entity, other_field, slot, status_yes))
            records.append(_record("fact", entities[(extra + 2) % len(entities)], other_field, slot, status_no, update=True))

    _add_distractors(records, names, field, other_field, status_yes, status_no, distractors, near, numeric=False)
    if level >= 4 and not compact:
        filler = 0
        while len(records) < 16:
            records.append(_record(
                "fact",
                names["distractors"][(filler + 1) % len(names["distractors"])],
                other_field,
                names["slots"][8 + filler],
                status_no if filler % 2 == 0 else status_yes,
            ))
            filler += 1
    query = {
        "type": "COUNT",
        "target": names["target"],
        "field": field,
        "condition": status_yes,
    }
    return records, query


def _records_for_sum(names, seed, level, index, attempt, aliases, updates, distractors, near, slots_used, compact=False):
    field = names["fields"][2]
    other_field = names["fields"][3]
    entities = _entity_cycle(names, aliases)
    records = []
    records.extend(_alias_records(names, aliases)[:1])
    slots = names["slots"]

    change_slots = min(updates, slots_used)
    for pos in range(slots_used):
        entity = entities[pos % len(entities)]
        final_value = _numeric(seed, level, index, pos, attempt, low=35, high=260)
        if pos < change_slots:
            old_value = max(10, final_value - 17 - pos)
            records.append(_record("fact", entity, field, slots[pos], old_value))
            records.append(_record("fact", entities[(pos + 1) % len(entities)], field, slots[pos], final_value, update=True))
        else:
            records.append(_record("fact", entity, field, slots[pos], final_value))

    if aliases >= 2:
        records.insert(max(1, len(records) // 2), _alias_records(names, aliases)[1])

    if level >= 3 and not compact:
        spare = slots[slots_used]
        records.append(_record("fact", entities[0], other_field, spare, _numeric(seed, level, index, 9, attempt)))
        records.append(_record("fact", entities[-1], other_field, spare, _numeric(seed, level, index, 10, attempt), update=True))

    _add_distractors(records, names, field, other_field, names["statuses"][2], names["statuses"][3], distractors, near, numeric=True)
    query = {
        "type": "SUM",
        "target": names["target"],
        "field": field,
    }
    return records, query


def _records_for_latest(names, seed, level, index, attempt, aliases, updates, distractors, near, latest_numeric, compact=False):
    field = names["fields"][4]
    other_field = names["fields"][1]
    slot = names["slots"][0]
    entities = _entity_cycle(names, aliases)
    records = []
    records.extend(_alias_records(names, aliases)[:1])

    if latest_numeric:
        old_value = 100 + ((seed * 19 + level * 23 + index * 31 + attempt * 7) % 400)
        final_value = 100 + ((seed * 43 + level * 47 + index * 53 + attempt * 11) % 850)
    else:
        old_value = names["statuses"][4]
        final_value = names["statuses"][5]

    if updates:
        records.append(_record("fact", entities[0], field, slot, old_value))
        records.append(_record("fact", entities[min(1, len(entities) - 1)], field, slot, final_value, update=True))
    else:
        records.append(_record("fact", entities[0], field, slot, final_value))

    filler_count = 1 + (level >= 3) + (2 if level >= 4 and not compact else 0)
    for pos in range(filler_count):
        filler_value = names["statuses"][(pos + 6) % len(names["statuses"])]
        records.append(_record("fact", entities[pos % len(entities)], field, names["slots"][pos + 1], filler_value))

    if aliases >= 2:
        records.insert(max(1, len(records) // 2), _alias_records(names, aliases)[1])

    for extra in range(max(0, updates - 1)):
        slot_extra = names["slots"][filler_count + extra + 2]
        records.append(_record("fact", entities[extra % len(entities)], other_field, slot_extra, names["statuses"][extra]))
        records.append(_record("fact", entities[(extra + 1) % len(entities)], other_field, slot_extra, names["statuses"][extra + 1], update=True))

    _add_distractors(records, names, field, other_field, names["statuses"][4], names["statuses"][6], distractors, near, numeric=False)
    query = {
        "type": "LATEST",
        "target": names["target"],
        "field": field,
        "slot": slot,
    }
    return records, query


def _add_distractors(records, names, field, other_field, yes, no, count, near, numeric):
    distractor_entities = list(names["distractors"])
    if near:
        distractor_entities.insert(0, names["near"])
    for pos in range(count):
        entity = distractor_entities[pos % len(distractor_entities)]
        use_query_field = (pos % 2 == 0)
        dfield = field if use_query_field else other_field
        slot = names["slots"][-1 - pos]
        if numeric:
            value = 700 + pos * 13 if use_query_field else 300 + pos * 7
        else:
            value = no if use_query_field else yes
        records.append(_record("fact", entity, dfield, slot, value, update=(pos % 3 == 2)))


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            chosen = min(root_left, root_right)
            other = root_right if chosen == root_left else root_left
            self.parent[other] = chosen


def _compute_gold(records, query):
    uf = _UnionFind()
    for record in records:
        if record["kind"] == "alias":
            uf.union(record["entity"], record["alias"])
        else:
            uf.find(record["entity"])

    state = {}
    value_kinds = {}
    for record in records:
        if record["kind"] != "fact":
            continue
        root = uf.find(record["entity"])
        key = (root, record["field"], record["slot"])
        state[key] = record["value"]
        value_kinds[key] = record["value_kind"]

    target = uf.find(query["target"])
    if query["type"] == "COUNT":
        total = 0
        for (entity, field, _slot), value in state.items():
            if entity == target and field == query["field"] and value == query["condition"]:
                total += 1
        return total, "int"

    if query["type"] == "SUM":
        total = 0
        for (entity, field, _slot), value in state.items():
            if entity == target and field == query["field"]:
                total += int(value)
        return total, "int"

    key = (target, query["field"], query["slot"])
    value = state[key]
    return value, value_kinds[key]


def _render_document(records, mode):
    lines = []
    for pos, record in enumerate(records, 1):
        if mode == "episode":
            if record["kind"] == "alias":
                lines.append(f"A {record['entity']}={record['alias']}")
            else:
                op = "U" if record.get("update") else "F"
                lines.append(f"{op} {record['entity']} {record['field']}/{record['slot']}={record['value']}")
        else:
            entry = f"{pos:02d}"
            if record["kind"] == "alias":
                lines.append(f"{entry}. {record['entity']}, also filed as {record['alias']}.")
            else:
                verb = "revised" if record.get("update") else "filed"
                lines.append(
                    f"{entry}. {record['entity']} {verb} "
                    f"{record['field']}/{record['slot']} = {record['value']}."
                )
    return "\n".join(lines)


def _render_query(query, mode):
    if mode == "episode":
        if query["type"] == "COUNT":
            tail = f"COUNT {query['field']} == {query['condition']}"
        elif query["type"] == "SUM":
            tail = f"SUM {query['field']}"
        else:
            tail = f"LATEST {query['field']}/{query['slot']}"
        return (
            f"Q target={query['target']}; include aliases; later same "
            f"entity+field+slot wins; {tail}."
        )

    base = (
        f"Query: Target {query['target']}; include all filed aliases. "
        "Later same entity+field+slot wins. "
    )
    if query["type"] == "COUNT":
        return base + (
            f"COUNT current {query['field']} slots equal {query['condition']}."
        )
    if query["type"] == "SUM":
        return base + f"SUM the current numeric values across field {query['field']}."
    return base + f"LATEST current value for {query['field']}/{query['slot']}."


def _observation(item):
    if item.get("mode") == "episode":
        intro = (
            "Sift ledger. Aliases are global/transitive; later same "
            "entity+field+slot wins. Codes: A alias, F fact, U update."
        )
    else:
        intro = (
            "Use aliases globally/transitively. Later same resolved "
            "entity+field+slot replaces earlier. Ignore near-miss names unless aliased."
        )
    return (
        intro + "\n"
        "Ledger:\n" + item["document"] + "\n"
        + item["query"] + "\n"
        "Reply with final line: ANSWER: <value>"
    )


def _extract_answer(action):
    if action is None:
        return None
    last = None
    for line in str(action).splitlines():
        text = line.strip()
        changed = True
        while changed:
            changed = False
            for prefix in (">", "-", "*"):
                if text.startswith(prefix):
                    text = text[len(prefix):].lstrip()
                    changed = True
            if text.startswith("**"):
                text = text[2:].lstrip()
                changed = True
        if text.endswith("**"):
            text = text[:-2].rstrip()
        match = _ANSWER_RE.match(text)
        if match:
            last = match.group(1).strip()
    return last


def _fallback_bare_answer(action, gold_type):
    if action is None:
        return None
    last = None
    for line in str(action).splitlines():
        text = line.strip()
        if text:
            last = text
    if last is None:
        return None
    if gold_type == "int":
        if re.fullmatch(r"[+-]?\d+", last):
            return last
        return None
    if re.fullmatch(r"[a-z]{4,}", last):
        return last
    return None


def _parse_int(text):
    stripped = text.strip()
    if not re.fullmatch(r"[+-]?\d+", stripped):
        return None
    return int(stripped, 10)
