import random
import re


META = {
    "name": "mirage",
    "capability": "calibrated abstention over fictional modular-cycle constraint puzzles",
    "paradigm": "single-turn",
    "action_format": "reply with a final line 'ANSWER: <token>' or 'ANSWER: IMPOSSIBLE'",
}


_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONSONANTS = "bdfghjklmnprstvyz"
_VOWELS = "aeiou"
_SYLLABLES = tuple(c + v for c in _CONSONANTS for v in _VOWELS)
_MODES = ("atom", "episode")
_LEVELS = {
    1: {"cycle": (12, 12), "entities": (3, 4), "target_steps": (1, 2), "ineq": 0, "cross": 0},
    2: {"cycle": (12, 14), "entities": (5, 6), "target_steps": (3, 3), "ineq": 0, "cross": 0},
    3: {"cycle": (14, 16), "entities": (7, 9), "target_steps": (4, 5), "ineq": 2, "cross": 0},
    4: {"cycle": (16, 18), "entities": (10, 12), "target_steps": (6, 7), "ineq": 3, "cross": 2},
}


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if level not in _LEVELS:
        raise ValueError("level must be 1, 2, 3, or 4")
    if mode not in _MODES:
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")

    rng = random.Random(_stable_seed(seed, level, mode))
    prefix = _seed_prefix(seed)
    items = []
    for pair_index in range(n // 2):
        solvable, unsolvable = _make_pair(seed, level, mode, pair_index, rng, prefix)
        items.append(solvable)
        items.append(unsolvable)
    if n % 2:
        solvable, unsolvable = _make_pair(seed, level, mode, n // 2, rng, prefix)
        items.append(solvable if rng.randrange(2) == 0 else unsolvable)
    rng.shuffle(items)
    return items


class Env:
    def __init__(self, item: dict):
        self.item = item

    def reset(self) -> str:
        return self.item.get("prompt", "")

    def step(self, action) -> tuple[str, bool]:
        try:
            answer = _extract_answer("" if action is None else str(action))
            if answer is None or answer.strip() == "":
                return "Use ANSWER: <token> or ANSWER: IMPOSSIBLE.", True
            return "Recorded.", True
        except Exception:
            return "Use ANSWER: <token> or ANSWER: IMPOSSIBLE.", True


def score(item: dict, transcript: list[dict]) -> dict:
    try:
        parts = []
        for entry in transcript or []:
            if isinstance(entry, dict):
                parts.append("" if entry.get("action") is None else str(entry.get("action")))
            else:
                parts.append("" if entry is None else str(entry))
        got_raw = _extract_answer("\n".join(parts))
        got = "" if got_raw is None else got_raw.strip()
        expected = item.get("forced_answer", "")
        if _is_impossible(expected):
            ok = _is_impossible(got)
        else:
            ok = got == expected
        return {"score": 1.0 if ok else 0.0, "expected": expected, "got": got}
    except Exception:
        return {"score": 0.0, "expected": item.get("forced_answer", ""), "got": ""}


def oracle_policy(item: dict, history: list[dict]) -> str:
    return "ANSWER: " + item["forced_answer"]


def random_policy(item: dict, history: list[dict], rng) -> str:
    candidates = list(item["cycle"]) + ["IMPOSSIBLE"] + list(item["offcycle"])
    return "ANSWER: " + candidates[rng.randrange(len(candidates))]


def _make_pair(seed, level, mode, pair_index, rng, prefix):
    cfg = _LEVELS[level]
    target_steps = rng.randint(cfg["target_steps"][0], cfg["target_steps"][1])
    contradiction_steps = target_steps + 1
    min_entities = contradiction_steps + 1
    if level >= 3:
        min_entities += 1
    if level == 1:
        entity_count = min_entities
    else:
        entity_count = rng.randint(max(cfg["entities"][0], min_entities), cfg["entities"][1])
    cycle_size = rng.randint(cfg["cycle"][0], cfg["cycle"][1])

    used = set()
    cycle = [_fresh_token(prefix, rng, used) for _ in range(cycle_size)]
    entities = [_fresh_token(prefix, rng, used) for _ in range(entity_count)]
    offcycle = [_fresh_token(prefix, rng, used) for _ in range(10)]

    deltas = _chain_deltas(level, contradiction_steps)
    offsets = [0]
    for delta in deltas:
        offsets.append((offsets[-1] + delta) % cycle_size)

    step = _answer_step(cycle_size)
    answer_index = (pair_index * step + level * 2 + (1 if mode == "episode" else 0)) % cycle_size
    start_index = (answer_index - offsets[target_steps]) % cycle_size
    end_index = (start_index + offsets[contradiction_steps]) % cycle_size
    bad_index = (end_index + 2 + (pair_index % (cycle_size - 3))) % cycle_size

    values = {}
    for index in range(contradiction_steps + 1):
        values[entities[index]] = (start_index + offsets[index]) % cycle_size

    solvable = [_anchor(entities[0], cycle[start_index])]
    unsolvable = [_anchor(entities[0], cycle[start_index])]
    link_pairs = set()
    for index, delta in enumerate(deltas, 1):
        kind = "shift" if delta == 1 else "eq"
        constraint = _link(kind, entities[index], entities[index - 1])
        solvable.append(constraint)
        unsolvable.append(dict(constraint))
        link_pairs.add(_pair_key(entities[index], entities[index - 1]))
    solvable.append(_anchor(entities[contradiction_steps], cycle[end_index]))
    unsolvable.append(_anchor(entities[contradiction_steps], cycle[bad_index]))

    _add_distractors(level, rng, cycle, entities[contradiction_steps + 1 :], values, solvable, unsolvable, link_pairs)
    _add_cross_links(cfg["cross"], entities[: contradiction_steps + 1], values, solvable, unsolvable, link_pairs, rng, cycle_size)
    _add_inequalities(cfg["ineq"], values, solvable, unsolvable, link_pairs, rng)

    order = list(range(len(solvable)))
    rng.shuffle(order)
    solvable = [solvable[index] for index in order]
    unsolvable = [unsolvable[index] for index in order]

    prompt_s = _render_prompt(cycle, entities, solvable, entities[target_steps])
    prompt_u = _render_prompt(cycle, entities, unsolvable, entities[target_steps])

    answer_s = _verified_answer(cycle, entities, solvable, entities[target_steps])
    answer_u = _verified_answer(cycle, entities, unsolvable, entities[target_steps])
    assert answer_s == cycle[answer_index]
    assert answer_u == "IMPOSSIBLE"
    assert len(prompt_s) <= 800
    assert len(prompt_u) <= 800

    skeleton = {
        "cycle_size": cycle_size,
        "entity_count": entity_count,
        "target_position": target_steps,
        "constraint_types": _type_multiset(solvable),
        "constraint_count": len(solvable),
        "proof_depth": target_steps,
        "contradiction_depth": contradiction_steps,
    }
    pair_id = "mirage-%s-L%s-%s-%s" % (mode, level, seed, pair_index)
    return (
        _item(pair_id, "S", seed, level, mode, cycle, entities, offcycle, solvable, entities[target_steps], prompt_s, "solvable", answer_s, skeleton),
        _item(pair_id, "U", seed, level, mode, cycle, entities, offcycle, unsolvable, entities[target_steps], prompt_u, "unsolvable", "IMPOSSIBLE", skeleton),
    )


def _item(pair_id, suffix, seed, level, mode, cycle, entities, offcycle, constraints, target, prompt, label, answer, skeleton):
    return {
        "id": pair_id + "-" + suffix,
        "level": level,
        "mode": mode,
        "max_turns": 1,
        "pair_id": pair_id,
        "prompt": prompt,
        "cycle": list(cycle),
        "entities": list(entities),
        "target": target,
        "constraints": [_copy_constraint(constraint) for constraint in constraints],
        "label": label,
        "forced_answer": answer,
        "offcycle": list(offcycle),
        "skeleton": dict(skeleton),
        "seed": seed,
    }


def _add_distractors(level, rng, cycle, extras, values, solvable, unsolvable, link_pairs):
    if level < 2 or not extras:
        return
    base = extras[0]
    base_value = rng.randrange(len(cycle))
    values[base] = base_value
    constraint = _anchor(base, cycle[base_value])
    solvable.append(constraint)
    unsolvable.append(dict(constraint))
    previous = base
    for entity in extras[1:]:
        delta = 1 if rng.randrange(3) else 0
        kind = "shift" if delta else "eq"
        values[entity] = (values[previous] + delta) % len(cycle)
        constraint = _link(kind, entity, previous)
        solvable.append(constraint)
        unsolvable.append(dict(constraint))
        link_pairs.add(_pair_key(entity, previous))
        previous = entity


def _add_cross_links(count, chain, values, solvable, unsolvable, link_pairs, rng, cycle_size):
    if count <= 0:
        return
    candidates = []
    for left in chain:
        for right in chain:
            if left == right or _pair_key(left, right) in link_pairs:
                continue
            diff = (values[left] - values[right]) % cycle_size
            if diff == 0:
                candidates.append(("eq", left, right))
            elif diff == 1:
                candidates.append(("shift", left, right))
    rng.shuffle(candidates)
    added = 0
    for kind, left, right in candidates:
        if added >= count:
            break
        if _pair_key(left, right) in link_pairs:
            continue
        constraint = _link(kind, left, right)
        solvable.append(constraint)
        unsolvable.append(dict(constraint))
        link_pairs.add(_pair_key(left, right))
        added += 1
    assert added == count


def _add_inequalities(count, values, solvable, unsolvable, link_pairs, rng):
    if count <= 0:
        return
    names = list(values)
    candidates = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if values[left] != values[right] and _pair_key(left, right) not in link_pairs:
                candidates.append((left, right))
    rng.shuffle(candidates)
    added = 0
    for left, right in candidates:
        if added >= count:
            break
        if _pair_key(left, right) in link_pairs:
            continue
        constraint = {"kind": "neq", "a": left, "b": right}
        solvable.append(constraint)
        unsolvable.append(dict(constraint))
        link_pairs.add(_pair_key(left, right))
        added += 1
    assert added == count


def _chain_deltas(level, count):
    if level == 1:
        base = (1, 0, 1)
    elif level == 2:
        base = (1, 0, 1, 1)
    elif level == 3:
        base = (1, 0, 1, 1, 0, 1)
    else:
        base = (1, 0, 1, 0, 1, 1, 0, 1)
    return [base[index % len(base)] for index in range(count)]


def _render_prompt(cycle, entities, constraints, target):
    lines = [
        "Cycle: " + " ".join(cycle) + " (wrap)",
        "Entities: " + " ".join(entities),
        "Rules (X=Y+1 means X is next after Y; X!=Y means different):",
    ]
    lines.extend(_render_constraint(constraint) for constraint in constraints)
    lines.append("Question: which cycle token does " + target + " hold?")
    lines.append("Reply final line ANSWER: <token>; if rules are unsatisfiable reply ANSWER: IMPOSSIBLE.")
    return "\n".join(lines)


def _render_constraint(constraint):
    kind = constraint["kind"]
    if kind == "anchor":
        return constraint["a"] + " = " + constraint["token"]
    if kind == "eq":
        return constraint["a"] + " = " + constraint["b"]
    if kind == "shift":
        return constraint["a"] + " = " + constraint["b"] + "+1"
    return constraint["a"] + " != " + constraint["b"]


def _verified_answer(cycle, entities, constraints, target):
    result = _verify(cycle, entities, constraints, target)
    if not result["satisfiable"]:
        return "IMPOSSIBLE"
    return result["forced"]


def _verify(cycle, entities, constraints, target):
    index = {name: pos for pos, name in enumerate(entities)}
    token_index = {token: pos for pos, token in enumerate(cycle)}
    uf = _ModUnionFind(len(entities), len(cycle))
    inequalities = []
    ok = True
    for constraint in constraints:
        kind = constraint["kind"]
        if kind == "anchor":
            ok = ok and uf.set_anchor(index[constraint["a"]], token_index[constraint["token"]])
        elif kind == "eq":
            ok = ok and uf.union(index[constraint["a"]], index[constraint["b"]], 0)
        elif kind == "shift":
            ok = ok and uf.union(index[constraint["a"]], index[constraint["b"]], 1)
        elif kind == "neq":
            inequalities.append((index[constraint["a"]], index[constraint["b"]]))
        else:
            ok = False
        if not ok:
            return {"satisfiable": False, "forced": None}
    for left, right in inequalities:
        diff = uf.diff(left, right)
        if diff == 0:
            return {"satisfiable": False, "forced": None}
    root, offset = uf.find(index[target])
    root_value = uf.anchor[root]
    if root_value is None:
        return {"satisfiable": True, "forced": None}
    return {"satisfiable": True, "forced": cycle[(root_value + offset) % len(cycle)]}


class _ModUnionFind:
    def __init__(self, size, modulus):
        self.parent = list(range(size))
        self.rank = [0] * size
        self.weight = [0] * size
        self.anchor = [None] * size
        self.modulus = modulus

    def find(self, node):
        parent = self.parent[node]
        if parent == node:
            return node, 0
        root, parent_weight = self.find(parent)
        self.weight[node] = (self.weight[node] + parent_weight) % self.modulus
        self.parent[node] = root
        return root, self.weight[node]

    def union(self, left, right, delta):
        left_root, left_weight = self.find(left)
        right_root, right_weight = self.find(right)
        if left_root == right_root:
            return (left_weight - right_weight - delta) % self.modulus == 0
        if self.rank[left_root] < self.rank[right_root]:
            return self._attach_left_to_right(left_root, right_root, left_weight, right_weight, delta)
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return self._attach_right_to_left(left_root, right_root, left_weight, right_weight, delta)

    def set_anchor(self, node, value):
        root, weight = self.find(node)
        root_value = (value - weight) % self.modulus
        if self.anchor[root] is not None and self.anchor[root] != root_value:
            return False
        self.anchor[root] = root_value
        return True

    def diff(self, left, right):
        left_root, left_weight = self.find(left)
        right_root, right_weight = self.find(right)
        if left_root != right_root:
            return None
        return (left_weight - right_weight) % self.modulus

    def _attach_right_to_left(self, left_root, right_root, left_weight, right_weight, delta):
        right_to_left = (left_weight - right_weight - delta) % self.modulus
        self.parent[right_root] = left_root
        self.weight[right_root] = right_to_left
        return self._merge_anchor(parent=left_root, child=right_root, child_to_parent=right_to_left)

    def _attach_left_to_right(self, left_root, right_root, left_weight, right_weight, delta):
        left_to_right = (delta - left_weight + right_weight) % self.modulus
        self.parent[left_root] = right_root
        self.weight[left_root] = left_to_right
        return self._merge_anchor(parent=right_root, child=left_root, child_to_parent=left_to_right)

    def _merge_anchor(self, parent, child, child_to_parent):
        child_anchor = self.anchor[child]
        parent_anchor = self.anchor[parent]
        if child_anchor is not None:
            implied_parent = (child_anchor - child_to_parent) % self.modulus
            if parent_anchor is not None and parent_anchor != implied_parent:
                return False
            self.anchor[parent] = implied_parent if parent_anchor is None else parent_anchor
        self.anchor[child] = None
        return True


def _extract_answer(text):
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def _is_impossible(value):
    return str(value).strip().upper() == "IMPOSSIBLE"


def _anchor(entity, token):
    return {"kind": "anchor", "a": entity, "token": token}


def _link(kind, left, right):
    return {"kind": kind, "a": left, "b": right}


def _copy_constraint(constraint):
    return dict(constraint)


def _type_multiset(constraints):
    counts = {}
    for constraint in constraints:
        kind = constraint["kind"]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _pair_key(left, right):
    return tuple(sorted((left, right)))


def _answer_step(cycle_size):
    for step in (5, 7, 11, 13):
        if _gcd(step, cycle_size) == 1:
            return step
    return 1


def _gcd(left, right):
    while right:
        left, right = right, left % right
    return abs(left)


def _seed_prefix(seed):
    return _SYLLABLES[int(seed) % len(_SYLLABLES)]


def _fresh_token(prefix, rng, used):
    for _attempt in range(10000):
        token = prefix + rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)
        if token not in used:
            used.add(token)
            return token
    raise RuntimeError("token pool exhausted")


def _stable_seed(seed, level, mode):
    text = "mirage|" + str(seed) + "|" + str(level) + "|" + mode
    value = 1469598103934665603
    for char in text:
        value ^= ord(char)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value
