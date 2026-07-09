import random
import re


META = {
    "name": "stockade",
    "capability": "bounded optimization - allocate fictional resources near-optimally under capacity/conflict/dependency constraints",
    "paradigm": "single-turn",
    "action_format": "Atom: final line ANSWER: comma-separated names or ANSWER: NONE; episode: TAKE comma-separated current-round names or TAKE NONE.",
}


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_SYLLABLES = tuple(a + b for a in _LETTERS for b in _LETTERS)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_TAKE_RE = re.compile(r"^\s*TAKE\s+(.+?)\s*$", re.IGNORECASE)

_LEVELS = {
    1: {
        "atom_resources": 6,
        "dims": 1,
        "atom_conflicts": 0,
        "atom_deps": 0,
        "tight": (0.40, 0.48),
        "episode_tight": (0.34, 0.39),
        "value": (2, 12),
        "rounds": 2,
        "offerings": (4, 4),
        "episode_conflicts": 1,
        "episode_deps": 0,
    },
    2: {
        "atom_resources": 8,
        "dims": 2,
        "atom_conflicts": 1,
        "atom_deps": 0,
        "tight": (0.35, 0.43),
        "episode_tight": (0.38, 0.40),
        "value": (3, 18),
        "rounds": 2,
        "offerings": (5, 5),
        "episode_conflicts": 2,
        "episode_deps": 0,
    },
    3: {
        "atom_resources": 10,
        "dims": 3,
        "atom_conflicts": 3,
        "atom_deps": 2,
        "tight": (0.30, 0.38),
        "episode_tight": (0.24, 0.29),
        "value": (4, 24),
        "rounds": 3,
        "offerings": (4, 4, 4),
        "episode_conflicts": 8,
        "episode_deps": 4,
    },
    4: {
        "atom_resources": 12,
        "dims": 3,
        "atom_conflicts": 5,
        "atom_deps": 3,
        "tight": (0.26, 0.34),
        "episode_tight": (0.08, 0.14),
        "value": (5, 30),
        "rounds": 3,
        "offerings": (5, 4, 4),
        "episode_conflicts": 10,
        "episode_deps": 6,
    },
}


def generate(seed, level, n, mode):
    if level not in _LEVELS:
        raise ValueError("level must be one of 1, 2, 3, 4")
    if mode not in ("atom", "episode"):
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")

    rng = random.Random(_stable_seed(seed, level, mode))
    items = []
    for index in range(n):
        for _attempt in range(200):
            item = _make_item(rng, seed, level, index, mode)
            if _acceptable_item(item):
                items.append(item)
                break
        else:
            raise RuntimeError("stockade generation exhausted 200 retries")
    return items


class Env:
    def __init__(self, item):
        self.item = item
        self.done = False
        self.round_index = 0
        self.remaining = list(item["caps"])
        self.acquired = set()
        self.value = 0

    def reset(self):
        self.done = False
        self.round_index = 0
        self.remaining = list(self.item["caps"])
        self.acquired = set()
        self.value = 0
        if self.item["mode"] == "atom":
            return _atom_observation(self.item)
        return _episode_reset_observation(self.item)

    def step(self, action):
        if self.done:
            return "Done.", True
        if self.item["mode"] == "atom":
            self.done = True
            _selected, malformed = _parse_answer(action, self.item)
            return ("Malformed answer." if malformed else "Recorded."), True

        valid, selected, _reason = _episode_parse_and_check(
            self.item, self.round_index, action, self.acquired, self.remaining
        )
        if valid:
            for name in selected:
                res = _resource_map(self.item)[name]
                self.value += res["value"]
                self.acquired.add(name)
                for dim in range(len(self.remaining)):
                    self.remaining[dim] -= res["costs"][dim]
            note = "OK."
        else:
            note = "Invalid action; round forfeited."

        self.round_index += 1
        if self.round_index >= self.item["max_turns"]:
            self.done = True
            return note + "\nDone.", True
        return _episode_turn_observation(self.item, self.round_index, self.remaining, note), False


def score(item, transcript):
    if item["mode"] == "atom":
        action = transcript[-1].get("action", "") if transcript else ""
        selected, malformed = _parse_answer(action, item)
        if malformed:
            return {
                "score": 0.0,
                "achieved": 0,
                "optimum": item["optimum"],
                "feasible": False,
                "malformed": True,
            }
        feasible, achieved = _atom_feasible_value(item, selected)
        if not feasible:
            return {
                "score": 0.0,
                "achieved": 0,
                "optimum": item["optimum"],
                "feasible": False,
                "malformed": False,
            }
        return {
            "score": _ratio(achieved, item["optimum"]),
            "achieved": achieved,
            "optimum": item["optimum"],
            "feasible": True,
            "malformed": False,
        }

    remaining = list(item["caps"])
    acquired = set()
    achieved = 0
    forfeits = 0
    malformed = False
    for round_index in range(item["max_turns"]):
        action = transcript[round_index].get("action", "") if round_index < len(transcript) else ""
        valid, selected, reason = _episode_parse_and_check(
            item, round_index, action, acquired, remaining
        )
        if not valid:
            forfeits += 1
            malformed = malformed or reason == "malformed"
            continue
        for name in selected:
            res = _resource_map(item)[name]
            achieved += res["value"]
            acquired.add(name)
            for dim in range(len(remaining)):
                remaining[dim] -= res["costs"][dim]
    return {
        "score": _ratio(achieved, item["optimum"]),
        "achieved": achieved,
        "optimum": item["optimum"],
        "feasible": forfeits == 0,
        "malformed": malformed,
        "forfeits": forfeits,
    }


def oracle_policy(item, history):
    if item["mode"] == "atom":
        selected = item["optimal_selection"]
        return "ANSWER: " + (", ".join(selected) if selected else "NONE")

    round_index = len([entry for entry in history if "action" in entry])
    if round_index >= len(item["optimal_rounds"]):
        return "TAKE NONE"
    selected = item["optimal_rounds"][round_index]
    return "TAKE " + (", ".join(selected) if selected else "NONE")


def random_policy(item, history, rng):
    if item["mode"] == "atom":
        names = [res["name"] for res in item["resources"]]
        verb = "ANSWER:"
    else:
        round_index = len([entry for entry in history if "action" in entry])
        names = list(item["rounds"][round_index]) if round_index < len(item["rounds"]) else []
        verb = "TAKE"

    if not names or rng.random() < 0.15:
        return verb + " NONE"
    selected = [name for name in names if rng.random() < 0.85]
    if not selected:
        return verb + " NONE"
    return verb + " " + ", ".join(selected)


def _stable_seed(seed, level, mode):
    text = str(seed) + "|" + str(level) + "|" + mode
    value = 1469598103934665603
    for char in text:
        value ^= ord(char)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value


def _make_item(rng, seed, level, index, mode):
    cfg = _LEVELS[level]
    dims = [_token(rng) for _ in range(cfg["dims"])]
    scenario = _token(rng)

    if mode == "atom":
        resources = _make_resources(
            rng, cfg["atom_resources"], cfg["dims"], cfg["value"], None, set()
        )
        conflict_count = cfg["atom_conflicts"]
        dep_count = cfg["atom_deps"]
        max_turns = 1
        rounds = []
        tight = cfg["tight"]
    else:
        resources = []
        used_names = set()
        for round_number, count in enumerate(cfg["offerings"], 1):
            resources.extend(
                _make_resources(rng, count, cfg["dims"], cfg["value"], round_number, used_names)
            )
        conflict_count = cfg["episode_conflicts"]
        dep_count = cfg["episode_deps"]
        max_turns = cfg["rounds"]
        rounds = [
            [res["name"] for res in resources if res["round"] == round_number]
            for round_number in range(1, cfg["rounds"] + 1)
        ]
        tight = cfg["episode_tight"]

    caps = _make_caps(rng, resources, cfg["dims"], tight)
    conflicts = _make_conflicts(rng, resources, conflict_count)
    deps = _make_dependencies(rng, resources, dep_count, conflicts)
    optimum, mask = _solve_optimum(resources, caps, conflicts, deps)

    item = {
        "id": "stockade-" + mode + "-L" + str(level) + "-" + str(seed) + "-" + str(index),
        "level": level,
        "mode": mode,
        "max_turns": max_turns,
        "scenario": scenario,
        "dims": dims,
        "caps": caps,
        "resources": resources,
        "rules": {"ban": conflicts, "need": deps},
        "optimum": optimum,
    }
    if mode == "atom":
        item["optimal_selection"] = [
            resources[i]["name"] for i in range(len(resources)) if mask & (1 << i)
        ]
    else:
        item["rounds"] = rounds
        item["optimal_selection"] = [
            resources[i]["name"] for i in range(len(resources)) if mask & (1 << i)
        ]
        item["optimal_rounds"] = [
            [name for name in round_names if name in set(item["optimal_selection"])]
            for round_names in rounds
        ]
    return item


def _acceptable_item(item):
    if item["optimum"] <= 0 or not item["optimal_selection"]:
        return False
    if item["mode"] == "atom":
        return len(_atom_observation(item)) <= 1200
    obs = _episode_reset_observation(item)
    return len(obs) <= 800


def _token(rng):
    return "qx" + "".join(rng.choice(_SYLLABLES) for _ in range(3)) + "q"


def _make_resources(rng, count, dims, value_range, round_number, used):
    resources = []
    for _ in range(count):
        name = _token(rng)
        while name in used:
            name = _token(rng)
        used.add(name)
        res = {
            "name": name,
            "value": rng.randint(value_range[0], value_range[1]),
            "costs": [rng.randint(1, 9 + dims) for _dim in range(dims)],
        }
        if round_number is not None:
            res["round"] = round_number
        resources.append(res)
    return resources


def _make_caps(rng, resources, dims, tight):
    caps = []
    for dim in range(dims):
        total = sum(res["costs"][dim] for res in resources)
        low = _ceil_float(total * tight[0])
        high = int(total * tight[1])
        if high < low:
            high = low
        cap = rng.randint(max(1, low), max(1, high))
        caps.append(cap)

    if not any(_costs_fit(res["costs"], caps) for res in resources):
        cheapest = min(resources, key=lambda res: sum(res["costs"]))
        caps = [max(caps[dim], cheapest["costs"][dim]) for dim in range(dims)]
    return caps


def _make_conflicts(rng, resources, count):
    pairs = set()
    n = len(resources)
    guard = 0
    while len(pairs) < count and guard < 2000:
        guard += 1
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        pairs.add(pair)
    return [[resources[a]["name"], resources[b]["name"]] for a, b in sorted(pairs)]


def _make_dependencies(rng, resources, count, conflicts):
    conflict_pairs = {tuple(sorted(pair)) for pair in conflicts}
    name_to_index = {res["name"]: i for i, res in enumerate(resources)}
    conflict_indices = {
        tuple(sorted((name_to_index[a], name_to_index[b]))) for a, b in conflict_pairs
    }
    deps = []
    dependents = set()
    n = len(resources)
    guard = 0
    while len(deps) < count and guard < 4000:
        guard += 1
        a = rng.randrange(1, n)
        if a in dependents:
            continue
        b = rng.randrange(0, a)
        if tuple(sorted((a, b))) in conflict_indices:
            continue
        deps.append([resources[a]["name"], resources[b]["name"]])
        dependents.add(a)
    return deps


def _solve_optimum(resources, caps, conflicts, deps):
    n = len(resources)
    name_to_index = {res["name"]: i for i, res in enumerate(resources)}
    values = [res["value"] for res in resources]
    costs = [res["costs"] for res in resources]
    rounds = [res.get("round", 1) for res in resources]

    conflict_masks = [0] * n
    for a_name, b_name in conflicts:
        a = name_to_index[a_name]
        b = name_to_index[b_name]
        conflict_masks[a] |= 1 << b
        conflict_masks[b] |= 1 << a

    dep_masks = [0] * n
    required_by = [0] * n
    for a_name, b_name in deps:
        a = name_to_index[a_name]
        b = name_to_index[b_name]
        dep_masks[a] |= 1 << b
        required_by[b] |= 1 << a

    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]

    best_value = 0
    best_mask = 0

    def dfs(i, used, selected, value):
        nonlocal best_value, best_mask
        if value + suffix[i] <= best_value:
            return
        if i == n:
            if _mask_dependencies_ok(selected, dep_masks, rounds) and value > best_value:
                best_value = value
                best_mask = selected
            return

        bit = 1 << i
        can_include = True
        new_used = []
        for dim in range(len(caps)):
            total = used[dim] + costs[i][dim]
            if total > caps[dim]:
                can_include = False
                break
            new_used.append(total)
        if can_include and conflict_masks[i] & selected:
            can_include = False
        if can_include:
            earlier = dep_masks[i] & ((1 << i) - 1)
            if earlier & ~selected:
                can_include = False
        if can_include:
            for dep_index in _bits(dep_masks[i]):
                if rounds[dep_index] > rounds[i]:
                    can_include = False
                    break
        if can_include:
            dfs(i + 1, new_used, selected | bit, value + values[i])

        if not (selected & required_by[i]):
            dfs(i + 1, used, selected, value)

    dfs(0, [0] * len(caps), 0, 0)
    return best_value, best_mask


def _mask_dependencies_ok(mask, dep_masks, rounds):
    for i, dep_mask in enumerate(dep_masks):
        if not (mask & (1 << i)):
            continue
        if dep_mask & ~mask:
            return False
        for dep_index in _bits(dep_mask):
            if rounds[dep_index] > rounds[i]:
                return False
    return True


def _bits(mask):
    index = 0
    while mask:
        if mask & 1:
            yield index
        mask >>= 1
        index += 1


def _atom_observation(item):
    lines = [
        "case " + item["scenario"],
        _cap_line(item["dims"], item["caps"]),
        "items:",
    ]
    lines.extend(_resource_line(res) for res in item["resources"])
    lines.append(_rules_line(item))
    lines.append("Reply with final line 'ANSWER: <comma-separated names>' or 'ANSWER: NONE'.")
    return "\n".join(lines)


def _episode_reset_observation(item):
    lines = ["case " + item["scenario"], _cap_line(item["dims"], item["caps"])]
    for round_index, round_names in enumerate(item["rounds"], 1):
        lines.append("R" + str(round_index) + ": " + _round_listing(item, round_names))
    lines.append(_rules_line(item))
    lines.append("Round 1 action: TAKE <comma-separated current names> or TAKE NONE.")
    return "\n".join(lines)


def _episode_turn_observation(item, round_index, remaining, note):
    round_names = item["rounds"][round_index]
    lines = [
        note,
        _cap_line(item["dims"], remaining, "left"),
        "R" + str(round_index + 1) + ": " + _round_listing(item, round_names),
        "Action: TAKE <comma-separated current names> or TAKE NONE.",
    ]
    return "\n".join(lines)


def _cap_line(dims, caps, label="cap"):
    return label + " " + " ".join(dims[i] + "=" + str(caps[i]) for i in range(len(caps)))


def _resource_line(res):
    return res["name"] + " v=" + str(res["value"]) + " c=" + "/".join(str(c) for c in res["costs"])


def _round_listing(item, names):
    by_name = _resource_map(item)
    return "; ".join(_resource_line(by_name[name]) for name in names)


def _rules_line(item):
    bans = item["rules"]["ban"]
    needs = item["rules"]["need"]
    ban_text = ";".join(a + "," + b for a, b in bans) if bans else "none"
    need_text = ";".join(a + ">" + b for a, b in needs) if needs else "none"
    return "ban:" + ban_text + " need:" + need_text


def _parse_answer(action, item):
    matches = _ANSWER_RE.findall(action or "")
    if not matches:
        return [], True
    return _parse_selection(matches[-1], [res["name"] for res in item["resources"]])


def _parse_take(action, legal_names):
    text = action or ""
    if "\n" in text.strip():
        return [], True
    match = _TAKE_RE.match(text)
    if not match:
        return [], True
    return _parse_selection(match.group(1), legal_names)


def _parse_selection(text, legal_names):
    raw = text.strip()
    if re.fullmatch(r"NONE", raw, re.IGNORECASE):
        return [], False
    if not raw or raw.startswith(",") or raw.endswith(",") or ",," in raw:
        return [], True
    legal = {name.lower(): name for name in legal_names}
    seen = set()
    selected = []
    for part in raw.split(","):
        lowered = part.strip().lower()
        if not lowered or lowered not in legal or lowered in seen:
            return [], True
        seen.add(lowered)
        selected.append(legal[lowered])
    return selected, False


def _atom_feasible_value(item, selected):
    selected_set = set(selected)
    if not selected_set:
        return True, 0
    by_name = _resource_map(item)
    used = [0] * len(item["caps"])
    value = 0
    for name in selected_set:
        res = by_name[name]
        value += res["value"]
        for dim in range(len(used)):
            used[dim] += res["costs"][dim]
            if used[dim] > item["caps"][dim]:
                return False, 0
    for a, b in item["rules"]["ban"]:
        if a in selected_set and b in selected_set:
            return False, 0
    for a, b in item["rules"]["need"]:
        if a in selected_set and b not in selected_set:
            return False, 0
    return True, value


def _episode_parse_and_check(item, round_index, action, acquired, remaining):
    if round_index >= len(item["rounds"]):
        return False, [], "malformed"
    current_names = item["rounds"][round_index]
    selected, malformed = _parse_take(action, current_names)
    if malformed:
        return False, [], "malformed"
    selected_set = set(selected)
    if not selected_set:
        return True, [], ""

    by_name = _resource_map(item)
    used = [0] * len(remaining)
    for name in selected_set:
        res = by_name[name]
        for dim in range(len(used)):
            used[dim] += res["costs"][dim]
            if used[dim] > remaining[dim]:
                return False, [], "illegal"
    combined = set(acquired) | selected_set
    for a, b in item["rules"]["ban"]:
        if a in combined and b in combined and (a in selected_set or b in selected_set):
            return False, [], "illegal"
    for a, b in item["rules"]["need"]:
        if a in selected_set and b not in combined:
            return False, [], "illegal"
    return True, selected, ""


def _resource_map(item):
    return {res["name"]: res for res in item["resources"]}


def _costs_fit(costs, caps):
    return all(costs[i] <= caps[i] for i in range(len(caps)))


def _ceil_float(value):
    integer = int(value)
    if integer < value:
        return integer + 1
    return integer


def _ratio(value, optimum):
    if optimum <= 0:
        return 0.0
    score_value = value / optimum
    if score_value < 0.0:
        return 0.0
    if score_value > 1.0:
        return 1.0
    return score_value
