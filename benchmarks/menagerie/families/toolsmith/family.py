import hashlib
import random
import re


META = {
    "name": "toolsmith",
    "capability": "tool orchestration: chaining dependent fictional tool calls where later calls consume opaque values returned by earlier calls",
    "paradigm": "multi-turn",
    "action_format": "episodes: one line per turn, CALL toolname(v1) / CALL toolname(v1,v2) / SUBMIT <value>; atoms: reply with final line 'ANSWER: tool1 -> tool2 -> ...'",
}


_GRAMMAR = {
    "call",
    "submit",
    "answer",
    "ok",
    "err",
    "tools",
    "have",
    "goal",
    "act",
}
_RARE = "qxzv"
_LETTERS = "bcdfghjklmnpqrstvwxyz"
_VOWELS = "aeiou"
_EXTRA = "bcdfhjklmnprstwy"
_ONSETS = tuple("bcdfghjklmnprstwy" + _RARE * 4)
_CODAS = tuple("bcdfghjklmnprst" + _RARE * 3)
_ATOM_DEPTH = {1: 2, 2: 3, 3: 4, 4: 5}
_DISTRACTORS = {1: 3, 2: 5, 3: 6, 4: 8}
_EPISODE_TURNS = {1: 4, 2: 4, 3: 5, 4: 7}

_CALL_RE = re.compile(r"^\s*call\s+([A-Za-z][A-Za-z0-9_]*)\s*\(\s*(.*?)\s*\)\s*$", re.I)
_SUBMIT_RE = re.compile(r"^\s*submit\b\s*(.*?)\s*$", re.I)
_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.*?)\s*$", re.I)
_OK_RE = re.compile(r"\bOK\s+([A-Za-z0-9_]+):([A-Za-z0-9_]+)\b")


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be 1, 2, 3, or 4")
    if mode not in ("atom", "episode"):
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")

    items = []
    salt = _seed_salt(seed)
    for index in range(n):
        item_id = "toolsmith-s%s-L%s-%s%s" % (seed, level, mode[0], index)
        item = None
        for attempt in range(100):
            rng = random.Random(_stable_int("item", seed, level, index, mode, attempt))
            used = set()
            try:
                if mode == "atom":
                    candidate = _build_atom(item_id, salt, rng, used, level)
                else:
                    candidate = _build_episode(item_id, salt, rng, used, level)
                _assert_item(candidate, used)
            except AssertionError:
                continue
            item = candidate
            break
        if item is None:
            raise AssertionError("could not generate invariant-satisfying item")
        items.append(item)
    return items


class Env:
    def __init__(self, item):
        self.item = item
        self.registry = {tool["name"]: tool for tool in item.get("registry", [])}
        self.reset()

    def reset(self) -> str:
        self.values = {entry["value"]: entry["type"] for entry in self.item.get("starts", [])}
        self.done = False
        self.submitted = None
        return _prompt(self.item)

    def step(self, action):
        if self.done:
            return "SUBMITTED", True
        text = "" if action is None else str(action)
        submit = _SUBMIT_RE.match(text)
        if submit:
            self.submitted = submit.group(1).strip().lower()
            self.done = True
            return "SUBMITTED", True

        call = _CALL_RE.match(text)
        if not call:
            return _syntax_error(), False

        name = call.group(1).lower()
        raw_args = call.group(2).strip()
        args = [] if raw_args == "" else [part.strip().lower() for part in raw_args.split(",")]
        if any(arg == "" for arg in args):
            return _syntax_error(), False
        if name not in self.registry:
            return "ERR unknown tool", False

        tool = self.registry[name]
        if len(args) != len(tool["args"]):
            return "ERR arity", False
        for arg in args:
            if arg not in self.values:
                return "ERR value", False
        for arg, need_type in zip(args, tool["args"]):
            if self.values[arg] != need_type:
                return "ERR type", False

        out = _output_value(self.item["id"], self.item["_salt"], name, args)
        self.values[out] = tool["out"]
        return "OK %s:%s" % (out, tool["out"]), False


def score(item: dict, transcript: list[dict]) -> dict:
    try:
        if item.get("mode") == "atom":
            return _score_atom(item, transcript)
        return _score_episode(item, transcript)
    except Exception as exc:
        return {
            "score": 0.0,
            "error": exc.__class__.__name__,
            "c": 0,
            "m": len(item.get("chain", [])) if isinstance(item, dict) else 0,
            "submitted": None,
            "target_hit": False,
        }


def oracle_policy(item: dict, history: list[dict]) -> str:
    if item.get("mode") == "atom":
        return "ANSWER: " + " -> ".join(item.get("answer", []))

    credited = _credited_steps_from_history(item, history)
    for index, step in enumerate(item.get("chain", [])):
        if index not in credited:
            return "CALL %s(%s)" % (step["tool"], ",".join(step["args"]))
    return "SUBMIT " + item.get("target", "")


def random_policy(item: dict, history: list[dict], rng) -> str:
    if item.get("mode") == "atom":
        if rng.random() < 0.5:
            names = [tool["name"] for tool in item.get("registry", [])]
            count = rng.randint(1, 3)
            chosen = [rng.choice(names) for _ in range(count)] if names else []
            return "ANSWER: " + " -> ".join(chosen)
        return rng.choice(("", "CALL", "ANSWER", "SUBMIT", "nargle"))

    roll = rng.random()
    if roll < 0.34:
        return rng.choice(("", "CALL", "CALL missing", "CALL tool value", "SUBMIT", "ANSWER: nope"))
    if roll < 0.67:
        registry = item.get("registry", [])
        if not registry:
            return "CALL tool(value)"
        tool = rng.choice(registry)
        known = _known_values_for_policy(item, history)
        args = []
        for _ in tool.get("args", []):
            if known and rng.random() < 0.55:
                args.append(rng.choice(known))
            else:
                args.append(_garbage_value(item, rng))
        if rng.random() < 0.15:
            if args and rng.random() < 0.5:
                args.pop()
            else:
                args.append(_garbage_value(item, rng))
        return "CALL %s(%s)" % (tool["name"], ",".join(args))

    known = _known_values_for_policy(item, history)
    if known and rng.random() < 0.4:
        value = rng.choice(known)
    else:
        value = _garbage_value(item, rng)
    return "SUBMIT " + value


def _build_atom(item_id, salt, rng, used, level):
    depth = _ATOM_DEPTH[level]
    types = [_new_type(rng, used, salt) for _ in range(depth + 1)]
    start = {"value": _new_start_value(rng, used, salt), "type": types[0]}
    registry = []
    true_defs = []
    for pos in range(depth):
        tool = {
            "name": _new_tool(rng, used, salt),
            "args": [types[pos]],
            "out": types[pos + 1],
            "true": True,
        }
        true_defs.append(tool)
        registry.append(tool)
    _add_distractors(registry, rng, used, salt, types, _DISTRACTORS[level])
    chain = _precompute_chain(item_id, salt, [start], true_defs)
    rng.shuffle(registry)
    item = {
        "id": item_id,
        "level": level,
        "mode": "atom",
        "max_turns": 1,
        "registry": registry,
        "chain": chain,
        "answer": [step["tool"] for step in chain],
        "target": chain[-1]["out"],
        "starts": [start],
        "goal_type": types[-1],
        "_salt": salt,
        "distractors": _DISTRACTORS[level],
    }
    _assert_graph(registry, [types[0]], types[-1], set(types))
    return item


def _build_episode(item_id, salt, rng, used, level):
    registry = []
    true_defs = []
    if level in (1, 2):
        depth = 2 if level == 1 else 3
        types = [_new_type(rng, used, salt) for _ in range(depth + 1)]
        starts = [{"value": _new_start_value(rng, used, salt), "type": types[0]}]
        for pos in range(depth):
            tool = {
                "name": _new_tool(rng, used, salt),
                "args": [types[pos]],
                "out": types[pos + 1],
                "true": True,
            }
            true_defs.append(tool)
            registry.append(tool)
        on_path_types = set(types)
        goal_type = types[-1]
    elif level == 3:
        main = [_new_type(rng, used, salt) for _ in range(3)]
        side = [_new_type(rng, used, salt) for _ in range(2)]
        goal_type = _new_type(rng, used, salt)
        starts = [
            {"value": _new_start_value(rng, used, salt), "type": main[0]},
            {"value": _new_start_value(rng, used, salt), "type": side[0]},
        ]
        specs = [
            ([main[0]], main[1]),
            ([main[1]], main[2]),
            ([side[0]], side[1]),
            ([main[2], side[1]], goal_type),
        ]
        for args, out in specs:
            tool = {"name": _new_tool(rng, used, salt), "args": args, "out": out, "true": True}
            true_defs.append(tool)
            registry.append(tool)
        on_path_types = set(main + side + [goal_type])
    else:
        main = [_new_type(rng, used, salt) for _ in range(4)]
        side = [_new_type(rng, used, salt) for _ in range(3)]
        goal_type = _new_type(rng, used, salt)
        starts = [
            {"value": _new_start_value(rng, used, salt), "type": main[0]},
            {"value": _new_start_value(rng, used, salt), "type": side[0]},
        ]
        specs = [
            ([main[0]], main[1]),
            ([side[0]], side[1]),
            ([main[1], side[1]], main[2]),
            ([main[2]], main[3]),
            ([side[1]], side[2]),
            ([main[3], side[2]], goal_type),
        ]
        for args, out in specs:
            tool = {"name": _new_tool(rng, used, salt), "args": args, "out": out, "true": True}
            true_defs.append(tool)
            registry.append(tool)
        on_path_types = set(main + side + [goal_type])

    _add_distractors(registry, rng, used, salt, sorted(on_path_types), _DISTRACTORS[level])
    chain = _precompute_chain(item_id, salt, starts, true_defs)
    rng.shuffle(registry)
    item = {
        "id": item_id,
        "level": level,
        "mode": "episode",
        "max_turns": _EPISODE_TURNS[level],
        "registry": registry,
        "chain": chain,
        "answer": [step["tool"] for step in chain],
        "target": chain[-1]["out"],
        "starts": starts,
        "goal_type": goal_type,
        "_salt": salt,
        "distractors": _DISTRACTORS[level],
    }
    _assert_graph(registry, [entry["type"] for entry in starts], goal_type, on_path_types)
    return item


def _add_distractors(registry, rng, used, salt, reachable_types, count):
    reachable_types = list(reachable_types)
    on_path = set(reachable_types)
    for _ in range(count):
        name = _new_tool(rng, used, salt)
        out_type = _new_type(rng, used, salt)
        assert out_type not in on_path
        if rng.random() < 0.5:
            in_type = _new_type(rng, used, salt)
            assert in_type not in on_path
        else:
            in_type = rng.choice(reachable_types)
        registry.append({"name": name, "args": [in_type], "out": out_type, "true": False})


def _precompute_chain(item_id, salt, starts, true_defs):
    by_type = {entry["type"]: entry["value"] for entry in starts}
    chain = []
    for tool in true_defs:
        args = [by_type[arg_type] for arg_type in tool["args"]]
        out = _output_value(item_id, salt, tool["name"], args)
        chain.append({"tool": tool["name"], "args": args, "out": out})
        by_type[tool["out"]] = out
    return chain


def _score_episode(item, transcript):
    env = Env(item)
    env.reset()
    submitted = None
    target_hit = False
    credited = set()
    entries = transcript if isinstance(transcript, list) else []
    for entry in entries:
        action = entry.get("action", "") if isinstance(entry, dict) else ""
        parsed = _parse_call(action)
        submit = _SUBMIT_RE.match("" if action is None else str(action))
        obs, done = env.step(action)
        if obs.startswith("OK ") and parsed:
            for index, step in enumerate(item.get("chain", [])):
                if index not in credited and parsed == (step["tool"], step["args"]):
                    credited.add(index)
                    break
        if submit:
            submitted = submit.group(1).strip().lower()
            target_hit = submitted == item.get("target")
        if done:
            break

    c = len(credited)
    m = len(item.get("chain", []))
    if target_hit:
        value = 1.0
    elif c == 0 or m == 0:
        value = 0.0
    else:
        value = 0.5 * c / m
    return {"score": value, "c": c, "m": m, "submitted": submitted, "target_hit": target_hit}


def _score_atom(item, transcript):
    entries = transcript if isinstance(transcript, list) else []
    final_action = ""
    if entries:
        last = entries[-1]
        if isinstance(last, dict):
            final_action = last.get("action", "")
    seq = _extract_answer(final_action)
    oracle = item.get("answer", [])
    m = len(oracle)
    if not seq:
        return {"score": 0.0, "p": 0, "m": m, "target_hit": False}
    if seq == oracle:
        return {"score": 1.0, "p": m, "m": m, "target_hit": True}
    prefix = 0
    for got, want in zip(seq, oracle):
        if got != want:
            break
        prefix += 1
    value = 0.0 if prefix == 0 or m == 0 else 0.5 * prefix / m
    return {"score": value, "p": prefix, "m": m, "target_hit": False}


def _credited_steps_from_history(item, history):
    if item.get("mode") == "atom":
        return set()
    env = Env(item)
    env.reset()
    credited = set()
    entries = history if isinstance(history, list) else []
    for entry in entries:
        action = entry.get("action", "") if isinstance(entry, dict) else ""
        parsed = _parse_call(action)
        obs, done = env.step(action)
        if obs.startswith("OK ") and parsed:
            for index, step in enumerate(item.get("chain", [])):
                if index not in credited and parsed == (step["tool"], step["args"]):
                    credited.add(index)
                    break
        if done:
            break
    return credited


def _known_values_for_policy(item, history):
    values = [entry["value"] for entry in item.get("starts", [])]
    seen = set(values)
    entries = history if isinstance(history, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("obs", "next_obs"):
            obs = entry.get(key, "")
            for match in _OK_RE.finditer(str(obs)):
                value = match.group(1)
                if value not in seen:
                    seen.add(value)
                    values.append(value)
    return values


def _extract_answer(action):
    text = "" if action is None else str(action)
    answer = None
    for line in text.splitlines():
        match = _ANSWER_RE.match(line)
        if match:
            answer = match.group(1).strip()
    if answer is None or answer == "":
        return []
    parts = [part.strip().lower() for part in re.split(r"\s*->\s*", answer)]
    if any(part == "" for part in parts):
        return []
    return parts


def _parse_call(action):
    match = _CALL_RE.match("" if action is None else str(action))
    if not match:
        return None
    raw_args = match.group(2).strip()
    args = [] if raw_args == "" else [part.strip().lower() for part in raw_args.split(",")]
    if any(arg == "" for arg in args):
        return None
    return match.group(1).lower(), args


def _prompt(item):
    sigs = []
    for tool in item.get("registry", []):
        sigs.append("%s(%s)->%s" % (tool["name"], ",".join(tool["args"]), tool["out"]))
    starts = ", ".join("%s:%s" % (entry["value"], entry["type"]) for entry in item.get("starts", []))
    if item.get("mode") == "atom":
        return (
            "Tools: %s\n"
            "Have: %s. Goal:%s.\n"
            "Give tool names in call order. Final line: ANSWER: tool1 -> tool2 -> tool3"
            % ("; ".join(sigs), starts, item.get("goal_type", ""))
        )
    return (
        "Tools: %s\n"
        "Have: %s. Goal:%s.\n"
        "Act one line: CALL name(value) or CALL name(value1,value2); SUBMIT value."
        % ("; ".join(sigs), starts, item.get("goal_type", ""))
    )


def _assert_item(item, used):
    assert item["target"] not in _prompt(item)
    if item["mode"] == "atom":
        assert len(_prompt(item)) <= 1200
        assert item["max_turns"] == 1
    else:
        assert len(_prompt(item)) <= 800
        cap = 4 if item["level"] in (1, 2) else 10 if item["level"] == 3 else 14
        assert item["max_turns"] <= cap
    tokens = set()
    for tool in item["registry"]:
        tokens.add(tool["name"])
        tokens.update(tool["args"])
        tokens.add(tool["out"])
    for start in item["starts"]:
        tokens.add(start["value"])
        tokens.add(start["type"])
    for step in item["chain"]:
        tokens.add(step["tool"])
        tokens.update(step["args"])
        tokens.add(step["out"])
    tokens.add(item["target"])
    tokens.add(item["goal_type"])
    assert all(token == token.lower() for token in tokens)
    assert all(token.lower() not in _GRAMMAR for token in tokens)
    assert all(5 <= len(token) <= 10 for token in tokens)
    values = [start["value"] for start in item["starts"]] + [step["out"] for step in item["chain"]]
    assert len(values) == len(set(values))
    assert not (set(values) & {tool["name"] for tool in item["registry"]})
    type_to_value = {start["type"]: start["value"] for start in item["starts"]}
    tools_by_name = {tool["name"]: tool for tool in item["registry"]}
    assert len(tools_by_name) == len(item["registry"])
    for step in item["chain"]:
        tool = tools_by_name[step["tool"]]
        type_to_value[tool["out"]] = step["out"]
    real_values = set(values)
    distractor_outputs = set()
    for tool in item["registry"]:
        if tool.get("true", False) or not all(arg_type in type_to_value for arg_type in tool["args"]):
            continue
        args = [type_to_value[arg_type] for arg_type in tool["args"]]
        out = _output_value(item["id"], item["_salt"], tool["name"], args)
        assert out not in real_values
        assert out not in distractor_outputs
        distractor_outputs.add(out)
    assert set(used).issuperset(
        {tool["name"] for tool in item["registry"]} | {start["value"] for start in item["starts"]}
    )


def _assert_graph(registry, start_types, goal_type, on_path_types):
    start_types = set(start_types)
    on_path_types = set(on_path_types)
    assert goal_type in on_path_types
    producers = {}
    for tool in registry:
        producers.setdefault(tool["out"], []).append(tool)
        if not tool.get("true", False):
            assert tool["out"] not in on_path_types
            for arg_type in tool["args"]:
                assert arg_type in on_path_types or arg_type not in _true_reachable_types(registry, start_types)
    for typ in on_path_types - start_types:
        made_by = producers.get(typ, [])
        assert len(made_by) == 1
        assert made_by[0].get("true", False)
    assert goal_type in _true_reachable_types(registry, start_types)


def _true_reachable_types(registry, start_types):
    reachable = set(start_types)
    changed = True
    while changed:
        changed = False
        for tool in registry:
            if tool.get("true", False) and all(arg in reachable for arg in tool["args"]):
                if tool["out"] not in reachable:
                    reachable.add(tool["out"])
                    changed = True
    return reachable


def _syntax_error():
    return "ERR syntax use: CALL name(value) / CALL name(value1,value2) / SUBMIT value"


def _stable_int(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest()[:16], "big")


def _seed_salt(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 7140:
        raise ValueError("toolsmith: seed must be an int in [0, 7140)")
    base = _RARE[seed % 4] + _LETTERS[(seed // 4) % 21] + _VOWELS[(seed // 84) % 5]
    q = seed // 420
    return base if q == 0 else base + _EXTRA[q - 1]


def _syllable(rng):
    text = rng.choice(_ONSETS) + rng.choice(_VOWELS)
    if rng.random() < 0.55:
        text += rng.choice(_CODAS)
    return text


def _new_tool(rng, used, salt):
    return _new_token(rng, used, salt, "q", 6, 10)


def _new_type(rng, used, salt):
    return _new_token(rng, used, salt, "x", 5, 7)


def _new_start_value(rng, used, salt):
    return _new_token(rng, used, salt, "z", 6, 9)


def _new_token(rng, used, salt, marker, min_len, max_len):
    for _ in range(1000):
        target = rng.randint(min_len, max_len)
        target = max(target, len(salt) + 2)
        text = salt + marker
        while len(text) < target:
            text += _syllable(rng)
        token = text[:target]
        if token not in used and token.lower() not in _GRAMMAR:
            used.add(token)
            return token
    raise RuntimeError("token generation exhausted")


def _garbage_value(item, rng):
    salt = item.get("_salt", "qaz")
    target = rng.randint(6, 9)
    target = max(target, len(salt) + 2)
    text = salt + "g"
    while len(text) < target:
        text += _syllable(rng)
    return text[:target]


def _hash_stock(salt):
    stock = []
    for index in range(32):
        digest = hashlib.sha256(("%s:%s" % (salt, index)).encode("utf-8")).digest()
        syll = _ONSETS[digest[0] % len(_ONSETS)] + _VOWELS[digest[1] % len(_VOWELS)]
        if digest[2] % 2:
            syll += _CODAS[digest[3] % len(_CODAS)]
        stock.append(syll)
    return stock


def _output_value(item_id, salt, tool_name, args):
    h = hashlib.sha256()
    h.update(str(item_id).encode("utf-8"))
    h.update(b"\0")
    h.update(str(tool_name).encode("utf-8"))
    for arg in args:
        h.update(b"\0")
        h.update(str(arg).encode("utf-8"))
    digest = h.digest()
    stock = _hash_stock(salt)
    target = 8 + digest[0] % 2
    text = salt + "v"
    pos = 1
    while len(text) < target:
        if pos >= len(digest):
            digest = hashlib.sha256(digest).digest()
            pos = 0
        text += stock[digest[pos] % len(stock)]
        pos += 1
    return text[:target]
