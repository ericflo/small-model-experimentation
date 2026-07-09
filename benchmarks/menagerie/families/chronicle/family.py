import hashlib
import random
import re


META = {
    "name": "chronicle",
    "capability": "state tracking: simulate an evolving fictional world from an ordered event stream and report the final state",
    "paradigm": "single-turn",
    "action_format": "reply with a final line 'ANSWER: <name>'",
}


EVENT_COUNTS = {
    "atom": {1: 5, 2: 12, 3: 20, 4: 32},
    "episode": {1: 7, 2: 12, 3: 18, 4: 20},
}

ENTITY_COUNTS = {
    "atom": {
        1: (3, 0, 4, 0),
        2: (5, 0, 5, 0),
        3: (6, 3, 5, 0),
        4: (8, 4, 7, 4),
    },
    "episode": {
        1: (4, 0, 5, 0),
        2: (5, 0, 5, 0),
        3: (6, 3, 5, 0),
        4: (7, 3, 6, 3),
    },
}

Q_SYLLABLES = ("qa", "qe", "qi", "qo", "qu")
OTHER_SYLLABLES = (
    "za",
    "ze",
    "zi",
    "zo",
    "zu",
    "xa",
    "xe",
    "xi",
    "xo",
    "xu",
    "va",
    "ve",
    "vi",
    "vo",
    "vu",
    "ka",
    "ke",
    "ki",
    "ko",
    "ku",
    "ma",
    "me",
    "mi",
    "mo",
    "mu",
    "ra",
    "re",
    "ri",
    "ro",
    "ru",
    "ta",
    "te",
    "ti",
    "to",
    "tu",
    "na",
    "ne",
    "ni",
    "no",
    "nu",
)
SYLLABLES = Q_SYLLABLES + OTHER_SYLLABLES
RESERVED = {
    "answer",
    "name",
    "events",
    "start",
    "where",
    "which",
    "object",
    "place",
    "agent",
    "line",
    "final",
    "end",
    "rules",
    "true",
    "false",
    "q",
}
ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE)
MAX_PROMPT_RETRY_SALTS = 50


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if mode not in EVENT_COUNTS:
        raise ValueError("mode must be atom or episode")
    if level not in EVENT_COUNTS[mode]:
        raise ValueError("level must be 1, 2, 3, or 4")
    items = []
    for index in range(n):
        items.append(_generate_with_retry(seed, level, mode, index))
    return items


class Env:
    def __init__(self, item: dict):
        self.item = item

    def reset(self) -> str:
        return self.item.get("prompt", "")

    def step(self, action) -> tuple[str, bool]:
        try:
            text = "" if action is None else str(action)
            got = _extract_answer(text)
            if got is None or got.strip() == "":
                return ("Reply with ANSWER: <name>.", True)
            return ("", True)
        except Exception:
            return ("Reply with ANSWER: <name>.", True)


def score(item: dict, transcript: list[dict]) -> dict:
    try:
        action = ""
        if transcript:
            last = transcript[-1]
            if isinstance(last, dict):
                action = last.get("action", "")
            else:
                action = str(last)
        action = "" if action is None else str(action)
        raw = _extract_answer(action)
        if raw is None:
            raw = _extract_bare_answer(action)
        got = "" if raw is None else raw.strip()
        got_canon = _canonical(got)
        accepted = {_canonical(value) for value in item.get("accepted", [])}
        expected = item.get("gold", "")
        return {
            "score": 1.0 if got_canon in accepted else 0.0,
            "expected": expected,
            "got": got,
        }
    except Exception:
        return {"score": 0.0, "expected": item.get("gold", ""), "got": ""}


def oracle_policy(item: dict, history: list[dict]) -> str:
    return "ANSWER: " + item["gold"]


def random_policy(item: dict, history: list[dict], rng) -> str:
    used = set(item.get("visible_tokens", []))
    return "ANSWER: " + _fresh_token(rng, used)


def _item_seed(seed, level, mode, index, salt=0):
    if salt == 0:
        payload = repr(("chronicle", seed, level, mode, index)).encode("utf-8")
    else:
        payload = repr(("chronicle", seed, level, mode, index, salt)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _generate_with_retry(seed, level, mode, index):
    limit = _prompt_limit(mode)
    last_length = None
    for salt in range(MAX_PROMPT_RETRY_SALTS + 1):
        rng = random.Random(_item_seed(seed, level, mode, index, salt))
        item = _generate_one(seed, level, mode, index, rng)
        last_length = len(item["prompt"])
        if last_length <= limit and not _shortcut_leak(item) and not _second_last_shortcut_leak(item):
            return item
    raise RuntimeError(
        f"chronicle retry window failed to find a prompt that fits the {limit}-char budget "
        f"and passes the shortcut guard after {MAX_PROMPT_RETRY_SALTS} retry salts for "
        f"seed={seed} level={level} mode={mode} index={index}; last length={last_length}"
    )


def _prompt_limit(mode):
    return 1200 if mode == "atom" else 800


def _generate_one(seed, level, mode, index, rng):
    objects_n, agents_n, places_n, aliases_n = ENTITY_COUNTS[mode][level]
    used = set()
    objects = [_fresh_token(rng, used) for _ in range(objects_n)]
    agents = [_fresh_token(rng, used) for _ in range(agents_n)]
    places = [_fresh_token(rng, used) for _ in range(places_n)]
    alias_tokens = [_fresh_token(rng, used) for _ in range(aliases_n)]
    initial = _initial_state(rng, objects, agents, places)
    state = _clone_state(initial)
    events = []
    event_count = EVENT_COUNTS[mode][level]
    forced_query = None
    if level == 1:
        forced_query = _build_l1(rng, state, events, event_count)
    elif level == 2:
        _build_l2(rng, state, events, event_count)
    elif level == 3:
        _build_l3(rng, state, events, event_count)
    else:
        _build_l4(rng, state, events, event_count, alias_tokens)
    assert len(events) == event_count

    replayed = _replay(initial, events)
    assert _state_core(replayed) == _state_core(state)
    query = _make_query(rng, state, forced_query)
    accepted = _accepted_answers(query["gold"], state)
    red_herrings = _red_herring_indices(initial, events, query)
    for idx in red_herrings:
        ablated = events[:idx] + events[idx + 1 :]
        assert _answer_for_query(_replay(initial, ablated), query) == query["gold"]

    prompt = _make_prompt(level, mode, initial, events, query)
    return {
        "id": f"chronicle-{mode}-L{level}-{seed}-{index}",
        "level": level,
        "mode": mode,
        "max_turns": 1,
        "prompt": prompt,
        "gold": query["gold"],
        "accepted": accepted,
        "accepted_aliases": [value for value in accepted if value != query["gold"]],
        "query": query,
        "events": list(events),
        "initial_state": _clone_state(initial),
        "final_state": _clone_state(state),
        "red_herring_indices": red_herrings,
        "visible_tokens": sorted(used),
    }


def _fresh_token(rng, used):
    for _ in range(10000):
        size = 2
        pieces = [rng.choice(SYLLABLES) for _ in range(size)]
        if not any(piece in Q_SYLLABLES for piece in pieces):
            pieces[rng.randrange(size)] = rng.choice(Q_SYLLABLES)
        token = "".join(pieces)
        if token not in used and token not in RESERVED:
            used.add(token)
            return token
    raise RuntimeError("token pool exhausted")


def _initial_state(rng, objects, agents, places):
    object_home = {}
    for idx, obj in enumerate(objects):
        if idx < len(agents):
            object_home[obj] = places[idx % len(places)]
        else:
            object_home[obj] = rng.choice(places)
    agent_place = {agent: places[idx % len(places)] for idx, agent in enumerate(agents)}
    return {
        "objects": list(objects),
        "agents": list(agents),
        "places": list(places),
        "object_home": object_home,
        "agent_place": agent_place,
        "aliases": {},
    }


def _clone_state(state):
    return {
        "objects": list(state["objects"]),
        "agents": list(state["agents"]),
        "places": list(state["places"]),
        "object_home": dict(state["object_home"]),
        "agent_place": dict(state["agent_place"]),
        "aliases": dict(state["aliases"]),
    }


def _state_core(state):
    return {
        "object_home": dict(state["object_home"]),
        "agent_place": dict(state["agent_place"]),
        "aliases": dict(state["aliases"]),
    }


def _append(events, state, event):
    events.append(event)
    _apply_event(state, event)


def _build_l1(rng, state, events, event_count):
    objects = state["objects"]
    target = objects[0]
    red = objects[1:]
    target_moves = event_count - (2 if event_count == 5 else 3)
    for idx in range(event_count):
        obj = target if idx < target_moves else red[(idx - target_moves) % len(red)]
        _append(events, state, _move_object_event(state, obj, rng))
    return {"type": "where", "subject": target}


def _build_l2(rng, state, events, event_count):
    places = state["places"]
    objects = state["objects"]
    _append(events, state, f"{places[0]}>>{places[2]}")
    _append(events, state, f"{objects[0]}~{objects[1]}")
    plan = ("move", "transfer", "swap", "move", "transfer", "swap", "move", "move", "transfer", "swap")
    _fill_plan(rng, state, events, event_count, plan, allow_agents=False)


def _build_l3(rng, state, events, event_count):
    objects = state["objects"]
    agents = state["agents"]
    places = state["places"]
    a0 = agents[0]
    o0, o1, o2 = objects[0], objects[1], objects[2]
    _append(events, state, f"{a0}+{o0}")
    _append(events, state, f"{a0}>{places[1]}")
    _append(events, state, f"?{o0}@{places[1]}:{o1}>{places[2]}")
    _append(events, state, f"?{o2}#{a0}:{o2}>{places[3]}")
    _append(events, state, f"{a0}-{o0}")
    _append(events, state, f"{places[2]}>>{places[3]}")
    _append(events, state, f"{o0}~{o1}")
    plan = (
        "move",
        "agent_move",
        "pickup",
        "drop",
        "transfer",
        "swap",
        "cond_true",
        "cond_false",
        "move",
        "transfer",
        "swap",
    )
    _fill_plan(rng, state, events, event_count, plan, allow_agents=True)


def _build_l4(rng, state, events, event_count, alias_tokens):
    objects = state["objects"]
    agents = state["agents"]
    places = state["places"]
    a0 = agents[0]
    o0, o1, o2, o3 = objects[0], objects[1], objects[2], objects[3]
    u_obj, u_place, u_agent = alias_tokens[0], alias_tokens[1], alias_tokens[2]
    _append(events, state, f"{u_obj}={o0}")
    _append(events, state, f"{u_place}={places[1]}")
    _append(events, state, f"{a0}+{u_obj}")
    _append(events, state, f"{a0}>{u_place}")
    _append(events, state, f"?{u_obj}@{u_place}:{o1}>{places[2]}")
    _append(events, state, f"?{o1}@{places[2]}:{o2}>{places[3]}")
    _append(events, state, f"?{o3}#{a0}:{o3}>{places[4]}")
    _append(events, state, f"{a0}-{u_obj}")
    _append(events, state, f"{places[2]}>>{places[4]}")
    _append(events, state, f"{o0}~{o1}")
    _append(events, state, f"{u_agent}={a0}")
    _append(events, state, f"{u_agent}>{places[0]}")
    if len(alias_tokens) > 3:
        u_place_2 = alias_tokens[3]
        _append(events, state, f"{u_place_2}={places[4]}")
        _append(events, state, f"{places[3]}>>{u_place_2}")
    else:
        _append(events, state, f"{places[3]}>>{places[4]}")
    plan = (
        "move",
        "pickup",
        "drop",
        "transfer",
        "swap",
        "cond_true",
        "cond_false",
        "agent_move",
        "move",
        "transfer",
        "swap",
    )
    _fill_plan(rng, state, events, event_count, plan, allow_agents=True)


def _fill_plan(rng, state, events, event_count, plan, allow_agents):
    idx = 0
    while len(events) < event_count:
        kind = plan[idx % len(plan)]
        idx += 1
        event = _sample_event(kind, state, rng, allow_agents)
        _append(events, state, event)


def _sample_event(kind, state, rng, allow_agents):
    if kind == "move":
        return _sample_move(state, rng, allow_agents)
    if kind == "agent_move":
        return _sample_agent_move(state, rng)
    if kind == "swap":
        a, b = rng.sample(state["objects"], 2)
        return f"{a}~{b}"
    if kind == "transfer":
        return _sample_transfer(state, rng)
    if kind == "pickup":
        event = _sample_pickup(state, rng)
        return event if event else _sample_move(state, rng, allow_agents)
    if kind == "drop":
        event = _sample_drop(state, rng)
        return event if event else _sample_move(state, rng, allow_agents)
    if kind == "cond_true":
        return _sample_conditional(state, rng, True, allow_agents)
    if kind == "cond_false":
        return _sample_conditional(state, rng, False, allow_agents)
    return _sample_move(state, rng, allow_agents)


def _sample_move(state, rng, allow_agents):
    choices = list(state["objects"])
    if allow_agents:
        choices += list(state["agents"])
    entity = rng.choice(choices)
    if entity in state["objects"]:
        return _move_object_event(state, entity, rng)
    return _move_agent_event(state, entity, rng)


def _sample_agent_move(state, rng):
    return _move_agent_event(state, rng.choice(state["agents"]), rng)


def _move_object_event(state, obj, rng):
    current = _effective_place(state, obj)
    target = _other_place(state, current, rng)
    return f"{obj}>{target}"


def _move_agent_event(state, agent, rng):
    current = state["agent_place"][agent]
    target = _other_place(state, current, rng)
    return f"{agent}>{target}"


def _other_place(state, current, rng):
    places = [place for place in state["places"] if place != current]
    return rng.choice(places or state["places"])


def _sample_transfer(state, rng):
    sources = [place for place in state["places"] if any(home == place for home in state["object_home"].values())]
    source = rng.choice(sources or state["places"])
    targets = [place for place in state["places"] if place != source]
    return f"{source}>>{rng.choice(targets)}"


def _sample_pickup(state, rng):
    pairs = []
    for agent in state["agents"]:
        place = state["agent_place"][agent]
        for obj in state["objects"]:
            if state["object_home"][obj] == place:
                pairs.append((agent, obj))
    if not pairs:
        return None
    agent, obj = rng.choice(pairs)
    return f"{agent}+{obj}"


def _sample_drop(state, rng):
    pairs = []
    for obj, home in state["object_home"].items():
        if home in state["agents"]:
            pairs.append((home, obj))
    if not pairs:
        return None
    agent, obj = rng.choice(pairs)
    return f"{agent}-{obj}"


def _sample_conditional(state, rng, truth, allow_agents):
    cond = _true_condition(state, rng) if truth else _false_condition(state, rng)
    inner = _sample_move(state, rng, allow_agents)
    return f"?{cond}:{inner}"


def _true_condition(state, rng):
    carried = [(obj, home) for obj, home in state["object_home"].items() if home in state["agents"]]
    if carried and rng.randrange(2) == 0:
        obj, agent = rng.choice(carried)
        return f"{obj}#{agent}"
    entities = list(state["objects"]) + list(state["agents"])
    entity = rng.choice(entities)
    return f"{entity}@{_effective_place(state, entity)}"


def _false_condition(state, rng):
    if state["agents"] and rng.randrange(2) == 0:
        obj = rng.choice(state["objects"])
        agents = [agent for agent in state["agents"] if state["object_home"][obj] != agent]
        if agents:
            return f"{obj}#{rng.choice(agents)}"
    entities = list(state["objects"]) + list(state["agents"])
    entity = rng.choice(entities)
    current = _effective_place(state, entity)
    return f"{entity}@{_other_place(state, current, rng)}"


def _resolve(state, token):
    seen = set()
    while token in state["aliases"]:
        if token in seen:
            break
        seen.add(token)
        token = state["aliases"][token]
    return token


def _apply_event(state, event):
    if event.startswith("?"):
        cond, inner = event[1:].split(":", 1)
        if _condition_true(state, cond):
            _apply_event(state, inner)
        return
    if ">>" in event:
        left, right = event.split(">>", 1)
        source = _resolve(state, left)
        target = _resolve(state, right)
        for obj, home in list(state["object_home"].items()):
            if home == source:
                state["object_home"][obj] = target
        return
    if "=" in event:
        left, right = event.split("=", 1)
        state["aliases"][left] = _resolve(state, right)
        return
    if "~" in event:
        left, right = event.split("~", 1)
        a = _resolve(state, left)
        b = _resolve(state, right)
        state["object_home"][a], state["object_home"][b] = state["object_home"][b], state["object_home"][a]
        return
    if "+" in event:
        left, right = event.split("+", 1)
        agent = _resolve(state, left)
        obj = _resolve(state, right)
        state["object_home"][obj] = agent
        return
    if "-" in event:
        left, right = event.split("-", 1)
        agent = _resolve(state, left)
        obj = _resolve(state, right)
        state["object_home"][obj] = state["agent_place"][agent]
        return
    if ">" in event:
        left, right = event.split(">", 1)
        entity = _resolve(state, left)
        place = _resolve(state, right)
        if entity in state["object_home"]:
            state["object_home"][entity] = place
        else:
            state["agent_place"][entity] = place


def _condition_true(state, cond):
    if "@" in cond:
        left, right = cond.split("@", 1)
        entity = _resolve(state, left)
        place = _resolve(state, right)
        return _effective_place(state, entity) == place
    if "#" in cond:
        left, right = cond.split("#", 1)
        obj = _resolve(state, left)
        agent = _resolve(state, right)
        return state["object_home"].get(obj) == agent
    return False


def _effective_place(state, entity):
    entity = _resolve(state, entity)
    if entity in state["object_home"]:
        home = state["object_home"][entity]
        if home in state["agent_place"]:
            return state["agent_place"][home]
        return home
    if entity in state["agent_place"]:
        return state["agent_place"][entity]
    return entity


def _replay(initial, events):
    state = _clone_state(initial)
    for event in events:
        _apply_event(state, event)
    return state


def _make_query(rng, state, forced):
    if forced:
        subject = forced["subject"]
        gold = _effective_place(state, subject)
        return {"type": "where", "subject": subject, "gold": gold}
    unique_places = []
    for place in state["places"]:
        objects = [obj for obj in state["objects"] if _effective_place(state, obj) == place]
        if len(objects) == 1:
            unique_places.append((place, objects[0]))
    if unique_places and rng.randrange(2) == 0:
        place, obj = rng.choice(unique_places)
        return {"type": "which", "place": place, "gold": obj}
    subject = rng.choice(state["objects"])
    return {"type": "where", "subject": subject, "gold": _effective_place(state, subject)}


def _answer_for_query(state, query):
    if query["type"] == "where":
        return _effective_place(state, query["subject"])
    objects = [obj for obj in state["objects"] if _effective_place(state, obj) == query["place"]]
    return objects[0] if len(objects) == 1 else None


def _accepted_answers(gold, state):
    accepted = [gold]
    for alias, target in state["aliases"].items():
        if target == gold:
            accepted.append(alias)
    return sorted(accepted)


def _surface_last_destination(events):
    for event in reversed(events):
        if ">" in event:
            return event.rsplit(">", 1)[1].strip()
    return None


def _shortcut_leak(item):
    dest = _surface_last_destination(item["events"])
    if dest is None:
        return False
    return _canonical(dest) in {_canonical(answer) for answer in item["accepted"]}


def _surface_second_last_destination(events):
    seen = 0
    for event in reversed(events):
        if ">" in event:
            seen += 1
            if seen == 2:
                return event.rsplit(">", 1)[1].strip()
    return None


def _second_last_shortcut_leak(item):
    dest = _surface_second_last_destination(item["events"])
    if dest is None:
        return False
    return _canonical(dest) in {_canonical(answer) for answer in item["accepted"]}


def _red_herring_indices(initial, events, query):
    red = []
    gold = _answer_for_query(_replay(initial, events), query)
    for idx in range(len(events)):
        try:
            state = _replay(initial, events[:idx] + events[idx + 1 :])
            if _answer_for_query(state, query) == gold:
                red.append(idx)
        except Exception:
            pass
    return red


def _make_prompt(level, mode, initial, events, query):
    parts = [_legend(level), _start_line(initial), "Events:"]
    parts.extend(events)
    if query["type"] == "where":
        parts.append(f"where is {query['subject']}?")
    else:
        parts.append(f"which object is at {query['place']}?")
    parts.append("End with final line ANSWER: <name>")
    return "\n".join(parts)


def _legend(level):
    if level == 1:
        return "Rules:\nX>P: move object X to place P."
    if level == 2:
        return (
            "Rules:\n"
            "X>P: move object X to place P.\n"
            "X~Y: swap object homes.\n"
            "P>>Q: move all loose objects at P to Q."
        )
    if level == 3:
        return (
            "X>P: move object/agent X to place P; held objects follow agents.\n"
            "A+X: A picks up loose object X at A's place; A-X: A drops carried X there.\n"
            "X~Y: swap object homes (place/carrier); P>>Q: move all loose objects at P to Q.\n"
            "?C:E: do E iff C true now; C: X@P effective place (carried object at carrier place), X#A carried by A."
        )
    return (
        "X>P: move object/agent X to place P; held objects follow agents.\n"
        "A+X: A picks up loose object X at A's place; A-X: A drops carried X there.\n"
        "X~Y: swap object homes (place/carrier); P>>Q: move all loose objects at P to Q.\n"
        "U=V: alias U for existing entity V.\n"
        "?C:E: do E iff C true now; C: X@P effective place (carried object at carrier place), X#A carried by A."
    )


def _start_line(state):
    objects = " ".join(f"{obj}@{state['object_home'][obj]}" for obj in state["objects"])
    if not state["agents"]:
        return "Start O:" + objects
    agents = " ".join(f"{agent}@{state['agent_place'][agent]}" for agent in state["agents"])
    return "Start O:" + objects + " A:" + agents


def _extract_answer(action):
    found = None
    for line in str(action).splitlines():
        match = ANSWER_RE.match(line)
        if match:
            found = match.group(1)
    return found


def _extract_bare_answer(action):
    last_line = None
    for line in str(action).splitlines():
        stripped = line.strip()
        if stripped:
            last_line = stripped
    if last_line is None:
        return None
    canon = _canonical(last_line)
    if re.fullmatch(r"[a-z]{4}", canon):
        return canon
    return None


def _canonical(value):
    return re.sub(r"[^0-9a-z]+", "", str(value).strip().lower())
