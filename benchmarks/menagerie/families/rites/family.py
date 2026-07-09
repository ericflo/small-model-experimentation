import random
import re
from collections import deque


META = {
    "name": "rites",
    "capability": "Protocol/state-machine compliance with hidden state and flag tracking.",
    "paradigm": "multi-turn",
    "action_format": "Episodes use ENACT <action>; atoms end with ANSWER: <a1> <a2> ...",
}


_MASK64 = 0xffffffffffffffff
_SYLLABLES = (
    "qa",
    "zu",
    "xi",
    "vo",
    "ky",
    "re",
    "ul",
    "ja",
    "ob",
    "ne",
    "fi",
    "go",
    "ha",
    "lu",
    "py",
    "se",
    "ti",
    "wa",
    "xo",
    "ye",
    "za",
    "bi",
    "cu",
    "do",
)
_DENYLIST = {
    "bibi",
    "dodo",
    "fido",
    "gobi",
    "gogo",
    "gone",
    "haha",
    "lulu",
    "lure",
    "pyre",
    "redo",
    "tire",
    "ware",
}
_LEVELS = {
    1: {"states": 4, "actions": 5, "flags": 0, "targets": (2,), "max_turns": 4, "cap": 0.10},
    2: {"states": 5, "actions": 7, "flags": 1, "targets": (3,), "max_turns": 4, "cap": 0.06},
    3: {"states": 7, "actions": 10, "flags": 2, "targets": (5, 6), "max_turns": 10, "cap": 0.02},
    4: {"states": 9, "actions": 12, "flags": 3, "targets": (8, 9), "max_turns": 14, "cap": 0.01},
}
_MODE_CODE = {"atom": 0, "episode": 1}
_ACTION_RE = re.compile(r"^\s*ENACT\s+([a-z]{4,6})\s*$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def generate(seed, level, n, mode):
    if level not in _LEVELS:
        raise ValueError("level must be one of 1, 2, 3, 4")
    if mode not in _MODE_CODE:
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be nonnegative")

    items = []
    for item_index in range(n):
        item_seed = _item_seed(seed, level, mode, item_index)
        last_reason = "not attempted"
        for attempt in range(10000):
            rng = random.Random(_attempt_seed(item_seed, attempt))
            item = _candidate(seed, level, mode, item_index, item_seed, rng)
            ok, reason = _finalize_and_check(item)
            if ok:
                items.append(item)
                break
            last_reason = reason
        else:
            raise RuntimeError(
                "rites generation exhausted: seed=%r level=%r mode=%r index=%r reason=%s"
                % (seed, level, mode, item_index, last_reason)
            )
    return items


class Env:
    def __init__(self, item):
        self.item = item
        self.done = False
        self.terminal_obs = None
        self.turns = 0
        self.state = item["_start"]
        self.flags = list(item["_initial_flags"])

    def reset(self):
        self.done = False
        self.terminal_obs = None
        self.turns = 0
        self.state = self.item["_start"]
        self.flags = list(self.item["_initial_flags"])
        return _initial_observation(self.item)

    def step(self, action):
        if self.done:
            return self.terminal_obs, True

        if self.item["mode"] == "atom":
            self.done = True
            result = _score_atom(self.item, action)
            if result["completed"]:
                self.terminal_obs = "THE RITE IS COMPLETE."
            else:
                self.terminal_obs = "THE RITE STALLS."
            return self.terminal_obs, True

        self.turns += 1
        parsed = _parse_episode_action(action)
        if parsed is None:
            obs = "MALFORMED. Reply exactly: ENACT <action-name>"
        else:
            next_state, next_flags, accepted, reason = _apply_action(
                self.item, self.state, self.flags, parsed
            )
            if accepted:
                self.state = next_state
                self.flags = next_flags
                if self.state == self.item["_goal"]:
                    self.done = True
                    self.terminal_obs = "THE RITE IS COMPLETE."
                    return self.terminal_obs, True
                obs = "ACCEPTED."
            elif reason == "unknown":
                obs = "REFUSED: unknown action."
            elif reason == "condition":
                obs = "REFUSED: condition unmet."
            else:
                obs = "REFUSED: wrong place."

        if self.turns >= self.item["max_turns"]:
            self.done = True
            self.terminal_obs = "THE RITE STALLS."
            return self.terminal_obs, True
        return obs, False


def score(item, transcript):
    if item["mode"] == "atom":
        last_answer = None
        for turn in transcript:
            for match in _ANSWER_RE.finditer(str(turn.get("action", ""))):
                last_answer = match.group(1)
        if last_answer is None:
            return {
                "score": 0.0,
                "turns_used": 0,
                "optimal_len": item["_optimal_len"],
                "completed": False,
            }
        return _score_atom(item, last_answer)
    return _score_episode(item, transcript)


def oracle_policy(item, history):
    if item["mode"] == "atom":
        path = _shortest_path(item, item["_start"], list(item["_initial_flags"]))
        return "ANSWER: " + " ".join(path)

    state = item["_start"]
    flags = list(item["_initial_flags"])
    turns = 0
    for turn in history:
        if turns >= item["max_turns"] or state == item["_goal"]:
            break
        turns += 1
        parsed = _parse_episode_action(turn.get("action", ""))
        if parsed is None:
            continue
        next_state, next_flags, accepted, _reason = _apply_action(item, state, flags, parsed)
        if accepted:
            state = next_state
            flags = next_flags
    path = _shortest_path(item, state, flags)
    if path:
        return "ENACT " + path[0]
    return "ENACT " + item["_actions"][0]


def random_policy(item, history, rng):
    if item["mode"] == "atom":
        length = rng.randint(1, item["_optimal_len"] + 2)
        return "ANSWER: " + " ".join(rng.choice(item["_actions"]) for _ in range(length))
    return "ENACT " + rng.choice(item["_actions"])


def _item_seed(seed, level, mode, item_index):
    mode_code = _MODE_CODE[mode]
    return (
        ((seed & 0xffffffff) * 1000003 + level * 10007 + mode_code * 101 + item_index)
        & _MASK64
    )


def _attempt_seed(item_seed, attempt):
    return (item_seed ^ ((attempt * 0x9E3779B97F4A7C15) & _MASK64)) & _MASK64


def _candidate(seed, level, mode, item_index, item_seed, rng):
    params = _LEVELS[level]
    target = params["targets"][rng.randrange(len(params["targets"]))]
    tokens = _tokens(
        rng,
        1 + params["states"] + params["actions"] + params["flags"],
    )
    pos = 0
    rite = tokens[pos]
    pos += 1
    states = tokens[pos : pos + params["states"]]
    pos += params["states"]
    actions = tokens[pos : pos + params["actions"]]
    pos += params["actions"]
    flags = tokens[pos : pos + params["flags"]]

    rules = _rules_for_level(level, target, states, actions, flags)
    rng.shuffle(rules)

    return {
        "id": "rites-%s-L%s-%s-%03d-%016x" % (seed, level, mode, item_index, item_seed),
        "level": level,
        "mode": mode,
        "max_turns": 1 if mode == "atom" else params["max_turns"],
        "_rite": rite,
        "_states": states,
        "_start": states[0],
        "_goal": states[-1],
        "_flags": flags,
        "_initial_flags": [0 for _flag in flags],
        "_rules": rules,
        "_actions": [rule["action"] for rule in rules],
        "_target_len": target,
        "_item_seed": item_seed,
    }


def _tokens(rng, count):
    used = set()
    out = []
    for _ in range(count):
        token = None
        for _try in range(64):
            cand = rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)
            if cand not in used and cand not in _DENYLIST:
                token = cand
                break
        if token is None:
            for _try in range(512):
                cand = rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES) + rng.choice(_SYLLABLES)
                if cand not in used and cand not in _DENYLIST:
                    token = cand
                    break
        if token is None:
            raise RuntimeError("token generation exhausted")
        used.add(token)
        out.append(token)
    return out


def _rules_for_level(level, target, s, a, f):
    if level == 1:
        return [
            _transit(a[0], s[0], s[1]),
            _transit(a[1], s[1], s[-1]),
            _transit(a[2], s[0], s[-2]),
            _transit(a[3], s[1], s[-2]),
            _transit(a[4], s[-2], s[0]),
        ]

    if level == 2:
        return [
            _transit(a[0], s[0], s[1]),
            _toggle(a[1], f[0], s[1], None),
            _transit(a[2], s[1], s[-1], [f[0], 1]),
            _transit(a[3], s[0], s[-2]),
            _transit(a[4], s[1], s[-2]),
            _transit(a[5], s[-2], s[0]),
            _transit(a[6], s[2], s[-2]),
        ]

    if level == 3 and target == 5:
        return [
            _transit(a[0], s[0], s[1]),
            _toggle(a[1], f[0], s[1], None),
            _transit(a[2], s[1], s[2], [f[0], 1]),
            _toggle(a[3], f[1], s[2], None),
            _transit(a[4], s[2], s[-1], [f[1], 1]),
            _transit(a[5], s[0], s[-2]),
            _transit(a[6], s[1], s[-2]),
            _transit(a[7], s[2], s[-2]),
            _transit(a[8], s[-2], s[0]),
            _transit(a[9], s[3], s[4]),
        ]

    if level == 3:
        return [
            _transit(a[0], s[0], s[1]),
            _toggle(a[1], f[0], s[1], None),
            _transit(a[2], s[1], s[2], [f[0], 1]),
            _toggle(a[3], f[1], s[2], None),
            _transit(a[4], s[2], s[3], [f[1], 1]),
            _transit(a[5], s[3], s[-1], [f[0], 1]),
            _transit(a[6], s[0], s[-2]),
            _transit(a[7], s[1], s[-2]),
            _transit(a[8], s[-2], s[0]),
            _transit(a[9], s[4], s[-2]),
        ]

    if level == 4 and target == 8:
        return [
            _transit(a[0], s[0], s[1]),
            _toggle(a[1], f[0], s[1], None),
            _transit(a[2], s[1], s[2], [f[0], 1]),
            _toggle(a[3], f[1], s[2], [f[0], 1]),
            _transit(a[4], s[2], s[3], [f[1], 1]),
            _toggle(a[5], f[2], s[3], [f[0], 1]),
            _transit(a[6], s[3], s[4], [f[2], 1]),
            _transit(a[7], s[4], s[-1], [f[1], 1]),
            _transit(a[8], s[0], s[-2]),
            _transit(a[9], s[2], s[-2]),
            _transit(a[10], s[-2], s[0]),
            _transit(a[11], s[5], s[6]),
        ]

    return [
        _transit(a[0], s[0], s[1]),
        _toggle(a[1], f[0], s[1], None),
        _transit(a[2], s[1], s[2], [f[0], 1]),
        _toggle(a[3], f[1], s[2], [f[0], 1]),
        _transit(a[4], s[2], s[3], [f[1], 1]),
        _toggle(a[5], f[2], s[3], [f[0], 1]),
        _transit(a[6], s[3], s[4], [f[2], 1]),
        _transit(a[7], s[4], s[5], [f[1], 1]),
        _transit(a[8], s[5], s[-1], [f[2], 1]),
        _transit(a[9], s[0], s[-2]),
        _transit(a[10], s[-2], s[0]),
        _transit(a[11], s[6], s[-2]),
    ]


def _transit(action, src, dst, guard=None):
    return {"action": action, "kind": "transit", "src": src, "dst": dst, "guard": guard}


def _toggle(action, flag, state, guard):
    return {"action": action, "kind": "toggle", "flag": flag, "state": state, "guard": guard}


def _finalize_and_check(item):
    path = _shortest_path(item, item["_start"], list(item["_initial_flags"]))
    if path is None:
        return False, "unreachable"
    if len(path) != item["_target_len"]:
        return False, "optimal length %s != target %s" % (len(path), item["_target_len"])
    item["_optimal_path"] = path
    item["_optimal_len"] = len(path)

    obs = _initial_observation(item)
    if item["mode"] == "episode" and len(obs) > 800:
        return False, "episode observation too long"
    if item["mode"] == "atom" and len(obs) > 1200:
        return False, "atom prompt too long"
    expected = _expected_random_score(item)
    if expected > _LEVELS[item["level"]]["cap"] + 1e-12:
        return False, "random expected score %.6f over cap" % expected
    return True, "ok"


def _initial_observation(item):
    if item["mode"] == "episode":
        return _episode_observation(item)
    return _atom_observation(item)


def _episode_observation(item):
    return (
        "RITE %s\n"
        "FORMAT: ENACT <action>\n"
        "S=%s;G=%s;F=%s\n"
        "RULES: %s\n"
        "KEY: a:x>y?f=1 moves x->y if f=1. a:!f@x?g=0 flips f at x if g=0. "
        "State/flags hidden after this. ACCEPTED applies; REFUSED/MALFORMED do not; "
        "all consume turns."
        % (
            item["_rite"],
            item["_start"],
            item["_goal"],
            _flag_text(item),
            "; ".join(_rule_text(rule) for rule in item["_rules"]),
        )
    )


def _atom_observation(item):
    return (
        "RITE %s\n"
        "S=%s;G=%s;F=%s\n"
        "RULES: %s\n"
        "KEY: a:x>y?f=1 moves x->y if f=1. a:!f@x?g=0 flips f at x if g=0. "
        "State/flags hidden after this. ACCEPTED applies; REFUSED/MALFORMED do not; "
        "all consume turns.\n"
        "Find a shortest legal sequence to reach G. Reply with one final line:\n"
        "ANSWER: <action1> <action2> ... <actionK>"
        % (
            item["_rite"],
            item["_start"],
            item["_goal"],
            _flag_text(item),
            "; ".join(_rule_text(rule) for rule in item["_rules"]),
        )
    )


def _flag_text(item):
    if not item["_flags"]:
        return "-"
    return ",".join("%s=0" % flag for flag in item["_flags"])


def _rule_text(rule):
    if rule["kind"] == "transit":
        text = "%s:%s>%s" % (rule["action"], rule["src"], rule["dst"])
    else:
        text = "%s:!%s" % (rule["action"], rule["flag"])
        if rule["state"] is not None:
            text += "@%s" % rule["state"]
    if rule["guard"] is not None:
        text += "?%s=%s" % (rule["guard"][0], rule["guard"][1])
    return text


def _parse_episode_action(action):
    match = _ACTION_RE.match(str(action))
    if not match:
        return None
    return match.group(1).lower()


def _rule_map(item):
    return dict((rule["action"], rule) for rule in item["_rules"])


def _apply_action(item, state, flags, action):
    rule = _rule_map(item).get(action)
    if rule is None:
        return state, list(flags), False, "unknown"
    return _apply_rule(item, state, flags, rule)


def _apply_rule(item, state, flags, rule):
    flag_values = list(flags)
    if rule["kind"] == "transit":
        if state != rule["src"]:
            return state, flag_values, False, "wrong"
        if not _guard_ok(item, flag_values, rule["guard"]):
            return state, flag_values, False, "condition"
        return rule["dst"], flag_values, True, "accepted"

    if rule["state"] is not None and state != rule["state"]:
        return state, flag_values, False, "wrong"
    if not _guard_ok(item, flag_values, rule["guard"]):
        return state, flag_values, False, "condition"
    index = item["_flags"].index(rule["flag"])
    flag_values[index] = 1 - flag_values[index]
    return state, flag_values, True, "accepted"


def _guard_ok(item, flags, guard):
    if guard is None:
        return True
    index = item["_flags"].index(guard[0])
    return flags[index] == guard[1]


def _shortest_path(item, start_state, start_flags):
    start = (start_state, tuple(start_flags))
    if start_state == item["_goal"]:
        return []
    queue = deque([(start_state, tuple(start_flags), [])])
    seen = {start}
    while queue:
        state, flags, path = queue.popleft()
        for action in item["_actions"]:
            next_state, next_flags, accepted, _reason = _apply_action(item, state, list(flags), action)
            if not accepted:
                continue
            next_key = (next_state, tuple(next_flags))
            if next_key in seen:
                continue
            next_path = path + [action]
            if next_state == item["_goal"]:
                return next_path
            seen.add(next_key)
            queue.append((next_state, tuple(next_flags), next_path))
    return None


def _score_episode(item, transcript):
    state = item["_start"]
    flags = list(item["_initial_flags"])
    turns = 0
    for turn in transcript:
        if turns >= item["max_turns"]:
            break
        turns += 1
        parsed = _parse_episode_action(turn.get("action", ""))
        if parsed is not None:
            next_state, next_flags, accepted, _reason = _apply_action(item, state, flags, parsed)
            if accepted:
                state = next_state
                flags = next_flags
                if state == item["_goal"]:
                    value = _clamp(float(item["_optimal_len"]) / float(turns))
                    return {
                        "score": value,
                        "turns_used": turns,
                        "optimal_len": item["_optimal_len"],
                        "completed": True,
                    }
    return {
        "score": 0.0,
        "turns_used": turns,
        "optimal_len": item["_optimal_len"],
        "completed": False,
    }


def _score_atom(item, answer_text):
    names = str(answer_text).strip().split()
    if not names:
        return {
            "score": 0.0,
            "turns_used": 0,
            "optimal_len": item["_optimal_len"],
            "completed": False,
        }
    state = item["_start"]
    flags = list(item["_initial_flags"])
    for raw_name in names:
        name = raw_name.lower()
        if name not in item["_actions"]:
            return {
                "score": 0.0,
                "turns_used": len(names),
                "optimal_len": item["_optimal_len"],
                "completed": False,
            }
        next_state, next_flags, accepted, _reason = _apply_action(item, state, flags, name)
        if not accepted:
            return {
                "score": 0.0,
                "turns_used": len(names),
                "optimal_len": item["_optimal_len"],
                "completed": False,
            }
        state = next_state
        flags = next_flags
    completed = state == item["_goal"]
    value = _clamp(float(item["_optimal_len"]) / float(len(names))) if completed else 0.0
    return {
        "score": value,
        "turns_used": len(names),
        "optimal_len": item["_optimal_len"],
        "completed": completed,
    }


def _expected_random_score(item):
    if item["mode"] == "atom":
        return _expected_random_atom(item)
    return _expected_random_episode(item)


def _expected_random_episode(item):
    memo = {}
    action_count = float(len(item["_actions"]))

    def value(state, flags, turns):
        key = (state, tuple(flags), turns)
        if key in memo:
            return memo[key]
        if turns >= item["max_turns"]:
            return 0.0
        total = 0.0
        for action in item["_actions"]:
            next_state, next_flags, accepted, _reason = _apply_action(item, state, list(flags), action)
            if accepted and next_state == item["_goal"]:
                total += float(item["_optimal_len"]) / float(turns + 1)
            elif turns + 1 < item["max_turns"]:
                total += value(next_state, tuple(next_flags), turns + 1)
        memo[key] = total / action_count
        return memo[key]

    return value(item["_start"], tuple(item["_initial_flags"]), 0)


def _expected_random_atom(item):
    action_count = float(len(item["_actions"]))
    expected = 0.0
    max_len = item["_optimal_len"] + 2
    for length in range(1, max_len + 1):
        states = {(item["_start"], tuple(item["_initial_flags"])): 1.0}
        for _step in range(length):
            next_states = {}
            for key, prob in states.items():
                state, flags = key
                for action in item["_actions"]:
                    next_state, next_flags, accepted, _reason = _apply_action(
                        item, state, list(flags), action
                    )
                    if not accepted:
                        continue
                    next_key = (next_state, tuple(next_flags))
                    next_states[next_key] = next_states.get(next_key, 0.0) + prob / action_count
            states = next_states
            if not states:
                break
        goal_prob = 0.0
        for key, prob in states.items():
            if key[0] == item["_goal"]:
                goal_prob += prob
        expected += (1.0 / float(max_len)) * goal_prob * (
            float(item["_optimal_len"]) / float(length)
        )
    return expected


def _clamp(value):
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
