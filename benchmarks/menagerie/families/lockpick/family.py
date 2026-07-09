import hashlib
import random
import re


META = {
    "name": "lockpick",
    "capability": "active rule induction: probe a hidden glyph machine with chosen experiments, then invert the induced rule to construct the opening input",
    "paradigm": "multi-turn",
    "action_format": "episodes: one line 'PROBE g1 g2 ... gL' (spend a probe) or 'OPEN g1 g2 ... gL' (one terminal attempt); atoms: final line 'ANSWER: g1 g2 ... gL'",
}


_CONSONANTS = ("z", "v", "k", "q", "x", "j", "r", "n", "m", "th", "sh", "zh")
_VOWELS = ("a", "e", "i", "o", "u", "y")
_BLOCKLIST = {
    "answer",
    "code",
    "door",
    "fail",
    "glyph",
    "key",
    "lock",
    "machine",
    "open",
    "pass",
    "probe",
    "target",
    "test",
}
_LEVELS = {
    1: {
        "alphabet_size": 6,
        "seq_len": 3,
        "depth": 1,
        "max_turns": 4,
        "probe_budget": 2,
        "atom_pairs": 2,
        "oracle_probes": 2,
    },
    2: {
        "alphabet_size": 6,
        "seq_len": 3,
        "depth": 2,
        "max_turns": 4,
        "probe_budget": 3,
        "atom_pairs": 3,
        "oracle_probes": 3,
    },
    3: {
        "alphabet_size": 8,
        "seq_len": 4,
        "depth": 3,
        "max_turns": 10,
        "probe_budget": 7,
        "atom_pairs": 4,
        "oracle_probes": 6,
    },
    4: {
        "alphabet_size": 8,
        "seq_len": 5,
        "depth": 4,
        "max_turns": 14,
        "probe_budget": 11,
        "atom_pairs": 5,
        "oracle_probes": 10,
    },
}
_MODES = {"atom", "episode"}
_MAX_ATTEMPTS = 512
_MAX_SEED_PERTURBATIONS = 8
_ACTION_RE = re.compile(r"^\s*(PROBE|OPEN)\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$", re.IGNORECASE)


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if level not in _LEVELS:
        raise ValueError("level must be one of 1, 2, 3, 4")
    if mode not in _MODES:
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")
    return [_make_item(seed, level, mode, idx) for idx in range(n)]


class Env:
    def __init__(self, item):
        self.item = item
        self.done = False
        self.probes_left = item["probe_budget"]

    def reset(self) -> str:
        self.done = False
        self.probes_left = self.item["probe_budget"]
        if self.item["mode"] == "atom":
            return _atom_prompt(self.item)
        return _episode_prompt(self.item, self.probes_left)

    def step(self, action) -> tuple[str, bool]:
        if self.item["mode"] == "atom":
            self.done = True
            return "", True
        if self.done:
            return "Episode over.", True

        parsed = _parse_episode_action(action, self.item)
        if parsed["error"]:
            return _corrective(parsed, self.item), False

        seq = parsed["seq"]
        if parsed["verb"] == "PROBE":
            if self.probes_left <= 0:
                return (
                    "No probes left. Make your final attempt.\n"
                    + _grammar_reminder(self.item)
                ), False
            self.probes_left -= 1
            out = _apply_rule(seq, self.item["rule_spec"], self.item["alphabet"])
            obs = (
                f"PROBE {_seq_text(seq)} -> {_seq_text(out)}. "
                f"Probes left: {self.probes_left}. Target: {_seq_text(self.item['target'])}."
            )
            return obs + "\n" + _grammar_reminder(self.item), False

        self.done = True
        out = _apply_rule(seq, self.item["rule_spec"], self.item["alphabet"])
        if out == self.item["target"]:
            return "The mechanism clicked open.", True
        return "The mechanism jammed.", True


def score(item: dict, transcript: list[dict]) -> dict:
    if item["mode"] == "atom":
        return _score_atom(item, transcript)
    return _score_episode(item, transcript)


def oracle_policy(item: dict, history: list[dict]) -> str:
    if item["mode"] == "atom":
        return f"ANSWER: {_seq_text(item['solution'])}"
    probes = _informative_sequences(item["alphabet"], item["seq_len"], _LEVELS[item["level"]]["oracle_probes"])
    limit = _LEVELS[item["level"]]["oracle_probes"]
    if len(history) < limit:
        return f"PROBE {_seq_text(probes[len(history) % len(probes)])}"
    return f"OPEN {_seq_text(item['solution'])}"


def random_policy(item: dict, history: list[dict], rng) -> str:
    seq = [rng.choice(item["alphabet"]) for _ in range(item["seq_len"])]
    if item["mode"] == "atom":
        return f"ANSWER: {_seq_text(seq)}"
    verb = "PROBE" if rng.random() < 0.5 else "OPEN"
    return f"{verb} {_seq_text(seq)}"


def _make_item(seed, level, mode, idx):
    cfg = _LEVELS[level]
    for perturb in range(_MAX_SEED_PERTURBATIONS):
        for attempt in range(_MAX_ATTEMPTS):
            rng = _rng_for(seed, level, mode, idx, attempt, perturb)
            alphabet = _make_alphabet(rng, cfg["alphabet_size"])
            rule = _make_rule(rng, level, alphabet, cfg["seq_len"])
            solution = _random_sequence(rng, alphabet, cfg["seq_len"])
            target = _apply_rule(solution, rule, alphabet)
            atom_pairs = _atom_probe_pairs(alphabet, cfg["seq_len"], rule, cfg["atom_pairs"])
            if mode == "atom" and level in (1, 2):
                solution = _determinate_atom_solution(level, alphabet, cfg["seq_len"], atom_pairs, target)
                if solution is None:
                    continue
            if _passes_generation_guards(mode, solution, target, rule, alphabet, atom_pairs):
                return {
                    "id": f"lockpick-{seed}-L{level}-{mode}-{idx}",
                    "level": level,
                    "mode": mode,
                    "max_turns": 1 if mode == "atom" else cfg["max_turns"],
                    "alphabet": alphabet,
                    "seq_len": cfg["seq_len"],
                    "rule_spec": rule,
                    "target": target,
                    "solution": solution,
                    "probe_budget": cfg["probe_budget"],
                    "atom_probe_pairs": atom_pairs,
                }
    raise RuntimeError("could not generate a non-degenerate lockpick item")


def _passes_generation_guards(mode, solution, target, rule, alphabet, atom_pairs):
    if _apply_rule(solution, rule, alphabet) != target:
        return False
    if _seq_text(solution) in _seq_text(alphabet):
        return False
    if solution == target:
        return False
    if _apply_rule(target, rule, alphabet) == target:
        return False
    if mode == "atom":
        for pair in atom_pairs:
            if pair["input"] == solution or pair["output"] == target or pair["output"] == solution:
                return False
    return True


def _rng_for(seed, level, mode, idx, attempt, perturb=0):
    text = f"lockpick|{seed}|{level}|{mode}|{idx}|{attempt}|{perturb}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _make_alphabet(rng, size):
    alphabet = []
    seen = set()
    for _ in range(10000):
        token = _syllable(rng) + _syllable(rng)
        if token not in seen and token not in _BLOCKLIST:
            alphabet.append(token)
            seen.add(token)
            if len(alphabet) == size:
                return alphabet
    raise RuntimeError("could not generate a distinct alphabet")


def _syllable(rng):
    return rng.choice(_CONSONANTS) + rng.choice(_VOWELS) + rng.choice(_CONSONANTS)


def _random_sequence(rng, alphabet, length):
    return [rng.choice(alphabet) for _ in range(length)]


def _make_rule(rng, level, alphabet, length):
    depth = _LEVELS[level]["depth"]
    conditional_positions = set()
    if level == 3 and rng.random() < 0.5:
        conditional_positions.add(rng.randrange(depth))
    elif level == 4:
        conditional_positions.add(rng.randrange(depth))

    rule = []
    for pos in range(depth):
        if pos in conditional_positions:
            rule.append(_sample_conditional(rng, alphabet, length))
        else:
            rule.append(_sample_nonconditional(rng, alphabet, length))
    return rule


def _sample_nonconditional(rng, alphabet, length):
    op = rng.choice(("ROTATE", "REVERSE", "SWAP", "SHIFT"))
    if op == "ROTATE":
        return {"op": "ROTATE", "k": rng.randrange(1, length)}
    if op == "REVERSE":
        return {"op": "REVERSE"}
    if op == "SWAP":
        i = rng.randrange(length)
        j = rng.randrange(length - 1)
        if j >= i:
            j += 1
        return {"op": "SWAP", "i": i, "j": j}
    scope = rng.choice(("all", "pos"))
    spec = {"op": "SHIFT", "scope": scope, "k": rng.randrange(1, len(alphabet))}
    if scope == "pos":
        spec["pos"] = rng.randrange(length)
    return spec


def _sample_conditional(rng, alphabet, length):
    pred_kind = rng.choice(("contains", "at"))
    if pred_kind == "contains":
        predicate = {"kind": "contains", "glyph": rng.choice(alphabet)}
    else:
        predicate = {"kind": "at", "pos": rng.randrange(length), "glyph": rng.choice(alphabet)}
    then = _sample_nonconditional(rng, alphabet, length)
    otherwise = _sample_nonconditional(rng, alphabet, length)
    for _ in range(12):
        if otherwise != then:
            break
        otherwise = _sample_nonconditional(rng, alphabet, length)
    return {"op": "CONDITIONAL", "predicate": predicate, "then": then, "else": otherwise}


def _apply_rule(seq, rule, alphabet):
    out = list(seq)
    for primitive in rule:
        out = _apply_primitive(out, primitive, alphabet)
    return out


def _apply_primitive(seq, primitive, alphabet):
    op = primitive["op"]
    if op == "ROTATE":
        k = primitive["k"] % len(seq)
        return list(seq[-k:] + seq[:-k])
    if op == "REVERSE":
        return list(reversed(seq))
    if op == "SWAP":
        out = list(seq)
        i = primitive["i"]
        j = primitive["j"]
        out[i], out[j] = out[j], out[i]
        return out
    if op == "SHIFT":
        mapping = _shift_mapping(alphabet, primitive["k"])
        out = list(seq)
        if primitive["scope"] == "all":
            return [mapping[g] for g in out]
        pos = primitive["pos"]
        out[pos] = mapping[out[pos]]
        return out
    if op == "CONDITIONAL":
        branch = primitive["then"] if _predicate_holds(seq, primitive["predicate"]) else primitive["else"]
        return _apply_primitive(seq, branch, alphabet)
    raise ValueError("unknown primitive")


def _shift_mapping(alphabet, k):
    size = len(alphabet)
    return {glyph: alphabet[(idx + k) % size] for idx, glyph in enumerate(alphabet)}


def _apply_inverse_rule(seq, rule, alphabet):
    out = list(seq)
    for primitive in reversed(rule):
        out = _apply_inverse_primitive(out, primitive, alphabet)
    return out


def _apply_inverse_primitive(seq, primitive, alphabet):
    op = primitive["op"]
    if op == "ROTATE":
        k = primitive["k"] % len(seq)
        return list(seq[k:] + seq[:k])
    if op == "REVERSE":
        return list(reversed(seq))
    if op == "SWAP":
        out = list(seq)
        i = primitive["i"]
        j = primitive["j"]
        out[i], out[j] = out[j], out[i]
        return out
    if op == "SHIFT":
        mapping = _shift_mapping(alphabet, -primitive["k"])
        out = list(seq)
        if primitive["scope"] == "all":
            return [mapping[g] for g in out]
        pos = primitive["pos"]
        out[pos] = mapping[out[pos]]
        return out
    raise ValueError("inverse only supports non-conditional primitives")


def _determinate_atom_solution(level, alphabet, length, pairs, target):
    consistent = []
    for rule in _enumerate_rule_space(level, alphabet, length):
        if all(_apply_rule(pair["input"], rule, alphabet) == pair["output"] for pair in pairs):
            consistent.append(rule)
    candidates = []
    seen = set()
    for rule in consistent:
        candidate = _apply_inverse_rule(target, rule, alphabet)
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    for candidate in candidates:
        if all(_apply_rule(candidate, rule, alphabet) == target for rule in consistent):
            return candidate
    return None


def _enumerate_rule_space(level, alphabet, length):
    depth = _LEVELS[level]["depth"]
    primitives = _enumerate_nonconditional_primitives(len(alphabet), length)
    rules = [[]]
    for _ in range(depth):
        rules = [rule + [primitive] for rule in rules for primitive in primitives]
    return rules


def _enumerate_nonconditional_primitives(alphabet_size, length):
    primitives = []
    for k in range(1, length):
        primitives.append({"op": "ROTATE", "k": k})
    primitives.append({"op": "REVERSE"})
    for i in range(length):
        for j in range(length):
            if i != j:
                primitives.append({"op": "SWAP", "i": i, "j": j})
    for k in range(1, alphabet_size):
        primitives.append({"op": "SHIFT", "scope": "all", "k": k})
        for pos in range(length):
            primitives.append({"op": "SHIFT", "scope": "pos", "k": k, "pos": pos})
    return primitives


def _predicate_holds(seq, predicate):
    if predicate["kind"] == "contains":
        return predicate["glyph"] in seq
    return seq[predicate["pos"]] == predicate["glyph"]


def _atom_probe_pairs(alphabet, length, rule, count):
    pairs = []
    for seq in _informative_sequences(alphabet, length, count)[:count]:
        pairs.append({"input": seq, "output": _apply_rule(seq, rule, alphabet)})
    return pairs


def _informative_sequences(alphabet, length, minimum):
    seqs = []
    seen = set()

    def add(seq):
        key = tuple(seq)
        if len(seq) == length and key not in seen:
            seen.add(key)
            seqs.append(list(seq))

    add([alphabet[0]] * length)
    add(alphabet[:length])
    add(list(reversed(alphabet[:length])))
    for shift in range(1, len(alphabet)):
        add([alphabet[(shift + i) % len(alphabet)] for i in range(length)])

    stride = 2
    shift = 0
    guard = 0
    while len(seqs) < minimum and guard < 1000:
        add([alphabet[(shift + i * stride) % len(alphabet)] for i in range(length)])
        shift += 1
        if shift >= len(alphabet):
            shift = 0
            stride += 1
        guard += 1
    return seqs


def _episode_prompt(item, probes_left):
    body = (
        "Fictional glyph lock: infer the hidden machine by probing, then open it. "
        f"Alphabet: {_seq_text(item['alphabet'])}. Sequence length: {item['seq_len']}. "
        "Machine: a fixed composition of hidden moves - position rotations, reversals, "
        "position swaps, and glyph shifts along the alphabet as listed (wrapping); "
        "some moves may be conditional on the input. "
        f"Hidden moves: {_LEVELS[item['level']]['depth']}. "
        f"Target output: {_seq_text(item['target'])}. Probes remaining: {probes_left}. "
        f"Max turns: {item['max_turns']}."
    )
    return body + "\n" + _grammar_reminder(item)


def _atom_prompt(item):
    pairs = "; ".join(
        f"{_seq_text(pair['input'])} -> {_seq_text(pair['output'])}" for pair in item["atom_probe_pairs"]
    )
    placeholder = _placeholder(item["seq_len"])
    body = (
        "Fictional glyph lock: infer the hidden machine from these probe pairs. "
        f"Alphabet: {_seq_text(item['alphabet'])}. Sequence length: {item['seq_len']}. "
        "Machine: a fixed composition of hidden moves - position rotations, reversals, "
        "position swaps, and glyph shifts along the alphabet as listed (wrapping); "
        "some moves may be conditional on the input. "
        f"Hidden moves: {_LEVELS[item['level']]['depth']}. "
        f"Probe pairs: {pairs}. Target output: {_seq_text(item['target'])}."
    )
    return body + "\n" + f"End with a final line in the form ANSWER: {placeholder}."


def _parse_episode_action(action, item):
    text = "" if action is None else str(action)
    candidate = None
    for line in text.splitlines():
        if _ACTION_RE.match(line):
            candidate = line.strip()
    if candidate is None:
        return {"error": "no_action", "verb": None, "seq": None}

    parts = candidate.split()
    verb = parts[0].upper()
    tokens = parts[1:]
    if len(tokens) != item["seq_len"]:
        return {"error": "wrong_length", "verb": verb, "seq": None, "got": len(tokens)}

    glyphs = {glyph.lower(): glyph for glyph in item["alphabet"]}
    seq = []
    for token in tokens:
        glyph = glyphs.get(token.lower())
        if glyph is None:
            return {"error": "unknown_glyph", "verb": verb, "seq": None, "token": token}
        seq.append(glyph)
    return {"error": None, "verb": verb, "seq": seq}


def _corrective(parsed, item):
    reminder = _grammar_reminder(item)
    if parsed["error"] == "wrong_length":
        return f"Wrong length: use {item['seq_len']} glyphs.\n{reminder}"
    if parsed["error"] == "unknown_glyph":
        token = parsed["token"]
        if len(token) > 24:
            token = token[:24] + "..."
        return f"Unknown glyph: {token}. Use alphabet glyphs.\n{reminder}"
    return f"Use one line.\n{reminder}"


def _score_episode(item, transcript):
    last_open = None
    probes_used = 0
    for entry in transcript:
        parsed = _parse_episode_action(entry.get("action", ""), item)
        if parsed["error"]:
            continue
        if parsed["verb"] == "PROBE":
            probes_used = min(item["probe_budget"], probes_used + 1)
        elif parsed["verb"] == "OPEN":
            last_open = parsed["seq"]
    if last_open is None:
        return {
            "score": 0.0,
            "opened": False,
            "reason": "no_open",
            "probes_used": probes_used,
            "submitted": None,
        }

    opened = _apply_rule(last_open, item["rule_spec"], item["alphabet"]) == item["target"]
    return {
        "score": 1.0 if opened else 0.0,
        "opened": opened,
        "reason": "opened" if opened else "wrong_open",
        "probes_used": probes_used,
        "submitted": last_open,
    }


def _score_atom(item, transcript):
    if not transcript:
        return {"score": 0.0, "opened": False, "reason": "no_answer", "submitted": None}
    seq, reason = _parse_answer(transcript[-1].get("action", ""), item)
    if seq is None:
        return {"score": 0.0, "opened": False, "reason": reason, "submitted": None}
    opened = _apply_rule(seq, item["rule_spec"], item["alphabet"]) == item["target"]
    return {
        "score": 1.0 if opened else 0.0,
        "opened": opened,
        "reason": "opened" if opened else "wrong_answer",
        "submitted": seq,
    }


def _parse_answer(action, item):
    text = "" if action is None else str(action)
    answer = None
    for line in text.splitlines():
        match = _ANSWER_RE.match(line)
        if match:
            answer = match.group(1)
    if answer is None:
        seq = _bare_glyph_sequence(text, item)
        if seq is not None:
            return seq, None
        return None, "no_answer"
    tokens = answer.split()
    if len(tokens) != item["seq_len"]:
        return None, "wrong_length"
    glyphs = {glyph.lower(): glyph for glyph in item["alphabet"]}
    seq = []
    for token in tokens:
        glyph = glyphs.get(token.lower())
        if glyph is None:
            return None, "unknown_glyph"
        seq.append(glyph)
    return seq, None


def _bare_glyph_sequence(text, item):
    glyphs = {glyph.lower(): glyph for glyph in item["alphabet"]}
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) != item["seq_len"]:
            return None
        seq = []
        for token in tokens:
            glyph = glyphs.get(token.lower())
            if glyph is None:
                return None
            seq.append(glyph)
        return seq
    return None


def _grammar_reminder(item):
    placeholder = _placeholder(item["seq_len"])
    return f"Grammar: reply ONE line, PROBE {placeholder} or OPEN {placeholder}; alphabet glyphs only."


def _placeholder(length):
    return " ".join(f"<g{i + 1}>" for i in range(length))


def _seq_text(seq):
    return " ".join(seq)
