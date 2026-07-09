import random
import re


META = {
    "name": "menders",
    "capability": "Debug and repair fictional mini-language programs from failing checks.",
    "paradigm": "multi-turn",
    "action_format": "episode: MEND <line#>: <replacement source line>; atom final line: ANSWER: <line#>: <replacement source line>",
}

__all__ = [
    "META",
    "generate",
    "Env",
    "score",
    "oracle_policy",
    "random_policy",
]


_LEVEL_BUGS = {1: 1, 2: 1, 3: 2, 4: 3}
_LEVEL_TURNS = {1: 3, 2: 3, 3: 6, 4: 7}
_GRID = list(range(-4, 7))
_OPS = ("set", "add", "sub", "mul", "emit", "rep", "done")
_ARITH = ("add", "sub", "mul")
_VOWELS = "aeiou"
_CONS = "bcdfghjklmnprstvyw"
_HARD = "qxz"
_DENYLIST = {
    "about",
    "after",
    "again",
    "answer",
    "array",
    "break",
    "class",
    "const",
    "debug",
    "done",
    "emit",
    "false",
    "float",
    "input",
    "int",
    "mend",
    "print",
    "raise",
    "range",
    "return",
    "set",
    "stack",
    "true",
    "value",
    "while",
}

_MEND_RE = re.compile(r"^\s*mend\s+([1-9][0-9]*)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*answer\s*:\s*([1-9][0-9]*)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_BAD_MEND = "Bad action. Use: MEND <line#>: <replacement source line>"
_BAD_ANSWER = "Bad action. Use: ANSWER: <line#>: <replacement source line>"


def _stable_seed(*parts):
    data = "|".join(str(part) for part in parts).encode("utf-8")
    h = 2166136261
    for byte in data:
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _seed_prefix(seed):
    value = abs(int(seed))
    c1 = _HARD[value % len(_HARD)]
    v1 = _VOWELS[(value // 3) % len(_VOWELS)]
    c2 = _CONS[(value // 15) % len(_CONS)]
    v2 = _VOWELS[(value // (15 * len(_CONS))) % len(_VOWELS)]
    return c1 + v1 + c2 + v2


def _token_stock(seed, rng, count):
    prefix = _seed_prefix(seed)
    out = []
    used = set()
    attempts = 0
    while len(out) < count:
        attempts += 1
        if attempts > 10000:
            raise RuntimeError("token generation exhausted")
        token = prefix + rng.choice(_CONS) + rng.choice(_VOWELS)
        if rng.randrange(3) == 0:
            token += rng.choice(_CONS)
        if (
            5 <= len(token) <= 7
            and any(ch in token for ch in _HARD)
            and token not in used
            and token not in _DENYLIST
        ):
            used.add(token)
            out.append(token)
    return out


def _make_vocab(seed, rng):
    stock = _token_stock(seed, rng, 17)
    return dict(zip(_OPS, stock[: len(_OPS)])), stock[len(_OPS) :]


def _line(tokens, op, *args):
    return " ".join([tokens[op]] + [str(arg) for arg in args])


def _canon(text):
    return " ".join(str(text).strip().lower().split())


def _is_int(text):
    if not text:
        return False
    if text[0] == "-":
        return len(text) > 1 and text[1:].isdigit()
    return text.isdigit()


def _operand_ok(text, variables, lo=-20, hi=20):
    if text in variables:
        return True
    return _is_int(text) and lo <= int(text) <= hi


def _parse_line(source, tokens, variables):
    parts = _canon(source).split()
    if not parts:
        return None
    reverse = {word: op for op, word in tokens.items()}
    op = reverse.get(parts[0])
    if op is None:
        return None
    args = parts[1:]
    if op == "set":
        if len(args) != 2 or args[0] not in variables or not _operand_ok(args[1], variables):
            return None
    elif op in _ARITH:
        if (
            len(args) != 3
            or args[0] not in variables
            or not _operand_ok(args[1], variables)
            or not _operand_ok(args[2], variables)
        ):
            return None
    elif op == "emit":
        if len(args) != 1 or not _operand_ok(args[0], variables):
            return None
    elif op == "rep":
        if len(args) != 1:
            return None
        arg = args[0]
        if arg in variables:
            pass
        elif not (_is_int(arg) and 0 <= int(arg) <= 6):
            return None
    elif op == "done":
        if args:
            return None
    else:
        return None
    return {"op": op, "args": args, "source": " ".join(parts)}


def _compile_program(program, tokens, variables):
    parsed = []
    for source in program:
        line = _parse_line(source, tokens, variables)
        if line is None:
            return None
        parsed.append(line)
    stack = []
    match = {}
    for idx, line in enumerate(parsed):
        op = line["op"]
        if op == "rep":
            if stack:
                return None
            stack.append(idx)
        elif op == "done":
            if not stack:
                return None
            start = stack.pop()
            match[start] = idx
            match[idx] = start
    if stack:
        return None
    return parsed, match


def _eval_arg(arg, state):
    if _is_int(arg):
        return int(arg)
    return state[arg]


def _execute(program, item, inputs):
    compiled = _compile_program(program, item["tokens"], item["variables"])
    if compiled is None:
        return "ERR"
    parsed, match = compiled
    state = {name: 0 for name in item["variables"]}
    for name in item["inputs"]:
        state[name] = int(inputs[name])
    out = []
    loops = []
    pc = 0
    steps = 0
    while pc < len(parsed):
        steps += 1
        if steps > 160:
            return "ERR"
        line = parsed[pc]
        op = line["op"]
        args = line["args"]
        if op == "set":
            state[args[0]] = _eval_arg(args[1], state)
            if abs(state[args[0]]) > 9999:
                return "ERR"
            pc += 1
        elif op == "add":
            state[args[0]] = _eval_arg(args[1], state) + _eval_arg(args[2], state)
            if abs(state[args[0]]) > 9999:
                return "ERR"
            pc += 1
        elif op == "sub":
            state[args[0]] = _eval_arg(args[1], state) - _eval_arg(args[2], state)
            if abs(state[args[0]]) > 9999:
                return "ERR"
            pc += 1
        elif op == "mul":
            state[args[0]] = _eval_arg(args[1], state) * _eval_arg(args[2], state)
            if abs(state[args[0]]) > 9999:
                return "ERR"
            pc += 1
        elif op == "emit":
            value = _eval_arg(args[0], state)
            if abs(value) > 9999:
                return "ERR"
            out.append(value)
            if len(out) > 12:
                return "ERR"
            pc += 1
        elif op == "rep":
            count = _eval_arg(args[0], state)
            if not 0 <= count <= 6:
                return "ERR"
            if count == 0:
                pc = match[pc] + 1
            else:
                loops.append([pc + 1, match[pc], count])
                pc += 1
        elif op == "done":
            if not loops or loops[-1][1] != pc:
                return "ERR"
            loops[-1][2] -= 1
            if loops[-1][2] > 0:
                pc = loops[-1][0]
            else:
                loops.pop()
                pc += 1
        else:
            return "ERR"
    return out


def _grid_points(inputs):
    if len(inputs) == 1:
        return [{inputs[0]: value} for value in _GRID]
    return [{inputs[0]: a, inputs[1]: b} for a in _GRID for b in _GRID]


def _build_correct(level, rng, tokens, variables):
    x = variables[0]
    y = variables[1]
    a, b, c, d, e, f, g = variables[2:9]
    if level == 1:
        c1 = rng.choice([-3, -2, -1, 1, 2, 3])
        c2 = rng.choice([2, 3, 4])
        c3 = rng.choice([-3, -2, -1, 1, 2, 3])
        return [x], [
            _line(tokens, "set", a, c1),
            _line(tokens, "add", b, x, a),
            _line(tokens, "mul", c, b, c2),
            _line(tokens, "sub", d, c, c3),
            _line(tokens, "emit", d),
        ]
    if level == 2:
        c1 = rng.choice([-4, -2, -1, 1, 3, 4])
        c2 = rng.choice([2, 3])
        return [x, y], [
            _line(tokens, "set", a, c1),
            _line(tokens, "add", b, x, y),
            _line(tokens, "mul", c, b, c2),
            _line(tokens, "sub", d, c, a),
            _line(tokens, "add", e, d, x),
            _line(tokens, "emit", e),
            _line(tokens, "emit", d),
        ]
    if level == 3:
        c1 = rng.choice([-2, -1, 1, 2])
        reps = rng.choice([2, 3, 4])
        c2 = rng.choice([2, 3])
        return [x, y], [
            _line(tokens, "set", a, c1),
            _line(tokens, "add", b, x, y),
            _line(tokens, "rep", reps),
            _line(tokens, "add", a, a, b),
            _line(tokens, "sub", c, a, y),
            _line(tokens, "emit", c),
            _line(tokens, "done"),
            _line(tokens, "mul", d, a, c2),
            _line(tokens, "emit", d),
        ]
    c1 = rng.choice([-2, -1, 1, 2])
    reps = rng.choice([2, 3, 4])
    m1 = rng.choice([2, 3])
    m2 = rng.choice([2, 3])
    return [x, y], [
        _line(tokens, "set", a, c1),
        _line(tokens, "add", b, x, y),
        _line(tokens, "sub", c, x, y),
        _line(tokens, "rep", reps),
        _line(tokens, "add", a, a, b),
        _line(tokens, "mul", d, c, m1),
        _line(tokens, "add", e, d, a),
        _line(tokens, "emit", e),
        _line(tokens, "done"),
        _line(tokens, "sub", f, a, c),
        _line(tokens, "mul", g, f, m2),
        _line(tokens, "emit", g),
    ]


def _source_operand_positions(op):
    if op == "set":
        return [1]
    if op in _ARITH:
        return [1, 2]
    if op == "emit":
        return [0]
    return []


def _line_mutations(source, tokens, variables, kinds):
    parsed = _parse_line(source, tokens, variables)
    if parsed is None:
        return []
    op = parsed["op"]
    args = parsed["args"]
    out = []
    if "const" in kinds and op != "rep":
        for pos, arg in enumerate(args):
            if _is_int(arg):
                old = int(arg)
                for delta in (1, -1, 2, -2, 3, -3):
                    new = old + delta
                    if -9 <= new <= 9 and new != old:
                        changed = args[:]
                        changed[pos] = str(new)
                        out.append(_line(tokens, op, *changed))
    if "repeat" in kinds and op == "rep" and args and _is_int(args[0]):
        old = int(args[0])
        for delta in (1, -1, 2, -2):
            new = old + delta
            if 0 <= new <= 6 and new != old:
                out.append(_line(tokens, op, new))
    if "var" in kinds:
        for pos in _source_operand_positions(op):
            if pos < len(args) and args[pos] in variables:
                for replacement in variables:
                    if replacement != args[pos]:
                        changed = args[:]
                        changed[pos] = replacement
                        out.append(_line(tokens, op, *changed))
    if "opcode" in kinds and op in _ARITH:
        for replacement in _ARITH:
            if replacement != op:
                out.append(_line(tokens, replacement, *args))
    seen = set()
    unique = []
    for candidate in out:
        if candidate != source and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _plant_bugs(correct, tokens, variables, level, bug_count, rng):
    if level == 1:
        kinds = ("const",)
    elif level == 2:
        kinds = ("var", "opcode")
    else:
        kinds = ("const", "var", "opcode", "repeat")
    program = correct[:]
    changed = []

    if level == 4 and bug_count >= 3:
        repeat_options = []
        for idx, source in enumerate(correct):
            for candidate in _line_mutations(source, tokens, variables, ("repeat",)):
                repeat_options.append((idx, candidate))
        if not repeat_options:
            return None, None
        idx, candidate = rng.choice(repeat_options)
        program[idx] = candidate
        changed.append(idx)

    tries = 0
    while len(changed) < bug_count:
        tries += 1
        if tries > 200:
            return None, None
        choices = []
        for idx, source in enumerate(correct):
            if idx in changed:
                continue
            for candidate in _line_mutations(source, tokens, variables, kinds):
                choices.append((idx, candidate))
        if not choices:
            return None, None
        idx, candidate = rng.choice(choices)
        program[idx] = candidate
        changed.append(idx)
    return program, changed


def _program_ok_on_grid(item, program):
    for point in _grid_points(item["inputs"]):
        output = _execute(program, item, point)
        if output == "ERR" or not output:
            return False
    return True


def _choose_checks(rng, inputs):
    points = _grid_points(inputs)
    return rng.sample(points, 4)


def _make_item(seed, level, mode, index):
    bug_count = 1 if mode == "atom" else _LEVEL_BUGS[level]
    max_turns = 1 if mode == "atom" else _LEVEL_TURNS[level]
    for attempt in range(400):
        rng = random.Random(_stable_seed(seed, level, mode, index, attempt))
        tokens, variables = _make_vocab(seed, rng)
        inputs, correct = _build_correct(level, rng, tokens, variables)
        base = {
            "tokens": tokens,
            "variables": variables,
            "inputs": inputs,
        }
        if not _program_ok_on_grid(base, correct):
            continue
        points = _choose_checks(rng, inputs)
        visible_input = points[0]
        hidden_inputs = points[1:]
        visible_expected = _execute(correct, base, visible_input)
        hidden = [
            {"inputs": point, "expected": _execute(correct, base, point)}
            for point in hidden_inputs
        ]
        if visible_expected == "ERR" or any(check["expected"] == "ERR" for check in hidden):
            continue
        buggy, bug_lines = _plant_bugs(correct, tokens, variables, level, bug_count, rng)
        if buggy is None:
            continue
        visible_actual = _execute(buggy, base, visible_input)
        if visible_actual == visible_expected:
            continue
        hidden_actual = [_execute(buggy, base, check["inputs"]) for check in hidden]
        if any(actual == check["expected"] for actual, check in zip(hidden_actual, hidden)):
            continue
        item = {
            "id": "menders-%s-%s-%s-%s" % (seed, level, mode, index),
            "level": level,
            "mode": mode,
            "max_turns": max_turns,
            "tokens": tokens,
            "variables": variables,
            "inputs": inputs,
            "program": buggy,
            "_correct_program": correct,
            "_visible": {
                "inputs": visible_input,
                "expected": visible_expected,
                "actual": visible_actual,
            },
            "_hidden": hidden,
            "_planted_bugs": bug_count,
            "_bug_lines": [idx + 1 for idx in bug_lines],
        }
        prompt = _render(item, buggy, atom=(mode == "atom"))
        if mode == "atom":
            if len(prompt) <= 1200:
                return item
        elif len(prompt) <= 800:
            return item
    raise RuntimeError("could not generate menders item")


def generate(seed, level, n, mode):
    if level not in (1, 2, 3, 4):
        raise ValueError("level must be 1, 2, 3, or 4")
    if mode not in ("atom", "episode"):
        raise ValueError("mode must be 'atom' or 'episode'")
    if n < 0:
        raise ValueError("n must be non-negative")
    return [_make_item(seed, level, mode, index) for index in range(n)]


def _fmt_seq(seq):
    if seq == "ERR":
        return "ERR"
    return "[" + ",".join(str(value) for value in seq) + "]"


def _fmt_inputs(inputs, order):
    return ",".join("%s=%s" % (name, inputs[name]) for name in order)


def _hidden_passing(item, program):
    passing = 0
    for check in item["_hidden"]:
        if _execute(program, item, check["inputs"]) == check["expected"]:
            passing += 1
    return passing


def _render(item, program, atom=False):
    tokens = item["tokens"]
    visible = item["_visible"]
    actual = _execute(program, item, visible["inputs"])
    passing = _hidden_passing(item, program)
    legend = (
        "Legend: %s=set %s=add %s=sub %s=mul %s=emit %s=rep %s=done."
        % (
            tokens["set"],
            tokens["add"],
            tokens["sub"],
            tokens["mul"],
            tokens["emit"],
            tokens["rep"],
            tokens["done"],
        )
    )
    lines = ["%d: %s" % (idx + 1, source) for idx, source in enumerate(program)]
    visible_text = "Visible: %s expected=%s actual=%s" % (
        _fmt_inputs(visible["inputs"], item["inputs"]),
        _fmt_seq(visible["expected"]),
        _fmt_seq(actual),
    )
    instruction = (
        "Reply with final line ANSWER: <line#>: <replacement source line>"
        if atom
        else "Use: MEND <line#>: <replacement source line>"
    )
    task = (
        "Program is buggy; exactly one line is wrong. Repair it so actual matches expected and all checks pass."
        if atom
        else "Program is buggy. MEND lines until actual matches expected and all checks pass."
    )
    return "\n".join(
        [legend, "CURRENT program:"]
        + lines
        + [visible_text, "checks passing: %d/3" % passing, task, instruction]
    )


def _extract_last(regex, text):
    found = None
    for line in str(text).splitlines():
        match = regex.match(line)
        if match:
            found = (int(match.group(1)), match.group(2))
    return found


def _extract_mend(text):
    return _extract_last(_MEND_RE, text)


def _extract_answer(text):
    return _extract_last(_ANSWER_RE, text)


def _apply_replacement(item, program, line_no, source):
    if line_no < 1 or line_no > len(program):
        return None
    parsed = _parse_line(source, item["tokens"], item["variables"])
    if parsed is None:
        return None
    candidate = program[:]
    candidate[line_no - 1] = parsed["source"]
    if _compile_program(candidate, item["tokens"], item["variables"]) is None:
        return None
    return candidate


def _program_from_mends(item, transcript):
    program = item["program"][:]
    for turn in transcript:
        action = turn.get("action", "") if isinstance(turn, dict) else ""
        parsed = _extract_mend(action)
        if parsed is None:
            continue
        updated = _apply_replacement(item, program, parsed[0], parsed[1])
        if updated is not None:
            program = updated
    return program


def _program_from_answer(item, transcript):
    answer = None
    for turn in transcript:
        action = turn.get("action", "") if isinstance(turn, dict) else ""
        parsed = _extract_answer(action)
        if parsed is not None:
            answer = parsed
    program = item["program"][:]
    if answer is None:
        return program
    updated = _apply_replacement(item, program, answer[0], answer[1])
    return updated if updated is not None else program


class Env:
    def __init__(self, item):
        self.item = item
        self.program = item["program"][:]

    def reset(self):
        self.program = self.item["program"][:]
        return _render(self.item, self.program, atom=(self.item["mode"] == "atom"))

    def step(self, action):
        if self.item["mode"] == "atom":
            parsed = _extract_answer(action)
            if parsed is None:
                return _BAD_ANSWER, False
        else:
            parsed = _extract_mend(action)
            if parsed is None:
                return _BAD_MEND, False
        updated = _apply_replacement(self.item, self.program, parsed[0], parsed[1])
        if updated is None:
            return _BAD_ANSWER if self.item["mode"] == "atom" else _BAD_MEND, False
        self.program = updated
        done = _hidden_passing(self.item, self.program) == 3
        return _render(self.item, self.program, atom=(self.item["mode"] == "atom")), done


def score(item, transcript):
    if item["mode"] == "atom":
        program = _program_from_answer(item, transcript)
    else:
        program = _program_from_mends(item, transcript)
    hidden_passing = _hidden_passing(item, program)
    changed = 0
    for before, after in zip(item["program"], program):
        if _canon(before) != _canon(after):
            changed += 1
    planted = int(item["_planted_bugs"])
    if hidden_passing == 3:
        if changed <= planted:
            bonus = 1.0
        else:
            bonus = planted / changed
        value = 0.8 + 0.2 * min(1.0, bonus)
    else:
        value = 0.5 * (hidden_passing / 3.0)
    value = max(0.0, min(1.0, value))
    return {
        "score": value,
        "hidden_passing": hidden_passing,
        "changed_lines": changed,
        "planted_bugs": planted,
        "all_hidden_pass": hidden_passing == 3,
    }


def oracle_policy(item, history):
    if item["mode"] == "atom":
        current = item["program"][:]
        prefix = "ANSWER"
    else:
        current = _program_from_mends(item, history)
        prefix = "MEND"
    correct = item["_correct_program"]
    for idx, (have, want) in enumerate(zip(current, correct), 1):
        if _canon(have) != _canon(want):
            return "%s %d: %s" % (prefix, idx, want) if prefix == "MEND" else "%s: %d: %s" % (prefix, idx, want)
    return "%s 1: %s" % (prefix, current[0]) if prefix == "MEND" else "%s: 1: %s" % (prefix, current[0])


def _random_operand(item, rng):
    if rng.randrange(2) == 0:
        return rng.choice(item["variables"])
    return str(rng.randint(-20, 20))


def _random_source_for_line(item, current_source, rng):
    parsed = _parse_line(current_source, item["tokens"], item["variables"])
    if parsed is not None and parsed["op"] == "rep":
        return _line(item["tokens"], "rep", rng.randint(0, 6))
    if parsed is not None and parsed["op"] == "done":
        return _line(item["tokens"], "done")
    op = rng.choice(("set", "add", "sub", "mul", "emit"))
    if op == "set":
        return _line(item["tokens"], op, rng.choice(item["variables"]), _random_operand(item, rng))
    if op in _ARITH:
        return _line(
            item["tokens"],
            op,
            rng.choice(item["variables"]),
            _random_operand(item, rng),
            _random_operand(item, rng),
        )
    return _line(item["tokens"], op, _random_operand(item, rng))


def random_policy(item, history, rng):
    if item["mode"] == "atom":
        current = item["program"][:]
        line_no = rng.randint(1, len(current))
        source = _random_source_for_line(item, current[line_no - 1], rng)
        return "ANSWER: %d: %s" % (line_no, source)
    current = _program_from_mends(item, history)
    line_no = rng.randint(1, len(current))
    source = _random_source_for_line(item, current[line_no - 1], rng)
    return "MEND %d: %s" % (line_no, source)
