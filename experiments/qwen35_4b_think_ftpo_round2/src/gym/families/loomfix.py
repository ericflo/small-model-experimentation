"""loomfix — program repair in SPOOL, an invented 4-register mini-language.

The generator authors a correct straight-line SPOOL routine, plants exactly
one single-line bug (wrong register, off-by-1..3 constant, or wrong op of the
same shape), and derives 2-3 tests (start registers -> expected finals on the
CORRECT routine) of which at least one fails on the buggy routine.

Atoms (two kinds, alternating):
  loc  — shown the buggy routine plus per-test correct-vs-buggy finals, name a
         line number whose single-line replacement can make every test pass
         (the gold set is precomputed by exhaustive search over the op menu).
  exec — trace a shown routine (labeled CORRECT or BUGGY copy) from given
         start registers and report one register's final value.

Episodes: observe the buggy routine and failing tests; PATCH lines and RUN
the tests until all pass or the turn budget runs out. Score is 1.0 on full
repair, else the pass fraction at the last RUN (0 if never run).
"""

from __future__ import annotations

from .. import base

FAMILY = "loomfix"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True

REGS = ("A", "B", "C", "D")
MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14}
_SEARCH_CONST_MAX = 12  # covers generated constants (<=9) plus +-3 mutations

_LEVEL_SHAPE = {
    # level: (n_lines, n_tests, allow_ifz, max_const)
    1: (4, 2, False, 5),
    2: (5, 2, True, 6),
    3: (6, 3, True, 8),
    4: (8, 3, True, 9),
}
# Reject loc atoms whose repairable-line set is larger than this (keeps the
# localization question sharp); after enough attempts the last candidate is
# accepted anyway, so generation always terminates.
_MAX_FIXABLE = {1: 2, 2: 2, 3: 2, 4: 3}

_SEMANTICS = (
    "SPOOL is a tiny register language. Four registers A, B, C, D hold\n"
    "non-negative integers. One instruction per line, executed top to bottom:\n"
    "ADD r n = add n to register r\n"
    "SUB r n = subtract n from r (floor at 0)\n"
    "COPY r s = set r to the current value of s\n"
    "SWAP r s = exchange the values of r and s\n"
    "ZERO r = set r to 0\n"
    "IFZ r = if r is 0, skip the next line"
)


# ---------------------------------------------------------------------------
# SPOOL interpreter and repair search
# ---------------------------------------------------------------------------


def _execute(program: list[list], start: dict[str, int]) -> dict[str, int]:
    regs = dict(start)
    pc = 0
    while pc < len(program):
        ins = program[pc]
        op = ins[0]
        if op == "ADD":
            regs[ins[1]] += ins[2]
        elif op == "SUB":
            regs[ins[1]] = max(0, regs[ins[1]] - ins[2])
        elif op == "COPY":
            regs[ins[1]] = regs[ins[2]]
        elif op == "SWAP":
            regs[ins[1]], regs[ins[2]] = regs[ins[2]], regs[ins[1]]
        elif op == "ZERO":
            regs[ins[1]] = 0
        elif op == "IFZ":
            if regs[ins[1]] == 0:
                pc += 2
                continue
        else:  # pragma: no cover - generator bug
            raise ValueError(f"unknown op {op!r}")
        pc += 1
    return regs


def _passes_all(program: list[list], tests: list[dict]) -> bool:
    return all(_execute(program, t["start"]) == t["expect"] for t in tests)


def _candidate_instructions():
    for op in ("ADD", "SUB"):
        for r in REGS:
            for n in range(0, _SEARCH_CONST_MAX + 1):
                yield [op, r, n]
    for op in ("COPY", "SWAP"):
        for r in REGS:
            for s in REGS:
                if r != s:
                    yield [op, r, s]
    for op in ("ZERO", "IFZ"):
        for r in REGS:
            yield [op, r]


def _fixable_lines(buggy: list[list], tests: list[dict]) -> list[int]:
    """0-based indices whose single-line replacement can pass every test."""
    fixable = []
    for i in range(len(buggy)):
        trial = [list(line) for line in buggy]
        for candidate in _candidate_instructions():
            trial[i] = candidate
            if _passes_all(trial, tests):
                fixable.append(i)
                break
    return fixable


def _render_ins(ins: list) -> str:
    return " ".join(str(part) for part in ins)


def _render_regs(regs: dict[str, int]) -> str:
    return " ".join(f"{r}={regs[r]}" for r in REGS)


def _render_program(program: list[list]) -> list[str]:
    return [f"{i + 1}: {_render_ins(ins)}" for i, ins in enumerate(program)]


# ---------------------------------------------------------------------------
# Instance generation (shared by atoms and episodes)
# ---------------------------------------------------------------------------


def _gen_program(rng, n_lines: int, allow_ifz: bool, max_const: int) -> list[list]:
    program: list[list] = []
    prev_ifz = False
    for i in range(n_lines):
        ops = ["ADD", "ADD", "SUB", "SUB", "COPY", "SWAP", "ZERO"]
        if allow_ifz and not prev_ifz and i < n_lines - 1:
            ops += ["IFZ", "IFZ"]
        op = rng.choice(ops)
        if op in ("ADD", "SUB"):
            ins = [op, rng.choice(REGS), rng.randint(1, max_const)]
        elif op in ("COPY", "SWAP"):
            r, s = rng.sample(REGS, 2)
            ins = [op, r, s]
        else:
            ins = [op, rng.choice(REGS)]
        prev_ifz = op == "IFZ"
        program.append(ins)
    return program


def _mutate(rng, ins: list) -> list:
    op = ins[0]
    if op in ("ADD", "SUB"):
        mode = rng.choice(["op", "reg", "const", "const"])
        if mode == "op":
            return ["SUB" if op == "ADD" else "ADD", ins[1], ins[2]]
        if mode == "reg":
            return [op, rng.choice([r for r in REGS if r != ins[1]]), ins[2]]
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        return [op, ins[1], max(0, ins[2] + delta)]
    if op in ("COPY", "SWAP"):
        mode = rng.choice(["op", "reg", "reg"])
        if mode == "op":
            return ["SWAP" if op == "COPY" else "COPY", ins[1], ins[2]]
        pos = rng.choice([1, 2])
        other = ins[2] if pos == 1 else ins[1]
        new = list(ins)
        new[pos] = rng.choice([r for r in REGS if r != other and r != ins[pos]])
        return new
    # ZERO / IFZ
    if rng.random() < 0.5:
        return ["IFZ" if op == "ZERO" else "ZERO", ins[1]]
    return [op, rng.choice([r for r in REGS if r != ins[1]])]


def _gen_tests(rng, program: list[list], n_tests: int) -> list[dict]:
    tests: list[dict] = []
    seen: set[tuple] = set()
    guard = 0
    while len(tests) < n_tests:
        guard += 1
        start = {r: (0 if rng.random() < 0.3 else rng.randint(1, 9)) for r in REGS}
        key = tuple(start[r] for r in REGS)
        if key in seen and guard < 50:
            continue
        seen.add(key)
        tests.append({"start": start, "expect": _execute(program, start)})
    return tests


def _gen_instance(rng, level: int) -> dict:
    """A correct program, a one-line mutant, and tests with >=1 buggy FAIL."""
    n_lines, n_tests, allow_ifz, max_const = _LEVEL_SHAPE[level]
    inst = None
    for _ in range(200):
        program = _gen_program(rng, n_lines, allow_ifz, max_const)
        idx = rng.randrange(n_lines)
        buggy = [list(line) for line in program]
        buggy[idx] = _mutate(rng, program[idx])
        tests = _gen_tests(rng, program, n_tests)
        inst = {"program": program, "buggy": buggy, "idx": idx, "tests": tests}
        if not _passes_all(buggy, tests):
            return inst
    return inst  # pragma: no cover - 200 rejections is practically unreachable


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        picked = None
        fallback = None
        item = None
        for attempt in range(40):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) > base.ATOM_PROMPT_CHAR_LIMIT:
                continue
            if fallback is None:
                fallback = item
            gold = item["gold"]
            if gold["kind"] == "loc" and len(gold["fixable"]) > _MAX_FIXABLE[level]:
                continue
            picked = item
            break
        items.append(picked or fallback or item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    inst = _gen_instance(rng, level)
    kind = "loc" if index % 2 == 0 else "exec"
    if kind == "loc":
        prompt, gold = _loc_atom(inst)
    else:
        prompt, gold = _exec_atom(rng, inst)
    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        # Lucky-guess guard: loc answers range over the program's line numbers;
        # exec answers range over a wide value space (nominal 50).
        "answer_domain": len(inst["buggy"]) if kind == "loc" else 50,
    }


def _loc_atom(inst: dict) -> tuple[str, dict]:
    fixable = sorted(i + 1 for i in _fixable_lines(inst["buggy"], inst["tests"]))
    test_lines = []
    for i, test in enumerate(inst["tests"]):
        got = _execute(inst["buggy"], test["start"])
        verdict = "pass" if got == test["expect"] else "FAIL"
        test_lines.append(
            f"T{i + 1}: start {_render_regs(test['start'])} | "
            f"correct {_render_regs(test['expect'])} | "
            f"buggy {_render_regs(got)} -> {verdict}"
        )
    lines = [
        _SEMANTICS,
        "",
        "This SPOOL routine was correct until exactly one line was changed",
        "(the planted bug):",
        "",
        *_render_program(inst["buggy"]),
        "",
        "Each test starts the registers as shown; 'correct' is what the",
        "original routine yields, 'buggy' is what the routine above yields.",
        "",
        *test_lines,
        "",
        "Which line number holds the planted bug? (If replacing some other",
        "single line would also make every test pass, that line number is",
        "accepted too.)",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    gold = {"kind": "loc", "fixable": fixable, "planted": inst["idx"] + 1}
    return "\n".join(lines), gold


def _exec_atom(rng, inst: dict) -> tuple[str, dict]:
    show_buggy = rng.random() < 0.5
    program = inst["buggy"] if show_buggy else inst["program"]
    start = {r: (0 if rng.random() < 0.3 else rng.randint(1, 9)) for r in REGS}
    finals = _execute(program, start)
    # Prefer a register with a nonzero final so a constant "0" reply stays a
    # floor, not a strategy.
    nonzero = [r for r in REGS if finals[r] > 0]
    pool = nonzero if nonzero and rng.random() > 0.10 else list(REGS)
    target = rng.choice(pool)
    if show_buggy:
        label = (
            "Below is the BUGGY copy of a loom-house routine (exactly one line"
            " differs from the correct copy). Trace it exactly as printed."
        )
    else:
        label = (
            "Below is the CORRECT copy of a loom-house routine, exactly as it"
            " should run."
        )
    lines = [
        _SEMANTICS,
        "",
        label,
        "",
        *_render_program(program),
        "",
        f"Start registers: {_render_regs(start)}.",
        f"After the routine finishes, what value does register {target} hold?",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    gold = {"kind": "exec", "value": finals[target]}
    return "\n".join(lines), gold


def score_atom(item: dict, reply_text: str) -> float:
    gold = item["gold"]
    if gold["kind"] == "exec":
        return base.score_exact_int(gold["value"], reply_text)
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    value = base.canon_int(answer)
    return 1.0 if value is not None and value in gold["fixable"] else 0.0


def oracle_atom(item: dict) -> str:
    gold = item["gold"]
    if gold["kind"] == "exec":
        return f"ANSWER: {gold['value']}"
    return f"ANSWER: {gold['planted']}"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

_GRAMMAR_HINT = "Use: PATCH <line#> <instruction> | RUN"


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        inst = _gen_instance(rng, level)
        self.level = level
        self.max_turns = MAX_TURNS[level]
        self._correct = inst["program"]
        self._program = [list(line) for line in inst["buggy"]]
        self._planted_idx = inst["idx"]
        self._tests = inst["tests"]
        self._turns = 0
        self._solved = False
        self._last_run: tuple[int, int] | None = None
        self.last_action_ok = True
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "buggy_program": [_render_ins(line) for line in inst["buggy"]],
            "correct_program": [_render_ins(line) for line in inst["program"]],
            "planted_line": inst["idx"] + 1,
            "tests": inst["tests"],
            "max_turns": self.max_turns,
        }

    def system_prompt(self) -> str:
        return (
            "You are repairing a routine for the loom-house's pattern engine.\n"
            + _SEMANTICS
            + "\n\nExactly one line of the routine was changed from the correct"
            " version. Each test starts the registers as shown and lists the"
            " finals the CORRECT routine yields. Repair the routine so every"
            " test passes, then RUN to confirm.\n"
            "Actions (one per turn):\n"
            "PATCH <line#> <instruction> — replace that line"
            " (e.g. PATCH 3 ADD B 2)\n"
            "RUN — rerun all tests and report pass/fail\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        test_lines = []
        for i, test in enumerate(self._tests):
            got = _execute(self._program, test["start"])
            verdict = "pass" if got == test["expect"] else "FAIL"
            test_lines.append(
                f"T{i + 1}: start {_render_regs(test['start'])} | "
                f"correct {_render_regs(test['expect'])} | "
                f"buggy {_render_regs(got)} -> {verdict}"
            )
        return (
            "Buggy routine:\n"
            + "\n".join(_render_program(self._program))
            + "\nTests (start | correct finals | this routine's finals):\n"
            + "\n".join(test_lines)
        )

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        observation = self._apply(action_line)
        done = self._solved or self._turns >= self.max_turns
        return observation, done

    def _apply(self, action_line: str) -> str:
        tokens = (action_line or "").strip().split()
        if not tokens:
            return "Empty action. " + _GRAMMAR_HINT
        verb = tokens[0].upper()
        if verb == "RUN":
            if len(tokens) > 1:
                return "RUN takes no arguments. " + _GRAMMAR_HINT
            self.last_action_ok = True
            return self._do_run()
        if verb == "PATCH":
            return self._do_patch(tokens)
        return "Unknown action. " + _GRAMMAR_HINT

    def _do_run(self) -> str:
        lines = []
        n_pass = 0
        for i, test in enumerate(self._tests):
            got = _execute(self._program, test["start"])
            if got == test["expect"]:
                n_pass += 1
                lines.append(f"T{i + 1}: got {_render_regs(got)} -> pass")
            else:
                lines.append(
                    f"T{i + 1}: got {_render_regs(got)} -> FAIL"
                    f" (want {_render_regs(test['expect'])})"
                )
        self._last_run = (n_pass, len(self._tests))
        if n_pass == len(self._tests):
            self._solved = True
            return (
                "Run results:\n" + "\n".join(lines)
                + "\nAll tests pass. Routine repaired."
            )
        return (
            "Run results:\n" + "\n".join(lines)
            + f"\n{n_pass}/{len(self._tests)} tests pass."
        )

    def _do_patch(self, tokens: list[str]) -> str:
        if len(tokens) < 3:
            return (
                "PATCH needs a line number and an instruction,"
                " e.g. PATCH 2 SUB C 1."
            )
        try:
            lineno = int(tokens[1])
        except ValueError:
            return "PATCH needs a numeric line number. " + _GRAMMAR_HINT
        if not 1 <= lineno <= len(self._program):
            return f"No line {lineno}. Lines are 1..{len(self._program)}."
        ins = _parse_instruction(tokens[2:])
        if ins is None:
            return (
                "Bad instruction. Grammar: ADD r n | SUB r n | COPY r s |"
                " SWAP r s | ZERO r | IFZ r (r,s in A,B,C,D; n an integer"
                " 0..99)."
            )
        self.last_action_ok = True
        self._program[lineno - 1] = ins
        return (
            f"Line {lineno} replaced. Current routine:\n"
            + "\n".join(_render_program(self._program))
            + "\n(Use RUN to test.)"
        )

    def score(self) -> float:
        if self._solved:
            return 1.0
        if self._last_run is None:
            return 0.0
        return self._last_run[0] / self._last_run[1]


def _parse_instruction(tokens: list[str]) -> list | None:
    if not tokens:
        return None
    op = tokens[0].upper()
    args = [t.upper() for t in tokens[1:]]
    if op in ("ADD", "SUB") and len(args) == 2 and args[0] in REGS:
        try:
            n = int(args[1])
        except ValueError:
            return None
        return [op, args[0], n] if 0 <= n <= 99 else None
    if op in ("COPY", "SWAP") and len(args) == 2 and args[0] in REGS and args[1] in REGS:
        return [op, args[0], args[1]]
    if op in ("ZERO", "IFZ") and len(args) == 1 and args[0] in REGS:
        return [op, args[0]]
    return None


class OraclePolicy:
    """PATCH the planted line back to its original form, then RUN."""

    def __init__(self, episode: Episode):
        self._episode = episode
        self._patched = False

    def act(self, observation_history: list[str]) -> str:
        if not self._patched:
            self._patched = True
            idx = self._episode._planted_idx
            fix = _render_ins(self._episode._correct[idx])
            return f"PATCH {idx + 1} {fix}"
        return "RUN"


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
    }
