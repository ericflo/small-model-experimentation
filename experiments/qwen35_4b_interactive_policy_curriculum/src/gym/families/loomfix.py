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

Frontier levels (L5/L6): longer routines (10/12 lines) with SUBTLER planted
bugs — constants drift by exactly 1, COPY operands flip, IFZ guards the wrong
register or "falls off" (the line becomes a duplicate of the instruction it
guarded). L6 episodes plant TWO independent bugs, each of which must be
repaired (fixing either alone still fails a test); atoms stay single-bug.
"""

from __future__ import annotations

import re

from .. import base

FAMILY = "loomfix"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = True

REGS = ("A", "B", "C", "D")
MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14, 5: 18, 6: 22}
_SEARCH_CONST_MAX = 12  # covers generated constants (<=9) plus +-3 mutations

_LEVEL_SHAPE = {
    # level: (n_lines, n_tests, allow_ifz, max_const)
    1: (4, 2, False, 5),
    2: (5, 2, True, 6),
    3: (6, 3, True, 8),
    4: (8, 3, True, 9),
    5: (10, 3, True, 9),
    6: (12, 4, True, 9),
}
# Reject loc atoms whose repairable-line set is larger than this (keeps the
# localization question sharp); after enough attempts the last candidate is
# accepted anyway, so generation always terminates.
_MAX_FIXABLE = {1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3}

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


def _mutate_subtle(rng, program: list[list], idx: int) -> list:
    """Frontier (L5+) single-line mutation menu: near-miss bugs only.

    Compared to :func:`_mutate`, constants drift by exactly 1 (never 2-3),
    COPY operands flip, and IFZ can 'fall off by one line': the guard line
    becomes a duplicate of the instruction it was supposed to protect, so the
    routine reads as if the IFZ slipped down a line.
    """
    ins = program[idx]
    op = ins[0]
    if op in ("ADD", "SUB"):
        mode = rng.choice(["const", "const", "reg", "op"])
        if mode == "const":
            return [op, ins[1], max(0, ins[2] + rng.choice([-1, 1]))]
        if mode == "reg":
            return [op, rng.choice([r for r in REGS if r != ins[1]]), ins[2]]
        return ["SUB" if op == "ADD" else "ADD", ins[1], ins[2]]
    if op == "COPY":
        mode = rng.choice(["flip", "flip", "reg"])
        if mode == "flip":
            return ["COPY", ins[2], ins[1]]
        pos = rng.choice([1, 2])
        other = ins[2] if pos == 1 else ins[1]
        new = list(ins)
        new[pos] = rng.choice([r for r in REGS if r != other and r != ins[pos]])
        return new
    if op == "SWAP":  # operand flip is a semantic no-op for SWAP; never use it
        if rng.random() < 0.35:
            return ["COPY", ins[1], ins[2]]
        pos = rng.choice([1, 2])
        other = ins[2] if pos == 1 else ins[1]
        new = list(ins)
        new[pos] = rng.choice([r for r in REGS if r != other and r != ins[pos]])
        return new
    if op == "IFZ":
        # _gen_program never emits consecutive IFZ, so the next line (when it
        # exists) is a plain instruction and the duplicate reads naturally.
        if idx + 1 < len(program) and rng.random() < 0.4:
            return list(program[idx + 1])
        return ["IFZ", rng.choice([r for r in REGS if r != ins[1]])]
    # ZERO
    if rng.random() < 0.3:
        return ["IFZ", ins[1]]
    return ["ZERO", rng.choice([r for r in REGS if r != ins[1]])]


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
        if level >= 5:
            buggy[idx] = _mutate_subtle(rng, program, idx)
        else:
            buggy[idx] = _mutate(rng, program[idx])
        tests = _gen_tests(rng, program, n_tests)
        inst = {"program": program, "buggy": buggy, "idx": idx, "tests": tests}
        if not _passes_all(buggy, tests):
            return inst
    return inst  # pragma: no cover - 200 rejections is practically unreachable


def _gen_instance_two_bug(rng, level: int) -> dict:
    """Two independent one-line mutants on distinct lines (L6 episodes).

    Both bugs must matter: the buggy routine fails a test, AND fixing either
    planted line alone still fails a test, so a full repair takes two PATCHes.
    """
    n_lines, n_tests, allow_ifz, max_const = _LEVEL_SHAPE[level]
    inst = None
    for _ in range(200):
        program = _gen_program(rng, n_lines, allow_ifz, max_const)
        idx_a, idx_b = sorted(rng.sample(range(n_lines), 2))
        buggy = [list(line) for line in program]
        buggy[idx_a] = _mutate_subtle(rng, program, idx_a)
        buggy[idx_b] = _mutate_subtle(rng, program, idx_b)
        tests = _gen_tests(rng, program, n_tests)
        inst = {
            "program": program,
            "buggy": buggy,
            "idxs": [idx_a, idx_b],
            "tests": tests,
        }
        only_a = [list(line) for line in buggy]
        only_a[idx_b] = list(program[idx_b])
        only_b = [list(line) for line in buggy]
        only_b[idx_a] = list(program[idx_a])
        if (
            not _passes_all(buggy, tests)
            and not _passes_all(only_a, tests)
            and not _passes_all(only_b, tests)
        ):
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
            if len(item["prompt"]) > base.atom_prompt_limit(level):
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
# Oracle traces (think-channel SFT text; truth-blind procedure narration)
# ---------------------------------------------------------------------------

_TRACE_SOFT_CAP_WORDS = 760

_PROGRAM_ROW_RE = re.compile(r"^(\d+): ([A-Z].*)$")
_TEST_ROW_RE = re.compile(
    r"^T(\d+): start (.+?) \| correct (.+?) \| buggy (.+?) -> (pass|FAIL)$"
)
_REG_PAIR_RE = re.compile(r"([A-D])=(\d+)")


def _parse_regs_text(text: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in _REG_PAIR_RE.finditer(text)}


def _parse_prompt_program(prompt: str) -> list[list]:
    program = []
    for line in prompt.splitlines():
        match = _PROGRAM_ROW_RE.match(line)
        if match:
            program.append(_parse_instruction(match.group(2).split()))
    return program


def _parse_prompt_tests(prompt: str) -> list[dict]:
    tests = []
    for line in prompt.splitlines():
        match = _TEST_ROW_RE.match(line)
        if match:
            tests.append(
                {
                    "label": f"T{match.group(1)}",
                    "start": _parse_regs_text(match.group(2)),
                    "expect": _parse_regs_text(match.group(3)),
                    "got": _parse_regs_text(match.group(4)),
                    "fail": match.group(5) == "FAIL",
                }
            )
    return tests


def _written_regs(ins: list) -> list[str]:
    op = ins[0]
    if op == "SWAP":
        return [ins[1], ins[2]]
    if op == "IFZ":
        return []
    return [ins[1]]


def _line_fixable(program: list[list], idx: int, tests: list[dict]) -> bool:
    trial = [list(line) for line in program]
    for candidate in _candidate_instructions():
        trial[idx] = candidate
        if _passes_all(trial, tests):
            return True
    return False


def _narrate_run(
    program: list[list], start: dict[str, int], style: int, verbose: bool
) -> tuple[list[str], dict[str, int], list[tuple]]:
    """Prose per executed line, final registers, and every register change."""
    sentences: list[str] = []
    regs = dict(start)
    history: list[tuple] = []  # (line_no, reg, new_value) whenever a value moves
    pc = 0
    while pc < len(program):
        ins = program[pc]
        op = ins[0]
        lineno = pc + 1
        jump = 1
        if op == "ADD":
            before = regs[ins[1]]
            regs[ins[1]] = before + ins[2]
            if ins[2] == 0:
                note = f"adding 0 leaves {ins[1]} at {before}"
            else:
                note = f"{ins[1]} goes from {before} to {regs[ins[1]]}"
                history.append((lineno, ins[1], regs[ins[1]]))
        elif op == "SUB":
            before = regs[ins[1]]
            after = max(0, before - ins[2])
            regs[ins[1]] = after
            if ins[2] == 0:
                note = f"subtracting 0 leaves {ins[1]} at {before}"
            elif before - ins[2] < 0:
                note = (
                    f"{before} minus {ins[2]} would go negative,"
                    f" so {ins[1]} floors at 0"
                )
            else:
                note = f"{ins[1]} drops from {before} to {after}"
            if after != before:
                history.append((lineno, ins[1], after))
        elif op == "COPY":
            before = regs[ins[1]]
            regs[ins[1]] = regs[ins[2]]
            note = f"{ins[1]} takes {ins[2]}'s value, so {ins[1]}={regs[ins[1]]}"
            if regs[ins[1]] != before:
                history.append((lineno, ins[1], regs[ins[1]]))
        elif op == "SWAP":
            first_before = regs[ins[1]]
            regs[ins[1]], regs[ins[2]] = regs[ins[2]], regs[ins[1]]
            note = f"now {ins[1]}={regs[ins[1]]} and {ins[2]}={regs[ins[2]]}"
            if regs[ins[1]] != first_before:
                history.append((lineno, ins[1], regs[ins[1]]))
                history.append((lineno, ins[2], regs[ins[2]]))
        elif op == "ZERO":
            before = regs[ins[1]]
            regs[ins[1]] = 0
            if before:
                note = f"{ins[1]} clears from {before} to 0"
                history.append((lineno, ins[1], 0))
            else:
                note = f"{ins[1]} is already 0 and stays 0"
        else:  # IFZ
            if regs[ins[1]] == 0:
                if pc + 1 < len(program):
                    note = (
                        f"{ins[1]} is 0, so line {pc + 2}"
                        f" ({_render_ins(program[pc + 1])}) is skipped"
                    )
                else:
                    note = f"{ins[1]} is 0, but there is no next line to skip"
                jump = 2
            else:
                note = (
                    f"{ins[1]} is {regs[ins[1]]}, not zero,"
                    " so nothing is skipped"
                )
        if style == 0:
            sentence = f"Line {lineno} is {_render_ins(ins)}: {note}."
            if verbose:
                sentence += f" Now {_render_regs(regs)}."
        else:
            sentence = f"Line {lineno}, {_render_ins(ins)} — {note}."
            if verbose:
                sentence += f" Registers: {_render_regs(regs)}."
        sentences.append(sentence)
        pc += jump
    return sentences, regs, history


def _exec_trace_text(item: dict, picks: dict, verbose: bool) -> str:
    prompt = item["prompt"]
    program = _parse_prompt_program(prompt)
    start = _parse_regs_text(
        re.search(r"Start registers: ([^.\n]+)\.", prompt).group(1)
    )
    target = re.search(r"what value does register ([A-D]) hold", prompt).group(1)

    openers = (
        f"I need to run this routine exactly as printed and report what"
        f" register {target} holds when it finishes.",
        f"The job is to trace this routine line by line from the given start"
        f" registers and read off register {target} at the end.",
        f"Let me execute this routine carefully, one line at a time, keeping"
        f" an eye on register {target}.",
    )
    starters = (
        f"The registers start at {_render_regs(start)}.",
        f"Starting state: {_render_regs(start)}.",
        f"I begin with {_render_regs(start)}.",
    )
    head = [openers[picks["opener"]], starters[picks["start"]]]
    if "BUGGY copy" in prompt:
        head.append(
            "The label says this is the buggy copy, but the question asks"
            f" what this printed routine leaves in {target}, so I trace it"
            " exactly as shown."
        )

    touched = sorted(
        {i + 1 for i, ins in enumerate(program) if target in _written_regs(ins)}
    )
    has_ifz = any(ins[0] == "IFZ" for ins in program)
    if not touched:
        plan = (
            f"Scanning first: no line ever touches {target} directly, so it"
            " should keep its start value — I'll trace the whole routine"
            " anyway to be sure."
        )
    else:
        if len(touched) == 1:
            plan = f"Scanning first: only line {touched[0]} touches {target}"
        else:
            listed = ", ".join(str(n) for n in touched)
            plan = f"Scanning first: lines {listed} touch {target}"
        if has_ifz:
            plan += (
                ", but an IFZ can change which lines actually run, so I trace"
                " everything in order."
            )
        else:
            plan += (
                "; I'll still trace the whole routine in order to keep every"
                " register straight."
            )

    sentences, finals, history = _narrate_run(
        program, start, picks["line_style"], verbose
    )
    tail = (
        f"That was the last line, so the routine ends with"
        f" {_render_regs(finals)}."
    )

    checks = (
        f"Before I commit, let me double-check {target}'s history.",
        f"Quick second pass on {target} to be safe.",
        f"Let me re-verify how {target} moved.",
    )
    target_moves = [(ln, v) for (ln, r, v) in history if r == target]
    if target_moves:
        walk = f"{target} begins at {start[target]}" + "".join(
            f", then line {ln} ({_render_ins(program[ln - 1])}) makes it {v}"
            for ln, v in target_moves
        ) + ", and it never changes again. That matches my trace."
    else:
        walk = (
            f"No line ever changed {target}, so it stays at {start[target]}"
            " from start to finish. That matches my trace."
        )
    conclusions = (
        f"So register {target} finishes holding {finals[target]}.",
        f"Register {target}'s final value is {finals[target]}.",
        f"So at the end of the routine, {target} holds {finals[target]}.",
    )
    paragraphs = [
        "\n".join(head),
        plan,
        "\n".join(sentences + [tail]),
        "\n".join([checks[picks["check"]], walk, conclusions[picks["conclusion"]]]),
    ]
    return "\n\n".join(paragraphs)


def _loc_trace_text(
    item: dict, picks: dict, verbose: bool, with_dead_end: bool
) -> str:
    prompt = item["prompt"]
    program = _parse_prompt_program(prompt)
    tests = _parse_prompt_tests(prompt)
    planted = item["gold"]["planted"]  # 1-based line number
    fails = [t for t in tests if t["fail"]]
    first = fails[0]

    openers = (
        "Exactly one line of this routine was changed, and I need to name a"
        " line whose single replacement makes every test pass. The"
        " correct-versus-buggy finals are my evidence.",
        "One planted bug, one line to find. I'll compare what the buggy"
        " routine produces against the correct finals, then test suspect"
        " lines by mental substitution.",
        "I need to find the changed line. Plan: read the test diffs, trace a"
        " failing test to see what the routine really does, then try"
        " single-line replacements until every test passes.",
    )

    diff_sentences = []
    mismatched: list[str] = []
    for t in tests:
        if not t["fail"]:
            diff_sentences.append(f"{t['label']} passes as-is.")
            continue
        bits = []
        for r in REGS:
            if t["got"][r] != t["expect"][r]:
                bits.append(
                    f"{r} ends {t['got'][r]} but should be {t['expect'][r]}"
                )
                if r not in mismatched:
                    mismatched.append(r)
        diff_sentences.append(f"{t['label']} fails: " + ", ".join(bits) + ".")
    mismatched.sort(key=REGS.index)
    if len(mismatched) == 1:
        reg = mismatched[0]
        diffs = [t["got"][reg] - t["expect"][reg] for t in fails]
        if all(d == diffs[0] for d in diffs):
            direction = "low" if diffs[0] < 0 else "high"
            summary = (
                f"So every failure is register {reg}, exactly"
                f" {abs(diffs[0])} {direction} each time."
            )
        else:
            summary = (
                f"So every failure lands on register {reg}, though the miss"
                " differs between tests."
            )
    else:
        summary = (
            f"So the failures touch registers {', '.join(mismatched)} — a"
            " pattern that can come from a copy, a swap, or a skipped guard."
        )

    intros = (
        f"Let me trace {first['label']} on the routine as printed to see the"
        " mechanics.",
        f"First I trace {first['label']} through the shown routine.",
        f"I'll walk {first['label']} line by line through the routine exactly"
        " as shown.",
    )
    run_sentences, run_finals, _ = _narrate_run(
        program, first["start"], picks["line_style"], verbose
    )
    confirm = (
        f"Final state {_render_regs(run_finals)} — exactly the buggy finals"
        f" listed for {first['label']}, so I am reading the routine"
        " correctly."
    )

    methods = (
        "Now the substitution test: change one line in my head, rerun every"
        " test, and see whether all the finals match the correct column.",
        "Now I test candidate lines one at a time: a line is the culprit only"
        " if some single replacement of it makes every test pass.",
        "Time to try repairs: pick a suspect line, imagine a replacement"
        " there, and rerun all the tests mentally.",
    )

    dead_end = None
    if with_dead_end:
        suspects = [
            i
            for i, ins in enumerate(program)
            if i + 1 != planted
            and any(r in mismatched for r in _written_regs(ins))
        ]
        for j in suspects[:3]:
            if _line_fixable(program, j, tests):
                continue
            showcase = None
            trial = [list(line) for line in program]
            for candidate in _candidate_instructions():
                if candidate == program[j]:
                    continue
                trial[j] = candidate
                if _execute(trial, first["start"]) == first["expect"]:
                    showcase = candidate
                    break
            lead = (
                f"Line {j + 1} ({_render_ins(program[j])}) is a tempting"
                " suspect."
            )
            if showcase is not None:
                trial = [list(line) for line in program]
                trial[j] = showcase
                mid = ""
                for t in tests:
                    got = _execute(trial, t["start"])
                    if got != t["expect"]:
                        bad = next(
                            r for r in REGS if got[r] != t["expect"][r]
                        )
                        mid = (
                            f"If it read {_render_ins(showcase)} instead,"
                            f" {first['label']} would come out fully correct,"
                            f" but then {t['label']} ends {bad}={got[bad]}"
                            f" where it needs {t['expect'][bad]}."
                        )
                        break
            else:
                mid = (
                    f"But no single replacement there even repairs"
                    f" {first['label']}: I swept other constants, the"
                    f" opposite op, copies and swaps, and {first['label']}"
                    " stays wrong."
                )
            dead_end = "\n".join(
                [
                    lead,
                    mid,
                    f"No one-line change at line {j + 1} satisfies every"
                    f" test, so line {j + 1} is out.",
                ]
            )
            break

    repair = None
    trial = [list(line) for line in program]
    for candidate in _candidate_instructions():
        if candidate == program[planted - 1]:
            continue
        trial[planted - 1] = candidate
        if _passes_all(trial, tests):
            repair = candidate
            break

    success = None
    if repair is not None:
        verify_bits = [
            f"Now line {planted} ({_render_ins(program[planted - 1])})."
            f" Suppose it read {_render_ins(repair)} instead."
        ]
        for i, t in enumerate(tests):
            got = _execute(trial, t["start"])
            if i == 0:
                verify_bits.append(
                    f"{t['label']} reruns to {_render_regs(got)} — the"
                    " correct finals exactly."
                )
            else:
                verify_bits.append(
                    f"{t['label']} reruns to {_render_regs(got)}, also"
                    " correct."
                )
        verify_bits.append(
            f"One replacement at line {planted} fixes every test at once."
        )
        success = "\n".join(verify_bits)

    conclusions = (
        f"So the planted bug is on line {planted}.",
        f"The changed line is line {planted}.",
        f"Everything points to line {planted} — that is where the bug sits.",
    )
    paragraphs = [
        openers[picks["opener"]],
        "\n".join(diff_sentences + [summary]),
        "\n".join([intros[picks["check"]], *run_sentences, confirm]),
        methods[picks["method"]],
    ]
    if dead_end:
        paragraphs.append(dead_end)
    if success:
        paragraphs.append(success)
    paragraphs.append(conclusions[picks["conclusion"]])
    return "\n\n".join(paragraphs)


def oracle_trace(item: dict) -> str:
    """First-person think-channel narration that derives the atom's answer.

    Truth-blind: the text re-parses the prompt and walks the family's own
    solving procedure (trace the routine line by line; for loc atoms, diff
    the tests then test candidate lines by mental substitution) until the
    answer falls out. Deterministic per item id; phrasing varies across
    items via base.rng_for.
    """
    rng = base.rng_for(FAMILY, "trace", item["id"])
    picks = {
        "opener": rng.randrange(3),
        "start": rng.randrange(3),
        "line_style": rng.randrange(2),
        "check": rng.randrange(3),
        "method": rng.randrange(3),
        "conclusion": rng.randrange(3),
    }
    if item["gold"]["kind"] == "exec":
        text = _exec_trace_text(item, picks, verbose=True)
        if len(text.split()) > _TRACE_SOFT_CAP_WORDS:
            text = _exec_trace_text(item, picks, verbose=False)
    else:
        text = _loc_trace_text(item, picks, verbose=True, with_dead_end=True)
        if len(text.split()) > _TRACE_SOFT_CAP_WORDS:
            text = _loc_trace_text(
                item, picks, verbose=False, with_dead_end=False
            )
    return text


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

_GRAMMAR_HINT = "Use: PATCH <line#> <instruction> | RUN"


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        if level >= 6:
            inst = _gen_instance_two_bug(rng, level)
            planted = list(inst["idxs"])
        else:
            inst = _gen_instance(rng, level)
            planted = [inst["idx"]]
        self.level = level
        self.max_turns = MAX_TURNS[level]
        self._correct = inst["program"]
        self._program = [list(line) for line in inst["buggy"]]
        self._planted_idxs = planted
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
        }
        if level >= 6:
            self.spec["planted_lines"] = [i + 1 for i in planted]
        else:
            self.spec["planted_line"] = planted[0] + 1
        self.spec["tests"] = inst["tests"]
        self.spec["max_turns"] = self.max_turns

    def system_prompt(self) -> str:
        if self.level >= 6:
            planted_note = (
                "Exactly two lines of the routine were changed from the"
                " correct version; repair BOTH."
            )
        else:
            planted_note = (
                "Exactly one line of the routine was changed from the correct"
                " version."
            )
        return (
            "You are repairing a routine for the loom-house's pattern engine.\n"
            + _SEMANTICS
            + "\n\n"
            + planted_note
            + " Each test starts the registers as shown and lists the"
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
    """PATCH every planted line back to its original form, then RUN."""

    def __init__(self, episode: Episode):
        self._queue = [
            f"PATCH {idx + 1} {_render_ins(episode._correct[idx])}"
            for idx in episode._planted_idxs
        ]

    def act(self, observation_history: list[str]) -> str:
        if self._queue:
            return self._queue.pop(0)
        return "RUN"


def _selftest_traces(module, *, n_per_level: int = 12, perfect_min: float = 0.95) -> dict:
    """Oracle-trace checks: score, word cap, forbidden words, determinism."""
    stats: dict = {"family": module.FAMILY, "levels": {}}
    for level in module.LEVELS:
        items = module.gen_atoms(7, level, n_per_level)
        n_perfect = 0
        max_words = 0
        for item in items:
            trace = module.oracle_trace(item)
            if trace != module.oracle_trace(item):
                raise base.SelftestError(
                    f"{module.FAMILY} L{level} {item['id']}: trace not deterministic"
                )
            words = len(trace.split())
            max_words = max(max_words, words)
            if words > 800:
                raise base.SelftestError(
                    f"{module.FAMILY} L{level} {item['id']}: trace is {words}"
                    " words (cap 800)"
                )
            lowered = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                if word in lowered:
                    raise base.SelftestError(
                        f"{module.FAMILY} L{level} {item['id']}: forbidden"
                        f" word {word!r} in trace"
                    )
            reply = trace + "\n\n" + module.oracle_atom(item)
            if module.score_atom(item, reply) == 1.0:
                n_perfect += 1
        frac = n_perfect / len(items)
        if frac < perfect_min:
            raise base.SelftestError(
                f"{module.FAMILY} L{level}: trace+answer scores 1.0 on only"
                f" {frac:.3f} < {perfect_min}"
            )
        stats["levels"][level] = {
            "n": len(items),
            "perfect_frac": round(frac, 4),
            "max_words": max_words,
        }
    return stats


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
        "traces": _selftest_traces(module),
    }
