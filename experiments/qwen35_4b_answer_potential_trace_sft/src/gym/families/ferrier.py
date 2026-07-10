"""ferrier — dependent tool chaining over wharf paperwork handles (atoms + episodes).

Per item an invented registry of 4-8 wharf-works paperwork tools forms a
dependency DAG: each tool consumes the literal cargo name and/or opaque handle
tokens returned by earlier tools, and returns a handle of its own result type.
The goal names a final artifact ("obtain a sealmark for the brinesilk cargo").

Atoms: given the signatures, the goal, and a partial transcript of correct
calls with their returned handles, answer the exact next call. Scored against
the machine-computed set of valid next calls (every uncalled chain tool whose
arguments are available lies on a shortest completion, because the chain is
the goal's full dependency closure).

Episodes: the model issues ``CALL name(args)`` actions; the env validates
tool name, arg count, and handle types, returning curt corrective
observations on bad calls (turn consumed). Score 1.0 when the goal artifact
is produced; at exhaustion, (distinct correct chain calls) / (chain length).

Levels scale chain length (1 / 3 / 4-5 / 6-7), distractor count, and DAG
branching (two-argument merge tools from L3).
"""

from __future__ import annotations

import re

from .. import base

FAMILY = "ferrier"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True

CARGO = (
    "brinesilk",
    "coalfern",
    "marrowkelp",
    "gladeamber",
    "frostpith",
    "wyrmflax",
    "shalenut",
    "tidegrain",
)
TOOL_NAMES = (
    "manifest",
    "berthing",
    "cranelift",
    "sealing",
    "gauging",
    "tariffing",
    "hoisting",
    "lading",
    "quayage",
    "weighdeck",
    "clerking",
    "ropework",
)
HANDLE_TYPES = (
    "docket",
    "gatepass",
    "sealmark",
    "voucher",
    "permit",
    "chit",
    "scrip",
    "tallycard",
    "crestmark",
    "berthslip",
    "quaytoken",
    "clearance",
)
_TOKEN_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

_LEVEL_SHAPE = {
    # level: (chain_length_choices, n_distractors, handle_arg_cap,
    #         extra_cargo_arg_prob, turn_cap)
    1: ((1,), 3, 1, 0.0, 4),
    2: ((3,), 2, 1, 0.0, 6),
    3: ((4, 5), 2, 2, 0.15, 10),
    4: ((6, 7), 1, 2, 0.25, 14),
}


# ---------------------------------------------------------------------------
# Spec construction (shared by atoms and episodes)
# ---------------------------------------------------------------------------


def _build_spec(rng, level: int) -> dict:
    chain_choices, n_distractors, cap, cargo_prob, _ = _LEVEL_SHAPE[level]
    chain_len = rng.choice(chain_choices)
    n_tools = chain_len + n_distractors
    names = rng.sample(TOOL_NAMES, n_tools)
    types = rng.sample(HANDLE_TYPES, n_tools)
    cargo = rng.choice(CARGO)
    chain_names = names[:chain_len]
    chain_types = types[:chain_len]

    # Wire the chain DAG: every chain tool's output is consumed by exactly one
    # later chain tool (a random in-tree converging on the goal tool), so the
    # goal's dependency closure is exactly the chain. Assign consumers
    # backward; capacity math guarantees a free slot always exists.
    counts = [0] * (chain_len + 1)  # 1-indexed by chain position
    consumed_by: list[list[int]] = [[] for _ in range(chain_len + 1)]
    for j in range(chain_len - 1, 0, -1):
        choices = [i for i in range(j + 1, chain_len + 1) if counts[i] < cap]
        i = rng.choice(choices)
        counts[i] += 1
        consumed_by[i].append(j)

    tools: dict[str, dict] = {}
    for i in range(1, chain_len + 1):
        producers = sorted(consumed_by[i])
        args = [chain_types[p - 1] for p in producers]
        if not args:
            args = ["cargo"]
        elif len(args) < 2 and rng.random() < cargo_prob:
            args.append("cargo")
            if rng.random() < 0.5:
                args.reverse()
        tools[chain_names[i - 1]] = {"args": args, "out": chain_types[i - 1]}

    for d in range(n_distractors):
        name = names[chain_len + d]
        pool = ["cargo"] + chain_types
        n_args = rng.randint(1, min(2, len(pool)))
        tools[name] = {
            "args": rng.sample(pool, n_args),
            "out": types[chain_len + d],
        }

    # One opaque token per result type, pre-issued (correct calls are
    # idempotent: same tool -> same token).
    used: set[str] = set()
    tokens: dict[str, str] = {}
    for out_type in types[:n_tools]:
        prefix = out_type[:2].upper()
        while True:
            tok = prefix + "-" + "".join(rng.choice(_TOKEN_ALPHABET) for _ in range(4))
            if tok not in used:
                used.add(tok)
                break
        tokens[out_type] = tok

    display_order = list(names)
    rng.shuffle(display_order)
    return {
        "cargo": cargo,
        "goal_type": chain_types[-1],
        "goal_tool": chain_names[-1],
        "chain": chain_names,
        "tools": tools,
        "tokens": tokens,
        "display_order": display_order,
        "chain_len": chain_len,
    }


def _arg_values(spec: dict, args: list[str]) -> list[str]:
    return [spec["cargo"] if a == "cargo" else spec["tokens"][a] for a in args]


def _render_call(spec: dict, name: str) -> str:
    vals = _arg_values(spec, spec["tools"][name]["args"])
    return f"{name}({', '.join(vals)})"


def _canon_call(text: str) -> str:
    text = text.strip().strip("`'\"")
    text = re.sub(r"^\s*call\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text).lower()
    return text.rstrip(".").strip("`'\"")


def _ready_chain(spec: dict, called: set[str]) -> list[str]:
    """Uncalled chain tools whose arguments are all available now.

    The chain is the goal's full dependency closure, so every ready uncalled
    chain tool lies on a shortest completion.
    """
    producer = {spec["tools"][n]["out"]: n for n in spec["chain"]}
    ready = []
    for name in spec["chain"]:
        if name in called:
            continue
        sig = spec["tools"][name]
        if all(a == "cargo" or producer[a] in called for a in sig["args"]):
            ready.append(name)
    return ready


def _signature_lines(spec: dict) -> list[str]:
    return [
        f"- {name}({', '.join(spec['tools'][name]['args'])}) -> {spec['tools'][name]['out']}"
        for name in spec["display_order"]
    ]


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        for attempt in range(20):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, "atom", seed, level, index, attempt)
    spec = _build_spec(rng, level)
    chain_len = spec["chain_len"]

    k = rng.randint(0, chain_len - 1) if chain_len > 1 else 0
    called: set[str] = set()
    transcript: list[str] = []
    for step in range(k):
        name = rng.choice(_ready_chain(spec, called))
        out = spec["tools"][name]["out"]
        transcript.append(
            f"{step + 1}. {_render_call(spec, name)} -> {out} {spec['tokens'][out]}"
        )
        called.add(name)

    valid_names = _ready_chain(spec, called)
    valid = sorted(_canon_call(_render_call(spec, name)) for name in valid_names)
    display = _render_call(spec, valid_names[0])

    lines = ["Wharf paperwork run. Tool signatures (call -> result type):"]
    lines += _signature_lines(spec)
    lines += [
        "",
        f"Goal: obtain a {spec['goal_type']} for the {spec['cargo']} cargo.",
        "Each correct call returns an opaque handle token of its result type;",
        "later calls take those tokens as arguments. Where a signature says",
        "cargo, pass the literal cargo name.",
        "",
    ]
    if transcript:
        lines.append("Calls so far, with returned handles:")
        lines += transcript
    else:
        lines.append("Calls so far: (none)")
    lines += [
        "",
        "What is the next call toward the goal? Write it exactly as",
        "name(arg) or name(arg1, arg2), using the literal cargo name or",
        "handle tokens already returned.",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": "\n".join(lines),
        "gold": {"valid": valid, "display": display},
    }


def score_atom(item: dict, reply_text: str) -> float:
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    return 1.0 if _canon_call(answer) in set(item["gold"]["valid"]) else 0.0


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']['display']}"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(
    r"^\s*(?:call\s+)?([a-zA-Z_]\w*)\s*\(([^()]*)\)\s*\.?\s*$", re.IGNORECASE
)


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        self._spec = _build_spec(rng, level)
        turn_cap = _LEVEL_SHAPE[level][4]
        self.max_turns = min(self._spec["chain_len"] + 3, turn_cap)
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "max_turns": self.max_turns,
            **self._spec,
        }
        self._turns = 0
        self._solved = False
        self._done = False
        self._called_ok: set[str] = set()
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "You are a wharf paperwork runner. Call tools to produce the goal "
            "document.\n"
            "Rules: each correct call returns '-> <type> <TOKEN>'; later calls "
            "consume tokens returned by earlier calls; where a signature says "
            "cargo, pass the literal cargo name. Bad calls waste the turn. The "
            "episode ends when the goal document is issued or turns run out.\n"
            "Action grammar: CALL name(arg1) | CALL name(arg1, arg2)\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        spec = self._spec
        lines = ["Tools:"] + _signature_lines(spec)
        lines.append(
            f"Goal: obtain a {spec['goal_type']} for the {spec['cargo']} cargo. "
            f"Turn budget: {self.max_turns}."
        )
        return "\n".join(lines)

    def _finish(self, observation: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            self._done = True
        return observation, self._done

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        spec = self._spec

        match = _ACTION_RE.match(base.extract_action(action_line or ""))
        if not match:
            return self._finish(
                "Bad action. Use exactly: CALL name(arg1) or CALL name(arg1, arg2)."
            )
        name = match.group(1).lower()
        raw = match.group(2).strip()
        args = [a.strip().strip("`'\"") for a in raw.split(",")] if raw else []

        if name not in spec["tools"]:
            return self._finish(f"No tool named '{name}'. Check the signatures.")
        sig = spec["tools"][name]
        if len(args) != len(sig["args"]):
            want = ", ".join(sig["args"])
            return self._finish(
                f"{name} takes {len(sig['args'])} argument(s): {name}({want})."
            )

        token_types = {tok.lower(): t for t, tok in spec["tokens"].items()}
        issued = {spec["tokens"][spec["tools"][n]["out"]].lower() for n in self._called_ok}
        got_types = []
        for arg in args:
            low = arg.lower()
            if low == spec["cargo"].lower():
                got_types.append("cargo")
            elif low in token_types and low in issued:
                got_types.append(token_types[low])
            else:
                return self._finish(
                    f"Unknown argument '{arg}'. Pass the literal cargo name or "
                    "a handle token already returned."
                )
        if got_types != sig["args"]:
            return self._finish(
                f"{name} needs ({', '.join(sig['args'])}) but got "
                f"({', '.join(got_types)})."
            )

        self.last_action_ok = True
        self._called_ok.add(name)
        out = sig["out"]
        token = spec["tokens"][out]
        if name == spec["goal_tool"]:
            self._solved = True
            return self._finish(f"-> {out} {token} (goal document produced)")
        return self._finish(f"-> {out} {token}")

    def score(self) -> float:
        if self._solved:
            return 1.0
        on_chain = sum(1 for n in self._spec["chain"] if n in self._called_ok)
        return on_chain / self._spec["chain_len"]


class OraclePolicy:
    """Executes the dependency chain in topological order, reusing tokens
    parsed from prior observations."""

    _TOKEN_RE = re.compile(r"->\s*([a-z]\w*)\s+([A-Z0-9]{2}-[A-Z0-9]{4})")

    def __init__(self, episode: Episode):
        self._chain = list(episode.spec["chain"])
        self._tools = episode.spec["tools"]
        self._cargo = episode.spec["cargo"]
        self._next = 0

    def act(self, observation_history: list[str]) -> str:
        seen: dict[str, str] = {}
        for obs in observation_history:
            for handle_type, token in self._TOKEN_RE.findall(obs):
                seen[handle_type] = token
        name = self._chain[min(self._next, len(self._chain) - 1)]
        self._next += 1
        vals = [
            self._cargo if a == "cargo" else seen[a]
            for a in self._tools[name]["args"]
        ]
        return f"CALL {name}({', '.join(vals)})"


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
    }
