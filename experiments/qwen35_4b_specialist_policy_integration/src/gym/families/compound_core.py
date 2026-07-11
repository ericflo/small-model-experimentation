"""Shared procedural mechanics for the specialist-integration compound gym.

The four public family modules below are deliberately thin wrappers around
these dependency-free environments.  Every task is generated from a seed,
admits an exact programmatic oracle, exposes only visible observations to the
model, and has explicit ablation policies that remove one required primitive.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any

from .. import base


PROTOCOL_STEPS = (
    "temper",
    "glaze",
    "quench",
    "anneal",
    "polish",
    "seal",
    "etch",
    "cool",
    "buff",
    "stamp",
)
PROTOCOL_CODES = ("zur", "pel", "nav", "keb", "fim", "wox", "dun", "ral", "teg", "hos")

MAZE_ROOMS = (
    "amber den",
    "cobalt nook",
    "ivory vault",
    "sable bay",
    "copper cell",
    "jade loft",
    "pearl hall",
    "ochre cave",
    "silver bend",
    "umber room",
)
MAZE_TOOLS = ("tagger", "binder", "notary", "press", "gauger", "folder", "marker", "issuer")
MAZE_TYPES = ("tag", "bundle", "note", "seal", "gauge", "folio", "mark", "grant")

PATCH_TOOLS = ("dovetail", "lacquer", "caliper", "spindlebox", "riveter", "platen", "finisher", "waxer")
PATCH_TYPES = ("stub", "lamina", "measure", "bobbin", "joint", "plate", "finish", "badge")

TRIPLE_TOOLS = ("miller", "caster", "planer", "borer", "joiner", "coater", "grader", "packer", "sender", "closer")
TRIPLE_TYPES = ("meal", "ingot", "plank", "bore", "joint", "coat", "grade", "pack", "send", "close")
TRIPLE_CODES = ("qir", "bex", "lum", "sov", "dax", "pir", "ken", "vut", "rag", "mep")

DIRECTIONS = ("north", "east", "south", "west", "up", "down")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east", "up": "down", "down": "up"}

_PROBE_RE = re.compile(r"^\s*PROBE\s+([a-z][a-z0-9 _-]*)\s*$", re.IGNORECASE)
_DO_RE = re.compile(r"^\s*DO\s+([a-z]+)\s*$", re.IGNORECASE)
_GO_RE = re.compile(r"^\s*GO\s+([a-z]+)\s*$", re.IGNORECASE)
_CALL_RE = re.compile(r"^\s*CALL\s+([a-z]+)\s*\(\s*([^()]*)\s*\)\s*$", re.IGNORECASE)
_PATCH_RE = re.compile(r"^\s*PATCH\s+([a-z]+)\s*\(\s*([a-z]+)\s*\)\s*$", re.IGNORECASE)


def _short_name(text: str) -> str:
    return " ".join(text.lower().split())


def _bfs(ports: dict[str, dict[str, str]], start: str, goal: str) -> list[str]:
    if start == goal:
        return []
    queue = deque([start])
    previous: dict[str, tuple[str, str]] = {}
    seen = {start}
    while queue:
        node = queue.popleft()
        for direction, nxt in ports[node].items():
            if nxt in seen:
                continue
            seen.add(nxt)
            previous[nxt] = (node, direction)
            if nxt == goal:
                path: list[str] = []
                cur = goal
                while cur != start:
                    cur, step = previous[cur]
                    path.append(step)
                return list(reversed(path))
            queue.append(nxt)
    raise AssertionError("connected graph lost a route")


def _tree(rng: Any, names: list[str]) -> dict[str, dict[str, str]]:
    """Build a connected direction-labelled tree with unique local ports."""
    for _ in range(100):
        ports = {name: {} for name in names}
        ok = True
        for index in range(1, len(names)):
            child = names[index]
            candidates = [name for name in names[:index] if len(ports[name]) < len(DIRECTIONS)]
            rng.shuffle(candidates)
            placed = False
            for parent in candidates:
                choices = [d for d in DIRECTIONS if d not in ports[parent] and OPPOSITE[d] not in ports[child]]
                if not choices:
                    continue
                direction = rng.choice(choices)
                ports[parent][direction] = child
                ports[child][OPPOSITE[direction]] = parent
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return ports
    raise RuntimeError("could not construct a direction-labelled tree")


class CipherProtocolEpisode:
    """Infer a cyclic code mapping, then execute a documented dependency chain."""

    def __init__(self, family: str, seed: int, level: int):
        rng = base.rng_for(family, "episode", seed, level)
        n = 5 + level
        steps = list(rng.sample(PROTOCOL_STEPS, n))
        codes = list(rng.sample(PROTOCOL_CODES, n))
        legal_order = list(steps)
        rng.shuffle(legal_order)
        offset = rng.randrange(1, n)  # zero-shift shortcut is impossible
        step_index = {name: i for i, name in enumerate(steps)}
        code_for = {name: codes[(step_index[name] + offset) % n] for name in steps}
        prerequisite = {legal_order[0]: None}
        prerequisite.update({legal_order[i]: legal_order[i - 1] for i in range(1, n)})
        doc_order = list(steps)
        rng.shuffle(doc_order)
        self.max_turns = n + 3
        self.spec = {
            "family": family,
            "seed": seed,
            "level": level,
            "max_turns": self.max_turns,
            "steps": steps,
            "codes": codes,
            "offset": offset,
            "code_for": code_for,
            "legal_order": legal_order,
            "prerequisite": prerequisite,
            "doc_order": doc_order,
        }
        self._turns = 0
        self._done = False
        self._solved = False
        self._probed = False
        self._completed: list[str] = []
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "Operate a coded finishing protocol. The step-to-code mapping is one cyclic offset between the published lists. "
            "Use PROBE step once to reveal a pair, infer the offset, then use DO code only when that step's prerequisite is complete. "
            "Bad or premature actions consume a turn.\nAction grammar: PROBE step | DO code\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        lines = [
            "Step index: " + ", ".join(f"{i}:{name}" for i, name in enumerate(self.spec["steps"])),
            "Code cycle: " + ", ".join(f"{i}:{code}" for i, code in enumerate(self.spec["codes"])),
            "Requirements:",
        ]
        for name in self.spec["doc_order"]:
            req = self.spec["prerequisite"][name]
            lines.append(f"- {name}: START" if req is None else f"- {name}: after {req}")
        lines.append(f"Complete every step. Turn budget: {self.max_turns}.")
        return "\n".join(lines)

    def _finish(self, text: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            self._done = True
        return text, self._done

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        action = base.extract_action(action_line or "")
        probe = _PROBE_RE.match(action)
        if probe:
            name = _short_name(probe.group(1))
            canonical = next((x for x in self.spec["steps"] if x.lower() == name), None)
            if canonical is None:
                return self._finish("Unknown protocol step.")
            if self._probed:
                return self._finish("The single diagnostic probe was already used.")
            self._probed = True
            self.last_action_ok = True
            return self._finish(f"Diagnostic: {canonical} uses code {self.spec['code_for'][canonical]}.")
        do = _DO_RE.match(action)
        if not do:
            return self._finish("Bad action. Use exactly PROBE step or DO code.")
        code = do.group(1).lower()
        decoded = next((name for name, value in self.spec["code_for"].items() if value == code), None)
        if decoded is None:
            return self._finish("Unknown code.")
        req = self.spec["prerequisite"][decoded]
        if decoded in self._completed:
            return self._finish(f"{decoded} was already completed.")
        if req is not None and req not in self._completed:
            return self._finish(f"{decoded} is premature; its prerequisite is incomplete.")
        self._completed.append(decoded)
        self.last_action_ok = True
        self._solved = len(self._completed) == len(self.spec["steps"])
        return self._finish(
            f"Accepted {decoded}. Completed {len(self._completed)}/{len(self.spec['steps'])}."
            + (" Protocol complete." if self._solved else "")
        )

    def score(self) -> float:
        return len(self._completed) / len(self.spec["steps"])


class MazeToolEpisode:
    """Explore a partial map to find and execute a typed tool chain."""

    def __init__(self, family: str, seed: int, level: int):
        rng = base.rng_for(family, "episode", seed, level)
        chain_len = 2 + level
        n_rooms = chain_len + 2
        rooms = list(rng.sample(MAZE_ROOMS, n_rooms))
        ports = _tree(rng, rooms)
        start = rooms[0]
        tools = list(rng.sample(MAZE_TOOLS, chain_len))
        types = list(rng.sample(MAZE_TYPES, chain_len))
        locations = dict(zip(tools, rng.sample(rooms[1:], chain_len)))
        cargo = rng.choice(("cinder", "vellum", "brass", "linen"))
        signatures = {}
        for i, name in enumerate(tools):
            signatures[name] = {"arg": "cargo" if i == 0 else types[i - 1], "out": types[i]}
        tokens = {typ: f"{typ[:2].upper()}-{rng.randrange(1000, 9999)}" for typ in types}
        oracle_moves = 0
        at = start
        for name in tools:
            oracle_moves += len(_bfs(ports, at, locations[name]))
            at = locations[name]
        self.max_turns = oracle_moves + chain_len + 4
        self.spec = {
            "family": family,
            "seed": seed,
            "level": level,
            "max_turns": self.max_turns,
            "rooms": rooms,
            "ports": ports,
            "start": start,
            "tools": tools,
            "types": types,
            "locations": locations,
            "cargo": cargo,
            "signatures": signatures,
            "tokens": tokens,
        }
        self._turns = 0
        self._done = False
        self._solved = False
        self._current = start
        self._called: list[str] = []
        self._issued: dict[str, str] = {}
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "Explore the rooms and execute the typed tool chain. Tool locations are revealed only when you enter a room. "
            "GO follows a visible exit; CALL must name the tool in the current room and pass the literal cargo or a returned handle.\n"
            "Action grammar: GO direction | CALL name(argument)\n" + base.EPISODE_ACTION_INSTRUCTION
        )

    def _room_observation(self) -> str:
        exits = ", ".join(sorted(self.spec["ports"][self._current]))
        here = next((name for name, room in self.spec["locations"].items() if room == self._current), None)
        tool = "No tool here."
        if here is not None:
            sig = self.spec["signatures"][here]
            tool = f"Tool here: {here}({sig['arg']}) -> {sig['out']}."
        return f"Room: {self._current}. Exits: {exits}. {tool}"

    def initial_observation(self) -> str:
        lines = ["Global chain signatures:"]
        for name in self.spec["tools"]:
            sig = self.spec["signatures"][name]
            lines.append(f"- {name}({sig['arg']}) -> {sig['out']}")
        lines.append(f"Goal: obtain {self.spec['types'][-1]} for cargo {self.spec['cargo']}. Turn budget: {self.max_turns}.")
        lines.append(self._room_observation())
        return "\n".join(lines)

    def _finish(self, text: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            self._done = True
        return text, self._done

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        action = base.extract_action(action_line or "")
        go = _GO_RE.match(action)
        if go:
            direction = go.group(1).lower()
            nxt = self.spec["ports"][self._current].get(direction)
            if nxt is None:
                return self._finish("No corridor in that direction. " + self._room_observation())
            self._current = nxt
            self.last_action_ok = True
            return self._finish(self._room_observation())
        call = _CALL_RE.match(action)
        if not call:
            return self._finish("Bad action. Use exactly GO direction or CALL name(argument).")
        name = call.group(1).lower()
        arg = call.group(2).strip().strip("`'\"")
        expected_name = self.spec["tools"][len(self._called)] if len(self._called) < len(self.spec["tools"]) else None
        if name != expected_name:
            return self._finish("That tool is not the next dependency in the chain. " + self._room_observation())
        if self.spec["locations"][name] != self._current:
            return self._finish("That tool is not in this room. " + self._room_observation())
        sig = self.spec["signatures"][name]
        expected_arg = self.spec["cargo"] if sig["arg"] == "cargo" else self._issued.get(sig["arg"], "")
        if arg.lower() != expected_arg.lower():
            return self._finish(f"Wrong argument for {name}. " + self._room_observation())
        self._called.append(name)
        token = self.spec["tokens"][sig["out"]]
        self._issued[sig["out"]] = token
        self.last_action_ok = True
        self._solved = len(self._called) == len(self.spec["tools"])
        return self._finish(
            f"-> {sig['out']} {token}." + (" Goal artifact produced." if self._solved else " ") + self._room_observation()
        )

    def score(self) -> float:
        return len(self._called) / len(self.spec["tools"])


class PatchToolEpisode:
    """Repair one corrupted type signature, then execute the chain."""

    def __init__(self, family: str, seed: int, level: int):
        rng = base.rng_for(family, "episode", seed, level)
        n = 3 + level
        tools = list(rng.sample(PATCH_TOOLS, n))
        types = list(rng.sample(PATCH_TYPES, n))
        cargo = rng.choice(("felt", "alloy", "paper", "resin"))
        true_args = ["cargo"] + types[:-1]
        bug_index = rng.randrange(1, n)
        wrong_pool = ["cargo"] + types
        wrong = rng.choice([x for x in wrong_pool if x != true_args[bug_index]])
        shown_args = list(true_args)
        shown_args[bug_index] = wrong
        tokens = {typ: f"{typ[:2].upper()}-{rng.randrange(1000, 9999)}" for typ in types}
        self.max_turns = n + 3
        self.spec = {
            "family": family,
            "seed": seed,
            "level": level,
            "max_turns": self.max_turns,
            "tools": tools,
            "types": types,
            "cargo": cargo,
            "true_args": true_args,
            "shown_args": shown_args,
            "bug_index": bug_index,
            "tokens": tokens,
        }
        self._turns = 0
        self._done = False
        self._solved = False
        self._patched = False
        self._called: list[str] = []
        self._issued: dict[str, str] = {}
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "Exactly one displayed input type was corrupted. Infer the unique chain from output types, repair it with PATCH, then call the tools in dependency order. "
            "Calls consume the literal cargo or a returned handle.\nAction grammar: PATCH name(correct_type) | CALL name(argument)\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        lines = ["Displayed registry (one input type is wrong):"]
        order = list(range(len(self.spec["tools"])))
        for i in order:
            lines.append(f"- {self.spec['tools'][i]}({self.spec['shown_args'][i]}) -> {self.spec['types'][i]}")
        lines.append(f"Goal: obtain {self.spec['types'][-1]} for cargo {self.spec['cargo']}. Turn budget: {self.max_turns}.")
        return "\n".join(lines)

    def _finish(self, text: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            self._done = True
        return text, self._done

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        action = base.extract_action(action_line or "")
        patch = _PATCH_RE.match(action)
        if patch:
            name, arg_type = patch.group(1).lower(), patch.group(2).lower()
            idx = self.spec["bug_index"]
            if name != self.spec["tools"][idx] or arg_type != self.spec["true_args"][idx]:
                return self._finish("That patch does not restore the unique type chain.")
            if self._patched:
                return self._finish("The corrupted signature was already repaired.")
            self._patched = True
            self.spec["shown_args"][idx] = arg_type
            self.last_action_ok = True
            return self._finish(f"Signature repaired: {name}({arg_type}) -> {self.spec['types'][idx]}.")
        call = _CALL_RE.match(action)
        if not call:
            return self._finish("Bad action. Use exactly PATCH name(type) or CALL name(argument).")
        index = len(self._called)
        if index >= len(self.spec["tools"]):
            return self._finish("Chain already complete.")
        name = call.group(1).lower()
        arg = call.group(2).strip().strip("`'\"")
        if name != self.spec["tools"][index]:
            return self._finish("That is not the next tool in the type chain.")
        if index == self.spec["bug_index"] and not self._patched:
            return self._finish("The next tool's displayed signature is corrupted; repair it first.")
        arg_type = self.spec["true_args"][index]
        expected = self.spec["cargo"] if arg_type == "cargo" else self._issued.get(arg_type, "")
        if arg.lower() != expected.lower():
            return self._finish(f"Wrong argument for {name}.")
        self._called.append(name)
        out = self.spec["types"][index]
        token = self.spec["tokens"][out]
        self._issued[out] = token
        self.last_action_ok = True
        self._solved = len(self._called) == len(self.spec["tools"])
        return self._finish(f"-> {out} {token}." + (" Goal artifact produced." if self._solved else ""))

    def score(self) -> float:
        return len(self._called) / len(self.spec["tools"])


class CodedToolEpisode:
    """Infer coded tool names, then execute a typed dependency order."""

    def __init__(self, family: str, seed: int, level: int):
        rng = base.rng_for(family, "episode", seed, level)
        n = 5 + level
        tools = list(rng.sample(TRIPLE_TOOLS, n))
        codes = list(rng.sample(TRIPLE_CODES, n))
        order = list(tools)
        rng.shuffle(order)
        types = list(rng.sample(TRIPLE_TYPES, n))
        out_for = dict(zip(order, types))
        true_arg: dict[str, str] = {order[0]: "cargo"}
        for i in range(1, n):
            true_arg[order[i]] = out_for[order[i - 1]]
        offset = rng.randrange(1, n)
        tool_index = {name: i for i, name in enumerate(tools)}
        code_for = {name: codes[(tool_index[name] + offset) % n] for name in tools}
        cargo = rng.choice(("quartz", "canvas", "timber", "clay"))
        tokens = {typ: f"{typ[:2].upper()}-{rng.randrange(1000, 9999)}" for typ in types}
        display = list(tools)
        rng.shuffle(display)
        self.max_turns = n + 3
        self.spec = {
            "family": family,
            "seed": seed,
            "level": level,
            "max_turns": self.max_turns,
            "tools": tools,
            "codes": codes,
            "order": order,
            "types": types,
            "out_for": out_for,
            "true_arg": true_arg,
            "offset": offset,
            "code_for": code_for,
            "cargo": cargo,
            "tokens": tokens,
            "display": display,
        }
        self._turns = 0
        self._done = False
        self._solved = False
        self._probed = False
        self._called: list[str] = []
        self._issued: dict[str, str] = {}
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "Tool names are encoded by one cyclic offset between the published tool index and code cycle. Use one PROBE to infer the mapping, then CALL code(argument) in typed dependency order. "
            "Bad codes, unavailable handles, and premature calls consume a turn.\nAction grammar: PROBE tool | CALL code(argument)\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        lines = [
            "Tool index: " + ", ".join(f"{i}:{name}" for i, name in enumerate(self.spec["tools"])),
            "Code cycle: " + ", ".join(f"{i}:{code}" for i, code in enumerate(self.spec["codes"])),
            "Typed signatures:",
        ]
        for name in self.spec["display"]:
            lines.append(f"- {name}({self.spec['true_arg'][name]}) -> {self.spec['out_for'][name]}")
        lines.append(f"Goal: obtain {self.spec['out_for'][self.spec['order'][-1]]} for cargo {self.spec['cargo']}. Turn budget: {self.max_turns}.")
        return "\n".join(lines)

    def _finish(self, text: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            self._done = True
        return text, self._done

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        action = base.extract_action(action_line or "")
        probe = _PROBE_RE.match(action)
        if probe:
            name = _short_name(probe.group(1))
            canonical = next((x for x in self.spec["tools"] if x.lower() == name), None)
            if canonical is None:
                return self._finish("Unknown tool name.")
            if self._probed:
                return self._finish("The single diagnostic probe was already used.")
            self._probed = True
            self.last_action_ok = True
            return self._finish(f"Diagnostic: {canonical} uses code {self.spec['code_for'][canonical]}.")
        call = _CALL_RE.match(action)
        if not call:
            return self._finish("Bad action. Use exactly PROBE tool or CALL code(argument).")
        code = call.group(1).lower()
        arg = call.group(2).strip().strip("`'\"")
        decoded = next((name for name, value in self.spec["code_for"].items() if value == code), None)
        index = len(self._called)
        expected_name = self.spec["order"][index] if index < len(self.spec["order"]) else None
        if decoded != expected_name:
            return self._finish("That coded tool is not the next available dependency.")
        arg_type = self.spec["true_arg"][decoded]
        expected_arg = self.spec["cargo"] if arg_type == "cargo" else self._issued.get(arg_type, "")
        if arg.lower() != expected_arg.lower():
            return self._finish(f"Wrong or unavailable argument for coded tool {code}.")
        self._called.append(decoded)
        out = self.spec["out_for"][decoded]
        token = self.spec["tokens"][out]
        self._issued[out] = token
        self.last_action_ok = True
        self._solved = len(self._called) == len(self.spec["order"])
        return self._finish(f"-> {out} {token}." + (" Goal artifact produced." if self._solved else ""))

    def score(self) -> float:
        return len(self._called) / len(self.spec["order"])


class OraclePolicy:
    def __init__(self, episode: Any):
        self.episode = episode

    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, CipherProtocolEpisode):
            if not ep._probed:
                return f"PROBE {ep.spec['steps'][0]}"
            next_step = next(name for name in ep.spec["legal_order"] if name not in ep._completed)
            return f"DO {ep.spec['code_for'][next_step]}"
        if isinstance(ep, MazeToolEpisode):
            name = ep.spec["tools"][len(ep._called)]
            room = ep.spec["locations"][name]
            if ep._current != room:
                return f"GO {_bfs(ep.spec['ports'], ep._current, room)[0]}"
            sig = ep.spec["signatures"][name]
            arg = ep.spec["cargo"] if sig["arg"] == "cargo" else ep._issued[sig["arg"]]
            return f"CALL {name}({arg})"
        if isinstance(ep, PatchToolEpisode):
            if not ep._patched:
                idx = ep.spec["bug_index"]
                return f"PATCH {ep.spec['tools'][idx]}({ep.spec['true_args'][idx]})"
            idx = len(ep._called)
            name = ep.spec["tools"][idx]
            arg_type = ep.spec["true_args"][idx]
            arg = ep.spec["cargo"] if arg_type == "cargo" else ep._issued[arg_type]
            return f"CALL {name}({arg})"
        if isinstance(ep, CodedToolEpisode):
            if not ep._probed:
                return f"PROBE {ep.spec['tools'][0]}"
            name = ep.spec["order"][len(ep._called)]
            arg_type = ep.spec["true_arg"][name]
            arg = ep.spec["cargo"] if arg_type == "cargo" else ep._issued[arg_type]
            return f"CALL {ep.spec['code_for'][name]}({arg})"
        raise TypeError(type(ep))


class NoDiscoveryPolicy(OraclePolicy):
    """Use all hidden task structure except the learned cyclic offset."""

    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, CipherProtocolEpisode):
            next_step = next(name for name in ep.spec["legal_order"] if name not in ep._completed)
            index = ep.spec["steps"].index(next_step)
            return f"DO {ep.spec['codes'][index]}"
        if isinstance(ep, CodedToolEpisode):
            name = ep.spec["order"][len(ep._called)]
            index = ep.spec["tools"].index(name)
            arg_type = ep.spec["true_arg"][name]
            arg = ep.spec["cargo"] if arg_type == "cargo" else ep._issued.get(arg_type, "missing")
            return f"CALL {ep.spec['codes'][index]}({arg})"
        return super().act(observation_history)


class NoControlPolicy(OraclePolicy):
    """Know the code map but ignore dependency order."""

    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, CipherProtocolEpisode):
            if not ep._probed:
                return f"PROBE {ep.spec['steps'][0]}"
            remaining = [name for name in ep.spec["steps"] if name not in ep._completed]
            return f"DO {ep.spec['code_for'][remaining[0]]}"
        if isinstance(ep, CodedToolEpisode):
            if not ep._probed:
                return f"PROBE {ep.spec['tools'][0]}"
            remaining = [name for name in ep.spec["tools"] if name not in ep._called]
            name = remaining[0]
            arg_type = ep.spec["true_arg"][name]
            arg = ep.spec["cargo"] if arg_type == "cargo" else ep._issued.get(arg_type, "missing")
            return f"CALL {ep.spec['code_for'][name]}({arg})"
        return super().act(observation_history)


class NoNavigationPolicy(OraclePolicy):
    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, MazeToolEpisode):
            name = ep.spec["tools"][len(ep._called)]
            sig = ep.spec["signatures"][name]
            arg = ep.spec["cargo"] if sig["arg"] == "cargo" else ep._issued.get(sig["arg"], "missing")
            return f"CALL {name}({arg})"
        return super().act(observation_history)


class NoToolsPolicy(OraclePolicy):
    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, MazeToolEpisode):
            name = ep.spec["tools"][len(ep._called)]
            room = ep.spec["locations"][name]
            if ep._current != room:
                return f"GO {_bfs(ep.spec['ports'], ep._current, room)[0]}"
            return "GO nowhere"
        if isinstance(ep, PatchToolEpisode):
            if not ep._patched:
                idx = ep.spec["bug_index"]
                return f"PATCH {ep.spec['tools'][idx]}({ep.spec['true_args'][idx]})"
            return "CALL nothing(missing)"
        if isinstance(ep, CodedToolEpisode):
            if not ep._probed:
                return f"PROBE {ep.spec['tools'][0]}"
            return "CALL nothing(missing)"
        return "XYZZY"


class NoRepairPolicy(OraclePolicy):
    def act(self, observation_history: list[str]) -> str:
        ep = self.episode
        if isinstance(ep, PatchToolEpisode):
            idx = len(ep._called)
            name = ep.spec["tools"][idx]
            arg_type = ep.spec["true_args"][idx]
            arg = ep.spec["cargo"] if arg_type == "cargo" else ep._issued.get(arg_type, "missing")
            return f"CALL {name}({arg})"
        return super().act(observation_history)


def _success_rate(module: Any, policy_class: type, n_per_level: int = 30) -> dict[int, float]:
    rates: dict[int, float] = {}
    for level in module.LEVELS:
        success = 0
        for index in range(n_per_level):
            episode = module.Episode(90000 + index, level)
            policy = policy_class(episode)
            history = [episode.initial_observation()]
            for _ in range(episode.max_turns):
                observation, done = episode.step(policy.act(history))
                history.append(observation)
                if done:
                    break
            success += episode.score() >= 0.999
        rates[level] = success / n_per_level
    return rates


def selftest_module(module: Any, ablations: tuple[type, ...]) -> dict[str, Any]:
    stats = {"episodes": base.selftest_episodes(module, n_per_level=16, random_max=0.15)}
    necessity = {}
    for policy in ablations:
        rates = _success_rate(module, policy)
        worst = max(rates.values())
        base._check(worst <= 0.20, f"{module.FAMILY}: {policy.__name__} success {worst:.3f} > 0.20")
        necessity[policy.__name__] = {str(level): round(value, 4) for level, value in rates.items()}
    stats["necessity"] = necessity
    return stats
