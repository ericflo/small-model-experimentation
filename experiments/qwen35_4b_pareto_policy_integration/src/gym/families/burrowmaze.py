"""burrowmaze — partially observable exploration with spatial memory.

An underground burrow network: chambers with invented names joined by
direction-labeled corridors (hidden graph = random spanning tree plus a few
extra tunnels; per corridor the way back is always the exact reverse
direction). Episodes show only the current chamber and its exit directions
each turn; the explorer must reach a named target chamber under a strict move
budget. Atoms ask memory questions (distinct-chamber counts, corridor
directions) and shortest-route questions over a written exploration log.

All content is invented; verification is an exact graph simulator. Episode
layouts are generated so the start-target distance fits the move budget with
slack ~2 (frontier levels: near-diameter targets with slack >= 3) and a
uniform random-direction walker succeeds rarely (exact hitting-probability
check at generation time).

Frontier levels (L5/L6) escalate to 20/26-chamber networks with horizons
18/22 and targets pushed toward the graph diameter (budget >= distance + 3),
plus longer logs and deeper route questions in atoms.
"""

from __future__ import annotations

import re
import sys

from .. import base

FAMILY = "burrowmaze"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = True

_OPP = {
    "north": "south",
    "south": "north",
    "east": "west",
    "west": "east",
    "up": "down",
    "down": "up",
}
_DIRS = ("north", "south", "east", "west", "up", "down")

CHAMBERS = (
    "Sootway",
    "Dripvault",
    "Amber Sump",
    "Fernwell",
    "Mosshollow",
    "Grubline",
    "Emberdeep",
    "Rootgall",
    "Tallowmere",
    "Quartzrun",
    "Palegrotto",
    "Bramblehole",
    "Cinderpit",
    "Loamgate",
    "Duskhollow",
    "Wormcourt",
    "Haldenook",
    "Mirebend",
    "Stonelap",
    "Gnarlden",
)

# Frontier chamber-name pool: the original 20 names with new invented names
# APPENDED. Only L5+ samples from it — random.sample's output depends on the
# population size, so L1-L4 must keep drawing from the original tuple to stay
# byte-identical across the frontier extension (longitudinal comparability).
FRONTIER_CHAMBERS = CHAMBERS + (
    "Veilscar",
    "Chalkreach",
    "Umberden",
    "Saltgleam",
    "Hushcavern",
    "Twinesump",
    "Marrowick",
    "Coldfen",
)


# Skin-shuffling: invented proper-noun lexemes that can be consistently
# renamed without changing mechanics (every chamber name, frontier pool
# included). The two-word chamber "Amber Sump" is listed as its two
# single-word halves so each entry matches as one whole word; renaming both
# renames the chamber. EXCLUDED: direction words (north/south/east/west/
# up/down) — they are the action grammar and the reverse-direction rule the
# verifier simulates — and the GO / ANSWER protocol words.
SKINNABLE: tuple[str, ...] = (
    "Sootway",
    "Dripvault",
    "Amber",
    "Sump",
    "Fernwell",
    "Mosshollow",
    "Grubline",
    "Emberdeep",
    "Rootgall",
    "Tallowmere",
    "Quartzrun",
    "Palegrotto",
    "Bramblehole",
    "Cinderpit",
    "Loamgate",
    "Duskhollow",
    "Wormcourt",
    "Haldenook",
    "Mirebend",
    "Stonelap",
    "Gnarlden",
    "Veilscar",
    "Chalkreach",
    "Umberden",
    "Saltgleam",
    "Hushcavern",
    "Twinesump",
    "Marrowick",
    "Coldfen",
)


def _name_pool(level: int) -> tuple:
    return CHAMBERS if level <= 4 else FRONTIER_CHAMBERS


_MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14, 5: 18, 6: 22}

# Episode profiles per level: n_chambers plus weighted layout profiles of
# (weight, extra_edge_choices, dist_lo, dist_hi, rwalk_cap). The cap bounds
# the EXACT hitting probability of a uniform random-direction walker within
# the move budget, enforced at generation time. L1's 4-move budget over 6
# chambers pins start-target distance at 2 (slack 2), where even the best
# sparse layouts leave a random walker ~0.20; the near profile therefore uses
# denser graphs (higher-degree chambers dilute the walker) and 40% of L1
# episodes use distance 3 to hold the family-wide random floor near 0.11.
_EP_CHAMBERS = {1: 6, 2: 8, 3: 12, 4: 16, 5: 20, 6: 26}
# Frontier profiles push the target toward the graph diameter: dist_hi is
# max_turns - 3, so the budget is always at least distance + 3 and the
# hardest layouts run at exactly diameter + 3. Sparse extra-edge choices keep
# long distances reachable; the rwalk cap holds the random floor down.
_EP_PROFILES = {
    1: (
        (0.6, (3, 4, 5), 2, 2, 0.14),
        (0.4, (0, 1), 3, 3, 0.10),
    ),
    2: ((1.0, (2,), 3, 4, 0.10),),
    3: ((1.0, (2, 3), 5, 8, 0.10),),
    4: ((1.0, (3, 4), 6, 12, 0.10),),
    5: ((1.0, (2, 3), 8, 15, 0.08),),
    6: ((1.0, (2, 3, 4), 9, 19, 0.08),),
}

# level: (n_chambers, n_extra_edges, walk_len, route_lo, route_hi)
_ATOM_SHAPE = {
    1: (6, 1, 4, 2, 2),
    2: (8, 2, 6, 2, 3),
    3: (12, 2, 9, 3, 4),
    4: (16, 3, 12, 3, 5),
    5: (20, 4, 16, 4, 6),
    6: (26, 5, 20, 5, 8),
}

# atom question mix per level: weights for (count, direction, route)
_KIND_WEIGHTS = {
    1: (40, 40, 20),
    2: (30, 35, 35),
    3: (25, 35, 40),
    4: (20, 35, 45),
    5: (15, 30, 55),
    6: (10, 30, 60),
}


# ---------------------------------------------------------------------------
# Hidden-graph machinery
# ---------------------------------------------------------------------------


def _build_ports(rng, names: list[str], n_extra: int) -> dict | None:
    """Random spanning tree + extra edges with direction-consistent ports.

    Returns {chamber: {direction: chamber}} where each corridor occupies one
    free direction port on each side and the reverse port is the opposite
    direction. Returns None if a node cannot be attached (regenerate).
    """
    ports: dict[str, dict[str, str]] = {name: {} for name in names}
    placed = [names[0]]
    for name in names[1:]:
        attached = False
        for _ in range(40):
            # bias toward the most recent chamber -> deeper, path-like trees
            parent = placed[-1] if rng.random() < 0.55 else rng.choice(placed)
            free = [d for d in _DIRS if d not in ports[parent]]
            if not free:
                continue
            direction = rng.choice(free)
            ports[parent][direction] = name
            ports[name][_OPP[direction]] = parent
            placed.append(name)
            attached = True
            break
        if not attached:
            return None
    added = 0
    for _ in range(30):
        if added >= n_extra:
            break
        a, b = rng.sample(names, 2)
        if b in ports[a].values():
            continue
        free = [d for d in _DIRS if d not in ports[a] and _OPP[d] not in ports[b]]
        if not free:
            continue
        direction = rng.choice(free)
        ports[a][direction] = b
        ports[b][_OPP[direction]] = a
        added += 1
    return ports


def _bfs(ports: dict, start: str) -> tuple[dict, dict]:
    """Breadth-first distances and predecessor (node, direction) links."""
    dist = {start: 0}
    prev: dict[str, tuple[str, str]] = {}
    frontier = [start]
    while frontier:
        nxt = []
        for node in frontier:
            for direction in _DIRS:
                dest = ports[node].get(direction)
                if dest is not None and dest not in dist:
                    dist[dest] = dist[node] + 1
                    prev[dest] = (node, direction)
                    nxt.append(dest)
        frontier = nxt
    return dist, prev


def _path_dirs(prev: dict, start: str, goal: str) -> list[str]:
    dirs: list[str] = []
    node = goal
    while node != start:
        parent, direction = prev[node]
        dirs.append(direction)
        node = parent
    return list(reversed(dirs))


def _hit_prob(ports: dict, start: str, target: str, budget: int) -> float:
    """Exact P(uniform random-direction walker reaches target within budget)."""
    prob = {name: 0.0 for name in ports}
    prob[start] = 1.0
    hit = 0.0
    for _ in range(budget):
        nxt = {name: 0.0 for name in ports}
        for name, mass in prob.items():
            if mass <= 0.0:
                continue
            exits = ports[name]
            share = mass / len(exits)
            for direction in _DIRS:
                dest = exits.get(direction)
                if dest is None:
                    continue
                if dest == target:
                    hit += share
                else:
                    nxt[dest] += share
        prob = nxt
    return hit


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def _walk(rng, ports: dict, start: str, length: int) -> list[tuple[str, str, str]]:
    """Exploration walk preferring unvisited chambers; returns (a, dir, b) steps."""
    steps: list[tuple[str, str, str]] = []
    visited = {start}
    current = start
    for _ in range(length):
        exits = ports[current]
        fresh = [d for d in _DIRS if d in exits and exits[d] not in visited]
        pool = fresh if fresh else [d for d in _DIRS if d in exits]
        direction = rng.choice(pool)
        dest = exits[direction]
        steps.append((current, direction, dest))
        visited.add(dest)
        current = dest
    return steps


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        item = None
        for attempt in range(40):
            item = _gen_one(seed, level, index, attempt)
            if item is not None and len(item["prompt"]) <= base.atom_prompt_limit(level):
                break
            item = None
        if item is None:  # pragma: no cover - generator bug
            raise RuntimeError(f"{FAMILY}: could not generate item {index} at L{level}")
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict | None:
    rng = base.rng_for(FAMILY, "atom", seed, level, index, attempt)
    n_chambers, n_extra, walk_len, route_lo, route_hi = _ATOM_SHAPE[level]
    names = list(rng.sample(_name_pool(level), n_chambers))
    ports = _build_ports(rng, names, n_extra)
    if ports is None:
        return None
    start = rng.choice(names)
    walk = _walk(rng, ports, start, walk_len)
    visited = [start]
    for _, _, dest in walk:
        if dest not in visited:
            visited.append(dest)

    kind = rng.choices(("count", "direction", "route"), weights=_KIND_WEIGHTS[level])[0]
    question = ""
    gold: dict = {}
    if kind == "route":
        # revealed subgraph: corridors named in the log, traversable both ways
        redges: list[list[str]] = []
        seen_pairs: set[frozenset] = set()
        radj: dict[str, dict[str, str]] = {name: {} for name in visited}
        for a, d, b in walk:
            pair = frozenset((a, b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            redges.append([a, d, b])
            radj[a][d] = b
            radj[b][_OPP[d]] = a
        pairs = [(u, v) for u in visited for v in visited if u != v]
        rng.shuffle(pairs)
        chosen = None
        for u, v in pairs:
            dist, prev = _bfs(radj, u)
            if route_lo <= dist.get(v, 999) <= route_hi:
                chosen = (u, v, dist[v], _path_dirs(prev, u, v))
                break
        if chosen is None:
            kind = "direction"  # rare fallback: no pair at the wanted distance
        else:
            src, dst, length, path = chosen
            question = (
                f"Using only corridors named in the log, what is the shortest "
                f"sequence of directions to go from {src} to {dst}?"
            )
            gold = {
                "kind": "route",
                "edges": redges,
                "src": src,
                "dst": dst,
                "length": length,
                "path": path,
            }
    if kind == "direction":
        a, d, b = walk[rng.randrange(len(walk))]
        if level >= 3 and rng.random() < 0.5:
            a, d, b = b, _OPP[d], a  # ask across the corridor; reverse rule applies
        question = f"From {a}, which single direction leads directly to {b}?"
        gold = {"kind": "direction", "value": d}
    elif kind == "count":
        question = "How many distinct chambers are named in the log?"
        gold = {"kind": "count", "value": len(visited)}

    lines = [
        "A scout kept a log while exploring an underground burrow network of",
        "chambers joined by corridors. Corridors run both ways: the return",
        "direction is always the exact reverse (north-south, east-west, up-down).",
        "",
        "Exploration log:",
    ]
    lines += [f"{i + 1}. From {a} we went {d} into {b}." for i, (a, d, b) in enumerate(walk)]
    lines += ["", question]
    if gold["kind"] == "route":
        lines.append(
            "Give the directions in order, separated by commas "
            "(for example: ANSWER: north, east)."
        )
    lines += ["", base.ATOM_ANSWER_INSTRUCTION]
    prompt = "\n".join(lines)

    # Lucky-guess guard: size of the plausible answer space for this item.
    if gold["kind"] == "route":
        answer_domain = 50  # ordered direction sequences: a wide space
    elif gold["kind"] == "count":
        answer_domain = n_chambers  # counts range over the hidden graph's size
    else:  # direction: the log's direction vocabulary, reverses included
        seen_dirs = {d for _, d, _ in walk}
        answer_domain = len(seen_dirs | {_OPP[d] for d in seen_dirs})

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        "answer_domain": answer_domain,
    }


def score_atom(item: dict, reply_text: str) -> float:
    gold = item["gold"]
    if gold["kind"] == "count":
        return base.score_exact_int(gold["value"], reply_text)
    if gold["kind"] == "direction":
        return base.score_exact_word(gold["value"], reply_text)
    # route: accept ANY valid path of the shortest length through the
    # revealed subgraph (corridors traversable both ways)
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    steps = [tok for tok in base.canon_list(answer) if tok in _OPP]
    if len(steps) != gold["length"]:
        return 0.0
    adj: dict[str, dict[str, str]] = {}
    for a, d, b in gold["edges"]:
        adj.setdefault(a, {})[d] = b
        adj.setdefault(b, {})[_OPP[d]] = a
    node = gold["src"]
    for direction in steps:
        dest = adj.get(node, {}).get(direction)
        if dest is None:
            return 0.0
        node = dest
    return 1.0 if node == gold["dst"] else 0.0


def oracle_atom(item: dict) -> str:
    gold = item["gold"]
    if gold["kind"] == "route":
        return "ANSWER: " + ", ".join(gold["path"])
    return f"ANSWER: {gold['value']}"


# ---------------------------------------------------------------------------
# Oracle trace: think-channel narration of the hand-coded solving procedure.
# Truth-blind: everything is re-derived from the prompt's exploration log
# (map rebuilt edge by edge, then counted / read off / breadth-first searched)
# rather than read from the gold answer.
# ---------------------------------------------------------------------------

_LOG_LINE_RE = re.compile(r"^\d+\.\s+From (.+?) we went ([a-z]+) into (.+?)\.\s*$")
_DIRECTION_Q_RE = re.compile(
    r"^From (.+?), which single direction leads directly to (.+?)\?\s*$"
)


def _parse_log(prompt: str) -> list[tuple[str, str, str]]:
    """Recover the (chamber, direction, chamber) steps from the prompt's log."""
    steps: list[tuple[str, str, str]] = []
    for raw in prompt.splitlines():
        match = _LOG_LINE_RE.match(raw.strip())
        if match and match.group(2) in _OPP:
            steps.append((match.group(1), match.group(2), match.group(3)))
    return steps


def oracle_trace(item: dict) -> str:
    rng = base.rng_for(FAMILY, "trace", item["id"])
    voice = rng.randrange(3)
    steps = _parse_log(item["prompt"])
    kind = item["gold"]["kind"]
    if kind == "count":
        return _trace_count(voice, steps)
    if kind == "direction":
        return _trace_direction(voice, steps, item["prompt"])
    return _trace_route(voice, steps, item["gold"]["src"], item["gold"]["dst"])


def _trace_count(voice: int, steps: list[tuple[str, str, str]]) -> str:
    start = steps[0][0]
    lines = [
        (
            "I need to count how many distinct chambers this log names. "
            "I'll replay the trek entry by entry, keeping a running list and "
            "counting each chamber only the first time it appears.",
            "The question is how many distinct chambers are named in the log. "
            "So let me walk the entries in order, adding each chamber to my "
            "list the first time it comes up and skipping repeats.",
            "So I have to work out how many different chambers this log "
            "mentions. I'll go through it one entry at a time and grow a list, "
            "counting a chamber only when it's new.",
        )[voice],
        (
            f"The trek starts in {start}, so that's one chamber before any move.",
            f"Before any move we're already standing in {start} — chamber number one.",
            f"{start} is where the trek begins, so my list opens with it: 1 so far.",
        )[voice],
    ]
    seen = [start]
    for i, (a, d, b) in enumerate(steps, start=1):
        if b not in seen:
            seen.append(b)
            n = len(seen)
            lines.append(
                (
                    f"Entry {i}: from {a} we went {d} into {b}. {b} is new — that makes {n}.",
                    f"Entry {i} goes {d} from {a} into {b}, a chamber I haven't "
                    f"listed yet, so the count is {n}.",
                    f"Entry {i}: {a}, {d}, into {b}. A new chamber — {n} so far.",
                )[(voice + i) % 3]
            )
        else:
            lines.append(
                (
                    f"Entry {i} goes {d} from {a} back into {b}, which is already "
                    f"on my list — nothing new.",
                    f"Entry {i} just returns to {b}; I've counted that one already.",
                    f"Entry {i} leads {d} into {b}, but {b} is old ground, so the "
                    f"count stays at {len(seen)}.",
                )[(voice + i) % 3]
            )
    roster = ", ".join(seen)
    lines.append(f"That's the whole log. Let me recheck my list: {roster}.")
    lines.append(
        "Scanning the entries once more, every name that appears is on that "
        f"list, and counting it off gives {len(seen)}."
    )
    lines.append(f"So the number of distinct chambers named in the log is {len(seen)}.")
    return "\n".join(lines)


def _trace_direction(voice: int, steps: list[tuple[str, str, str]], prompt: str) -> str:
    a = b = None
    for raw in prompt.splitlines():
        match = _DIRECTION_Q_RE.match(raw.strip())
        if match:
            a, b = match.group(1), match.group(2)
    lines = [
        (
            f"I need the single direction that leads from {a} directly to {b}. "
            "First I'll rebuild the map from the log, edge by edge, remembering "
            "that the way back along any corridor is the exact reverse direction.",
            f"The question asks which direction goes from {a} straight into {b}. "
            "Let me reconstruct the burrow map from the log first; every corridor "
            "runs both ways, with the return being the exact reverse.",
            f"So: from {a}, which way leads directly to {b}? I'll read the log "
            "in order and lay out the corridors, keeping the reverse-direction "
            "rule in mind for the way back.",
        )[voice]
    ]
    adj: dict[str, dict[str, str]] = {}
    pair_entry: tuple[int, str, str] | None = None
    for i, (u, d, v) in enumerate(steps, start=1):
        known = adj.get(u, {}).get(d) == v
        adj.setdefault(u, {})[d] = v
        adj.setdefault(v, {})[_OPP[d]] = u
        if known:
            line = (
                f"Entry {i} retraces the corridor between {u} and {v} that I "
                f"already have.",
                f"Entry {i} walks the {u}-{v} corridor again; nothing new to map.",
                f"Entry {i} re-covers {u} to {v}, already on my map.",
            )[(voice + i) % 3]
        else:
            line = (
                f"Entry {i}: {u} leads {d} to {v}, so from {v} the way back is "
                f"{_OPP[d]}.",
                f"Entry {i}: from {u}, going {d}, we reach {v}; the return "
                f"corridor from {v} runs {_OPP[d]}.",
                f"Entry {i} maps {u} {d} to {v} (and {v} {_OPP[d]} back to {u}).",
            )[(voice + i) % 3]
        if pair_entry is None and {u, v} == {a, b}:
            pair_entry = (i, u, d)
            line += " That's the very pair the question asks about."
        lines.append(line)
    exits = adj[a]
    listing = ", ".join(f"{d} to {exits[d]}" for d in _DIRS if d in exits)
    lines.append(f"Now I read my map at {a}. Its known exits: {listing}.")
    answer = next(d for d in _DIRS if exits.get(d) == b)
    others = [d for d in _DIRS if d in exits and exits[d] != b]
    if others:
        listed = (
            " and ".join(others)
            if len(others) <= 2
            else ", ".join(others[:-1]) + ", and " + others[-1]
        )
        verb = "leads" if len(others) == 1 else "lead"
        lines.append(
            f"Of those, {listed} {verb} elsewhere; the exit that reaches {b} "
            f"is {answer}."
        )
    else:
        lines.append(f"The only exit recorded there is the one into {b}: {answer}.")
    idx, logged_from, logged_dir = pair_entry
    if logged_from == b:
        lines.append(
            f"Note that entry {idx} recorded this corridor from {b}'s side, "
            f"going {logged_dir}. The reverse pairs are north-south, east-west, "
            f"up-down, so from {a} it must run {answer}. That checks out."
        )
    else:
        lines.append(
            f"Entry {idx} recorded it in exactly this orientation: from {a}, "
            f"going {answer}, into {b}."
        )
    lines.append(
        f"So from {a}, the single direction that leads directly to {b} is {answer}."
    )
    return "\n".join(lines)


def _trace_route(
    voice: int, steps: list[tuple[str, str, str]], src: str, dst: str
) -> str:
    lines = [
        (
            f"I need the shortest sequence of directions from {src} to {dst}, "
            "using only corridors named in the log. First, the map: I'll take "
            "the log edge by edge, and since every corridor runs both ways, "
            "each entry also gives me the reverse direction back.",
            f"The task is a shortest route from {src} to {dst} over the "
            "corridors the log reveals. Let me rebuild that map first, entry "
            "by entry; the way back along any corridor is the exact reverse "
            "direction.",
            f"So I want the shortest run of directions from {src} to {dst}, "
            "staying on logged corridors. Step one is reconstructing the map "
            "from the log, and each corridor is two-way with a reversed return.",
        )[voice]
    ]
    radj: dict[str, dict[str, str]] = {}
    seen_pairs: set[frozenset] = set()
    for i, (u, d, v) in enumerate(steps, start=1):
        pair = frozenset((u, v))
        if pair in seen_pairs:
            lines.append(
                (
                    f"Entry {i} retraces the {u}-{v} corridor I already mapped.",
                    f"Entry {i} walks {u} to {v} again; already on the map.",
                    f"Entry {i} re-covers the corridor between {u} and {v}.",
                )[(voice + i) % 3]
            )
            continue
        seen_pairs.add(pair)
        radj.setdefault(u, {})[d] = v
        radj.setdefault(v, {})[_OPP[d]] = u
        lines.append(
            (
                f"Entry {i}: {u} leads {d} to {v}, so back from {v} is {_OPP[d]}.",
                f"Entry {i}: from {u}, going {d}, we reach {v}; return from {v} "
                f"runs {_OPP[d]}.",
                f"Entry {i} maps {u} {d} to {v} (reverse: {_OPP[d]}).",
            )[(voice + i) % 3]
        )
    lines.append(
        (
            f"Map done. Now I search outward from {src} one move at a time — "
            f"breadth-first — so the first time {dst} appears, that distance "
            "is the shortest possible.",
            f"That's the revealed map. Now I fan out from {src} move by move; "
            f"whichever wave first touches {dst} gives the shortest distance.",
            f"With the map rebuilt, I explore from {src} in waves of one move "
            f"each, so {dst} first shows up at its shortest distance.",
        )[voice]
    )
    dist = {src: 0}
    prev: dict[str, tuple[str, str]] = {}
    frontier = [src]
    layer = 0
    while frontier and dst not in dist:
        layer += 1
        discoveries: list[tuple[str, list[tuple[str, str]]]] = []
        stuck: list[str] = []
        nxt: list[str] = []
        for node in frontier:
            found: list[tuple[str, str]] = []
            for d in _DIRS:
                dest = radj.get(node, {}).get(d)
                if dest is not None and dest not in dist:
                    dist[dest] = layer
                    prev[dest] = (node, d)
                    found.append((d, dest))
                    nxt.append(dest)
            if found:
                discoveries.append((node, found))
            else:
                stuck.append(node)
        parts = []
        for node, found in discoveries:
            reached = " and ".join(f"{dest} ({d})" for d, dest in found)
            parts.append(f"from {node} I newly reach {reached}")
        if layer == 1:
            lines.append(f"One move out: {'; '.join(parts)}.")
        else:
            lines.append(f"At {layer} moves: {'; '.join(parts)}.")
        if stuck:
            named = ", ".join(stuck[:3])
            tail = f" and {len(stuck) - 3} more" if len(stuck) > 3 else ""
            lines.append(
                f"{named}{tail} only lead back to chambers I've already "
                "reached, so nothing new comes from there."
            )
        frontier = nxt
    parent, last_dir = prev[dst]
    lines.append(
        f"There it is: {dst} turns up at distance {layer}, entered from "
        f"{parent} going {last_dir}, and no earlier wave touched it, so "
        f"{layer} moves is the minimum."
    )
    path = _path_dirs(prev, src, dst)
    nodes = [src]
    for d in path:
        nodes.append(radj[nodes[-1]][d])
    walkthrough = ", then ".join(
        f"{d} to {node}" for d, node in zip(path, nodes[1:])
    )
    lines.append(f"Reading the route back off the search: from {src} go {walkthrough}.")
    lines.append(f"That's {len(path)} moves, matching the distance.")
    lines.append(
        f"So the shortest sequence of directions from {src} to {dst} is: "
        f"{', '.join(path)}."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class Episode:
    def __init__(self, seed: int, level: int):
        self.level = level
        self.max_turns = _MAX_TURNS[level]
        n_chambers = _EP_CHAMBERS[level]

        # profile is a property of the episode (seed), not of the search
        profile_rng = base.rng_for(FAMILY, "episode-profile", seed, level)
        roll = profile_rng.random()
        acc = 0.0
        profile = _EP_PROFILES[level][-1]
        for candidate in _EP_PROFILES[level]:
            acc += candidate[0]
            if roll < acc:
                profile = candidate
                break
        # profile dist ranges already encode the intended budget slack
        # (slack 2 everywhere except L1's far profile, which runs at slack 1);
        # clamp only for raw reachability within the move budget.
        _, extra_choices, dist_lo, dist_hi, rwalk_cap = profile
        dist_hi = min(dist_hi, self.max_turns - 1)

        chosen = None
        for attempt in range(2000):
            rng = base.rng_for(FAMILY, "episode", seed, level, attempt)
            n_extra = rng.choice(extra_choices)
            names = list(rng.sample(_name_pool(level), n_chambers))
            ports = _build_ports(rng, names, n_extra)
            if ports is None:
                continue
            starts = list(names)
            rng.shuffle(starts)
            if level >= 5:
                # Frontier: try diameter-endpoint starts first so the trek
                # spans the graph's full depth (budget >= diameter + 3 by the
                # dist_hi = max_turns - 3 cap). Sorting draws no randomness;
                # the shuffle above tie-breaks equal eccentricities, so the
                # L1-L4 generation paths stay byte-identical.
                ecc = {s: max(_bfs(ports, s)[0].values()) for s in starts}
                starts.sort(key=lambda s: -ecc[s])
            for start in starts:
                dist, prev = _bfs(ports, start)
                cands = [t for t in names if dist_lo <= dist.get(t, 999) <= dist_hi]
                rng.shuffle(cands)
                if level >= 5:
                    # Frontier: push the target toward the graph diameter so
                    # the budget runs at distance + small slack. The shuffle
                    # above tie-breaks equal distances; the sort itself draws
                    # no randomness, so L1-L4 generation is byte-identical.
                    cands.sort(key=lambda t: -dist[t])
                for target in cands:
                    if _hit_prob(ports, start, target, self.max_turns) <= rwalk_cap:
                        chosen = (
                            ports,
                            start,
                            target,
                            dist[target],
                            _path_dirs(prev, start, target),
                        )
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen is None:  # pragma: no cover - generator bug
            raise RuntimeError(f"{FAMILY}: no layout for seed={seed} level={level}")
        ports, start, target, distance, path = chosen
        self._ports = ports
        self._start = start
        self._target = target
        self._path = path
        self._reset_state()
        edges = [
            [name, direction, ports[name][direction]]
            for name in sorted(ports)
            for direction in _DIRS
            if direction in ports[name] and name < ports[name][direction]
        ]
        self.spec = {
            "family": FAMILY,
            "level": level,
            "chambers": sorted(ports),
            "edges": edges,
            "start": start,
            "target": target,
            "distance": distance,
            "max_turns": self.max_turns,
        }

    def _reset_state(self) -> None:
        """(Internal) rewind to the initial state; used by the selftest."""
        self._current = self._start
        self._turns = 0
        self._done = False
        self._reached = False
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            "You are exploring an underground burrow network of connected "
            "chambers.\n"
            "Each turn you see only your current chamber and its exit "
            "directions; where an exit leads stays unknown until you walk "
            "through it, so remember what you discover.\n"
            "Corridors are consistent: the way back is always the exact "
            "reverse direction (north-south, east-west, up-down).\n"
            f"Reach the target chamber within {self.max_turns} moves; entering "
            "it ends the trek at once. Every action costs one move, even an "
            "invalid one.\n"
            "Action grammar: GO <direction>   "
            "(directions: north, south, east, west, up, down)\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def _view(self) -> str:
        dirs = [d for d in _DIRS if d in self._ports[self._current]]
        return (
            f"You are in {self._current}. Exits: {', '.join(dirs)}. "
            f"Moves left: {self.max_turns - self._turns}."
        )

    def initial_observation(self) -> str:
        return f"Target chamber: {self._target}. {self._view()}"

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            return ("The trek is already over.", True)
        self._turns += 1
        match = re.match(r"^\s*go\s+([a-z]+)[\s.!]*$", str(action_line).strip().lower())
        exits = self._ports[self._current]
        if not match or match.group(1) not in _OPP:
            self.last_action_ok = False
            observation = f"Bad action. Use exactly: GO <direction>. {self._view()}"
        elif match.group(1) not in exits:
            self.last_action_ok = False
            observation = (
                f"No corridor leads {match.group(1)} from {self._current}. "
                f"{self._view()}"
            )
        else:
            self.last_action_ok = True
            self._current = exits[match.group(1)]
            if self._current == self._target:
                self._reached = True
                self._done = True
                return (
                    f"You are in {self._current}. Target reached: the trek is "
                    f"complete.",
                    True,
                )
            observation = self._view()
        if self._turns >= self.max_turns:
            self._done = True
            return (observation + " Out of moves.", True)
        return (observation, False)

    def score(self) -> float:
        return 1.0 if self._reached else 0.0


class OraclePolicy:
    """Walks a precomputed shortest start-to-target path."""

    def __init__(self, episode: Episode):
        self._moves = list(episode._path)
        self._index = 0

    def act(self, observation_history: list[str]) -> str:
        if self._index < len(self._moves):
            direction = self._moves[self._index]
            self._index += 1
            return f"GO {direction}"
        return "GO north"  # pragma: no cover - path always fits the budget


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def _random_direction_rollout(episode: Episode, rng) -> float:
    """One rollout of a policy that picks a uniformly random VALID exit."""
    episode._reset_state()
    episode.initial_observation()
    for _ in range(episode.max_turns):
        exits = [d for d in _DIRS if d in episode._ports[episode._current]]
        _, done = episode.step(f"GO {rng.choice(exits)}")
        if done:
            break
    return episode.score()


def selftest() -> dict:
    module = sys.modules[__name__]
    stats = {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
        "random_direction": {},
    }
    # Beyond base's garbage-action floor (which never moves), also hold down a
    # random-VALID-direction walker. The exact hitting probability IS that
    # policy's true mean; empirical rollouts exercise step() as a smoke test.
    for level in LEVELS:
        exact = []
        empirical = []
        for index in range(12):
            seed = 1000 + index
            episode = Episode(seed, level)
            exact.append(
                _hit_prob(episode._ports, episode._start, episode._target, episode.max_turns)
            )
            rng = base.rng_for(FAMILY, "rdir-check", seed, level)
            for _ in range(100):
                empirical.append(_random_direction_rollout(episode, rng))
        exact_mean = sum(exact) / len(exact)
        empirical_mean = sum(empirical) / len(empirical)
        if exact_mean > 0.15:
            raise base.SelftestError(
                f"{FAMILY} L{level}: random-direction exact mean {exact_mean:.3f} > 0.15"
            )
        if empirical_mean > 0.15:
            raise base.SelftestError(
                f"{FAMILY} L{level}: random-direction empirical mean "
                f"{empirical_mean:.3f} > 0.15"
            )
        stats["random_direction"][level] = {
            "exact_mean": round(exact_mean, 4),
            "empirical_mean": round(empirical_mean, 4),
        }
    # Oracle traces: think-channel narrations must actually solve the atoms
    # (trace + terse ANSWER scores 1.0), stay inside the deploy think budget,
    # avoid firewalled vocabulary, be deterministic, and end by stating the
    # very answer value the terse line carries.
    stats["traces"] = {}
    for level in LEVELS:
        items = gen_atoms(11, level, 12)
        scores = []
        word_counts = []
        for item in items:
            trace = oracle_trace(item)
            if trace != oracle_trace(item):
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace not deterministic"
                )
            n_words = len(trace.split())
            word_counts.append(n_words)
            if n_words > 800:
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace is {n_words} words > 800"
                )
            lowered = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                if word in lowered:
                    raise base.SelftestError(
                        f"{FAMILY} L{level}: forbidden word {word!r} in trace"
                    )
            gold = item["gold"]
            expected = (
                ", ".join(gold["path"]) if gold["kind"] == "route" else str(gold["value"])
            )
            last_line = trace.strip().splitlines()[-1].lower()
            if expected.lower() not in last_line:
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace does not end by "
                    f"stating the answer value"
                )
            scores.append(score_atom(item, trace + "\n\n" + oracle_atom(item)))
        trace_mean = sum(scores) / len(scores)
        if trace_mean < 0.95:
            raise base.SelftestError(
                f"{FAMILY} L{level}: trace+answer score {trace_mean:.3f} < 0.95"
            )
        stats["traces"][level] = {
            "score": round(trace_mean, 4),
            "words_mean": round(sum(word_counts) / len(word_counts), 1),
            "words_max": max(word_counts),
        }
    return stats
