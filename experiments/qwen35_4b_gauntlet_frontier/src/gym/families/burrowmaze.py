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
    return stats
