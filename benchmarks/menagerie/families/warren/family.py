from collections import deque
import random
import re


META = {
    "name": "warren",
    "capability": "partially-observable exploration and spatial memory: navigate an unseen burrow graph to a target chamber under a move budget",
    "paradigm": "multi-turn",
    "action_format": "episodes reply 'MOVE <tunnel-token>', atoms reply final line 'ANSWER: <tunnel-token> <tunnel-token> ...'",
}


SYLLABLES = (
    "dax",
    "vok",
    "mur",
    "pel",
    "zin",
    "qeb",
    "rax",
    "nul",
    "feg",
    "jiv",
    "lom",
    "saz",
    "tir",
    "wup",
    "yeg",
    "bok",
    "cim",
    "gud",
    "hax",
    "kiv",
    "paz",
    "rul",
    "siv",
    "tov",
)
DIGITS = "abcdefghijklmnopqrstuvwx"
SEED_LIMIT = 24**10

EPISODE_LEVELS = {
    1: {"chambers": 8, "degree": 3.0, "max_degree": 4, "sp": 2, "max_turns": 4, "hint_prob": 1.0},
    2: {"chambers": 12, "degree": 4.5, "max_degree": 6, "sp": 2, "max_turns": 4, "hint_prob": 0.7},
    3: {"chambers": 16, "degree": 4.0, "max_degree": 5, "sp": 5, "max_turns": 10, "hint_prob": 0.4},
    4: {"chambers": 24, "degree": 4.5, "max_degree": 6, "sp": 8, "max_turns": 14, "hint_prob": 0.2},
}

ATOM_LEVELS = {
    1: {"chambers": 5, "sp": 2},
    2: {"chambers": 6, "sp": 3},
    3: {"chambers": 8, "sp": 4},
    4: {"chambers": 9, "sp": 6},
}

_MOVE_RE = re.compile(r"^MOVE\s+(\S+)$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^ANSWER\s*:\s*(.*)$", re.IGNORECASE)
_MASK64 = (1 << 64) - 1


def generate(seed: int, level: int, n: int, mode: str) -> list[dict]:
    if not isinstance(seed, int) or abs(seed) >= SEED_LIMIT:
        raise ValueError(f"seed must be an int with abs(seed) < 24**10, got {seed!r}")
    if level not in EPISODE_LEVELS:
        raise ValueError("level must be one of 1, 2, 3, 4")
    if mode not in ("episode", "atom"):
        raise ValueError("mode must be 'episode' or 'atom'")
    if n < 0:
        raise ValueError("n must be non-negative")
    return [_make_item(seed, level, mode, i) for i in range(n)]


class Env:
    def __init__(self, item: dict):
        self.item = item
        self.current = item["start"]
        self.moves_left = item["max_turns"]
        self.done = False

    def reset(self) -> str:
        self.current = self.item["start"]
        self.moves_left = self.item["max_turns"]
        self.done = False
        if self.item["mode"] == "atom":
            return _render_atom_prompt(self.item)
        return _render_episode_obs(self.item, self.current, self.moves_left)

    def step(self, action: str) -> tuple[str, bool]:
        if self.item["mode"] == "atom":
            self.done = True
            return "Answer recorded.", True
        if self.done:
            if self.current == self.item["target"]:
                return f"You reached the target chamber {self.item['target']}.", True
            return _render_episode_obs(self.item, self.current, self.moves_left), True
        self.current, self.moves_left, self.done, obs = _episode_step_state(
            self.item, self.current, self.moves_left, action
        )
        return obs, self.done


def score(item: dict, transcript: list[dict]) -> dict:
    if item["mode"] == "atom":
        return _score_atom(item, transcript)
    return _score_episode(item, transcript)


def oracle_policy(item: dict, history: list[dict]) -> str:
    if item["mode"] == "atom":
        path = _shortest_tunnel_path(item["journal"], item["start"], item["target"])
        return "ANSWER: " + " ".join(path)
    current, _moves_left = _replay_episode(item, history)
    path = _shortest_tunnel_path(item["graph"], current, item["target"])
    if not path:
        tunnels = sorted(item["graph"].get(current, {}))
        return "MOVE " + (tunnels[0] if tunnels else "void")
    return "MOVE " + path[0]


def random_policy(item: dict, history: list[dict], rng) -> str:
    if item["mode"] == "atom":
        stock = sorted(tunnel for chamber in item["journal"].values() for tunnel in chamber)
        length = rng.randint(1, 4)
        return "ANSWER: " + " ".join(rng.choice(stock) for _ in range(length))
    current, _moves_left = _replay_episode(item, history)
    tunnels = sorted(item["graph"][current])
    return "MOVE " + rng.choice(tunnels)


def _make_item(seed, level, mode, index):
    for attempt in range(256):
        rng = random.Random(_stable_seed(seed, level, mode, index, attempt))
        if mode == "episode":
            item = _make_episode_item(seed, level, mode, index, rng)
            if all(
                len(_render_episode_obs(item, chamber, item["max_turns"], "No such tunnel here. ")) <= 800
                for chamber in item["graph"]
            ):
                return item
        else:
            item = _make_atom_item(seed, level, mode, index, rng)
            if len(_render_atom_prompt(item)) <= 1200:
                return item
    raise AssertionError(f"could not generate valid warren item for {seed} L{level} {mode} {index}")


def _make_episode_item(seed, level, mode, index, rng):
    params = EPISODE_LEVELS[level]
    prefix = _seed_prefix(seed)
    used = set()
    chambers = [_draw_token(rng, prefix, used) for _ in range(params["chambers"])]
    target_edges = int(round(params["chambers"] * params["degree"] / 2.0))
    start, target, edges = _build_edges(
        rng, chambers, params["sp"], target_edges, params["max_degree"]
    )
    if len(edges) < target_edges:
        raise AssertionError("episode graph did not reach target degree")
    graph = _label_graph(rng, prefix, used, chambers, edges)
    distances = _distances(graph, target)
    if distances.get(start) != params["sp"]:
        raise AssertionError("episode graph shortest path mismatch")
    hints = _make_hints(rng, graph, target, params["hint_prob"])
    item = {
        "id": f"warren-{seed}-L{level}-{mode}-{index}",
        "level": level,
        "mode": mode,
        "max_turns": params["max_turns"],
        "graph": graph,
        "start": start,
        "target": target,
        "sp": params["sp"],
        "hints": hints,
    }
    if _distances(item["graph"], item["target"])[item["start"]] != item["sp"]:
        raise AssertionError("episode item failed BFS verification")
    return item


def _make_atom_item(seed, level, mode, index, rng):
    params = ATOM_LEVELS[level]
    prefix = _seed_prefix(seed)
    used = set()
    chambers = [_draw_token(rng, prefix, used) for _ in range(params["chambers"])]
    start, target, edges = _build_edges(rng, chambers, params["sp"], params["chambers"] - 1, 4)
    if len(edges) != params["chambers"] - 1:
        raise AssertionError("atom graph is not a tree")
    graph = _label_graph(rng, prefix, used, chambers, edges)
    distances = _distances(graph, target)
    if distances.get(start) != params["sp"]:
        raise AssertionError("atom graph shortest path mismatch")
    item = {
        "id": f"warren-{seed}-L{level}-{mode}-{index}",
        "level": level,
        "mode": mode,
        "max_turns": 1,
        "graph": graph,
        "start": start,
        "target": target,
        "sp": params["sp"],
        "hints": {},
        "journal": graph,
    }
    return item


def _stable_seed(seed, level, mode, index, attempt):
    mode_value = 0
    for ch in mode:
        mode_value = (mode_value * 257 + ord(ch)) & _MASK64
    x = _mix64(seed & _MASK64)
    x ^= _mix64((level + 0x9E3779B97F4A7C15) & _MASK64)
    x ^= _mix64((mode_value + 0xBF58476D1CE4E5B9) & _MASK64)
    x ^= _mix64((index + 0x94D049BB133111EB) & _MASK64)
    x ^= _mix64((attempt + 0xD6E8FEB86659FD93) & _MASK64)
    return _mix64(x)


def _mix64(x):
    x &= _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _seed_prefix(seed):
    base = len(DIGITS)
    if seed == 0:
        return DIGITS[0]
    if seed < 0:
        return DIGITS[0] + _seed_prefix(-seed)
    # Tokens are prefix + exactly 6 suffix chars: equal-length tokens from
    # different seeds differ inside the prefix region, and different-length
    # prefixes give different token lengths. Positive signatures never start
    # with 'a' because there is no leading zero, so the negative marker cannot
    # collide.
    digits = []
    value = seed
    while value:
        digits.append(DIGITS[value % base])
        value //= base
    return "".join(reversed(digits))


def _draw_token(rng, prefix, used):
    choices = [
        prefix + left + right
        for left in SYLLABLES
        for right in SYLLABLES
        if prefix + left + right not in used
    ]
    if not choices:
        raise AssertionError("token space exhausted")
    token = rng.choice(choices)
    used.add(token)
    return token


def _build_edges(rng, chambers, sp, target_edges, max_degree):
    path = rng.sample(chambers, sp + 1)
    start = path[0]
    target = path[-1]
    edges = set()
    degree = {chamber: 0 for chamber in chambers}

    for left, right in zip(path, path[1:]):
        _add_edge(edges, degree, left, right)

    attached = list(path)
    rest = [chamber for chamber in chambers if chamber not in set(path)]
    rng.shuffle(rest)
    for chamber in rest:
        parents = [node for node in attached if degree[node] < max_degree]
        if not parents:
            parents = list(attached)
        parent = rng.choice(parents)
        _add_edge(edges, degree, chamber, parent)
        attached.append(chamber)

    while len(edges) < target_edges:
        candidates = []
        for i, left in enumerate(chambers):
            if degree[left] >= max_degree:
                continue
            for right in chambers[i + 1 :]:
                if degree[right] >= max_degree:
                    continue
                edge = _edge_key(left, right)
                if edge in edges:
                    continue
                trial = set(edges)
                trial.add(edge)
                if _edge_distance(chambers, trial, start, target) == sp:
                    candidates.append(edge)
        if not candidates:
            break
        left, right = rng.choice(candidates)
        _add_edge(edges, degree, left, right)

    if _edge_distance(chambers, edges, start, target) != sp:
        raise AssertionError("edge graph shortest path mismatch")
    return start, target, edges


def _add_edge(edges, degree, left, right):
    edge = _edge_key(left, right)
    if edge not in edges:
        edges.add(edge)
        degree[left] += 1
        degree[right] += 1


def _edge_key(left, right):
    if left < right:
        return (left, right)
    return (right, left)


def _edge_distance(chambers, edges, start, target):
    graph = {chamber: [] for chamber in chambers}
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        chamber, dist = queue.popleft()
        if chamber == target:
            return dist
        for neighbor in graph[chamber]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    return None


def _label_graph(rng, prefix, used, chambers, edges):
    graph = {chamber: {} for chamber in chambers}
    for left, right in sorted(edges):
        left_tunnel = _draw_token(rng, prefix, used)
        right_tunnel = _draw_token(rng, prefix, used)
        graph[left][left_tunnel] = right
        graph[right][right_tunnel] = left
    return graph


def _make_hints(rng, graph, target, hint_prob):
    distances = _distances(graph, target)
    hints = {}
    for chamber in sorted(graph):
        hints[chamber] = {}
        for tunnel in sorted(graph[chamber]):
            neighbor = graph[chamber][tunnel]
            if rng.random() < hint_prob:
                if distances[neighbor] < distances[chamber]:
                    hints[chamber][tunnel] = "warm trail"
                else:
                    hints[chamber][tunnel] = "cold"
            else:
                hints[chamber][tunnel] = "none"
    return hints


def _distances(graph, target):
    distances = {target: 0}
    queue = deque([target])
    while queue:
        chamber = queue.popleft()
        for neighbor in graph[chamber].values():
            if neighbor not in distances:
                distances[neighbor] = distances[chamber] + 1
                queue.append(neighbor)
    return distances


def _render_episode_obs(item, current, moves_left, prefix=""):
    lines = [
        "Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.",
        f"Target chamber: {item['target']}",
        f"Current chamber: {current}",
        f"Moves left: {moves_left}",
        "Tunnels:",
    ]
    for tunnel in sorted(item["graph"][current]):
        lines.append(f"- {tunnel} | scent: {item['hints'][current][tunnel]}")
    lines.append("Reply with one line: MOVE <tunnel-token>")
    return prefix + "\n".join(lines)


def _render_atom_prompt(item):
    lines = [
        "Field journal: find the shortest tunnel-token path.",
        "Map lines read: <chamber>: <tunnel> -> <destination chamber>",
    ]
    for chamber in sorted(item["journal"]):
        for tunnel in sorted(item["journal"][chamber]):
            neighbor = item["journal"][chamber][tunnel]
            lines.append(f"{chamber}: {tunnel} -> {neighbor}")
    lines.extend(
        [
            f"Current chamber: {item['start']}",
            f"Target chamber: {item['target']}",
            "Reply with final line: ANSWER: <tunnel-token> <tunnel-token> ...",
        ]
    )
    return "\n".join(lines)


def _episode_step_state(item, current, moves_left, action):
    moves_left -= 1
    token = _last_move_token(action)
    resolved = _resolve_tunnel(item["graph"], current, token)
    invalid = resolved is None
    if resolved is not None:
        current = resolved[1]
    if current == item["target"]:
        return current, moves_left, True, f"You reached the target chamber {item['target']}."
    done = moves_left <= 0
    prefix = "No such tunnel here. " if invalid else ""
    return current, moves_left, done, _render_episode_obs(item, current, moves_left, prefix)


def _last_move_token(action):
    token = None
    text = action if isinstance(action, str) else "" if action is None else str(action)
    for line in text.splitlines():
        match = _MOVE_RE.match(line.strip())
        if match:
            token = match.group(1)
    return token


def _resolve_tunnel(graph, current, token):
    if token is None:
        return None
    lowered = token.lower()
    for tunnel, neighbor in graph[current].items():
        if tunnel.lower() == lowered:
            return tunnel, neighbor
    return None


def _score_episode(item, transcript):
    current = item["start"]
    moves_left = item["max_turns"]
    turns_used = 0
    for turn in transcript:
        if current == item["target"] or moves_left <= 0:
            break
        action = turn.get("action", "") if isinstance(turn, dict) else ""
        current, moves_left, _done, _obs = _episode_step_state(item, current, moves_left, action)
        turns_used += 1
        if current == item["target"]:
            return {"score": item["sp"] / turns_used, "reached": True, "turns_used": turns_used, "sp": item["sp"]}
    return {"score": 0.0, "reached": False, "turns_used": turns_used, "sp": item["sp"]}


def _score_atom(item, transcript):
    tokens = None
    for turn in transcript:
        if isinstance(turn, dict):
            candidate = _last_answer_tokens(turn.get("action", ""))
            if candidate is not None:
                tokens = candidate
    if tokens is None:
        known_lower = {tunnel.lower() for chamber in item["journal"].values() for tunnel in chamber}
        for turn in transcript:
            if isinstance(turn, dict):
                candidate = _bare_path_tokens(turn.get("action", ""), known_lower)
                if candidate is not None:
                    tokens = candidate
    if tokens is None:
        return {"score": 0.0, "valid": False, "reached": False, "turns_used": 0, "sp": item["sp"]}

    current = item["start"]
    valid = True
    for token in tokens:
        resolved = _resolve_tunnel(item["journal"], current, token)
        if resolved is None:
            valid = False
            break
        current = resolved[1]

    reached = valid and current == item["target"]
    value = 0.0
    if reached:
        value = 1.0 if len(tokens) == item["sp"] else 0.5
    return {
        "score": value,
        "valid": valid,
        "reached": reached,
        "turns_used": len(tokens),
        "sp": item["sp"],
    }


def _last_answer_tokens(action):
    tokens = None
    text = action if isinstance(action, str) else "" if action is None else str(action)
    for line in text.splitlines():
        match = _ANSWER_RE.match(line.strip())
        if match:
            tokens = match.group(1).split()
    return tokens


def _bare_path_tokens(action, known_lower):
    tokens = None
    text = action if isinstance(action, str) else "" if action is None else str(action)
    for line in text.splitlines():
        parts = line.strip().split()
        if parts and all(part.lower() in known_lower for part in parts):
            tokens = parts
    return tokens


def _replay_episode(item, history):
    current = item["start"]
    moves_left = item["max_turns"]
    for turn in history:
        if current == item["target"] or moves_left <= 0:
            break
        action = turn.get("action", "") if isinstance(turn, dict) else ""
        current, moves_left, _done, _obs = _episode_step_state(item, current, moves_left, action)
    return current, moves_left


def _shortest_tunnel_path(graph, start, target):
    distances = _distances(graph, target)
    if start not in distances:
        return []
    current = start
    path = []
    while current != target:
        options = []
        for tunnel, neighbor in graph[current].items():
            if distances.get(neighbor) == distances[current] - 1:
                options.append((neighbor, tunnel))
        if not options:
            return []
        neighbor, tunnel = sorted(options)[0]
        path.append(tunnel)
        current = neighbor
    return path
