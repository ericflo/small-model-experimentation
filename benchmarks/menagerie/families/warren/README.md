# warren

## Capability

Partially observable exploration with spatial memory: navigate an unseen, text-described chamber graph to a named target under a strict move budget.

## Task description

`warren` generates a connected undirected graph of burrow chambers. Chamber tokens and tunnel tokens are fictional strings made from a seed signature followed by two seeded syllables, such as `hdaxvok` or `hmurpel`. The seed signature renders the seed as base-24 over letter digits `abcdefghijklmnopqrstuvwx`: seed `0` is `a`, seed `7` is `h`, and negative seeds prefix `a` to the positive signature. This is injective for token stock because all tokens are the seed prefix plus exactly 6 suffix characters; equal-length tokens from different seeds differ inside the prefix region, different-length prefixes give different token lengths, and positive signatures never start with `a` because there is no leading zero, so the negative marker cannot collide. The supported seed domain is `int` values with `abs(seed) < 24**10`; `generate` raises `ValueError` outside that domain. The generator must derive all vocabulary from a local `random.Random` seeded by the item seed, and different seeds should use disjoint token stock. No chamber name, tunnel name, direction, object, or landmark is a real word.

Edges are undirected, but tunnel labels are local to the chamber. The same physical edge can be shown as tunnel `hfegmur` from chamber `hhaxhax` and tunnel `hvoksiv` from the other endpoint. A tunnel's destination is hidden until the agent moves through that tunnel.

### Episode mode

The agent starts in a chamber, is told the target chamber token, and must reach it before `max_turns` replies have been spent. Every reply costs one move, including malformed actions and unknown tunnel labels.

Each episode observation has this shape:

```text
Target chamber: <target-token>
Current chamber: <current-token>
Tunnels:
- <tunnel-token> | scent: warm trail
- <tunnel-token> | scent: cold
- <tunnel-token> | scent: none
Reply with one line: MOVE <tunnel-token>
```

The scent field is generated per `(chamber, tunnel)` and fixed for the item. With the level's hint probability, the tunnel displays either `warm trail` if the far-end chamber is strictly closer to the target by shortest-path distance, or `cold` otherwise. If that tunnel did not receive a hint, it displays `none`.

The only valid action is:

```text
MOVE <tunnel-token>
```

Action parsing is case-insensitive and whitespace-tolerant. If the action is malformed or names no tunnel from the current chamber, the environment gives a curt corrective observation, the agent stays in place, and the move is still spent. The episode ends when the target chamber is reached or when the move budget is exhausted.

### Atom mode

Atom mode is single-shot with `max_turns == 1`. The prompt presents a field journal from a previous exploration: a fully revealed sub-map using lines of this form:

```text
Map lines read: <chamber>: <tunnel> -> <destination chamber>
<chamber-token>: <tunnel-token> -> <neighbor-token>
```

The prompt then gives:

```text
Current chamber: <current-token>
Target chamber: <target-token>
Reply with final line: ANSWER: <tunnel-token> <tunnel-token> ...
```

The answer must be the shortest tunnel path from the current chamber to the target. Each token in the path is the tunnel label visible in the chamber occupied at that step, not a global edge id.

## Novelty statement

The nearest public benchmarks found in the brief literature search are:

- [TextWorld](https://arxiv.org/abs/1806.11532): TextWorld is a procedural text-game environment with parser commands, rooms, objects, quests, and English-like world descriptions. `warren` removes the interactive-fiction surface entirely: no objects, inventories, compass directions, common-sense affordances, or English room names. It uses per-seed fictional chamber/tunnel vocabulary, one-line `MOVE` actions over local tunnel labels, scent-gradient hints, and efficiency-ratio scoring against a verified BFS optimum.
- [LMRL-Gym Maze](https://arxiv.org/abs/2311.18232): LMRL-Gym includes a maze task inside a broader multi-turn reinforcement-learning benchmark for language models, with agents making moves from cues supplied by the task simulator. `warren` is an offline Menagerie evaluation family, not an RL training benchmark with offline datasets. Its instances are anonymous generated graphs with disjoint fictional vocabulary, hidden per-chamber tunnel destinations, level-scaled sparse scent hints, atom-mode shortest-path probes, and deterministic machine scoring by BFS ratio or path validity.
- [MazeEval](https://arxiv.org/abs/2507.20395): MazeEval evaluates coordinate-based grid navigation through a function-calling interface with coordinate feedback and distance-to-wall information. `warren` has no coordinates, compass directions, grid geometry, or wall-distance readings. The agent must build a spatial memory over an anonymous labeled graph whose local edge names change by chamber, while scoring rewards efficient target reach rather than only final navigation success.

Compared with broader candidates such as BabyAI/MiniGrid, Jericho/Zork, ALFWorld, MazeBench, and GraphArena, these three are closest because they directly involve text-game exploration, multi-turn maze movement, or text-only maze navigation. `warren` is kept distinct through fictional seeded vocabulary, local tunnel labels with hidden destinations, sparse target-gradient scent hints, a single-line action grammar, an atom journal variant, and ratio scoring against an explicit shortest-path oracle.

## Level ladder

Episode levels:

| Level | Chambers | Typical degree | Required shortest path | `max_turns` | Scent hint probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | about 8 | about 3 | 2 | 4 | 1.0 |
| L2 | about 12 | about 4-5 | 2 | 4 | 0.7 |
| L3 | about 16 | about 4 | 5 | 10 | 0.4 |
| L4 | about 24 | about 4-5 | 8 | 14 | 0.2 |

Atom levels:

| Level | Revealed map size | Required shortest path |
| --- | ---: | ---: |
| L1 | about 5 chambers | 2 |
| L2 | about 6 chambers | 3 |
| L3 | about 7-8 chambers | 4 |
| L4 | about 9 chambers | 5-6 |

Atom prompts must remain within the 1200-character cap. Episode observations must remain within the 800-character per-turn cap. Episode `max_turns` follows the contract caps: L1-L2 at most 4, L3 at most 10, and L4 at most 14.

## Scoring

Episode scoring is ratio-scored. If the target is reached on turn `k`, the score is:

```text
shortest_path_length / k
```

The oracle reaches the target in exactly `shortest_path_length` moves and scores `1.0`. If the target is never reached, the score is `0.0`. There is no other partial credit.

The episode oracle is a perfect-play upper bound that reads the hidden graph in the item and replays a BFS-shortest tunnel path from the current chamber. This is intentionally stronger than any real agent, which must spend moves to discover hidden destinations. The move budget supplies exploration slack for real agents.

Atom scoring walks the submitted path over the revealed journal map:

- `1.0` if the sequence is a valid walk from current chamber to target and its length equals the BFS-shortest path length.
- `0.5` if the sequence is a valid walk to the target but is longer than shortest.
- `0.0` otherwise.

Whenever an optimum is claimed, generation must verify the BFS shortest-path length for the item. Because this family uses ratio scoring, a random floor above `0.05` but at most `0.15` is contract-sanctioned if measurement lands there: occasional random success on small graphs receives a fractional efficiency score rather than being a calibrated answer-space chance rate. Degenerate empty, constant, or echo policies still must remain below the contract's degenerate-resistance threshold.

## Example item + oracle transcript

### Atom example

Generated with `item = family.generate(7, 1, 2, 'atom')[0]`.

Prompt from `family.Env(item).reset()`:

```text
Field journal: find the shortest tunnel-token path.
Map lines read: <chamber>: <tunnel> -> <destination chamber>
hhaxhax: hfegmur -> hlomhax
hhaxhax: hpazpel -> htirfeg
hhaxhax: hwuprul -> hsazdax
hlomhax: hqebgud -> hnulpel
hlomhax: hvoksiv -> hhaxhax
hnulpel: hsazhax -> hlomhax
hsazdax: hrullom -> hhaxhax
htirfeg: htovkiv -> hhaxhax
Current chamber: htirfeg
Target chamber: hsazdax
Reply with final line: ANSWER: <tunnel-token> <tunnel-token> ...
```

Oracle reply from `family.oracle_policy(item, [])`:

```text
ANSWER: htovkiv hwuprul
```

Score from `family.score(item, [{'obs': <the prompt>, 'action': <the oracle reply>}])`:

```text
{'score': 1.0, 'valid': True, 'reached': True, 'turns_used': 2, 'sp': 2}
```

### Episode example

Generated with `item = family.generate(7, 3, 1, 'episode')[0]`. The item has `max_turns == 10` and `sp == 5`.

Turn 1 observation:

```text
Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.
Target chamber: hrultov
Current chamber: hqebtir
Moves left: 10
Tunnels:
- hdaxtov | scent: warm trail
- hjivhax | scent: none
Reply with one line: MOVE <tunnel-token>
```

Oracle action:

```text
MOVE hdaxtov
```

Turn 2 observation:

```text
Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.
Target chamber: hrultov
Current chamber: hgudrul
Moves left: 9
Tunnels:
- hboklom | scent: none
- hmurfeg | scent: none
- hraxpel | scent: none
Reply with one line: MOVE <tunnel-token>
```

Oracle action:

```text
MOVE hraxpel
```

Turn 3 observation:

```text
Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.
Target chamber: hrultov
Current chamber: hlomsiv
Moves left: 8
Tunnels:
- hbokdax | scent: warm trail
- hgudkiv | scent: none
- hsivqeb | scent: cold
- htovyeg | scent: none
- hzinyeg | scent: none
Reply with one line: MOVE <tunnel-token>
```

Oracle action:

```text
MOVE hzinyeg
```

Turn 4 observation:

```text
Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.
Target chamber: hrultov
Current chamber: hpellom
Moves left: 7
Tunnels:
- hfegfeg | scent: none
- hlomfeg | scent: none
- hlommur | scent: none
- hwupcim | scent: none
- hyeghax | scent: none
Reply with one line: MOVE <tunnel-token>
```

Oracle action:

```text
MOVE hlomfeg
```

Turn 5 observation:

```text
Burrow hunt: reach the target chamber before your moves run out. Every reply costs one move.
Target chamber: hrultov
Current chamber: hqebcim
Moves left: 6
Tunnels:
- hfegtov | scent: warm trail
- hgudzin | scent: cold
- htirbok | scent: cold
Reply with one line: MOVE <tunnel-token>
```

Oracle action:

```text
MOVE hfegtov
```

Final score from `family.score(item, transcript)`:

```text
{'score': 1.0, 'reached': True, 'turns_used': 5, 'sp': 5}
```
