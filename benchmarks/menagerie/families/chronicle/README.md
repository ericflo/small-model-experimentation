# chronicle

## Capability Axis

STATE TRACKING: simulate an evolving fictional world from an ordered event
stream and report one final-state fact.

## Task Description

Each item is a single-turn prompt. The model receives a compact event chronicle
over freshly generated fictional tokens: objects, agents, places, and aliases.
It must apply every event in order, maintain the current world state, and answer
one final query.

The world state has:

- Places containing loose objects.
- Agents located at places.
- Agents carrying zero or more objects; carried objects move with their agent.
- Aliases that become valid only after their introduction event.

The event stream is deliberately terse. Entity tokens are invented per item and
are never real words. Event syntax uses punctuation rather than familiar puzzle
wording:

- `X>P`: move object `X` to place `P`, or move agent `X` to place `P` if `X`
  is an agent. A moved object becomes loose at `P`; a moved agent carries its
  held objects along.
- `A+X`: agent `A` picks up co-located object `X`.
- `A-X`: agent `A` drops carried object `X` at the agent's current place.
- `X~Y`: swap the current homes of objects `X` and `Y`. A home is either a
  place or a carrying agent.
- `P>>Q`: transfer all loose objects currently at place `P` to place `Q`.
- `U=V`: introduce alias `U` for existing entity `V`; later events may use `U`.
- `?C:E`: execute event `E` only if condition `C` is true in the current state.
  Conditions are `X@P` for object/agent effective location and `X#A` for object
  carried by agent.

Conditionals are evaluated when encountered, not after the stream. False
conditionals leave the state unchanged, but they still require tracking because
the reader must know that the consequent did not happen.

Queries have exactly two forms:

- `where is X?` Answer with the final place containing object `X`; if `X` is
  carried, answer the carrier agent's current place.
- `which object is at P?` Answer with the unique object whose final effective
  place is `P`. The generator only emits this query when exactly one object is
  there.

The prompt ends with: `End with final line ANSWER: <name>`.

## Novelty Statement

The nearest public benchmarks are:

- Facebook bAbI tasks 1-3
  (https://arxiv.org/abs/1502.05698): synthetic QA stories generated from a
  simulated world, including people moving, objects being picked up, and
  one-, two-, or three-supporting-fact questions. `chronicle` differs by using
  fresh nonword tokens per item, a compact symbolic event surface, free-text
  final-state answers, swaps, transfer-all operations, state conditionals,
  aliases, and no fixed train/test vocabulary.
- BIG-bench `tracking_shuffled_objects`
  (https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/tracking_shuffled_objects):
  initial person-object assignments followed by pairwise swaps, scored as
  multiple choice over the final object held by a named person. `chronicle`
  keeps the ordered-swap pressure but adds places, carrying agents, transfer-all
  operations, conditional execution, aliases, two query types, and exact
  canonical free-text scoring.
- Kim and Schuster 2023, Entity Tracking in Language Models
  (https://arxiv.org/abs/2305.02363): box-and-object worlds with initial box
  contents and state-changing operations such as putting, removing, moving, and
  moving contents, then cloze-style prediction of a final box state. `chronicle`
  uses per-item fictional entity vocabulary rather than common nouns and fixed
  boxes, asks one exact final query, includes agents whose carried objects move
  with them, introduces aliases during the stream, and uses current-state
  conditionals as a core difficulty axis.

These differences are intended to make memorized benchmark items, real-world
entity priors, and fixed-format multiple-choice heuristics unhelpful. The task
is about ordered causal simulation, not retrieval under textual noise.

## Level Ladder

Both modes are single-turn. `max_turns` is always `1`.

Atom mode uses the smallest prompt that isolates the level's new operation
types:

| Level | Events | Planned mix |
| --- | ---: | --- |
| L1 | 5 | Object moves only, including two causally valid red-herring moves in a disjoint component. |
| L2 | 12 | Object moves, object swaps, and transfer-all events. |
| L3 | 20 | L2 mix plus agents, agent moves, pickups, drops, and true/false conditionals. |
| L4 | 32 | L3 mix plus alias introductions and chained conditionals whose outcomes affect later conditions. |

Episode mode has the same single-shot shape but denser composition under the
800-character observation budget:

| Level | Events | Planned mix |
| --- | ---: | --- |
| L1 | 7 | Longer moves-only chain with more disjoint causal red herrings. |
| L2 | 12 | Moves, swaps, and transfer-alls, with at least one transfer-all affecting a later swap. |
| L3 | 18 | Agent carrying, drops, transfer-alls, swaps, and a balanced true/false conditional mix. |
| L4 | 20 | Full mix: agents, swaps, transfer-alls, aliases, and chained conditionals. |

Red-herring events are not filler text. They mutate valid parts of the same
world and would matter for other possible queries. For the selected query, the
generator marks them as red herrings only after dependency checks confirm that
removing the event would not change the canonical answer.

## Generator Design

Generation is deterministic and local to the item. The future implementation
will derive an integer item seed from `sha256("chronicle", seed, level, mode,
index)` and construct a private `random.Random` from that integer. It will never
use global RNG state, wall-clock time, I/O, network, or non-stdlib packages.

Entity tokens are always 4 lowercase characters formed from two artificial syllables
such as `qa`, `qe`, `qi`, `qo`, `za`, `ze`, `xi`, `xo`, `va`, and `vo`, with at
least one `q` syllable in every token. The generator rejects duplicates,
reserved protocol strings, and a small internal forbidden list. Because every
item gets a fresh token pool, a constant-answer policy should score near zero
over a batch.

Ground truth is produced by exact forward simulation. The generator builds an
initial state, samples only events that are semantically valid under the current
state, applies them immediately, and then independently replays the serialized
event stream from the initial state. The replayed final state must match the
builder state. Query generation then filters for a valid answer: object-location
queries require a known final effective place, and place-object queries require
exactly one final effective object at the queried place.

Shortcut resistance: the deterministic salt-retry rejects any item whose final or
second-to-last `>`-bearing event's surface destination canonically matches any
accepted answer, so both the parrot-the-last-destination and
parrot-the-second-to-last-destination lazy policies score 0 by construction. The
residual ceiling for a guard-aware adversary on where-is queries is therefore
roughly `1 / (#places - 2)` when the two excluded destinations are distinct
places (about 0.4-0.5 at L1 where there are 4 places, measured 0.425; L1 is
intentionally the easy rung), while realistic recency-parroting policies score
exactly 0.

## Budget Plan

The implementation uses compact event lines without spaces and a level-scoped
rules legend: each level includes only the operation rules that can appear at
that level, while L4 includes the full alias and conditional grammar.

The original hand budget underestimated real prompts because the fixed legend,
query text, and exact event mix are longer than the rough arithmetic assumed.
In a 150-seed scan, observed maxima were approximately:

- L4 atom: 958 / 1200 characters.
- L4 episode: 794 / 800 characters.

Prompt budgets and shortcut resistance are enforced in generation. For each
`(seed, level, mode, index)`, the generator first builds the legacy salt-0 item.
If its prompt would exceed the mode limit or fail the shortcut guard, generation
deterministically retries with salt 1, then salt 2, and so on up to the bounded
retry limit. This preserves byte-identical outputs for items that already fit
and pass the guard, prevents `generate()` from emitting an over-budget prompt,
and ensures a prompt-budget or shortcut-guard miss never crashes any seed on the
former bare assertion path. Only failure to find an acceptable prompt within the
bounded retry window raises `RuntimeError`.

## Answer Protocol

The model may reason however it likes, but the final answer must include a last
line of the form:

```text
ANSWER: <name>
```

Only the last `ANSWER:` line is scored. Matching is case-insensitive and
whitespace-tolerant.

## Scoring Rules

The scorer extracts the last line matching `ANSWER:` case-insensitively, strips
surrounding whitespace, lowercases the value, and removes punctuation and
internal whitespace for canonical comparison. The score is `1.0` if the
canonical value matches the correct canonical entity name or any alias of that
entity introduced in the prompt; otherwise the score is `0.0`.

Alias acceptance applies to both answer types. If the correct final place or
object has aliases introduced in the event stream, any introduced alias is an
accepted answer. Aliases that never appear in the prompt are not accepted.

The `random_policy` emits a fresh random syllable token built from the provided
RNG using the same fictional-token scheme, not a token sampled from the item's
visible tokens. This is not the same as an informed type-aware guess. The
type-aware guessing ceiling is `1 / #visible_places` for `where is X?` queries
and `1 / #visible_objects` for `which object is at P?` queries.

## Example Item + Oracle Transcript

Reproduction line: `generate(7, 3, 1, 'atom')[0]` (seed 7, level 3, atom mode,
index 0).

Prompt:

```text
X>P: move object/agent X to place P; held objects follow agents.
A+X: A picks up loose object X at A's place; A-X: A drops carried X there.
X~Y: swap object homes (place/carrier); P>>Q: move all loose objects at P to Q.
?C:E: do E iff C true now; C: X@P effective place (carried object at carrier place), X#A carried by A.
Start O:kuqe@qine xoqu@xeqo veqe@qare ziqa@qare qoxi@riqe qoxu@riqe A:quqo@qine qaxa@xeqo qevi@qare
Events:
quqo+kuqe
quqo>xeqo
?kuqe@xeqo:xoqu>qare
?veqe#quqo:veqe>riqe
quqo-kuqe
qare>>riqe
kuqe~xoqu
xoqu>riqe
qevi>riqe
qevi+qoxu
qevi-qoxu
riqe>>qine
veqe~kuqe
?ziqa@qine:xoqu>riqe
?veqe@qare:qoxu>xeqo
veqe>riqe
riqe>>qine
kuqe~qoxi
kuqe>qare
quqo>qine
which object is at qare?
End with final line ANSWER: <name>
```

Gold answer: `kuqe`

Accepted aliases: `[]`

Accepted answers: `['kuqe']`

Oracle policy action:

```text
ANSWER: kuqe
```

`score()` result for transcript
`[{'obs': prompt, 'action': oracle_action}]`:

```python
{'score': 1.0, 'expected': 'kuqe', 'got': 'kuqe'}
```
