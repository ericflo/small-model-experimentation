# mirage

## Capability Axis

Calibrated abstention: answer when the prompted constraint system forces a
unique target value, and reply `IMPOSSIBLE` when the constraint system is
provably unsatisfiable.

## Task Description

`mirage` is a single-turn family of value-assignment constraint puzzles over
fictional entities. Each item defines:

- a cyclic ordering of fictional value tokens;
- a set of fictional entity names;
- a list of constraints; and
- one target entity whose value token is queried.

All puzzle-specific names are procedurally generated from a seeded syllable
stock. Entity names, value tokens, and flavor nouns are nonce strings, not real
English words. The prompt states the whole value cycle, so the family is
knowledge-free.

The constraint grammar is:

- direct anchor: entity `X` holds token `T`;
- equality link: entity `X` matches entity `Y`;
- cyclic shift link: entity `X` holds the token immediately after entity `Y` in
  the stated cycle;
- inequality link, from higher levels onward: entity `X` differs from entity
  `Y`.

The question is always: what token does the named target entity hold? Solvable
items have a target token forced in every satisfying assignment. Unsolvable
items have no satisfying assignment because of a planted contradiction chain.

The required reply protocol is one final line:

```text
ANSWER: <token>
```

for a solvable item, or:

```text
ANSWER: IMPOSSIBLE
```

for a provably unsatisfiable item. The scorer extracts the last
case-insensitive `ANSWER:` line and compares the canonical payload.

`mode="atom"` and `mode="episode"` use the same task shape. `episode` remains
single-turn with `max_turns == 1`; it is not a harder composited variant.

## Novelty Statement

The three nearest public benchmarks or datasets are:

- [AbstentionBench](https://ar5iv.labs.arxiv.org/html/2506.09038): a broad
  abstention benchmark over 20 datasets, including unknown-answer,
  underspecified, false-premise, subjective, stale, math, and science
  questions. It scores abstention with judged precision, recall, and F1, and
  response correctness with a judge. `mirage` instead uses procedurally
  generated fictional constraint puzzles, contains no real-world QA surface, and
  uses exact string scoring over a balanced answer-or-`IMPOSSIBLE` label.
- [SQuAD 2.0](https://ar5iv.labs.arxiv.org/html/1806.03822): extractive
  reading comprehension where unanswerability arises because a Wikipedia
  paragraph does not support any answer span, despite plausible distractor
  spans. It reports exact match and F1, with no-answer predictions scored as
  correct on negative examples. `mirage` has no passage reading, no Wikipedia
  content, no span extraction, and no natural-language plausible-answer
  artifacts; unsolvability is a formal inconsistency in prompt-internal
  constraints.
- [UMWP](https://arxiv.org/abs/2403.03558): unanswerable math word problems
  built from real math-problem datasets, with unanswerability caused by missing,
  ambiguous, unrealistic, unrelated, or absent question information. It
  identifies abstention-like behavior using text similarity and mathematical
  expression detection, then reports F1 on unanswerability detection. `mirage`
  is not a math word problem benchmark: it uses fictional tokens, modular
  value-cycle constraints, exact satisfiability checks, and symmetric
  one-point scoring for both forced answers and abstentions.

Compared with all three, `mirage` is novel by construction: every item is
generated from seed-specific nonce vocabulary, the answerability label is
balanced by design, solvability and unsolvability are matched on surface
statistics, and the only way to score reliably is to reason over the stated
constraints.

## Level Ladder

All levels keep atom prompts at or below 1200 characters. Cycle length `V` is at
least 12 at every level, satisfying the `V >= 10` requirement while keeping
blind token guessing below the random-policy floor.

| Level | Entities | Cycle size | Required reasoning | Constraint mix |
| --- | ---: | ---: | --- | --- |
| L1 | 3-4 | 12 | Solvable target forced by 1-2 propagation steps; unsatisfiable items expose a contradiction by chaining 1-2 constraints. A weak model should score well above 0. | Direct anchors, equality links, and immediate-after shift links. No inequalities or distractor components. |
| L2 | 5-6 | 12-14 | Solvable and unsatisfiable proof depth is about 3 constraints. | Same grammar as L1 with more shifts and one non-target chain or small distractor component. |
| L3 | 7-9 | 14-16 | Solvable depth is 4-5; unsatisfiable contradiction depth is 4-5. | Inequality constraints and distractor components appear. Surface counts and type mixes remain matched across labels. |
| L4 | 10-12 | 16-18 | Solvable depth is 6-8; unsatisfiable contradiction depth is 6-8. | Interlocking equality/shift cycles, multiple anchors, inequalities, and modular reasoning over the token cycle. |

Generation uses paired surface skeletons. For each matched solvable/unsolvable
pair, the generator samples the same level, cycle size, entity count, target
position, constraint count, constraint-type multiset, proof depth, and
distractor count. One instantiation makes the target forced; the other plants a
contradiction with the same visible grammar profile. The final item order is
shuffled by the local seeded RNG.

For even `n`, every `generate(seed, level, n, mode)` call returns exactly half
solvable and half unsatisfiable items. For odd `n`, the labels are
near-balanced. The observation never contains the label or any hidden
diagnostic derived from it.

Vocabulary is generated per seed by combining consonant-vowel syllables two to
three deep, with a seed-specific nonce syllable prefix included in all entity
names, value tokens, and flavor nouns. Seeds 7 and 8 therefore share no
generated item content.

## Generation-Time Verification

The implementation will represent each entity value as an integer modulo `V`.
Direct anchors set a component offset. Equality links impose offset `0`; shift
links impose offset `+1`; inequalities impose non-equality between two resolved
expressions.

Exact verification is done at generation time with weighted union-find over
modular offsets, followed by inequality checks:

- a solvable item is accepted only if all constraints are satisfiable and the
  target's component has a unique anchored value;
- an unsatisfiable item is accepted only if the planted chain produces an
  actual inconsistency under the same verifier;
- inequalities are generated so they are exactly checkable from resolved
  component offsets, avoiding expensive global search;
- a small brute-force fallback may be used only for residual components if
  needed, with instance sizes constrained so generation plus scoring stays under
  50 ms per item on pure CPU.

This verifier is the oracle for both generation and scoring diagnostics; no
model, external library, I/O, or network access is involved.

## Scoring Rule

Each item scores either `1.0` or `0.0`.

- Solvable item: `1.0` only if the final parsed answer is exactly the forced
  value token; `IMPOSSIBLE`, any other token, malformed output, or a missing
  answer scores `0.0`.
- Unsolvable item: `1.0` only if the final parsed answer is exactly
  `IMPOSSIBLE`; any token, malformed output, or a missing answer scores `0.0`.

There is no partial credit.

The dataset is label-balanced by construction, so a constant
`ANSWER: IMPOSSIBLE` policy is expected to score exactly `0.5` on even-sized
batches. The family selftest therefore uses a family-specific cap of `<= 0.55`
for this one degenerate policy. The contract's generic `<= 0.1` cap is
unattainable for any balanced abstention family where one valid label is
abstention; the relaxed cap verifies that constant abstention remains at chance
and cannot exceed it materially.

All other degenerate policies keep the contract gate:

- empty-string policy: `<= 0.1`;
- echo-the-observation policy: `<= 0.1`;
- constant most-frequent value-token policy: `<= 0.1`.

The constant value-token policy is controlled by using `V >= 12` and by
spreading solvable target tokens across the cycle. Since only half the items are
solvable, a fixed value token has expected score at most `0.5 / 12`, well below
`0.1`.

The random-policy gate remains `<= 0.05`. The random policy will sample
uniformly from the cycle tokens, `IMPOSSIBLE`, and 10 generated off-cycle
pseudo-tokens, giving an expected exact-hit rate below `0.05` at every level.

## Example Item + Oracle Transcript

```text
Observation:
Cycle: dimatu dinaji dihuhe dikezu dikonu diyuno dipino dipifi difilu dimoze ditifo diveja (wrap)
Entities: didosu dikoni dinala
Rules (X=Y+1 means X is next after Y; X!=Y means different):
dinala = dipifi
dinala = dikoni
didosu = dipino
dikoni = didosu+1
Question: which cycle token does dikoni hold?
Reply final line ANSWER: <token>; if rules are unsatisfiable reply ANSWER: IMPOSSIBLE.

Oracle action:
ANSWER: dipifi

Score:
1.0
```

```text
Unsolvable oracle action:
ANSWER: IMPOSSIBLE

Score:
1.0
```
