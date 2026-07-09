# stockade

Capability axis: bounded optimization under explicit constraints.

Paradigm: single-turn. Atom items are one-shot. Episode mode is a disclosed
sequential variant with one terse action per round; all offerings and shared
budgets are visible in the initial observation.

## Task Description

`stockade` asks the model to choose a subset of procedurally generated fictional
resource tokens that maximizes total integer value while respecting small
integer capacity limits and, at higher levels, interaction rules.

All task-specific content vocabulary is synthetic. Resource names, dimension
labels, unit labels, scenario labels, and rule labels are generated only from a
local `random.Random(seed)` using a closed syllable bank and a reserved token
shape such as `qx` + 2-3 syllables + `q`. The implementation must not use real
English words or real-world nouns for item content. Fixed parser keywords such
as `ANSWER`, `TAKE`, `cap`, `ban`, and `need` are protocol markers, not content
vocabulary.

### Atom Mode

Each atom item is one allocation instance. The observation gives:

- one fictional scenario label;
- generated capacity-dimension labels and integer caps;
- one line per resource with generated name, value, and cost vector;
- optional `ban:a,b` conflict rules;
- optional `need:a>b` dependency rules, meaning selecting `a` requires also
  selecting `b`;
- the required final answer grammar.

Atom answer grammar:

```text
ANSWER: <selection>
```

`<selection>` is either `NONE` or a comma-separated list of resource names from
the item:

```text
ANSWER: qxvazumq, qxkirnexq
ANSWER: NONE
```

The scorer extracts the last line matching `ANSWER:` case-insensitively.
Whitespace around names and commas is ignored. Resource names are generated in
lowercase and matched case-insensitively; canonical oracle output uses the
lowercase names. Order is ignored. Unknown names, duplicate names, extra
unparsed text in the selection, or malformed comma structure make the answer
malformed and score 0.0.

### Episode Mode

Episodes have 2 rounds at L1-L2 and 3 rounds at L3-L4. The initial `reset()`
observation shows the full scenario upfront: all round offerings, all shared
capacity dimensions, and all conflict/dependency rules. Capacity is shared
across the episode and carries over after every valid acquisition.

At each round the model may choose only resources offered in the current round.
The action grammar is:

```text
TAKE <selection>
```

`<selection>` is either `NONE` or a comma-separated list of current-round
resource names:

```text
TAKE qxvazumq, qxkirnexq
TAKE NONE
```

The verb and `NONE` are case-insensitive. Resource names are matched
case-insensitively against generated lowercase names. Order is ignored.
Duplicate names, unknown names, selecting a resource from a different round,
exceeding remaining capacity, violating a conflict with already acquired or
same-round resources, or selecting a resource whose prerequisite is not already
acquired or also selected in the same round forfeits that round. A forfeited
round adds no value and spends no capacity; the environment returns a curt
corrective note and proceeds to the next round.

## Novelty Statement

The three nearest public benchmarks found in the novelty scan are:

1. NPHardEval
   - Source: https://arxiv.org/abs/2312.14890 and
     https://github.com/casmlab/NPHardEval
   - Surface and scoring: dynamic LLM reasoning benchmark organized by
     computational complexity classes, with tasks including knapsack and meeting
     scheduling. It uses generated algorithmic questions, automatic answer
     checking, and reports average weighted accuracy.
   - Difference: `stockade` is not a known named problem suite and does not ask
     for textbook algorithm answers. It uses fictional tokens, compact bounded
     allocation instances, ratio-to-brute-force-optimum scoring, and an episode
     variant with shared budget carryover.

2. NL4Opt
   - Source: https://arxiv.org/abs/2303.08233
   - Surface and scoring: natural-language linear-programming word problems for
     optimization formulation. The competition separates entity labeling from
     meaning-representation generation; reported systems are scored with NER
     F1 and generation accuracy/logical-form correctness.
   - Difference: `stockade` does not test translation from natural language to a
     mathematical program. The complete discrete instance is already formalized
     in terse text, the model must directly choose a feasible high-value subset,
     and scoring gives objective partial credit by achieved value divided by the
     brute-forced optimum.

3. GraphArena
   - Source: https://arxiv.org/abs/2407.00379 and
     https://github.com/squareRoot3/GraphArena
   - Surface and scoring: LLM graph-computation benchmark with polynomial tasks
     and NP-complete graph challenges such as TSP, maximum independent set, and
     minimum vertex cover. Evaluation distinguishes correct, suboptimal
     feasible, hallucinatory/infeasible, and missing outputs, and reports
     accuracy-style metrics by task and difficulty.
   - Difference: `stockade` is not graph-reasoning over real public graph
     datasets. It uses fresh per-seed fictional allocation objects, multiple
     capacity dimensions, explicit conflict/dependency constraints, and small
     brute-force-verified optima designed for pure-stdlib generation and
     scoring under the Menagerie item budget.

## Level Ladder

Capacity tightness is defined per dimension as a sampled cap divided by the sum
of that dimension's costs across all resources in the atom, or across all
episode resources for the shared episode budget. Caps are clamped only as needed
to keep at least one feasible non-empty subset, and the generator rejects any
instance whose brute-forced optimum is empty or has value 0.

| Level | Atom resources | Dimensions | Atom interactions | Atom tightness | Episode tightness | Value range | Episode rounds and offerings | Episode interactions |
| --- | ---: | ---: | --- | --- | --- | --- | --- | --- |
| L1 | 6 | 1 | none | 40-48% | 34-39% | 2-12 | 2 rounds, 4 per round | 1 conflict |
| L2 | 8 | 2 | 1 conflict | 35-43% | 38-40% | 3-18 | 2 rounds, 5 per round | 2 conflicts |
| L3 | 10 | 3 | 3 conflicts, 2 dependencies | 30-38% | 24-29% | 4-24 | 3 rounds, 4 per round | 8 conflicts, 4 dependencies |
| L4 | 12 | 3 | 5 conflicts, 3 dependencies | 26-34% | 8-14% | 5-30 | 3 rounds, 5/4/4 offerings | 10 conflicts, 6 dependencies |

Brute-force verification remains small: atom search enumerates at most `2^12`
subsets, and episode search enumerates at most `2^16` global acquisition
subsets before storing the optimum. Scoring uses the stored optimum and does not
need to recompute it. The generator rejects items whose atom prompt exceeds
1200 characters or whose episode reset observation exceeds 800 characters.
Terse formatting keeps resource rows to forms such as `name v=9 c=2/4/1`,
with interaction rows like `ban:a,b` and
`need:c>d`.

## Scoring

Atom score:

```text
score = total_value(chosen_set) / brute_force_optimum
```

This ratio is awarded only if parsing succeeds and all constraints hold:

- every selected name exists in the item;
- no selected name appears twice;
- each capacity dimension total is at or below its cap;
- no conflict pair is jointly selected;
- every dependency `a>b` is satisfied by selecting `b` whenever `a` is selected.

Any malformed answer or constraint violation scores 0.0. `ANSWER: NONE` is
always feasible, but its achieved value is 0, so it scores 0.0 because the
generator guarantees a positive non-empty optimum.

Episode score:

```text
score = total_value(acquired_resources) / global_brute_force_optimum
```

The score replay follows the environment semantics round by round. A malformed
or illegal round action forfeits only that round. Valid previous acquisitions
remain, no capacity is spent on the forfeited round, and later rounds can still
earn value. The global optimum is computed over the full disclosed scenario
with the same sequential feasibility rules and shared capacity carryover.

Degenerate guard:

- the empty set is feasible but scores 0.0;
- the generator rejects zero-optimum instances;
- the generator rejects instances whose optimum is a trivial empty choice;
- constant `NONE`, empty-string, and echo-style policies should have mean score
  at or below the contract's 0.1 degenerate-resistance gate.

Random-floor gate:

This is a ratio-scored family and will use the contract's justified `<= 0.15`
random-policy gate rather than the stricter `<= 0.05` gate. L1 must remain
solvable enough to avoid measuring pure floor behavior, so a uniformly random
syntactically valid subset can occasionally be feasible and earn partial ratio
credit. The tuned generator measured a pooled random-policy mean of 0.0703 at
`n=60` per `(level, mode)` cell over three generation seeds and three policy RNG
seeds. The design keeps that floor low by giving episodes tighter decreasing
capacity ranges, increasing dimensions and interactions with level, shrinking
later-level episode offerings, and using value/cost decoys so random feasible
subsets are usually not near the optimum. The generator is tuned against the
selftest random policy with a measured mean at or below 0.10, and the selftest
fails if the measured mean exceeds 0.15.

Lazy-play floor (measured):

Adversarial verification probed four attack policies on 96 fresh items
(generation seeds 202 and 203, all levels, both modes). Measured pooled mean
scores were:

| Policy | Mean | Atom | Episode |
| --- | ---: | ---: | ---: |
| repeat-last-observation-token | 0.000 | - | - |
| constant NONE | 0.000 | - | - |
| alphabetically-first single legal name | 0.318 | 0.230 | 0.407 |
| highest-value single legal name each round | 0.497 | 0.366 | 0.628 |

Content-blind degenerate play (empty, constant, echo, token-parroting) scores
~0 and is what the contract's degenerate gate covers. Any syntactically
legal single-item pick earns partial credit that is inherent to
achieved/optimum ratio scoring on instances of 6-13 resources, because the
brute-forced optimum contains only 2-5 resources, so one legal pick is bounded
below by roughly 1/k of optimum. Pushing this class under the 0.15 random-floor
bar would require either optima spread over 15+ resources, which is impossible
under the item budgets, or baseline-subtracted scoring that would zero out
legitimate weak play and violate the contract's L1 requirement that a weak
model scores well above 0.

Scores in the ~0.3-0.5 band indicate value-blind or greedy single-pick
heuristics, not chance. The capability signal of this family is the climb from
greedy-heuristic play (~0.5) toward the brute-forced optimum (1.0), and the
random/degenerate floors (~0.0-0.09) only anchor the bottom of the scale.

## Example Item + Oracle Transcript

### Atom example (generated, seed 42, L1)

Reset observation:

```text
case qxuypkfgq
cap qxjqpeobq=11
items:
qxfojcdxq v=2 c=3
qxjatjyuq v=7 c=6
qxynsyxoq v=3 c=1
qxomwmshq v=8 c=3
qxfabtprq v=5 c=7
qxiqiskkq v=3 c=7
ban:none need:none
Reply with final line 'ANSWER: <comma-separated names>' or 'ANSWER: NONE'.
```

Oracle answer:

```text
ANSWER: qxjatjyuq, qxynsyxoq, qxomwmshq
```

Score:

```python
{'score': 1.0, 'achieved': 18, 'optimum': 18, 'feasible': True, 'malformed': False}
```

### Episode example (generated, seed 42, L2)

Reset observation:

```text
case qxgqwgkmq
cap qxropglvq=18 qxjbxfraq=27
R1: qxalmyzuq v=3 c=1/10; qxqjbjwhq v=10 c=1/7; qxcttrgdq v=10 c=6/10; qxdfbgieq v=12 c=4/6; qxjmyfznq v=8 c=9/7
R2: qxnbqxkxq v=10 c=5/8; qxrdmntiq v=3 c=3/7; qxyzyqurq v=16 c=10/6; qxjowkdpq v=3 c=6/6; qxftpuwyq v=6 c=1/2
ban:qxalmyzuq,qxqjbjwhq;qxcttrgdq,qxyzyqurq need:none
Round 1 action: TAKE <comma-separated current names> or TAKE NONE.
```

Oracle action:

```text
TAKE qxqjbjwhq, qxdfbgieq
```

Following observation:

```text
OK.
left qxropglvq=13 qxjbxfraq=14
R2: qxnbqxkxq v=10 c=5/8; qxrdmntiq v=3 c=3/7; qxyzyqurq v=16 c=10/6; qxjowkdpq v=3 c=6/6; qxftpuwyq v=6 c=1/2
Action: TAKE <comma-separated current names> or TAKE NONE.
```

Oracle action:

```text
TAKE qxyzyqurq, qxftpuwyq
```

Following observation:

```text
OK.
Done.
```

Final score:

```python
{'score': 1.0, 'achieved': 44, 'optimum': 44, 'feasible': True, 'malformed': False, 'forfeits': 0}
```
