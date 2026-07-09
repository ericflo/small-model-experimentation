# lockpick

Capability axis: active rule induction and exploitation -- choose informative probes against a hidden symbol machine under a tight probe budget, induce the input->output rule, then invert it to construct an opening input for a target output.

## Task Description

`lockpick` is a knowledge-free, procedurally generated task family for the Menagerie atoms and episodes contract. Each item creates a fictional glyph machine. The model sees a per-item alphabet and must infer how the hidden machine maps a fixed-length glyph sequence to another fixed-length glyph sequence.

The alphabet contains 6 glyph tokens at levels 1-2 and 8 glyph tokens at levels 3-4. Tokens are generated from seeded fictional syllables using an uncommon consonant inventory `z v k q x j r n m th sh zh` and vowel inventory `a e i o u y`; each token is two CVC-like syllables joined as a single lowercase word, such as `zarvok`. The generator rejects duplicates and any token that appears in a small built-in blocklist of obvious English words. Tokens, machine labels, and item identifiers are fictional and local to the item.

Machine inputs are fixed-length glyph sequences: length 3 for L1-L2, length 4 for L3, and length 5 for L4. The hidden machine applies a deterministic seeded composition of invertible sequence transforms and returns the transformed glyph sequence. All randomness uses local `random.Random(seed)` only.

The model is scored only on exploitation. It never receives credit for stating the rule. Given a target output, it must submit an input that the hidden machine maps exactly to that target.

## Rule Space

Primitive transforms are cheap, deterministic, invertible, and parameterized as follows:

- `ROTATE k`: cyclically rotate the whole sequence right by `k`, where `1 <= k < sequence_length`.
- `REVERSE`: reverse the full sequence.
- `SWAP i j`: swap two distinct zero-based positions.
- `SHIFT k scope`: advance each affected glyph `k` steps along the alphabet in the order listed in the prompt, wrapping around, where `1 <= k < alphabet_size`. `scope=all` shifts every position; `scope=pos p` shifts only zero-based position `p`.
- `CONDITIONAL predicate A B`: for L3+, if the original input to the conditional satisfies the predicate, apply primitive `A`; otherwise apply primitive `B`. Predicates are `contains g` or `at p is g`. Branch primitives are non-conditional primitives sampled from the same item alphabet and length, including shifts.

Levels draw from this space as follows:

- L1: one primitive.
- L2: composition of two primitives.
- L3: composition of three primitives, with at most one conditional.
- L4: composition of four primitives, including at least one conditional.

## Interaction

### Episodes

Episode mode is multi-turn active probing. The initial observation gives the alphabet, sequence length, mechanism family, number of hidden moves, target output, probe budget, max turns, and action grammar. It describes the hidden machine only as a fixed composition of position rotations, reversals, swaps, and glyph shifts along the listed alphabet, with possible conditionals; it does not reveal parameters, order, predicates, or scopes.

- `PROBE g1 g2 ... gL`: returns the machine output for that input while probes remain.
- `OPEN g1 g2 ... gL`: terminal. The episode ends after this one attempt whether right or wrong.

Probe budgets and turn caps:

- L1: K=2 probes, `max_turns=4`.
- L2: K=3 probes, `max_turns=4`.
- L3: K=7 probes, `max_turns=10`.
- L4: K=11 probes, `max_turns=14`.

Probing after the budget is exhausted returns a curt corrective observation. Malformed actions also return a curt corrective observation, never crash, and are never terminal. Corrective observations truncate over-long unknown tokens to 24 characters plus `...`, so observations stay within the contract's 800-character gate.

### Atoms

Atom mode is single-shot passive induction with `max_turns=1`. The prompt shows the alphabet, sequence length, mechanism family, number of hidden moves, target output, and generator-chosen informative probe input->output pairs:

- L1: 2 pairs.
- L2: 3 pairs.
- L3: 4 pairs.
- L4: 5 pairs.

The atom prompt asks for a final line:

`ANSWER: g1 g2 ... gL`

Atom prompts must stay within the contract's 1200-character gate, and parsing follows the Menagerie atom rule: use the last whitespace-tolerant, case-insensitive `ANSWER:` line.

## Solvability

L1-L2 atom items are guaranteed determinate from the shown passive pairs. During generation, the family enumerates the full non-conditional rule space at the level depth, filters to every rule consistent with all shown pairs, and emits the item only when there is an opening input that maps to the target under every consistent rule. The stored oracle solution is that determinate witness.

L3-L4 atom prompts do not enumerate the full conditional composition space. Their passive pairs may under-determine the rule; active probing in episode mode is the intended path at those levels.

## Scoring

Scoring is binary and machine-checkable. The scorer applies the hidden machine to the submitted opening input and returns `1.0` iff the output equals the target exactly; otherwise it returns `0.0`.

Any input whose image is the target counts. The scorer does not compare against a stored solution and does not award partial credit, because partial credit would reward echoing the target. The generator chooses a target by applying the machine to a sampled input, stores an opening input `X*` that maps to that target, and resamples if `machine(T) == T`, so echoing the target can never open the lock.

Anti-leak generation guards also reject any item whose stored solution text is a contiguous substring of the listed alphabet order. In atom mode, generation additionally rejects any shown probe pair whose output exactly equals the stored solution, so the oracle answer cannot appear verbatim in passive examples.

## Oracle

The oracle knows the item's private fields. In episode mode it plays a level-scaled informative probe plan, then opens with `X*`:

- L1: 2 probes, then `OPEN X*`.
- L2: 3 probes, then `OPEN X*`.
- L3: 6 probes, then `OPEN X*`.
- L4: 10 probes, then `OPEN X*`.

In atom mode the oracle returns `ANSWER: X*`. Because the noisy-oracle policy has more opportunities to be disrupted at higher levels and the probe plans grow with the transform space, the epsilon=0.5 noisy-oracle gate is expected to decline monotonically with level.

## Level Ladder

- L1, easy: alphabet size 6, length 3, one primitive, 2 active probes or 2 passive examples. A competent human should solve it in under 30 seconds.
- L2, moderate: alphabet size 6, length 3, two composed primitives, 3 active probes or 3 passive examples, and a tight 4-turn episode cap.
- L3, hard: alphabet size 8, length 4, three composed primitives, may include one conditional, 7 active probes or 4 passive examples.
- L4, frontier: alphabet size 8, length 5, four composed primitives, includes at least one conditional, 11 active probes or 5 passive examples.

The ladder increases alphabet size, sequence length, composition depth, conditional branching, and the amount of hypothesis testing required before exploitation.

## Novelty Statement

Nearest public benchmarks:

- RULEARN/IDEA evaluates LLM agents in interactive simulated environments where they gather observations, form and refine rules, and solve problems; `lockpick` narrows the surface to fictional glyph-sequence machines, uses a strict `PROBE`/`OPEN` vocabulary, and scores only exact inverse construction rather than rule statements or broader task success.
- Oracle black-box environment interaction defines hidden input->output functions that agents explore over limited turns; `lockpick` uses a smaller, fully invertible compositional transform grammar, per-item fictional alphabets, atom and episode modes, and a scorer that verifies only whether the submitted input maps to the target.
- DeepMind Alchemy is a public meta-RL benchmark with procedurally resampled latent causal structure, hypothesis testing, and action sequencing in a Unity/symbolic chemistry environment; `lockpick` removes RL reward maximization and visuomotor or chemistry vocabulary, replacing them with stdlib-only textual glyph probes and binary target opening.

Novelty by construction: every glyph token, the machine framing, and all naming are procedurally generated fictional vocabulary; the task uses no real-world nouns and is not a known puzzle format verbatim.

## Example Item + Oracle Transcript

```text
lockpick-7-L1-atom-0

Fictional glyph lock: infer the hidden machine from these probe pairs. Alphabet: zyvzhash jeqqax vezhzeq xemnan nonmar jashvim. Sequence length: 3. Machine: a fixed composition of hidden moves - position rotations, reversals, position swaps, and glyph shifts along the alphabet as listed (wrapping); some moves may be conditional on the input. Hidden moves: 1. Probe pairs: zyvzhash zyvzhash zyvzhash -> zyvzhash zyvzhash zyvzhash; zyvzhash jeqqax vezhzeq -> vezhzeq jeqqax zyvzhash. Target output: jeqqax zyvzhash xemnan. End with a final line in the form ANSWER: <g1> <g2> <g3>.

ANSWER: xemnan zyvzhash jeqqax

{'score': 1.0, 'opened': True, 'reason': 'opened', 'submitted': ['xemnan', 'zyvzhash', 'jeqqax']}
```

```text
lockpick-7-L2-episode-0

Observation 1:
Fictional glyph lock: infer the hidden machine by probing, then open it. Alphabet: themxek xethzhum shiqshyz thyqzhyv shuzshun rymmyth. Sequence length: 3. Machine: a fixed composition of hidden moves - position rotations, reversals, position swaps, and glyph shifts along the alphabet as listed (wrapping); some moves may be conditional on the input. Hidden moves: 2. Target output: rymmyth shiqshyz thyqzhyv. Probes remaining: 3. Max turns: 4. Grammar: submit one line like PROBE <g1> <g2> <g3> or OPEN <g1> <g2> <g3>; use alphabet glyphs only.
Action 1:
PROBE themxek themxek themxek
Observation 2:
PROBE themxek themxek themxek -> themxek themxek themxek. Probes left: 2. Target: rymmyth shiqshyz thyqzhyv.
Action 2:
PROBE themxek xethzhum shiqshyz
Observation 3:
PROBE themxek xethzhum shiqshyz -> shiqshyz xethzhum themxek. Probes left: 1. Target: rymmyth shiqshyz thyqzhyv.
Action 3:
PROBE shiqshyz xethzhum themxek
Observation 4:
PROBE shiqshyz xethzhum themxek -> themxek xethzhum shiqshyz. Probes left: 0. Target: rymmyth shiqshyz thyqzhyv.
Action 4:
OPEN thyqzhyv shiqshyz rymmyth
Observation 5:
The mechanism clicked open.

{'score': 1.0, 'opened': True, 'reason': 'opened', 'probes_used': 3, 'submitted': ['thyqzhyv', 'shiqshyz', 'rymmyth']}
```
