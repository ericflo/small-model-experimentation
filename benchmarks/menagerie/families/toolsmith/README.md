# toolsmith

Capability axis: orchestrating dependent tool calls when later calls require opaque values returned by earlier calls.

## Task Description

`toolsmith` items present a compact registry of fictional typed tools. Every tool name, type name, and value token is generated from seed-local fictional syllables; no real API names, domain words, or reusable cross-seed vocabulary appear. A tool signature is displayed as `qorvex(brimq)->soltr`, or as `qorvex(brimq,zenth)->soltr` for a two-argument join.

In episode mode, the model starts with one value token, or two at higher levels, each with a known fictional type. The goal is to obtain and submit the opaque value of a stated goal type. Tool outputs are produced only by the environment as deterministic hashes of the item id, tool name, and argument values. The generator simulates the unique true path to precompute the target final value, but that target is never shown in any observation.

The type graph is constructed so exactly one dependency path leads from the start type(s) to the goal type. Distractor tools either require unreachable input types or produce dead-end output types, so they cannot create an alternate goal path. Callable distractor outputs are asserted at generation time to be distinct from all true-path values, preserving the unique-path guarantee. Valid calls return a new value token and its type. Malformed actions and unknown tools, arity mismatches, type mismatches, or unknown values return terse `ERR ...` observations and never crash.

Episode initial observation template:

```text
Tools: <sig>; <sig>; ...
Have: <value>:<type>[, <value>:<type>]. Goal:<type>.
Act one line: CALL name(value) or CALL name(value1,value2); SUBMIT value.
```

Episode actions are parsed as one terse line:

```text
CALL toolname(value)
CALL toolname(value1,value2)
SUBMIT <value>
```

Valid call observation template:

```text
OK <value>:<type>
```

Error observation templates are `ERR syntax`, `ERR unknown tool`, `ERR arity`, `ERR value`, and `ERR type`.

Episode scoring is `1.0` iff the submitted value equals the precomputed target. Otherwise, let `m` be the number of true-path calls and let `c` be the number of true-path calls executed with the exact required tool and argument values after their dependencies were available, counting each true-path call at most once and allowing independent join prerequisites in either order. The score is `0.5 * c / m`. If there are no valid calls, or no true-path calls are credited, the score is `0.0`.

Token matching for tool names, argument values, submitted values, and atom answer tool names is case-insensitive because all generated tokens are lowercase.

In atom mode, the same registry, start value(s), and goal type are shown, but no tool execution occurs. Atom items use linear chains only, with no joins, so the call-order answer is unique.

Atom prompt template:

```text
Tools: <sig>; <sig>; ...
Have: <value>:<type>. Goal:<type>.
Give tool names in call order. Final line: ANSWER: tool1 -> tool2 -> tool3
```

Atom scoring is `1.0` for an exact oracle sequence. Otherwise, a malformed or missing `ANSWER:` line scores `0.0`; a non-empty answer scores `0.5 * p / m`, where `p` is the longest initial prefix of tool names matching the oracle sequence and `m` is the oracle chain length. An empty sequence scores `0.0`.

## Novelty

The three nearest public benchmarks are:

| Benchmark | Task surface | Vocabulary source | Scoring method | How `toolsmith` differs |
| --- | --- | --- | --- | --- |
| [NESTFUL](https://arxiv.org/abs/2409.03797) | Nested API sequences where earlier API outputs are passed into later calls. | Real RapidAPI-derived APIs plus curated/synthetic function schemas. | Partial/full sequence match and executable API pass rate. | `toolsmith` uses seed-disjoint fictional syllables, not real APIs; the environment deterministically hashes every call result; a unique type-graph path defines the target; scoring is exact final value plus objective dependency progress. |
| [Berkeley Function-Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) | Single-turn, parallel, multiple, and multi-turn function-calling tasks across language, REST, memory, and web-search categories. | Human, public, enterprise, or benchmark-contributed real-world function surfaces and domain terms. | AST matching, executable checks, state-based multi-turn checks, and category accuracy aggregation. | `toolsmith` has no real language/API surface, no domain semantics, no state database, and no alternative valid call set; it isolates typed dependency chaining over opaque generated values. |
| [ToolBench / ToolLLM](https://arxiv.org/abs/2307.16789) | Multi-round tool use over large collections of REST APIs, including API retrieval and multi-tool instructions. | Real RapidAPI documentation, names, parameters, and responses. | ToolEval pass rate and win rate, using an LLM-based evaluator over solution paths. | `toolsmith` has no API retrieval, no real endpoints, no natural-language tool semantics, and no LLM judging; every item is procedural, executable offline, and scored by exact machine-checkable values. |

## Level Ladder

| Level | Episode structure | Episode distractors | Episode max turns | Atom structure |
| --- | --- | ---: | ---: | --- |
| L1 | 2 true calls, linear chain, one start value | 3 | 4 | Linear depth 2 |
| L2 | 3 true calls, linear chain, one start value | 5 | 4 | Linear depth 3 |
| L3 | 4 true calls: main chain plus one side branch from a second start value feeding one two-argument join | 6 | 5 | Linear depth 4 |
| L4 | 6 true calls with two start values and two two-argument joins | 8 | 7 | Linear depth 5 |

The tight turn budgets are deliberate: chain length plus submit, with at most 1 spare turn, keeps noisy-oracle difficulty monotone in level and makes L3/L4 demand near-perfect play while staying within the contract caps of 10/14.

## Generation Notes

Seed-local syllable generation uses compact pseudo-syllables with mandatory rare anchors (`q`, `x`, `z`, or `v`) supplied by the category marker in every generated tool, type, and value token. The salt first zig-zag maps any signed Python integer seed to a non-negative integer, then encodes it with a no-leading-zero variable-length base alphabet that excludes the category markers. This is injective by construction, accepts negative seeds and seeds at or beyond the old 7140 boundary, and keeps cross-seed token disjointness because the marker immediately follows the salt and cannot be mistaken for a salt character. Tokens are checked for uniqueness within the item and against the action grammar words.

## Example Item + Oracle Transcript

### Episode Example

Item id: `toolsmith-s7-L2-e0`
Max turns: `4`

Initial observation:

```text
Tools: vcaqzixdon(vcaxba)->vcaxv; vcaqziszex(vcaxcam)->vcaxpuq; vcaqbezovs(vcaxpuq)->vcaxz; vcaqzexikz(vcaxpuq)->vcaxqox; vcaqvos(vcaxbiz)->vcaxziz; vcaqraxahi(vcaxcam)->vcaxqi; vcaqxuqbig(vcaxv)->vcaxcam; vcaqzodo(vcaxzed)->vcaxke
Have: vcazro:vcaxba. Goal:vcaxpuq.
Act one line: CALL name(value) or CALL name(value1,value2); SUBMIT value.
```

Oracle transcript:

```text
CALL vcaqzixdon(vcazro)
OK vcavjahju:vcaxv
CALL vcaqxuqbig(vcavjahju)
OK vcavjara:vcaxcam
CALL vcaqziszex(vcavjara)
OK vcavsubbu:vcaxpuq
SUBMIT vcavsubbu
SUBMITTED
```

Score:

```python
{'score': 1.0, 'c': 3, 'm': 3, 'submitted': 'vcavsubbu', 'target_hit': True}
```

### Atom Example

Item id: `toolsmith-s7-L1-a0`

Atom prompt:

```text
Tools: vcaqlenur(vcaxy)->vcaxw; vcaqsovl(vcaxb)->vcaxqiv; vcaqjiz(vcaxpa)->vcaxfu; vcaqme(vcaxpa)->vcaxroq; vcaqxogva(vcaxne)->vcaxpa
Have: vcazwu:vcaxne. Goal:vcaxroq.
Give tool names in call order. Final line: ANSWER: tool1 -> tool2 -> tool3
```

Oracle answer:

```text
ANSWER: vcaqxogva -> vcaqme
```

Score:

```python
{'score': 1.0, 'p': 2, 'm': 2, 'target_hit': True}
```
