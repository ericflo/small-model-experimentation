# menders

## Capability Axis

`menders` evaluates debugging and program repair: the model must localize and
repair a small broken program from failing execution evidence, then use
rerun feedback across a bounded multi-turn episode.

## Task Description

Each item contains a program in an invented, line-oriented mini-language. The
language has no stable vocabulary: every keyword and variable name is generated
from the item seed using fictional consonant-vowel syllable combinations. Tokens
are 5-7 lowercase ASCII letters, always include at least one of `q`, `x`, or
`z`, and are rejected if they match a built-in denylist of English words,
language keywords, or programming mnemonics. Accepted keyword and variable names
are therefore fictional rather than English, Forth, BF, or assembly-like tokens.
Different seeds have disjoint program text and disjoint keyword and variable
tokens.

Programs are shown as numbered lines. The replacement source line in an action
does not include the line number.

The keyword legend is shown in every observation. The concrete words change per
item, but the semantics are:

| Form | Meaning |
| --- | --- |
| `<set> A X` | assign integer literal or variable `X` to variable `A` |
| `<add> A X Y` | assign `X + Y` to `A` |
| `<sub> A X Y` | assign `X - Y` to `A` |
| `<mul> A X Y` | assign `X * Y` to `A` |
| `<emit> X` | append integer literal or variable `X` to the output sequence |
| `<rep> K` | L3+ only: begin a non-nested repeat block run `K` times |
| `<done>` | L3+ only: end the repeat block |

`X`, `Y`, and `K` are small integer literals or item-local variables where the
form permits them. Source integer literals accepted from model actions are in
`[-20, 20]`; generated constants use the smaller `[-9, 9]` range. Repeat counts
accepted from model actions are in `[0, 6]`; generated correct repeat counts are
in `[1, 4]`. Input variables are ordinary variables initialized before line 1;
non-input variables start at 0. Programs use one or two integer input variables.
Each item has one visible check and exactly three hidden checks. The visible
check is fixed across turns and is not part of the score. Hidden checks reuse
the same program with different input values.
Outputs are exact sequences of emitted integers.

Generated visible and hidden inputs are distinct points from the grid
`[-4, 6]` for each input variable. Candidate correct and buggy programs are
brute-force executed over the whole grid for their arity. Any candidate whose
execution would exceed 12 emitted integers, produce an absolute value above
9999, or create invalid repeat structure is rejected during generation. During
model interaction, a syntactically valid patch that causes the execution cap to
be exceeded yields actual output `ERR` and fails every check, but does not crash
the environment.

Generation builds a private correct program from a level template, executes it
with a small brute-force reference interpreter over the candidate input grid,
then plants bugs by source-level mutation. Allowed mutations are wrong constant,
wrong variable reference, wrong opcode, and off-by-one repeat count. The
generator rejects and retries any instance where the buggy program does not fail
the visible check and all three hidden checks, so an empty or do-nothing policy
scores exactly 0.

Each episode observation is at most 800 characters and contains:

- a terse legend mapping the item keywords to the semantics above,
- the numbered current program,
- one fixed visible failing check: input values, expected output, actual output,
- hidden-check pass count, formatted like `checks passing: 0/3`,
- the exact action instruction.

Episode actions are single-line repairs and must parse within the contract's
96-token action budget. The bounded vocabulary, line count, literal ranges, and
execution caps above are chosen so generation plus scoring stays under
50 ms per item in pure stdlib Python.

The episode action grammar is:

```text
MEND <line#>: <replacement source line>
```

The verb is case-insensitive and surrounding whitespace is tolerated. Each valid
`MEND` replaces exactly one existing source line, then the environment reruns
the whole program and reports the new visible actual output plus hidden-check
pass count. Unknown line numbers, unknown tokens, invalid arity, invalid repeat
structure, out-of-range constants, or malformed actions produce only:

```text
Bad action. Use: MEND <line#>: <replacement source line>
```

Malformed actions do not change state, but the harness still consumes the turn.
Line numbers are 1-based decimal integers. Source text is canonicalized to
single spaces before diffing for the minimal-edit bonus.

Atom mode is the single-turn form. For every level, atom items have exactly one
planted bug and `max_turns == 1`. The atom prompt ends with:

```text
Reply with final line ANSWER: <line#>: <replacement source line>
```

The atom scorer reads the last case-insensitive `ANSWER:` line, applies that one
replacement if it is valid, and scores the resulting program. Source lines are
kept short enough for the required answer to parse within 64 new tokens, and
the full atom prompt must remain at or below 1200 characters.

## Scoring

Scoring uses the three hidden checks only. A program passes only if all hidden
checks match their expected output sequences exactly.

If all hidden checks pass:

```text
score = 0.8 + 0.2 * minimal_edit_bonus
```

The minimal edit bonus is `1.0` when the number of distinct final source lines
that differ from the buggy original is less than or equal to the number of
planted bugs. If there are extra final edits, the bonus is:

```text
planted_bug_count / distinct_changed_line_count
```

If not all hidden checks pass:

```text
score = 0.5 * (hidden_checks_passing / 3)
```

This partial credit is safe because the generator requires the buggy baseline
to pass 0 hidden checks. Empty output, no action, malformed-only transcripts,
and echoing the observation therefore score exactly 0 unless a later valid
repair changes the program.

The oracle policy reconstructs the current program from the item plus the
history of valid `MEND` actions, diffs it against the private correct program,
and patches the first differing line. In atom mode it emits the matching
`ANSWER:` line. This remains perfect even when noisy-oracle testing interleaves
random actions before oracle actions.

The random policy for selftest chooses a random existing line and emits a
syntactically valid replacement assembled from the current item vocabulary,
allowed opcodes, and allowed literal ranges. It never reads the private correct
program.

## Level Ladder

| Level | Episode program shape | Planted bugs in episode | Episode max_turns | Atom rule |
| --- | --- | ---: | ---: | --- |
| L1 | About 5 lines, one input, straight-line arithmetic and one or two emits; obvious wrong constant | 1 | 3 | Same level shape, exactly 1 bug, `max_turns == 1` |
| L2 | About 7 lines, one or two inputs, straight-line arithmetic with intermediate variables; wrong variable reference or wrong opcode | 1 | 3 | Same level shape, exactly 1 bug, `max_turns == 1` |
| L3 | About 9-10 lines, one bounded non-nested repeat block, two inputs, multiple emitted values | 2 | 6 | Same level shape, exactly 1 bug, `max_turns == 1` |
| L4 | About 12 lines, one bounded repeat block, aliased intermediate variables, near-miss opcodes, and an off-by-one repeat count | 3 | 7 | Same level shape, exactly 1 bug, `max_turns == 1` |

The noisy-oracle difficulty gate depends mostly on required bug count and turn
budget rather than semantic subtlety. The default ladder uses `(bugs, turns)` of
`(1,3)`, `(1,3)`, `(2,6)`, `(3,7)`, so L1 and L2 may tie while L3 and L4 should
decline. If implementation testing shows L4 is not monotone under the epsilon
0.5 noisy-oracle gate, the planned fallback is L4 with 2 bugs and 5 turns.
All choices remain within the contract caps: L1-L2 <= 4, L3 <= 10, L4 <= 14.

## Novelty Statement

Nearest public benchmarks checked:

| Benchmark | Task surface | Language | Scoring | How `menders` differs |
| --- | --- | --- | --- | --- |
| DebugBench | Repair/debug buggy programming snippets with optional runtime feedback; bugs are implanted into collected code snippets | C++, Java, Python | Pass-rate style functional correctness | `menders` uses a per-seed invented language with no public corpus items, bounded patch actions, hidden checks, and a minimal-edit bonus |
| QuixBugs | Repair small buggy algorithm programs specified by input-output tests | Python and Java | Test-suite adequate repair, with later studies checking overfitting | `menders` generates fresh programs and fictional tokens procedurally, supports multi-turn patch-and-rerun feedback, and requires every buggy baseline to fail all hidden checks |
| Defects4J | Repair real-world Java bugs with at least one failing test and a regression test suite | Java | Test-suite based patch adequacy, with known overfitting risk | `menders` is knowledge-free and corpus-free, has tiny documented semantics in every observation, and scores hidden behavioral checks plus final-edit minimality |

The family is novel by construction because the model never sees a real
programming language or public bug instance. The task surface is a generated
mini-language whose keyword and variable vocabulary changes with the seed. The
interaction is iterative patch-and-rerun rather than one-shot code generation.
The scoring combines hidden checks with a minimal-edit bonus, so a patch must
both repair behavior and avoid broad unnecessary rewrites.

Sources used for novelty research:

- DebugBench: https://arxiv.org/abs/2401.04621
- QuixBugs APR study: https://arxiv.org/abs/1805.03454
- Defects4J APR study: https://arxiv.org/abs/1811.02429

## Example Item + Oracle Transcript

Episode example generated with
`generate(seed=11, level=3, n=1, mode='episode')[0]`; reproducible coordinates
are `(seed=11, level=3, mode='episode', index=0)`.

Initial observation:

```text
Legend: zobapa=set zobaveg=add zobayo=sub zobapu=mul zobade=emit zobajod=rep zobanu=done.
CURRENT program:
1: zobapa zobageb 1
2: zobaveg zobaga zobageb zobadom
3: zobajod 3
4: zobaveg zobageb zobageb zobaga
5: zobayo zobatu zobageb zobadom
6: zobade zobatu
7: zobanu
8: zobapu zobaya zobageb 4
9: zobade zobaya
Visible: zobahu=-3,zobadom=1 expected=[-2,-4,-6,-10] actual=[2,4,6,28]
checks passing: 0/3
Program is buggy. MEND lines until actual matches expected and all checks pass.
Use: MEND <line#>: <replacement source line>
```

Turn 1 oracle action:

```text
MEND 2: zobaveg zobaga zobahu zobadom
```

Turn 1 next observation:

```text
Legend: zobapa=set zobaveg=add zobayo=sub zobapu=mul zobade=emit zobajod=rep zobanu=done.
CURRENT program:
1: zobapa zobageb 1
2: zobaveg zobaga zobahu zobadom
3: zobajod 3
4: zobaveg zobageb zobageb zobaga
5: zobayo zobatu zobageb zobadom
6: zobade zobatu
7: zobanu
8: zobapu zobaya zobageb 4
9: zobade zobaya
Visible: zobahu=-3,zobadom=1 expected=[-2,-4,-6,-10] actual=[-2,-4,-6,-20]
checks passing: 0/3
Program is buggy. MEND lines until actual matches expected and all checks pass.
Use: MEND <line#>: <replacement source line>
```

```text
done=False
```

Turn 2 oracle action:

```text
MEND 8: zobapu zobaya zobageb 2
```

Turn 2 next observation:

```text
Legend: zobapa=set zobaveg=add zobayo=sub zobapu=mul zobade=emit zobajod=rep zobanu=done.
CURRENT program:
1: zobapa zobageb 1
2: zobaveg zobaga zobahu zobadom
3: zobajod 3
4: zobaveg zobageb zobageb zobaga
5: zobayo zobatu zobageb zobadom
6: zobade zobatu
7: zobanu
8: zobapu zobaya zobageb 2
9: zobade zobaya
Visible: zobahu=-3,zobadom=1 expected=[-2,-4,-6,-10] actual=[-2,-4,-6,-10]
checks passing: 3/3
Program is buggy. MEND lines until actual matches expected and all checks pass.
Use: MEND <line#>: <replacement source line>
```

```text
done=True
{'score': 1.0, 'hidden_passing': 3, 'changed_lines': 2, 'planted_bugs': 2, 'all_hidden_pass': True}
```

Atom sub-example generated with
`generate(seed=11, level=1, n=1, mode='atom')[0]`; reproducible coordinates are
`(seed=11, level=1, mode='atom', index=0)`.

Atom prompt:

```text
Legend: zobabo=set zobaki=add zobaja=sub zobapo=mul zobaje=emit zobakit=rep zobasul=done.
CURRENT program:
1: zobabo zobagi 4
2: zobaki zobapu zobame zobagi
3: zobapo zobavab zobapu 2
4: zobaja zobakec zobavab 1
5: zobaje zobakec
Visible: zobame=5 expected=[13] actual=[17]
checks passing: 0/3
Program is buggy; exactly one line is wrong. Repair it so actual matches expected and all checks pass.
Reply with final line ANSWER: <line#>: <replacement source line>
```

Atom oracle answer:

```text
ANSWER: 1: zobabo zobagi 2
```
