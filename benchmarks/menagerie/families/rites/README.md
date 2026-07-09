# rites

## 1. Capability

Protocol/state-machine compliance: execute a compact, freshly generated procedure while mentally tracking hidden-but-documented state and flags.

## 2. Task description

`rites` is a multi-turn family with an atom variant. Each item defines one fictional rite as a finite state machine. A marker starts at a documented start state and the goal is to move it to a documented goal state in the minimum number of legal actions. After the initial observation, the environment never echoes the current state or flag values; the solver must update them from the spec plus its own accepted/refused/malformed action history.

All content tokens for the rite name, state names, action names, and flag names are generated per item from a fixed nonword syllable pool such as `qa, zu, xi, vo, ky, re, ul, ja, ob, ne, fi, go, ha, lu, py, se, ti, wa, xo, ye, za, bi, cu, do`. Tokens use two syllables by default and a third only for deterministic collision repair, keeping generated content tokens 4-6 lowercase ASCII letters after denylist checks. The item RNG is local: `item_seed = ((seed & 0xffffffff) * 1000003 + level * 10007 + mode_code * 101 + item_index) & 0xffffffffffffffff`, where `mode_code` is `0` for atom and `1` for episode. The generator uses `random.Random(item_seed)` only; it never uses Python `hash()`, global RNG state, wall clock, I/O, or network.

Spec notation grammar:

```text
token       := [a-z]{4,6}
bit         := "0" | "1"
flags       := "-" | flag "=" bit ("," flag "=" bit)*
rule        := action ":" transit | action ":" toggle
transit     := state ">" state guard?
toggle      := "!" flag state_limit? guard?
state_limit := "@" state
guard       := "?" flag "=" bit
```

Sample rule line:

```text
voqa:zubi>relu?xigo=1
```

This means action `voqa` moves the marker from state `zubi` to state `relu` only when flag `xigo` is `1`; otherwise the action is refused and changes nothing.

Episode initial observation template, with all generated rules on the `RULES:` line:

```text
RITE <rite>
FORMAT: ENACT <action>
S=<start>;G=<goal>;F=<flag=0,...|->
RULES: <rule>; <rule>; ...
KEY: a:x>y?f=1 moves x->y if f=1. a:!f@x?g=0 flips f at x if g=0. State/flags hidden after this. ACCEPTED applies; REFUSED/MALFORMED do not; all consume turns.
Reply: ENACT <action>
```

Action grammar:

```text
ENACT <action-name>
```

Parsing is case-insensitive and tolerant of surrounding or repeated whitespace. `<action-name>` must be one documented fictional action token; a syntactically valid but unknown name is refused.

Rules have two action kinds:

- Transit: `act:src>dst` moves the marker from `src` to `dst`; `?flag=bit` adds a required flag value.
- Toggle: `act:!flag` flips that flag; `@state` restricts it to one marker state; `?flag=bit` adds a guard. State restrictions are checked before guards.

Per-turn feedback strings are exactly:

- `ACCEPTED. Reply: ENACT <action>` for a legal action that does not complete the rite.
- `THE RITE IS COMPLETE.` for a legal action that moves the marker to the goal; `done=True`.
- `REFUSED: wrong place. Reply: ENACT <action>` when the action's source state or state restriction is unsatisfied.
- `REFUSED: condition unmet. Reply: ENACT <action>` when the state requirement is satisfied but a guard is false.
- `REFUSED: unknown action. Reply: ENACT <action>` when the line is well-formed but the action name is not in the spec.
- `MALFORMED. Reply exactly: ENACT <action-name>` when the line does not match the action grammar.
- `THE RITE STALLS.` when `max_turns` is hit without reaching the goal; `done=True`.

All refused, malformed, and unknown actions consume a turn and leave state unchanged. If a final turn does not reach the goal, the terminal observation is `THE RITE STALLS.` rather than revealing the last hidden state.

Episode scoring:

```text
score = 0.0                                      if the goal is not reached within max_turns
score = clamp(optimal_len / turns_used, 0.0, 1.0) otherwise
```

`turns_used` counts every accepted, refused, unknown, and malformed action up to the first completion. Actions after completion are ignored by the scorer. `optimal_len` is the shortest legal action count computed by BFS over `(marker_state, flag_values)`.

Atom mode uses the same generated spec, `max_turns=1`, and a prompt ending:

```text
Find a shortest legal sequence to reach G. Reply with one final line:
ANSWER: <action1> <action2> ... <actionK>
```

The atom scorer extracts the last `ANSWER:` line case-insensitively, splits the action names on whitespace, canonicalizes names case-insensitively, and simulates the whole sequence from the initial state. Atom scoring:

```text
score = clamp(optimal_len / K, 0.0, 1.0)
```

only if `K > 0`, every listed action is known and legal at the moment it is executed, and the marker ends at the goal after the final listed action; otherwise `score = 0.0`. Any valid shortest path scores `1.0`; a longer legal path to the goal receives ratio credit.

Optimality ground truth is exact. During generation, BFS explores the product graph of at most `9 * 2^3 = 72` states with at most 12 actions, so reachability and shortest paths are well under the 50 ms/item contract. The generator rejects and regenerates machines where the goal is unreachable or the BFS shortest length is not in the level target range.

Distractors are generated as extra source-specific or guarded actions, plus toggles that are useful only at the right time or legal only in selected states. A cheap exact dynamic program under the family `random_policy` (uniform over documented action names) rejects items with excessive random expected score; target caps are `<=0.10` for L1, `<=0.06` for L2, `<=0.02` for L3, and `<=0.01` for L4, keeping the suite mean below the `0.05` random-floor gate while leaving L1 solvable.

Character budgets are guaranteed by construction. Tokens are capped at 6 chars; L4 caps are 9 states, 12 actions/rules, and 3 flags; each rule is a single punctuation-coded fragment with no prose. The worst-case episode template with 12 guarded/restricted rules fits under 800 chars, and atom mode only appends the short `ANSWER:` instruction, leaving it under 1200 chars. Implementation must still assert `len(reset_obs) <= 800` for episodes and `len(atom_prompt) <= 1200`, rejecting/regenerating any item that violates the cap.

## 3. Novelty statement

- [TextWorld](https://arxiv.org/abs/1806.11532) is a sandbox for training and evaluating agents in text-based games, including generated games with state tracking and reward assignment. `rites` differs by giving the complete finite-state protocol in the first prompt, using only per-item fictional tokens rather than adventure-game objects or natural language affordances, hiding state after reset, and scoring exact shortest-protocol execution rather than game reward.
- [PlanBench / PDDL-style planning evaluations](https://arxiv.org/abs/2206.10498) test LLM plan generation and reasoning about actions/change across automated-planning domains. `rites` differs by avoiding standard PDDL/IPC domains, generating fresh fictional action/state vocabularies, supporting an interactive episode where illegal actions consume turns, and scoring by simulation ratio against a BFS optimum.
- [SmartPlay](https://arxiv.org/abs/2310.01557) evaluates LLM agents across multiple games such as Rock-Paper-Scissors, Tower of Hanoi, and Minecraft variants. `rites` differs by using no known game rules or reusable domain semantics: every item is a new compact state-machine spec with nonce vocabulary, hidden-but-documented state, deterministic legality feedback, and exact optimality-based scoring.

## 4. Level ladder

| Level | States | Actions | Flags and guards | Optimal path length | max_turns |
| --- | ---: | ---: | --- | ---: | ---: |
| L1 | 4 | 5 | 0 flags; unguarded transits plus source-specific distractors | 2 | 4 |
| L2 | 5 | 7 | 1 flag; at least one required guarded transit; toggle unguarded or state-restricted | 3 | 4 |
| L3 | 7 | 10 | 2 flags; several guarded transits; at least one state-restricted toggle | 5-6 | 10 |
| L4 | 9 | 12 | 3 flags; interacting guards; some toggles both guarded and state-restricted | 8-9 | 14 |

The exact L3 and L4 target length is selected deterministically from the item seed, and generation rejects machines outside the selected target.

## 5. Example item + oracle transcript

Seed: `7`.

L2 episode item from `generate(7, 2, 1, "episode")[0]`, reset observation:

```text
RITE goti
FORMAT: ENACT <action>
S=pyne;G=zulu;F=wawa=0
RULES: haza:bifi>zulu?wawa=1; kyxi:bifi>zuvo; xoxo:pyne>zuvo; goqa:zuvo>pyne; doha:ticu>zuvo; lune:!wawa@bifi; zucu:pyne>bifi
KEY: a:x>y?f=1 moves x->y if f=1. a:!f@x?g=0 flips f at x if g=0. State/flags hidden after this. ACCEPTED applies; REFUSED/MALFORMED do not; all consume turns.
Reply: ENACT <action>
```

Oracle transcript:

```text
Turn 1 action: ENACT zucu
Turn 1 observation: ACCEPTED. Reply: ENACT <action>
Turn 2 action: ENACT lune
Turn 2 observation: ACCEPTED. Reply: ENACT <action>
Turn 3 action: ENACT haza
Turn 3 observation: THE RITE IS COMPLETE.
```

L1 atom item from `generate(7, 1, 1, "atom")[0]`, prompt:

```text
RITE qase
S=hati;G=xobi;F=-
RULES: haxi:hazu>fiul; xiul:hazu>xobi; gose:fiul>hati; luti:hati>fiul; hane:hati>hazu
KEY: a:x>y?f=1 moves x->y if f=1. a:!f@x?g=0 flips f at x if g=0. State/flags hidden after this. ACCEPTED applies; REFUSED/MALFORMED do not; all consume turns.
Find a shortest legal sequence to reach G. Reply with one final line:
ANSWER: <action1> <action2> ... <actionK>
```

Oracle answer:

```text
ANSWER: hane xiul
```
