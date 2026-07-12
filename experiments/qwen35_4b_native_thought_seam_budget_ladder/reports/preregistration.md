# Preregistration: Native-Thought Seam Budget Ladder

Frozen before any model call. CPU task generation, exhaustive identifiability
checks, gate-reachability checks, and unit tests may precede the immutable design
commit. No decoded model output informed this design.

## 1. Scientific purpose and boundary

The direct parent, `qwen35_4b_native_thought_jacobian_value_transport`, opened
its 160-token seam once and found all 48 traces cap-bound. It therefore made no
value or J-space inference. This new experiment answers only:

> What is the smallest frozen budget in 256/512/1024 that exposes a naturally
> reached close-and-answer seam on the inherited task, and does it replicate?

This is an interface calibration, not a capability comparison. It fits no
probe, performs no intervention, trains nothing, and allocates no claim ID.

## 2. Fixed model and inference contract

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Transformers, bf16, SDPA, unpadded batch one, `use_cache=True`.
- Native-thinking chat template with exactly one `<think>` token and no
  pre-existing `</think>` token.
- Exact token IDs: open 248068, close 248069; smoke must verify them.
- Temperature 0.6, top-p 0.95, top-k 20; every value explicit.
- Maximum 16 answer tokens after natural close; EOS may stop earlier.
- Prompt cap 768 and total sequence cap 2048. The CPU envelope verifies
  `768 + 1024 + 16 <= 2048`.
- A forward-input-length audit must show the complete prompt on the prefill and
  one token on every later cached forward. Any scientific row failing this
  contract invalidates its whole stage.

Transformers is intentionally used despite the seam itself not requiring
activations: the immediate successor does. vLLM and cache-free samples are not
mixed into any cell. The failed cache-free parent remains lineage only.

## 3. Fixed task and freshness contract

The prompt grammar, operation menu, one-token aliases, and depth-two task family
are inherited exactly from the direct parent. Tasks are generated under a fresh
seed into two disjoint splits:

- 16 `budget_selection` tasks;
- 24 `seam_confirmation` tasks.

Each task contains eight visible and eight hidden I/O examples. Exhaustive
enumeration over every concrete depth-two DSL pipeline requires all candidates
matching the visible examples to share one first-operation type. `negate` may
appear second or as a distractor but cannot be the target because its first-step
compositions are algebraically reorderable. First-operation counts differ by at
most one inside each split.

All 40 fingerprints must be unique and have zero overlap with both
`qwen35_4b_jacobian_value_transport` and
`qwen35_4b_native_thought_jacobian_value_transport`. No file in `benchmarks/` is
read or imported.

## 4. Exact generation and right-censoring semantics

Three traces are sampled per task with stable task/trace seeds. On selection,
each trace is generated once with the maximum rung of 1024 thought-generation
steps. Let `c` be the one-indexed generation step at which the close token was
emitted. A trace is naturally closed at rung `B` iff `c <= B`. If no close is
emitted by `B`, the row is a cap contact at that rung and has no answer, even if
the same sampled path closes at a later rung.

Thus lower-rung cells are nested right-censored views of the same 48 traces, not
independent generations. No between-rung p-value or nominal independent sample
size is permitted. This construction makes the smallest-cap rule exact and
avoids spending different random traces at each cap.

If a close is naturally emitted, generation continues for at most 16 answer
tokens without injecting any delimiter. A parse succeeds only when the natural
answer contains the inherited exact `First: <alias>` grammar. Cap-bound and
EOS-before-close traces are neither parsed nor assigned an answer.

Raw rows are held in memory and written only after the full stage completes.
The automated decision is computed before any human inspection of decoded
thought content. Partial rows cannot select a cap.

## 5. Usable trace and headroom definitions

A trace is usable at cap `B` iff:

1. its close step is at most `B`;
2. its naturally generated answer parses; and
3. it contains at least 16 thought tokens before close.

Correctness means the parsed alias equals the exhaustively identifiable first
operation. Usable success is computed only over usable traces. A mixed usable
task has at least one correct and one incorrect usable trace among its three
samples. Cap-bound rows cannot manufacture incorrect members of a mixed task.

Both parse denominators are reported: parse/all traces and parse/natural closes.
Only the latter enters the frozen parse gate.

## 6. Selection gate and cap freeze

For each of `[256, 512, 1024]`, compute on the 16-task/48-trace selection split:

- natural close rate >= 0.80 (at least 39/48);
- parse rate conditional on natural close >= 0.90;
- usable traces >= 32;
- usable exact success in [0.05, 0.95]; and
- mixed usable tasks >= 6.

Select the smallest rung satisfying every condition. The anticipated cap does
not enter the rule. If none passes, write terminal `NO_BUDGET_SELECTED`; the
confirmation split remains unopened and no new rung may be appended.

The gate is mathematically reachable: the CPU receipt freezes 48 traces, minimum
close count 39, minimum usable count 32, and minimum mixed count six, with a
feasible 0.5 success construction.

## 7. Untouched confirmation

Only `B*`, the selected smallest cap, is run on the 24-task/72-trace confirmation
split, with a disjoint base seed. No other rung is generated or scored there.
Confirmation requires:

- natural close rate >= 0.80 (at least 58/72);
- 95% Wilson lower bound for natural close >= 0.70;
- parse rate conditional on natural close >= 0.90;
- usable traces >= 48;
- usable exact success in [0.05, 0.95]; and
- mixed usable tasks >= 8.

Passing yields `NATURAL_SEAM_REPLICATED`. Any failure yields
`SEAM_NOT_REPLICATED`. A failed selected cap cannot be rescued by opening a
larger cap on confirmation. The selection row hash is verified before the
confirmation model is loaded.

The confirmation gate is mathematically reachable: 72 traces, minimum close
count 58, minimum usable count 48, and minimum mixed count eight. Perfect close
has Wilson lower bound above 0.70.

## 8. Frozen terminal interpretation

- `NO_BUDGET_SELECTED`: this prompt/task interface still lacks a valid natural
  seam through 1024; no conclusion about J-space or continuation value.
- `SEAM_NOT_REPLICATED`: the selected seam was unstable under fresh tasks/seeds;
  no value study is licensed.
- `NATURAL_SEAM_REPLICATED`: a fixed natural interface exists. This licenses a
  separate experiment at `B*`; it is not value decodability, causal transport,
  or capability gain.

No terminal label can be upgraded using older forced-close results, parent rows,
another backend, additional samples, decoded qualitative examples, or a post hoc
threshold.

## 9. Contract for the next experiment

Only `NATURAL_SEAM_REPLICATED` permits a successor. That successor must:

1. use the exact selected cap, backend, prompt grammar, and natural-close rule;
2. create new fit and confirmation tasks/seeds;
3. label a prefix with disjoint continuation outcomes, never whole-trace labels;
4. replay each exact prefix as its own causal sequence rather than assuming a
   historical activation is invariant to suffix length;
5. build and audit post-bf16 random/non-J controls separately at every live
   prefix/sequence length; and
6. remain oracle-only until a later non-oracle method beats frozen and
   matched-compute sampling on untouched procedural tasks.

## 10. Artifact and publication contract

Commit the task splits, manifest, gate-reachability receipt, model smoke,
selection/confirmation summaries, complete rows, derived metrics, and terminal
narrative. Preserve negatives. Update both program ledgers and shared synthesis
when the terminal label is known. Run `make check`, synchronize with main, push,
and inspect CI at every publication boundary.
