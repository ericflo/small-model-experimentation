# Qwen3.5-4B Verified Macro Invention Experiment Log

## Scaffold

Created as a new experiment scaffold after the user selected verified macro invention from the
2026-07-09 forest review.

## Design freeze, before GPU

- Attached to `operator_and_skill_inventories`; no new program needed.
- Related-work search found fixed human-authored inventory scaling and failed large-bank
  shortlisting, but no result-bearing experiment that derives executable composite operators
  from a prior solved-program corpus.
- Chose the experiment-local vLLM runner for every proposal and solver sample. No Transformers
  inference comparison is permitted.
- Added a frozen latent-motif source grammar because uniform independent primitives contain no
  genuine abstraction distribution to recover.
- Adversarial review required paired no-reuse tasks, exhaustive true-depth checks, multiple
  matched random libraries, a highlighted-but-not-callable control, and conjunctive verdicts.
- Scoped the Qwen arm honestly as proposal/ranking within the closed length-2/3 macro language;
  Qwen-specific invention requires exclusive verified entries that carry unique correct solves.
- No model generation was launched before saving the intake, preregistration, and design review.

## CPU preflight

- Full generation produced 932 unique concrete programs and 932 unique frozen-probe behavior
  signatures: 800 construction, 12 smoke, and 120 scored tasks.
- Every scored task is behaviorally verified at true depth 5 by exhaustive search through depth
  4. Construction/evaluation concrete and behavioral overlap are both zero.
- The designed library reduces reuse tasks by exactly two surface calls (80/80); paired no-reuse
  median reduction is zero. These are substrate gates, not evidence.
- Repeated preparation was byte-identical after sorting multiset permutations before seeded
  shuffling.
- A tokenizer-only preflight found the original 200-program proposal view produced an 18,007-token
  prompt (18,903 with the registered generation reserve), exceeding the 16,384 vLLM context. Before
  any model load, reduced the frozen proposal view to 150 programs and regenerated every dependent
  library/hash. No scored output existed.
- Clarified before GPU that all arms share the same parent run seed and decode configuration, while
  the generic vLLM runner intentionally derives deterministic effective seeds from arm-qualified
  record ids; this is not a common-random-numbers design.

## Smoke v1: failed interface gate

- Ran the macro-proposal stage and solver smoke entirely through the experiment-local vLLM runner
  with the pinned `Qwen/Qwen3.5-4B` revision. No full generation was launched.
- The registered matched base/designed pool had 0.5972 overall parse rate: base 0.6111 and
  designed ceiling 0.5694, all above the 0.50 parser threshold.
- The same pool had 0.40046 answer truncation, far above the 0.05 ceiling. Every one of the 1,440
  all-arm solver samples force-closed its thinking stage, and 607 answer stages truncated.
- The designed ceiling produced zero valid macro-using candidates and zero oracle solves. Base
  oracle coverage was 1/12 = 0.0833, with the sole solve on a no-reuse task. The smoke gate failed.
- The strict whole-answer macro-proposal parser accepted 0/16 samples. A post-failure line-local
  audit found 18 behaviorally unique, train-supported candidate expansions in those same raw
  outputs. This audit is exploratory only: it does not populate the v1 Qwen arm or alter the
  failed-v1 verdict.
- Interpretation: v1 did not establish a usable macro surface. It did not test the full macro
  hypothesis. Saved the complete diagnosis in `reports/smoke_v1_failure.md`.

## V1 preservation and v2 refreeze, before another GPU call

- Preserved the historical config, smoke data, failed proposal, failed solver outputs, analyses,
  and exact source under `configs/smoke_v1.yaml`, `data/smoke_v1_frozen/`,
  `runs/proposal_v1_failed/`, `runs/smoke_v1_failed/`, `analysis/smoke_v1_failed/`, and
  `archive/smoke_v1_source/`.
- Left the construction corpus, proposal view, libraries, full tasks and hashes, hidden boundary,
  analyzer, controls, and full decision rules unchanged.
- Froze amendment 1 before v2 generation. V2 uses fresh seed `20260710` and ids
  `smoke-v2-reuse-NNN` / `smoke-v2-no-reuse-NNN`, with explicit disjointness against train, v1
  smoke, and full evaluation.
- Matched the v2 smoke to the preregistered full think@768 budget while retaining the 128-token
  answer cap. Added the same surface-first procedure and abstract alias-use example to both arms,
  while keeping the solver parser strict.
- Restricted the scored v2 gate to base and designed ceiling at matched K=12. The macro-use gate
  now requires valid alias use on at least two distinct reuse tasks.
- Added a non-scored train-only plan-given interface probe. It may diagnose mechanical formatting
  and alias calling only; it is not hypothesis evidence.
- Repaired the train-only proposal interface with a compact program-only prompt and a frozen
  line-local first-eight extraction rule. Full generation remains blocked until v2 passes.

## Interface attempt 2: failed before fresh smoke

- Ran only the non-scored, train-only, plan-given designed-alias transcription gate from
  amendment 1. It contained 4 records and n=4, for 16 total vLLM samples.
- Two records succeeded: `interface-v2-00::designed_ceiling` and
  `interface-v2-02::designed_ceiling`. A successful record had at least one strict completion
  that used a macro, had optimal surface length, and expanded exactly to the supplied plan.
- Four samples were strictly valid and all four used a macro. This shows some alias calling, but
  only 2/4 records cleared the required 3/4 reliability gate.
- Answer truncation was 12/16 = 0.75, failing the below-0.05 gate. All 16 samples exhausted the
  768-token thinking allowance and force-closed before the answer stage.
- The runner raised the registered gate failure and stopped. No fresh `smoke-v2-*` evaluation
  prompt was generated or shown to the model; the fresh smoke tasks remain model-unseen. Full
  generation also remains unrun.
- Preserved the exact attempt under `runs/interface_v2_failed/`,
  `analysis/interface_v2_gate_failed.json`, `configs/interface_v2.yaml`, and
  `archive/interface_v2_source/`.

## Amendment 2: interface attempt 3, frozen before GPU

- Classified the plan-given gate as transcription/formatting rather than induction: the verified
  primitive plan is already present in the prompt, so extended reasoning is not part of the
  capability being tested.
- Froze a retry of only that gate using the copied vLLM runner's exact `thinking: off` mode, n=4,
  and answer cap 128. Prompt contents, four targets, designed aliases, parser, executor, parent
  seed family, and success definition remain unchanged.
- Retained the same requirements: exact macro-using optimal transcription on at least 3/4 records
  and answer truncation below 0.05 across 16 samples. Failure stops before induction smoke.
- If attempt 3 passes, the scientific induction smoke still uses think@768 on the same unseen
  fresh tasks under amendment 1. No full metric or decision rule changes.

## Interface attempt 3: final gate failure and stop

- Retried only the same 4 task-independent, plan-given records through the experiment-local vLLM
  runner with `thinking: off`, n=4, and answer cap 128.
- All 16/16 samples passed the strict program parser, all 16/16 used at least one supplied macro,
  and 0/16 truncated. No-think therefore repaired the formatting, termination, and raw alias-use
  failures seen under think@768.
- Only `interface-v3-00::designed_ceiling` succeeded under the full exact criterion. Record
  coverage was 1/4, below the frozen at-least-3/4 gate, so attempt 3 failed.
- A committed post-gate audit regenerated the error taxonomy from raw rows. All 13 failed samples
  used multiple aliases and expanded beyond depth five (depth 6--10); 10/13 included the correct
  designated alias but appended unrelated aliases, while 3/13 omitted it. This audit describes
  the already-failed gate and does not change its decision.
- Preserved the exact config, outputs, verdict, and source under `configs/interface_v3.yaml`,
  `runs/interface_v3_failed/`, `analysis/interface_v3_gate_failed.json`, and
  `archive/interface_v3_source/`.
- Amendment 2 required a stop on failure. No fresh `smoke-v2-*` induction prompt and no full prompt
  was ever generated or shown to the model. The fresh scientific question remains untested.
- Closed the experiment as **interface gate failed; macro hypothesis unresolved**. No claim-ledger
  update is warranted. Any additional interface design is a material follow-up and must receive a
  new experiment directory, intake, design review, and preregistration rather than amendment 4.
