# Gauntlet round 1: breadth-first agentic expert iteration Experiment Log

## Scaffold

Created as a new experiment scaffold (program: agentic_breadth_installation).

## 2026-07-09 — gym built, design reviewed, round-1 harvest launched

- Designed the 12-family gym (10 trained + 2 held-out) against the public
  menagerie axis descriptions only; family module contract in
  `reports/gym_design.md`. All 12 families implemented with generators,
  verifiers, oracle/random policies, selftests: oracle 1.0 everywhere,
  degenerate/random floors <= 0.15 (runeward's 0.25 constant-IMPOSSIBLE
  floor documented, gate <= 0.30).
- Pipeline: `src/harness.py` (batched atoms + lockstep episode driver over
  the template `src/vllm_runner.py`), `scripts/harvest.py` (per-family
  sharded, resumable), `scripts/build_sft.py` (verified + naturally-closed +
  terse-answer filter, lucky-guess gate, action_ok gate, per-item/rollout
  caps, (level,kind) round-robin family cap), `scripts/train_think.py`
  (C48 think-channel QLoRA recipe, revision-pinned), `scripts/eval_gym.py`,
  `scripts/bench.py` (fresh-seed enforcement, aggregate-only storage,
  null-calibration support).
- Smoke harvest 1 (think 2048, 240 atom gens, 72 rollouts): keeper rate
  0.146; five families at zero. Diagnosis: (a) terminal-marker pollution —
  the runner generates through `<|im_end|>`; gym parsers kept the literal
  marker, so word/glyph/list answers and ALL episode actions failed
  (mechanism lens verified: stripping markers raises correct 40->94/240);
  (b) at think 2048 the model solves hard items in-think, force-closes while
  double-checking, and the stage-2 answer restarts an explanation truncated
  before ANSWER.
- Three-lens adversarial design review (`reports/design_review.md`): all
  three lenses sound_with_fixes; every must-fix applied pre-GPU (marker
  stripping + selftest enforcement, lucky-guess answer_domain gate,
  last_action_ok turn gate, round-robin family cap, pre-registration
  reconciliation, null calibration + three-way decision rule, yield-gate
  fallback).
- Smoke harvest 2 (think 4096, fixed parsers): atom keeper rate 0.146 ->
  0.433; episodes from 0.00 success everywhere to 0.50-1.00 in most cells
  (ferrier 1.00 with zero forced closes). Remaining thin: glyphgate atoms
  0.08, loomfix atoms 0.08, stallwright atoms 0.00 (the model never
  concludes its optimization deliberation even at 4096 — matches the
  public stockade floor; yield gate will handle).
- Launched the full round-1 harvest (~8.5 h projected) — then KILLED it ~35
  min in on user direction: iteration speed is the research budget; a
  multi-hour-to-first-feedback loop wastes a 44-second instrument. Doctrine
  codified in gym_design.md ("Iteration doctrine") and the program charter.
- Cut over to the fast profile (`configs/fast.yaml`): atoms 40/40 L1-L2 K=2
  with family-adaptive think budgets (4096 only for slow-closing families),
  episodes 16/level K=2; harvest ~80 min; menagerie quick base-vs-adapter
  every round on a fresh seed. Full profile reserved for decision rounds.
  Fast harvest running under `runs/harvest_round1_fast/`.

## 2026-07-10 — round 1 fast: FLAT on both instruments; mechanism diagnosed

- Fast harvest yields: 849 SFT examples across 9/10 families (stallwright 0 —
  never closes optimization deliberation even at 4096; pre-registered
  starved-family handling). Atom targets canonicalized to terse
  `ANSWER: <value>` after the filter audit showed the ≤96-token answer filter
  rejecting the model's verbose-but-correct answers wholesale.
- Training required length-bucketed batching (random batching padded to
  ~max_length: 60 s/step -> mixed 10-50 s/step).
- First menagerie event ever recorded for an install (quick, seed 52001,
  arms base/base/adapter): base0 0.1156, base1 0.1500, adapter 0.1271 —
  FLAT inside the same-seed null spread. NULL CALIBRATION KEY FINDING:
  base-vs-base same-seed spread is ~0.034 (three base realizations 0.116 /
  0.150 / 0.171) — ~3x the published cross-backend 0.011; single quick arms
  only detect large effects. Seven of ten families are hard-zero in all arms;
  quick variance lives entirely in toolsmith/warren.
- Gym-internal eval (greedy, think 1024, held-out seeds): base mean 0.184 vs
  adapter 0.187 — flat even in-distribution, held-out families flat, and the
  dominant failure UNCHANGED: parse_fail ~90/100 atoms in most families (the
  forced-close cascade; e.g. caravan 98/100 -> 95/100).
- MECHANISM READ: round-1 training was near-self-distillation — loss 0.27 ->
  0.20; ~1,400 own-text think tokens per example carry ~zero gradient and the
  ~10 tokens encoding the new behavior (terse ANSWER) are diluted AND
  conditioned on a naturally-completed chain, a state deployment rarely
  reaches at budget 1024 (the naturally-closed-only filter excluded the
  deployment-critical post-force-close state entirely).
- ROUND 2 (same harvest, sharper signal): (a) forced-close recovery arm —
  76 correct-after-cut samples train truncated_think + </think> + terse
  ANSWER with think-as-context (weight 0); (b) per-token loss weights
  (prompt 0 / think 0.2 / close+answer 1.0, echo-QLoRA precedent);
  (c) bs4/ga4. 925 examples (467 atom / 76 recovery / 382 episode turns).
  Primary readout: does gym parse_fail collapse; then a fresh-seed quick
  event.

## 2026-07-10 (cont.) — vLLM runtime LoRA is a SILENT NO-OP; rounds 1-2 were never measured

- Round-2 gym-eval came back BYTE-IDENTICAL to round-1's (1200/1200 identical
  generations from two different adapters, sha-identical row files) — a
  behavioral impossibility that unmasked the real bug: **vLLM 0.24 runtime
  LoRA silently does not apply Qwen3.5-4B PEFT adapters.** In-process probe
  (same engine, same prompts, greedy, lora_request on vs off): token-identical
  outputs. Mechanism: adapter tensors are named `base_model.model.model.
  layers.*` but the served composite keeps text layers under
  `model.language_model.layers.*`; vLLM's mapping matches nothing, no error.
  ALL adapter arms so far (gym-evals and menagerie events at seeds
  52001/52002) actually measured the BASE model. Codified in
  docs/vllm_inference.md with the on-vs-off behavioral gate; menagerie's
  --adapter flag is presumed equally affected (same vLLM path) — merged
  checkpoints via --model-id are the verified deployment.
- Fix: scripts/merge_adapter.py merges LoRA deltas into the FULL COMPOSITE
  checkpoint by explicit name mapping (a text-only merge_and_unload
  checkpoint does not load — vLLM's Qwen3.5 class requires the composite
  config); runner gains model_override; bench gains a merged arm.
- Merged round-2 probe on training items: the trained recovery behavior IS
  installed and visible (immediate terse `ANSWER: <value>` after a
  force-closed chain, where base emits a verbose re-explanation). Rounds 1-2
  "flat" results were measurement artifacts, not nulls.

## 2026-07-10 (cont.) — REAL round-2 results: gym +0.52, MENAGERIE QUICK +0.22

- Gym-internal (greedy, think 1024, held-out seeds; merged deployment):
  mean 0.184 -> 0.701 (+0.518). Parse failures collapsed (caravan 98/100 ->
  8/100; kilnrite/runeward/loomfix -> ~0). BREADTH TRANSFERS IN-GYM: held-out
  never-trained families brinework +0.540 and spindle +0.608; harvest-starved
  stallwright (zero training examples) 0.000 -> 0.395 by pure transfer.
- Menagerie --model-id guard rejects local checkpoint paths (string-prefix
  one-model check) despite the README documenting checkpoint runs —
  instrument bug flagged for the user; worked around via the HF parity
  backend (--backend qwen), where PEFT adapters genuinely apply and base-vs-
  adapter on the same seed is deterministic and noise-free.
- **FIRST VALID BLACKBOX EVENT (quick, seed 52004, HF backend both arms):
  base 0.1396 -> adapter 0.3625, DELTA +0.2229** — 7x the pre-registered
  +0.03 bar. Per-family deltas align with trained axes (chronicle +0.750,
  siftstack +0.625, toolsmith +0.354 to 1.000, mirage +0.250, lockpick/rites/
  warren +0.125) while untrained/starved axes stay flat or dip (menders 0,
  sirens 0, stockade -0.125) — the signature of genuine transfer, not
  leakage. Replication (quick seed 52005) + medium confirmation (seed 52006)
  running per the decision rule.
