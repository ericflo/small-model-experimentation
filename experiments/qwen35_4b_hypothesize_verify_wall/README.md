# Does the installable hypothesize-and-verify skill move the structure wall?

## Research Program

- Program: `structured_execution_and_compilers` (× `posttraining_and_adaptation`)
- Program question: the structure-wall arc terminates in C36 ("the fixed 4B cannot PROPOSE deep op-structure — a model-level law across substrates"); the induction-install arc terminates in C45 ("a GENERAL hypothesize-and-verify serial strategy is installable and transfers to a held-out family"). These two terminal laws make OPPOSITE predictions about one untested cell: aim the strategy at the wall.
- Prior anchors: C32/C36 (`qwen35_4b_structure_or_values`, `qwen35_4b_crosssubstrate_structure` — the wall + the behavioral skeleton metric), C45 (`qwen35_4b_meta_induction` — the installable strategy, committed train_general.jsonl), C44 (serial-compute law; CoT mandatory), C39 (execute-given-rule gate discipline), C25/C34 (guided search never beat brute — but guidance was logit-level, never a TAUGHT strategy).

## Question

Can the model's own serial hypothesize-and-verify loop — elicited by prompt scaffold, transferred
zero-shot from the C45 adapter, or installed by a small DSL-native reasoning-SFT dose — lift
op-STRUCTURE proposal on the list/string DSL substrates where the wall was measured? C36 predicts
no arm moves (the wall is a capability bound); C45's logic predicts the serial strategy converts
proposal into enumerate-and-check, which the model CAN execute (C32: execution-given-structure ≈ 1.0).

## Hypothesis

The wall arc never tested a TAUGHT serial strategy — its negatives are direct sampling (C17),
logit-guided search (C25/C34), and steering (C20). C44 says the missing ingredient for induction is
serial compute in the CoT channel; C45 says the loop template generalizes across (affine) families.
If the wall is a proposal-PROCEDURE deficit, arms 1–3 climb (scaffold ≥ some lift; DSL-native SFT
most); if it is a hypothesis-SPACE deficit (the model cannot even enumerate candidate op-types
against I/O evidence), all arms stay at base and C36 hardens. Falsifiers are asymmetric and both
informative: any significant lift breaks C36's "un-installable" reading; a flat result with a
passing trap-gate scopes C45's strategy to retrieval-adjacent families.

## Setup

- Model: Qwen3.5-4B; QLoRA r32/α64 for SFT arms (C45 recipe: bs 2 + grad-accum 8, 2 epochs).
- Dataset/task source: contamination-free procedural DSL identification tasks from the C36
  `families.py` (byte-identical copy): **list** (16 prims) and **string** (13 prims) families;
  behavioral min-depth verification rejects shallower-equivalent compositions (C13 discipline).
- Train/eval split: frozen eval set, committed (`data/eval_tasks.jsonl`): n=30 tasks per
  family × depth {2,3} (120 total), generated once (seed 71), each with 8 visible + 6 hidden
  examples. SFT-arm training tasks are depth-1/2 compositions with op-composition dedup vs ALL
  eval tasks (0 leakage, verified at build time). The C45-adapter arm trains only on the committed
  digit-affine `train_general.jsonl` (different substrate entirely — zero leakage by construction).
- Arms (shared eval; matched K=12 think samples at budget 1024, identical decode params; the ONLY
  variable is strategy provenance):
  - **base**: C36's `ident_prompt` unchanged — the sample-more baseline (known: skeleton-coverage
    falls off a cliff by depth; replicates the wall).
  - **scaffold** (training-free, the pure-elicitation arm): same task rendering + an explicit
    enumerate-and-verify procedure in the prompt ("shortlist candidate op types consistent with
    the I/O evidence; compose a candidate pipeline; mentally execute it on example 1; check
    against example 2; revise; only then write the function").
  - **c45_zero**: C45 adapter (regenerated from committed train_general.jsonl), zero-shot on the
    DSLs with the base prompt — does the installed meta-skill transfer across substrates?
  - **dsl_sft**: reasoning-SFT on ~1,500 PROGRAMMATIC hypothesize-and-verify CoT traces (C45's
    trace-template method — interpreter-generated, no teacher model) on depth-1/2 list+string
    tasks disjoint from eval; deploy think-mode. The install ceiling.
- Baseline to beat: base arm at matched K (sample-more); secondary anchor: rand-skelfill@R
  (random skeletons + value-fill at matched interpreter budget — C32's value-fungibility control,
  CPU-only).
- Controls / gates (pre-registered, run.py hard-stops):
  1. **Trap gate (C43-mandatory)**: oracle-skelfill (true skeleton + value search) must solve
     ≥ 0.85 of eval tasks per family×depth — otherwise a proposal null is uninterpretable.
  2. **Skill-installed gate for dsl_sft**: the trained model must reproduce the taught trace
     format on ≥ 0.8 of held-out depth-1 tasks AND its depth-1 correctness must not collapse vs
     base in the deploy channel (C29/C43 forgetting check) — else the arm tests a failed install.
  3. **c45_zero regen-sanity gate**: the regenerated C45 adapter must reproduce C45's committed
     headline (held-out a7 induction-via-generation 0.905; gate = historical − 2×SE at n=200 →
     0.87; initially miscalibrated to 0.95 from C44's shift number, corrected — see log) BEFORE
     its DSL eval is spent — else a flat arm is a failed rebuild, not evidence. Measured: 0.920.
  4. Leakage: dedup at op-TYPE-SEQUENCE (skeleton) level between all SFT training tasks and all
     eval tasks (0 exact-skeleton overlap at any depth, verified at build); depth-3 window overlap
     is stratified, not excluded (see decision rule).
  5. Smoke path with `_smoke` artifact suffixes throughout (smoke can never poison a full run);
     run.py sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself; launch with
     pipefail per docs/compute_environment.md.
  6. Base-arm anchors: a cheap NO-think base pass ties this frozen eval to C36's historical
     numbers, with a pre-registered branch: if base-THINK@1024 itself clears the historical wall
     (skeleton-coverage@K ≥ 0.25 at depth 3), the experiment's frame shifts from "can the strategy
     move the wall" to "serial-compute dose already moves it" (a C44 result) and arm contrasts
     are interpreted against that elevated base. Decode pinned in configs/default.yaml
     (temperature 0.8/top_p 0.95 sampling, greedy pass separate; answer_max 512).
- Primary metric: **probe-robust skeleton-coverage@K** per cell — ≥1 of K samples structure-correct
  under a MIMICRY-ROBUST extension of the C36 metric (review must-fix): some fill of the TRUE
  skeleton must reproduce the model program's behavior on the 8 visible inputs AND on 6 fresh probe
  INPUTS (labels never shown to the model) — lookup-table/hardcoded programs match visible but
  diverge on probes and are counted as a per-arm **mimicry rate**, reported alongside the legacy
  C36 metric for continuity. Secondary (characterization only, non-verdictive): greedy skeleton@1,
  full-solve coverage@K, per-family cells, depth-2 cells, code-parse rate per arm (a null with
  parse-rate far below base is pre-registered as "format-confounded", not "no transfer").
- Decision rule (pre-registered; power-corrected per review): PRIMARY contrast = each arm vs base
  on probe-robust skeleton-coverage@K, **pooled list+string at depth 3** (n=60 tasks/arm; min
  detectable ~0.067, power 0.90 at true +0.15), one-sided paired bootstrap, **Holm-corrected
  across the 3 arm-vs-base contrasts**. **C36's "un-installable/un-elicitable" reading falsified**
  iff any corrected contrast has CI lower bound > 0 AND point diff ≥ +0.10 — with depth-3 results
  STRATIFIED by trained-window overlap (both/one/zero depth-2 sub-skeletons of the eval task seen
  in dsl_sft training): the falsification claim requires lift in the zero-or-one-window stratum
  (both-windows lift = composition of banked shallow structure, C24/C28 territory, reported as
  such). **C45 scoped** iff no corrected contrast moves (honestly: "no effect ≥ ~0.15 detectable")
  with all gates passing. Scaffold lift without SFT lift = elicitation-sufficient; SFT-only lift =
  installable-only (noting the pre-registered asymmetry: dsl_sft carries vocabulary+strategy, the
  scaffold strategy only). Prior-art conditioning: C28/C33 already installed depth-3 structure
  FROM oracle depth-3 content; the novel cell here is PROCEDURE-only transfer across depth
  (traces are depth-1/2 only; C21 is the negative prior for answer-banking across depth).
- Compute-parity (review must-fix): report per-arm mean prompt tokens, generated tokens/sample,
  total tokens/task, forced-close rate; base is additionally re-presented at K′ matched to the
  scaffold arm's total generated tokens. Pre-committed contingency: if a NULL arm shows
  forced-close > 50% at depth 3, run a budget-2048 probe on the depth-3 subset before concluding.
- Treatment freezing (review must-fix): the scaffold prompt text is checked in VERBATIM at
  `configs/scaffold_prompt.txt` (zero op names / substrate hints — C30/C31 hint-leak discipline;
  includes an explicit do-not-hardcode instruction) and the dsl_sft trace generator implements the
  SAME factorized procedure: evidence extraction (length/sign/order/duplicate deltas) → per-stage
  candidate SHORTLIST from a fixed truth-independent feature→candidate rulebook → compose from
  shortlists in fixed order → mentally execute on example 1 with intermediate states → on failure
  localize the divergent stage and revise → verify on examples 2–3 → only then emit code. Traces
  are truth-BLIND by construction (candidate order and shortlists are pure functions of visible
  I/O; verified at build time by regenerating with the oracle blinded and byte-comparing); tasks
  the blind procedure cannot solve within the trace budget are dropped (C45 cot→None pattern,
  drop rate reported); kept traces' final candidate must also solve the hidden set (train-data
  purity filter, C45-standard). Traces train into the THINK channel with the final code block as
  the answer (bank-the-thoughts `train_lora_think` pattern — a stated recipe change vs C45, which
  aligns train and deploy channels).
- Oracle-only metrics: oracle-skelfill gate, purity/leakage checks, rand-skelfill anchor. No
  oracle signal reaches any arm's prompt, training data selection, or decoding.
- Hidden-label boundary: hidden examples grade full-solve only; skeleton metric uses visible
  behavior + the true skeleton (evaluation-side oracle, standard for the wall arc).
- Known limits (pre-registered): single LoRA seed per SFT arm; K=12/n=30 resolves ~0.25 effects,
  not 0.10 subtleties; budget 1024 may truncate deep enumerate loops (report forced-close rates —
  C45 needed ≥400 tokens, C47 saw 99% truncation still work); register family deferred.

## Run

Smoke:

```bash
python scripts/run.py --smoke
```

Full (measured ~6 h main pipeline + ~3 h budget-2048 contingency probes on the RTX 4090;
pre-run estimate was 9–11 h):

```bash
python scripts/run.py            # idempotent; safe to re-run after interruption
python scripts/analyze.py
```

## Results

Full narrative in `reports/report.md`; stats in `runs/verdict.json`; saga in `experiment_log.md`.

1. **The wall holds (C36 hardens).** No arm clears the pre-registered pooled depth-3 Holm
   contrast: dsl_sft +0.033 (p_holm 0.55), scaffold +0.033 (0.41), c45_zero −0.017 (0.82); base
   depth-3 probe-robust cov@12 = 0.05 (wall replicates at think@1024; no C44 frame-shift —
   `base_think_clears_wall=false`). Zero-or-one-trained-window stratum: +0.019, n.s.
2. **The procedure installs, dramatically, WITHIN taught depths.** dsl_sft (1,476 truth-blind
   d1/d2 traces, think-channel) doubles depth-2 structure proposal: list 0.37→0.70, string
   0.33→0.57; deployable greedy@1 list 0.10→0.37; parse 1.00; zero d1 forgetting (0.85→0.85).
   Flat at depth-3 (0.10/0.07 vs 0.03/0.07): **serial-strategy installs are depth-local** —
   C21's cross-depth negative extends from banked answers to banked procedures.
3. **C45's skill is substrate-local.** The regenerated adapter beats its own in-family headline
   (0.920 vs 0.905) yet transfers at ~zero to the DSLs, with active interference (list d2
   0.37→0.00) and degraded parsing (0.70–0.79).
4. **Scaffold null is format-confounded** (pre-registered reading): parse collapses to 0.44–0.67
   vs base ~0.89 while coverage matches/edges base — the treatment text disrupts formatting more
   than it elicits search.
5. **Guards:** mimicry ≤0.014 everywhere (probe-robust metric armed, barely needed);
   oracle-skelfill 1.00 all cells; rand-skelfill ≤0.07; traces byte-verified truth-blind; 0
   task-level skeleton leakage (51/1,476 trace found-pipelines coincide with eval d2 skeletons,
   0 with d3).
6. **Budget-2048 contingency (pre-committed): the d3 wall is partially BUDGET-limited for
   everyone.** c45_zero 0.017 (null doubly confirmed); scaffold 0.150 — but the post-hoc
   base@2048 control doubles too (0.05→0.10, forced-close 0.84→0.75), and the paired
   matched-budget scaffold edge is +0.050 (CI-lo +0.000, p 0.043–0.051 across seeds —
   borderline, post-hoc, not verdict-grade). Serial-compute dose is the operative depth-3
   margin; a small procedure margin beyond dose is unresolved at this n; even 2048 truncates
   75–100% of thinking (dose-curve unfinished).

## Interpretation

The wall is not a missing procedure. The model demonstrably learns and executes the full
hypothesize-shortlist-check-revise loop (that is what 0.37→0.70 at parse 1.00 with zero
forgetting means) and gains nothing one composition step past where the loop was practiced. With
C21 (answers don't climb), C24 (diversity is in-depth dose), and C44 (induction is serial-compute
limited): **depth itself is the resource, and neither answers, diversity, nor procedure banks it
across.** C36 survives its strongest challenger; C45 gains a regime clause (family-general within
its substrate, zero across substrates). Deployable takeaway: DSL-native procedure-SFT is the
strongest in-depth structure-proposal lever measured on this substrate; the depth frontier still
belongs to external search (C34/C35).

## Knowledgebase Update

- Program evidence updated: `structured_execution_and_compilers` × `posttraining_and_adaptation`
- Claim ledger updated: C48 (this experiment); C21 annotated (cross-depth negative extends to
  procedures); C45 annotated (substrate-local scope clause)

## Artifacts

- `src/` — `families.py` (byte-identical C36 copy), `gen_lib.py` (C36 copy + the torch-2.12
  `OOM_ERRORS` AcceleratorError patch from `qwen35_4b_verifier_free_banking`)
- `scripts/` — `make_tasks.py` (frozen eval + SFT-train tasks, leakage check), `gen_traces.py`
  (programmatic DSL hypothesize-and-verify CoT), `train_lora.py` (C45 recipe), `eval_arms.py`
  (shared K-sample eval + skeleton metric), `run.py` (orchestrator, gates), `analyze.py`
- `data/` — frozen eval sets, SFT training pairs, trace corpus
- `runs/` — adapters (moved external before commit), eval JSONs, verdict
- `reports/` — design_review.md, report.md, artifact_manifest.yaml
