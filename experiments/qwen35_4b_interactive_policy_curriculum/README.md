# Interactive policy curriculum: oracle DAgger to execution-reward RL

**Status:** finished

This completed preregistered study tested whether correction at the model's
own live multi-turn states could improve Qwen3.5-4B beyond the incumbent C53
blend and unlock guarded whole-episode reward training. It did not: the
DAgger warm start failed its mechanism gate decisively, so RL, controls, and
Menagerie were correctly cancelled.

## Research Program

- Primary program: `agentic_breadth_installation`
- Supporting programs: `process_control_and_tool_use`,
  `posttraining_and_adaptation`
- Program question: after C53's one-time emission-policy install, can training
  on live state transitions and their consequences buy a second increment of
  substrate-general agentic capability?
- Prior anchors: C50/C53 (blackbox install and second wall), C5 (adaptation
  must beat frozen alternatives), C11 (test-time feedback alone does not beat
  matched sampling), and `qwen35_4b_oracle_process_grpo` (compact executable
  process state is learnable).

## Question

Can Qwen3.5-4B learn one shared
`observe -> state -> probe/tool/revise -> verify -> commit` policy across
firewall-clean interactive environments when supervision follows the states
the policy actually visits and the final update is driven by multi-turn
execution reward? Does that policy transfer to unseen proxy families and beat
the incumbent C53 blend on fresh paired Menagerie events?

## Hypothesis

C53 saturated because it trains successful actions as isolated completions;
it does not optimize the state distribution induced by the model's own prior
actions. Programmatic-oracle DAgger should repair recoverable visited states,
and guarded group-relative policy optimization on complete episodes should
then favor sequences of decisions that cause success rather than merely look
like successful traces.

The mechanism is falsified if a matched additional-DAgger/SFT control performs
as well as execution-reward training, shuffled trajectory rewards perform as
well as real rewards, or gains stay confined to trained environment families.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Starting policy: a regenerated merged checkpoint of the committed C53
  `sft_blend.jsonl` recipe. It is the incumbent for every comparison.
- Environment source: a self-contained copy of the C53 procedural gym. No
  benchmark family implementation, item, transcript, or result detail is read
  or imported.
- DAgger/RL train families: `kilnrite`, `glyphgate`, `loomfix`, `ferrier`, and
  `burrowmaze`.
- Proxy-transfer families never used for DAgger or RL: `gatepost`,
  `patchwheel`, and `spindle`.
- Train/eval seed namespaces and exact level mixtures are frozen in
  `configs/curriculum.yaml`.
- Inner-loop primary metric: greedy terminal episode score on frozen unseen
  seeds, macro-averaged by family; exact-success, action validity, natural
  close, turn use, and train-vs-transfer slices are co-primary diagnostics.
- Blackbox primary metric: paired Menagerie medium aggregate against the C53
  blend on two fresh seeds. Quick is the regression guard; slow is
  confirmation only.
- Oracle-only quantities: expert action, expert state summary, pass@K
  coverage, and hidden environment reward. They label training or establish
  ceilings but are never model inputs at deployment.

## Registered Curriculum

1. Reproduce and merge the C53 blend from its committed training rows.
2. Roll out that policy on fresh train-family episodes. At every model-visited
   nonterminal state, a state-aware programmatic expert emits a compact common
   process trace and the next action. Mix these DAgger rows with a frozen C53
   replay anchor and train an emission-weighted QLoRA update.
3. Evaluate the DAgger checkpoint on frozen train-family and unseen-family
   proxy episodes. If its mechanism gate fails, diagnose the interface and
   make at most one preregistered repair before any RL spend.
4. From the gated DAgger checkpoint, sample grouped on-policy trajectories.
   Use exact terminal environment score, group-relative advantages, PPO
   clipping, a reference-KL tether, and supervised replay guards. Constant-
   reward groups contribute diagnostics but no fabricated dense reward.
5. Compare the RL checkpoint against the DAgger checkpoint, a compute-
   overmatched additional-DAgger/SFT update, shuffled trajectory rewards, and
   frozen matched-compute sampling.
6. Open the Menagerie firewall only if all whitebox mechanism and retention
   gates pass. Benchmark invocations are CLI-only and stored aggregate-only.

Full frozen decisions, reward definition, entropy routing, and stop rules are
in `reports/preregistration.md`; adversarial review is in
`reports/design_review.md`.

## Gates

- DAgger gate: train-family macro score improves at least `+0.08` over the
  incumbent, proxy-transfer macro is nonnegative with at least one family
  improving `>=+0.03`, and atom/closure/parse retention regressions are each
  no worse than `-0.03` absolute.
- RL mechanism gate: real-reward RL beats both DAgger and matched
  additional-SFT by `>=+0.05` train-family macro, is nonnegative on the
  unseen-family macro, improves at least three train families, and exceeds
  shuffled-reward training by `>=+0.03`; retention guards remain green.
- Menagerie strategy win: candidate minus incumbent is `>=+0.05` on the mean
  of two fresh paired medium seeds, quick is no worse than `-0.02`, and a slow
  confirmation is positive. Anything weaker is recorded as partial or
  negative evidence, not a breakthrough.

## Run

CPU-only smoke:

```bash
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --smoke
```

The result-bearing pipeline is staged so every expensive phase can stop at
its registered gate:

```bash
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage incumbent
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage dagger-collect
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage dagger-train
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage proxy-eval
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage dagger-gate
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage rl-collect
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage rl-train
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage controls
.venv/bin/python experiments/qwen35_4b_interactive_policy_curriculum/scripts/run.py --stage rl-gate
```

Menagerie is a separate, conditional CLI-only stage documented after the
whitebox decision receipt is written.

## Results

Verdict: **negative training-recipe result; DAgger gate failed.** The frozen
paired proxy produced:

| Metric | C53 incumbent | DAgger | Delta | Registered decision |
| --- | ---: | ---: | ---: | --- |
| train-family episode macro | 0.6048 | 0.3517 | **-0.2531** | `>= +0.08` — fail |
| unseen-family episode macro | 0.6850 | 0.3519 | **-0.3331** | `>= 0`, one `>= +0.03` — fail |
| atom-retention macro | 0.6926 | 0.6711 | -0.0215 | `>= -0.03` — pass |
| atom parse macro | 1.0000 | 1.0000 | 0.0000 | `>= -0.03` — pass |

Paired bootstrap 95% intervals exclude zero by a wide margin: train macro
`[-0.2954, -0.2103]`, unseen macro `[-0.3804, -0.2869]`. Mean action validity
fell 5.7pp on trained families and 25.2pp on unseen families, even though
natural thinking closure improved 10.3pp and 12.9pp. This is not a global
parser or atom-capability collapse; it is a multi-turn policy collapse.

The collection itself was valid: 400 live trajectories yielded 1,386 unique
visited-state corrections, 203 expert demonstrations (12.8%, below the 20%
cap), and 681 C53 replay rows. All 2,270 training rows fit the 4,096-token
window. The diagnostic failure is semantic-operator imbalance: only 55/2,270
targets (2.4%) were `VERIFY`, including 19/1,386 visited rows. At evaluation
the DAgger model emitted `PATCH` on all 600 loomfix turns and `RULE` on 599/600
unseen patchwheel turns, effectively deleting the `RUN` verify/commit pivot.
Glyphgate fell 0.667 to 0.017 and unseen spindle 0.977 to 0.322.

Entropy/varentropy were useful only within their actual information regime:
67/200 collection groups were confident failures and 25/200 had outcome
variance, while only 2/200 diverged at the coarse first semantic operator.
With two siblings, outcome varentropy is degenerate; exact outcome variance
still identifies forks. Do not turn either diagnostic into token pressure.

This was not classified as the preregistered mechanical-repair exception:
parsing, truncation, expert selftests, merge hashes, and atom retention were
all healthy. Rebalancing operators, adding a behavior/KL tether, or replacing
full-sequence SFT with context-local pivot control is a new training design
and belongs in a standalone follow-up. No RL, matched controls, or Menagerie
seed was run.

## Interpretation

Live-state labels were not empty, but broad full-sequence imitation installed
the common trace/closure surface more strongly than the state-conditional
decision policy. The result sharpens C52's locality lesson: supervising the
right next action does not make a shared-weight update local to that state,
and operator-frequency skew can erase precisely the pivot tokens needed for
looping agents. A future warm start must prove semantic-pivot retention and
neighboring-policy locality before execution-reward training is licensed.

## Knowledgebase Update

The owning program evidence, backlog, scorecard, synthesis, practitioner
brief, and result visualization record the negative. The concurrently active
specialist-policy experiment owns the next integration test; this mixed-policy
arm should not be rerun independently.

## Artifacts

- `src/gym/`: copied firewall-clean procedural environments.
- `src/curriculum.py`: state-aware experts and shared process representation.
- `src/rollout.py`: live multi-turn collection and grouped trajectory schema.
- `scripts/`: staged orchestration, DAgger, RL, controls, analysis, merging,
  and aggregate-only benchmark wrapper.
- `configs/curriculum.yaml`: frozen seed namespaces, splits, budgets, and
  gates.
- `data/`, `runs/`, `analysis/`, `reports/`: small reproducible artifacts.
- Large adapters and merged checkpoints remain under the gitignored
  `large_artifacts/qwen35_4b_interactive_policy_curriculum/` path declared in
  `reports/artifact_manifest.yaml`.
