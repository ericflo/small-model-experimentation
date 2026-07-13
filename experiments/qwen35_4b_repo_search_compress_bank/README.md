# Repository search-compress-bank coding curriculum

**Status:** finished

This experiment asks whether `Qwen/Qwen3.5-4B` can acquire broader coding-agent competence by searching real procedural repositories, verifying candidate repairs with executable tests, compressing successful tool trajectories to their replay-necessary core, and banking compact plans and actions with explicit operator balance.

## Research program

Primary program: `agentic_breadth_installation`. The experiment follows C53's queued scaffold-distillation direction, composes C12/C22 verified banking with C54 compression advantage, and directly guards the semantic-operator collapse found by the interactive-policy curriculum.

The closest near-duplicate is `qwen35_4b_think_ftpo_round2`: it introduced a six-family procedural repository agent only as a fresh evaluation gate. It did not search repositories for training data, replay-minimize successful edits, bank tool states, balance operators, or train on coding-agent actions.

## Hypothesis

Long successful coding trajectories contain a small causal core: issue-directed inspection, necessary edits, post-edit verification, and commit. Executable minimization should discard failed branches and redundant patches; compact state-specific planning should install the useful transition structure; equal INSPECT/PATCH/VERIFY/COMMIT loss mass should preserve the rare terminal operators that broad DAgger erased.

The hypothesis fails if the compact arm cannot beat (a) an identical action-only bank, (b) the regenerated C54 apex policy on family-disjoint repositories, and (c) two shorter apex samples under the same eight-call/token reservation.

## Result

`PRIMARY GATE FAILED`. The compact arm improved trained-family success from 40/48 to 48/48 (+16.7 points) and reduced the learned path to exactly four valid calls on every task, but this was family-specific policy cloning rather than transferable coding competence. On four wholly held-out repository families it fell from 49/72 to 25/72 (−33.3 points; paired 95% CI [−44.4, −22.2]) and also lost to matched-compute sampling by 18.1 points. Median unrelated-context centered-logit drift was 0.386 against the frozen 0.15 ceiling.

The transition audit localizes the failure. After a failed visible test, the apex control patched again on 24/26 transitions; compact did so on 0/48, instead re-testing, re-inspecting, or emitting an invalid action. All 18 recursive-overlay tasks repeated the same exact patch after the patch tool had already rejected it. Operator-marginal balance preserved the happy-path vocabulary, but success-only compression deleted failure-conditioned recovery behavior.

The necessary gate stopped action-only training, confirmation, and Menagerie. Therefore the result rejects the registered compact-bank recipe as a whole; it does not identify compact plan gradients rather than action gradients as the cause. See the [final report](reports/report.md), [primary gate](analysis/repo_primary_gate.json), and [result receipt](reports/result_receipt.json).

## Firewall-clean substrate

Ten fresh Python repository generators materialize source files and a visible test. Six families are eligible for search/training; four algorithmically distinct families are never harvested or trained and are used only for transfer. Hidden test programs and oracle edits live in host memory, are never written into the repository, never enter a model message, and are reduced to pass/fail booleans in receipts. The benchmark directory is never read or imported.

The agent has real bounded tools: `tree`, `read`, literal `search`, exact single-replacement `patch`, subprocess `test`, and `submit`. Source/test paths are constrained to the temporary repository; visible tests are readable but immutable.

## Arms and controls

- `apex_replay`: regenerate C54 apex from its committed training data with this experiment's fixed optimizer budget.
- `compact`: the same C54 data plus replay-minimized repository rows with compact state-specific plans.
- `action_only`: identical repository contexts, actions, row weights, and optimizer budget, but zero loss on the compact plan span.
- C53 blend: fixed search teacher and contextual deployment comparator, never a source of private labels.
- Matched sampling: two independent four-turn apex rollouts versus one eight-turn candidate rollout; both reserve eight model calls and 6,144 sampled tokens per task.

Entropy and varentropy may be recorded later to route which live states deserve more search, but verifier outcomes supply correctness and operator class supplies balancing. They never scale token loss or choose a push-up/push-down target.

## Frozen stages

1. Self-test every generator, path boundary, and hidden-label firewall.
2. Search 24 tasks from each of six training families with four eight-turn C53 trajectories.
3. Keep private-test successes, greedily delete replay-unnecessary patches, then reconstruct and replay inspect→patch→test→submit traces.
4. Train the three registered arms from the pinned base at the same 584 optimizer steps.
5. Gate trained-family retention, family-disjoint transfer, compact-vs-action-only advantage, matched-call sampling, verification/commit retention, invalid actions, and unrelated-context logit locality.
6. Confirm on a second transfer block. Only then assign fresh union-checked Menagerie seeds and compare aggregate quick/medium scores through the benchmark CLI.

The run stopped during stage 5. Stages 6 and Menagerie were not authorized.

Exact thresholds, seeds, and interpretation rules are frozen in [the preregistration](reports/preregistration.md), its pre-harvest [token-mass implementation amendment](reports/preregistration_amendment.md), the [memory-feasibility audit](reports/preregistration_amendment_2.md), the final exact [chunked-loss amendment](reports/preregistration_amendment_3.md), and [configuration](configs/default.yaml). The adversarial review is in [reports/design_review.md](reports/design_review.md).

## Run

CPU smoke (no model or benchmark):

```bash
.venv/bin/python experiments/qwen35_4b_repo_search_compress_bank/scripts/run.py --smoke
```

GPU smoke after the preregistration commit:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_repo_search_compress_bank/scripts/run.py --gpu-smoke
```

The staged full command is intentionally gate-stopping:

```bash
.venv/bin/python experiments/qwen35_4b_repo_search_compress_bank/scripts/run.py --full
```

Result-bearing model outputs and weights live under `large_artifacts/qwen35_4b_repo_search_compress_bank`; small summaries and receipts are committed. No PR is created: accepted work is rebased and pushed directly to `main`.

## Status

`PRIMARY GATE FAILED; EXPERIMENT STOPPED`. Both licensed 584-step arms trained and merged successfully. The compact arm overfit the trained procedural families, regressed family-disjoint transfer, lost to matched sampling, failed verification/invalid-action retention, and exceeded the locality ceiling. Action-only, confirmation, and Menagerie were correctly cancelled; zero benchmark seeds were consumed.
