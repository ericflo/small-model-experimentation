# Qwen Verifier-Guided Slot Repair Experiment Log

## Objective

Test whether verifier-guided local search over compiled slots can recover exact long-chain execution from near-miss Qwen numeric-copy programs.

## Experiment Question

If a compiler produces high-accuracy but imperfect modular programs, can a small repair search over likely slot alternatives substantially improve length-24 exact execution when selected by a state-trajectory verifier?

## Planned Conditions

1. Tiny-model smoke test for repair metrics and checkpoint writing.
2. Qwen smoke test for 4-bit QLoRA training with repair evaluation enabled.
3. Main Qwen light-state compiler with verifier-guided repair search and checkpoint selection by repaired paired length-24 accuracy.
4. Fresh selected-checkpoint retest on length-24 programs.
5. One-edit repair-budget ablation on the selected checkpoint.

## Primary Selection Rule

Select the saved checkpoint with the highest `paired_len24_repair_executor_accuracy`.

## Primary Metrics

- `paired_len24_executor_accuracy`
- `paired_len24_repair_executor_accuracy`
- `fresh_paired_len24_executor_accuracy`
- `fresh_paired_len24_repair_executor_accuracy`
- `repair_found_fraction`
- `repair_changed_fraction`
- `repair_pair_state_consistency`

## Artifact Policy

Lightweight outputs stay in:

```text
experiments/qwen_verifier_guided_slot_repair/runs/
experiments/qwen_verifier_guided_slot_repair/analysis/
experiments/qwen_verifier_guided_slot_repair/reports/
```

Large adapters and head checkpoints stay in:

```text
large_artifacts/qwen_verifier_guided_slot_repair/checkpoints/
```

## Log

### 2026-06-22

- Created standalone experiment directory.
- Forked the checkpoint-selected Qwen numeric-copy compiler harness.
- Added verifier-guided local repair search over initial value, operation, argument, same-step operation/argument, and two-argument edits.
- Added repair metrics to ordinary evaluation, selected-checkpoint tracking, analysis, and fresh retesting.
- Stopped the first main run after the step-1 checkpoint showed that answer-only verification was degenerate: with many local candidates in a 97-way answer space, even an untrained compiler could find spurious final-answer matches. Changed the primary verifier to require the full intermediate state trajectory.
- Reran `smoke_tiny_repair` with the state-trajectory verifier. The smoke completed and no longer produced fake repair gains on length-3 paired examples.
- Reran `smoke_qwen3_4b_repair` with `Qwen/Qwen3-4B`, 4-bit QLoRA, and the state-trajectory verifier. The smoke completed and saved checkpoints.
- Ran `main_state_w025_repair_s900` with light state supervision, paired training, top-3 repair candidates, up to two edits, and checkpoint selection by `paired_len24_repair_executor_accuracy`.
- Main validation selected step 800. At that checkpoint, paired length-24 exact execution was 30.5% unrepaired and 88.3% repaired. Repaired paired state consistency was 90.6%, repaired changed fraction was 57.8%, and repair found fraction was 88.3%. Step 900 dropped to 82.0% repaired paired length-24.
- Ran a fresh selected-checkpoint retest with 256 fresh standard length-24 programs, 256 fresh paraphrase length-24 programs, and 256 paired length-24 programs. Fresh paired length-24 exact execution was 27.5% unrepaired and 91.0% repaired. Fresh repaired program exact was 90.0%, repaired pair state consistency was 92.2%, and repaired pair both-correct was 89.5%.
- Ran a one-edit ablation on the same selected checkpoint and fresh seed. Fresh paired length-24 exact execution improved from 27.5% unrepaired to 70.5% with one-edit repair, versus 91.0% with two-edit repair.

## Result

State-trajectory verifier repair reveals large local-search headroom. With a top-3/two-edit repair budget, the selected Qwen compiler improves from 27.5% to 91.0% fresh paired length-24 exact execution. The one-edit ablation reaches 70.5%, so many failures are one local slot edit away, while the remaining lift depends on two-slot repair.

This is not yet a deployable verifier, because the state trajectory is an oracle training/evaluation signal. It is a strong headroom result: the compiled programs usually contain enough local evidence that a small verifier-guided search can recover the true long-chain execution.
