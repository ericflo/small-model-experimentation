# Adaptive Cognitive Kernel Experiment Log

## Setup

Created a fresh experiment directory and separate large-artifact checkpoint root.

Experiment root:

```text
/workspace/experiments/adaptive_cognitive_kernel
```

Large artifacts:

```text
/workspace/large_artifacts/adaptive_cognitive_kernel/checkpoints
```

## Initial Design

The mechanism under test is a task-conditioned recurrent runtime. A prompt encoder reads an initial two-register state plus a sequence of operations. The ACK arm maps each operation token to coefficients over a learned bank of low-rank transition atoms. At runtime, those coefficients temporarily edit the recurrent transition used for that step.

Primary controls:

- direct sequence transformer;
- fixed recurrent controller;
- ACK ablation without dynamic low-rank deltas;
- shuffled-code evaluation for ACK.

Primary readouts:

- trained-length final pair accuracy;
- held-out longer-length final pair accuracy;
- held-out adjacent-composition final pair accuracy;
- state-step accuracy;
- ACK ordered versus shuffled-code gap.

## Iteration Notes

Smoke `smoke_ack_v1` validated the code path and artifact generation.

Pilot `pilot_ack_v1` showed that 450 steps on a finite training set was too weak to test the mechanism. A longer focused run, `pilot_ack_long_v2`, drove training loss down but left validation near chance, indicating memorization of a finite random program table rather than learned operation semantics.

Patch: training now defaults to online generation of fresh programs each step. This better matches the benchmark: the target is to learn reusable operation semantics, not memorize a fixed table.

Pilot `pilot_ack_online_v3` confirmed that online generation made the task learnable: short-length final-answer and state accuracy rose meaningfully, and ACK ordered conditioning separated from shuffled/random controls.

Pilot `pilot_ack_drive_v4` added an operation drive vector to the ACK shell. Dynamic ACK improved clearly over no-delta ACK, but fixed GRU remained competitive or stronger.

## Main Run

Run: `main_ack_v1`

Configuration:

- seeds: `101,202,303`
- arms: `ack_dynamic`, `ack_no_delta`, `fixed_gru`, `direct_transformer`
- train steps per arm/seed: `2500`
- online train examples: enabled
- eval examples per split: `768`
- model width: `128`
- ACK atoms: `24`
- ACK atom rank: `16`

Result:

- ACK ordered length-12 final-answer accuracy: `7.7%`
- ACK shuffled-code length-12 final-answer accuracy: `5.7%`
- ACK no-delta length-12 final-answer accuracy: `6.5%`
- fixed GRU length-12 final-answer accuracy: `8.3%`
- direct transformer length-12 final-answer accuracy: `5.8%`
- ACK ordered length-12 state-step accuracy: `11.6%`
- ACK shuffled-code length-12 state-step accuracy: `2.0%`
- ACK no-delta length-12 state-step accuracy: `4.0%`
- fixed GRU length-12 state-step accuracy: `14.4%`

Interpretation: dynamic ACK computation is not inert. Shuffling/randomizing the conditioning stream collapses state accuracy, and disabling dynamic deltas weakens the runtime. However, the dynamic ACK does not beat a conventional fixed recurrent controller on the longest held-out split, and all arms degrade steeply with length. The mechanism is therefore mixed but not a breakthrough under this test.
