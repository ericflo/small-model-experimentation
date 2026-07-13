# State-Carry Versus State-Bag Counterfactual Report

## Status

`PILOT_MECHANISM_MISS`: the fixed-source rank-32 LoRA pilot is complete and valid. It failed the
preregistered joint-state-formation gate, so the experiment stopped before confirmation. The earlier
analysis-dispatch attempt remains preserved but is not pooled with this result.

## Question

Does a serially inherited internal state produce deeper, causally useful representations than an equal-compute collection of independent shallow states?

## Validity

The canonical G0 receipt passed exact direct-model and Carry/Bag K=1 parity (`0.0`), both-arm gradient
checks, and finite worst-format K=12 forwards. The pilot used only seed 7401 and the dedicated pilot
validation, depth, joint-holdout, and counterfactual splits. Carry and Bag each trained for the fixed
300 steps. Their initialization, ordered-row digest, 2,594,937 prompt tokens, and 145,316,472
decoder-layer-token applications matched exactly. Both final checkpoints repeated K=1 parity at
`0.0`. All registered pilot cells, K=4 diagnostics, joint-holdout cells, and 64 bidirectional swap
pairs were complete; the configured gain was mathematically reachable.

The analysis receipt is source/config/data/lock bound:

- verdict receipt: `c9fec5b584bef5aa3de40844a8552395f3c3f0f95f79285700a9e4e375aed0ef`;
- config: `70e4a2d6df7acb0c5a21c7c945c66499a0ede8e98321c7b56da1c080c819744b`;
- source: `ef2dd25107cb306490e30dba8ac3035c8c69c76173cec62390557cf1add7a28d`;
- data manifest: `2cf9a4d008d0990873928424170ab5daf0a53473a9f97e542d8739ac6de92879`;
- training lock: `05546fe977583116d6169ea0dfa7b27e1184dd4a2b61d556dfb3f889d5b2b7b1`.

## Result

| Pilot endpoint | Result | Registered interpretation |
|---|---:|---|
| Carry joint-state step accuracy | `0.0045948` | Fail versus `0.40` |
| Carry node step accuracy | `0.0641912` | Chance-like diagnostic |
| Carry minus Bag, matched depth | `+0.04296875`, CI `[-0.0078125, 0.09375]` | Positive point estimate; uncertain |
| Positive depth cells | `5 / 8` | Diagnostic only at pilot |
| Carry unseen-K gain over K=4 | `+0.01171875`, CI `[-0.03515625, 0.05859375]` | Complete; uncertain |
| Joint family+surface holdout | `+0.05078125`, CI `[0.0078125, 0.09765625]` | Pilot diagnostic passed |
| Swap donor-follow gain | `+0.0078125`, CI `[-0.0234375, 0.0390625]` | Causal diagnostic failed |
| Donor follow minus recipient preserve | `-0.0546875` | Causal diagnostic failed |
| Carry answer-mode rate | `1.0` | Interface valid |

Both node and checksum strata had positive Carry-minus-Bag point estimates (`+0.0234375` and
`+0.0625`). Those answer-level and holdout signs cannot rescue a state that did not encode the
registered node+phase+checksum trajectory. The only failed promotion check was
`joint_state_sufficient`; the analyzer therefore emitted `PILOT_MECHANISM_MISS` and `promote: false`.

## Scope and Next Experiment

No confirmation seed (7411–7413), same-checkpoint edge cut, or explicit-CoT sample-more comparator
was run, because the pilot stop prohibited them. This result says the registered rank-32 extra-call
LoRA recipe did not form the required deep joint state in its valid pilot. It does not distinguish a
serial-state limitation from insufficient low-rank plasticity.

Under preregistration section 10, the next step is mandatory: create and execute a fresh successor
that replaces extra-call LoRA with zero-initialized full-rank deltas on Qwen layers 12–19. The base
weights remain frozen, deltas are active only on extra R calls, and the first pass, coda, K=1 logits,
Carry/Bag equality, procedural substrate, pilot firewall, crossed analysis, and causal gates remain
fixed.

## Artifact Manifest

See `artifact_manifest.yaml`.
