# Locality-first recovery-reason interpolation report

## Summary

**`LOCAL_BUT_NO_BEHAVIOR`: stopped at the frozen calibration policy gate.**
Every action→reason mixture passed exact-logit/entropy locality, and λ=.18
reached 96.7% trained-family recovery, but no point met the invalid-turn and
immediate rejected-patch transition guards. Independent locality confirmation,
held-out families, and Menagerie were never opened.

## Research Program Fit

The parent found a useful action endpoint (85.0% recovery, 0.098 drift) and a
behaviorally stronger but non-local reason endpoint (91.7%, 0.303). This
follow-up resolved whether the weight segment contains a locality-safe useful
region before spending any fresh family or benchmark evidence.

## Method

Both parent LoRA deltas were reconstructed per module in float32. Four frozen
points used
`delta_action + λ(delta_reason − delta_action)` for λ=.10/.18/.24/.30, were
added once to the common apex checkpoint, and cast once to bfloat16. Full action
and reason endpoints anchored the screen.

The full ladder first faced 48 unrelated contexts with median centered
non-target logit drift ≤0.15 and mean entropy change ≥−0.05. Only passing points
then ran the checksum-frozen 60-case training-family recovery block. Eligibility
also required base-level tool validity, immediate rejected-patch change ≥0.60,
and failed-test changed patch within two turns ≥0.60.

## Results

### Interpolation geometry

The endpoint contrast is strongly cancelling, not a scalar dose direction.
Summed mixed-delta norms fell from 29.17 at action to 26.57/.24.94/.24.05/.23.47
at the four points, even while behavior changed sharply. All 128 adapted modules
were covered and every output hash was recorded.

| Point | Drift | Entropy Δ | Varentropy Δ | Locality |
| --- | ---: | ---: | ---: | --- |
| action | 0.0982 | +0.0060 | −0.0204 | pass |
| λ=.10 | 0.0999 | −0.0107 | −0.0171 | pass |
| λ=.18 | 0.1039 | −0.0212 | −0.0185 | pass |
| λ=.24 | 0.1107 | −0.0267 | −0.0196 | pass |
| λ=.30 | 0.1207 | −0.0412 | −0.0232 | pass |
| reason | 0.3031 | −0.1058 | −0.0139 | fail |

The safe region extends through λ=.30, far beyond the endpoint-linear
prediction. Because selection failed later, the independent confirmation block
correctly remained unused.

### Calibration behavior

| Arm | Overall | Failed | Rejected | Invalid/turn | Rejected immediate | Failed changed≤2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | .483 | .533 | .433 | .057 | .000 | .400 |
| happy | .817 | .800 | .833 | .020 | .333 | .967 |
| action | .850 | .733 | .967 | .191 | .167 | .833 |
| λ=.10 | .883 | .767 | 1.000 | .116 | .167 | .833 |
| λ=.18 | **.967** | **.933** | **1.000** | .104 | .333 | **.967** |
| λ=.24 | .950 | .900 | 1.000 | .102 | .333 | .833 |
| λ=.30 | .883 | .767 | 1.000 | .100 | .333 | .867 |
| reason | .917 | 1.000 | .833 | .059 | .833 | 1.000 |

λ=.18 was a large behavior optimum: +.483 over base, +.150 over happy, and
+.117 over action. Yet every candidate failed two frozen checks. Invalid turns
had to be ≤.077 (base +.02), and immediate rejected change had to be ≥.60. No
candidate was selected, so the registered run stopped.

## Failure Forensics (Exploratory)

The stop is valid, but the two failed metrics have different meanings:

- All 24 invalid steps at λ=.18 had closed thinking and consumed exactly all
  256 answer tokens. Every parse status was `no_json_tool_call_in_answer`; raw
  tails ended inside long exact-replacement JSON payloads. Nine were also
  force-closed at the 512-token thinking limit. This is a real deployability
  failure under the registered harness, caused by payload truncation rather
  than wandering prose.
- Invalids occurred in 11 trajectories, nine of which still ended with a
  correct workspace. They waste turns and cause two failures, but do not imply
  absence of the repair capability.
- All 30 rejected-patch cases made a changed patch within two generated turns
  and all 30 solved. Twenty used the sensible `INSPECT→PATCH` sequence; ten used
  `PATCH→VERIFY`. The immediate-only metric therefore rejects successful,
  context-seeking recovery even though the requested conditional transition is
  retained within one intervening inspection.

These diagnostics were computed only after the registered stop and cannot
rescue this experiment.

## Controls

The full action endpoint proves the gain is not merely conditional action
training: λ=.18 adds 11.7 points. The happy arm proves it also exceeds generic
balanced action SFT by 15.0 points. The reason endpoint verifies the known
locality failure exactly. Matched sampling and external scaffold controls were
not funded because no candidate cleared the prerequisite policy gate.

## Oracle Versus Deployable Evidence

Hidden tests remain host-only and support calibration performance only. No
unseen-family task was generated or evaluated in the result-bearing stages, and
no Menagerie seed was assigned. This is a mechanism result, not a breadth or
black-box capability claim.

## Interpretation

The parent trade-off is partially separable: locality-safe interpolation can
outperform both endpoints on recovery. The remaining obstacle is now concrete
and harness-facing—long exact-replacement actions do not fit a 256-token answer
slot—rather than generic collateral or lack of repair knowledge.

The immediate-transition guard also encoded the wrong behavioral preference.
After rejection, re-reading the changed file before patching is defensible; the
faithful metric is changed patch within two turns, while still measuring the
intervening operator and final success.

## Next Experiment

Create a separate harness experiment that freezes λ=.18, increases tool-answer
payload capacity for every arm under matched total compute, and records
rejected-patch changed-within-two as the primary transition. It must retain the
existing locality receipt, compare all controls under the same budget/interface,
and still pass two untouched family blocks before Menagerie. Do not alter this
experiment's thresholds or run its transfer seeds.

## Artifact Manifest

Large checkpoints and trajectories remain under
`large_artifacts/qwen35_4b_recovery_reason_locality_interpolation`. The committed
`result_receipt.json` contains compact metrics, failure forensics, weight hashes,
and source checksums.
