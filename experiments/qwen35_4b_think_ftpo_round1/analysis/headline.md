# Round-1 headline table

| arm | budget | success | Δsuccess | natural_close | loop | unresolved | answer_limit |
|---|---|---:|---:|---:|---:|---:|---:|
| base | think@1024 | 0.498 | +0.000 | 0.014 | 0.002 | 0.984 | 0.443 |
| base | think@2048 | 0.541 | +0.000 | 0.167 | 0.004 | 0.829 | 0.420 |
| pivot | think@1024 | 0.459 | -0.039 | 0.002 | 0.002 | 0.996 | 0.475 |
| pivot | think@2048 | 0.465 | -0.075 | 0.075 | 0.002 | 0.922 | 0.490 |
| shuffled | think@1024 | 0.475 | -0.022 | 0.002 | 0.002 | 0.996 | 0.480 |
| shuffled | think@2048 | 0.486 | -0.055 | 0.075 | 0.008 | 0.916 | 0.484 |

## Verdicts
```json
{
  "P0_census": {
    "eligible_rate": null,
    "note": "combined across main+extension; see report"
  },
  "P1_mechanism": {
    "bar": "+0.05 absolute greedy success on held-out band tasks",
    "measured_1024": -0.0388,
    "measured_2048": -0.0755,
    "verdict": "FAIL"
  },
  "control_read": {
    "shuffled_delta_1024": -0.0224,
    "shuffled_delta_2048": -0.0551,
    "note": "shuffled ~= pivot degradation -> damage is generic to the training regime, not the outcome-conditioned signal"
  },
  "collapse_guard": {
    "greedy_rel_change": 0.1429,
    "pass8_rel_change": -0.0769,
    "flag_damaging": false
  },
  "nothink_guard": {
    "base": 0.3667,
    "pivot": 0.4083,
    "flag": false
  },
  "P3_gym": {
    "base_aggregate": 0.5172,
    "pivot_aggregate": 0.4845,
    "shuffled_aggregate": 0.5139,
    "guard_fail": true
  },
  "P4_menagerie": {
    "verdict": "NOT RUN \u2014 preregistered rule: mechanism prediction (P1) failed, so the round is a training-recipe failure with no capability read; blackbox spend cancelled"
  }
}
```