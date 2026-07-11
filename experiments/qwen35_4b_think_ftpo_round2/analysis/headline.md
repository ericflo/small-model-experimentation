# Round-2 headline

**Verdict: LOW_DOSE_NULL.**

## Whitebox success

| budget | base | demote | uplift | shuffled | uplift-base | uplift-shuffled |
|---|---:|---:|---:|---:|---:|---:|
| think@1024 | 0.536 | 0.508 | 0.538 | 0.513 | +0.003 | +0.026 |
| think@2048 | 0.582 | 0.548 | 0.551 | 0.571 | -0.031 | -0.020 |

## Repository agent

| arm | success | patch-correct | submit | mean sampled tokens |
|---|---:|---:|---:|---:|
| base | 0.597 | 0.597 | 0.458 | 2510 |
| demote | 0.472 | 0.472 | 0.389 | 2597 |
| uplift | 0.542 | 0.542 | 0.389 | 2642 |
| uplift_shuffled | 0.403 | 0.403 | 0.264 | 2816 |
| base sample-more | 0.306 | 0.306 | 0.014 | 2341 |

## Gates

```json
{
  "P0": true,
  "P1": {
    "demote": false,
    "uplift": false,
    "uplift_shuffled": false
  },
  "P2": false,
  "P2_by_budget": {
    "think@1024": false,
    "think@2048": false
  },
  "P3": false,
  "P4": true,
  "menagerie_eligible": false
}
```
