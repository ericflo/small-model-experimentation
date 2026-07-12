# G0 analysis summary

The averaged targeted J lens is directly writable at Qwen3.5-4B layer 24 but
does not transport through an arbitrary prompt-local mapping.

## Confirmation headline

| condition | direct target rate | consequence target rate |
| --- | ---: | ---: |
| baseline | 0/24 | 0/24 |
| layer-24 J swap | 18/24 | 0/24 |
| layer-24 random coordinate swap | 0/24 | 0/24 |
| layer-24 logit-lens swap | 5/24 | 0/24 |

Earlier J layers achieved direct target rates of 0/24, 0/24, 0/24, and 1/24
at layers 8, 12, 16, and 20. Mapped-consequence target rate was 0/24 at all
five layers.

## Decision

G0 failed its consequence and adjacent-layer requirements. The prefix-value and
causal-task stages were not run. The next design must test context-local
transport and true coordinate clamping, not retune this result-bearing experiment.
