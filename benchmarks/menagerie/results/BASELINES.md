# Base Qwen3.5-4B thinking baselines

These are the honest starting lines install experiments must beat. All
measurements use the base `Qwen/Qwen3.5-4B` model with no adapter, thinking
mode, greedy decoding, and seed `31337`. Thinking is the deployment default.
The backend is the default `qwen_vllm` backend: vLLM 0.24.0 with
`gpu_memory_utilization=0.85` and `max_model_len=16384`, running on an RTX 4090
with the model resident; the approximately 35-second load is excluded. Scores
are per-family means, and aggregate is the mean of family means.

## Four-tier scores

| family | quick | medium | slow | deep | what it measures |
| --- | ---: | ---: | ---: | ---: | --- |
| chronicle | 0.000 | 0.300 | 0.154 | 0.222 | event-stream state tracking |
| lockpick | 0.000 | 0.000 | 0.115 | 0.028 | active rule induction → exploit |
| menders | 0.000 | 0.100 | 0.038 | 0.111 | program repair from failing traces |
| mirage | 0.000 | 0.000 | 0.000 | 0.056 | calibrated abstention (provable unsolvability) |
| rites | 0.000 | 0.100 | 0.077 | 0.111 | state-machine / spec compliance |
| siftstack | 0.000 | 0.100 | 0.038 | 0.083 | information triage under noise/contradiction |
| sirens | 0.500 | 0.400 | 0.462 | 0.389 | goal fidelity under prompt injection |
| stockade | 0.120 | 0.191 | 0.080 | 0.055 | bounded optimization vs brute-forced optimum |
| toolsmith | 0.375 | 0.272 | 0.355 | 0.357 | dependent tool-call chaining |
| warren | 0.125 | 0.000 | 0.064 | 0.264 | partially-observable exploration + memory |
| **aggregate** | **0.112** | **0.146** | **0.138** | **0.168** | |

Absolute scores are **not** comparable across tiers because tiers use different
level mixes and budgets. Tiers are longitudinal instruments: compare the same
tier across checkpoints.

## Wall time versus budget

| tier | measured wall | budget | headroom | items | rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| quick | 43.8 s | 60 s | 27% | 80 | 1 |
| medium | 187.2 s | 300 s | 38% | 100 | 4 |
| slow | 548.3 s | 1200 s | 54% | 260 | 10 |
| deep | 1190.6 s | 3600 s | 67% | 360 | 14 |

## Think budgets: floor and escalate

| tier | atom think budget | episode think budget per turn |
| --- | ---: | ---: |
| quick | 1024 | 1024 |
| medium | 2048 | 2048 |
| slow | 2048 | 2048 |
| deep | 4096 | 2048 |

The 1024-token floor exists because 256 and 512 tokens sit in the
empirically-proven truncation-harm zone: approximately 100% forced-close and
approximately 90% no-answer. Budgets escalate above that floor because the
model consumes essentially any budget given. The measured forced-close counts
were 79/80 generations on quick; medium's `forced_think_closes` was 131 across
its multi-turn generations; and the counts were 399/414 on slow and 726/787 on
deep. Deep had a mean think length of 2588 tokens, and 222/240 atoms hit the
full 4096-token budget. Score keeps responding to compute on the harder tiers:
slow scored 0.138 at 2048 and deep scored 0.168 at 4096, versus quick at 0.112
with 1024. Deep's `episode_think_budget=2048` is a wall-clock bound on the
14-round horizon, not a context workaround. `--think-budget N` overrides both
atom and episode budgets for explicit compute-response studies.

### Medium is a fast subsample of slow

The original medium ran at think 1024 with L3 atoms and scored 0.052—below
quick's 0.112—because its hardest items were truncated with no compute: it was
a scaled-up quick. Medium was rebuilt as a scaled-down slow at the same 2048
budget and with the same construct, L1-L4 atoms plus multi-turn episodes. Its
aggregate is 0.146, and its per-family rank Spearman against slow is 0.717, the
highest of any tier pair. Medium now tracks slow as an approximately 3×-faster
proxy.

Medium reads marginally above slow (0.146 versus 0.138) because both share the
2048 think budget while medium carries less hard content: no L4-heavy atom mass
and no L3/10-turn episodes. At equal compute, the lighter tier scores slightly
higher. The ladder is therefore monotonic along the compute axis—quick (1024) <
medium/slow (2048) < deep (4096)—with medium and slow statistically tied: their
0.008 gap is within the instrument noise floor, while the HF/vLLM aggregate gap
on quick alone is 0.011. The pathological medium-below-quick dip is eliminated.
Medium uses representative mid-difficulty L2 episodes at 6-turn depth and a
2048 budget. It cannot afford slow's hard L3/8-turn episodes within the 300 s
wall: a tested L3-episode medium variant hit 278 s, leaving 7% headroom, and
re-inverted to 0.108, below quick.

## Truncation history

Earlier revisions ran quick/medium at 256/512 think budgets, squarely in the
truncation-harm zone: approximately 100% of thinking chains were force-closed
and most items produced no parseable answer. The current floor-and-escalate
design (1024 minimum) eliminates that failure mode, and the two-phase runner
still guarantees an answer pass even when a chain is force-closed.

## Sizing invariant: worst case fits the budget

Tiers are sized so the worst case—every generation binding its full think
budget and every episode running its full horizon, at a conservative 1500
tokens/second floor—fits inside `budget_s`.

| tier | estimator `worst_s` | `budget_s` |
| --- | ---: | ---: |
| quick | 58.0 | 60 |
| medium | 288.0 | 300 |
| slow | 1139.2 | 1200 |
| deep | 3066.9 | 3600 |

Measured walls land at 33–73% of budget because base-model episodes terminate
early. Measured horizon use was 0.69 of maximum turns on medium, 0.37 on slow,
and 0.33 on deep. The asymmetric headroom is deliberate: a stronger policy
that survives to the full horizon must still fit. Quick has the least headroom
because `validate_suite.py`'s random-floor gate pins it at at least eight atoms
per family.

## Backend policy

`qwen_vllm` is the default. On quick at think 1024 it delivered a measured 4.0×
wall-speedup: 176.0 seconds with HF versus 43.8 seconds with vLLM.

vLLM continuous batching is not bit-reproducible. Historically, approximately
one item in 80 flips between identical no-think runs, and at think 1024 the
divergence compounds over the chain. On quick with the same seed, measured
cross-backend item-score agreement was 72/80 = 0.90 exact; aggregate was 0.123
with HF versus 0.112 with vLLM, a delta of 0.011. Disagreements ran in both
directions—five favored HF and three favored vLLM—which indicates noise rather
than bias.

The `qwen` (HF) backend is the deterministic parity oracle: a token-for-token
two-phase mirror that is approximately four times slower. Use it for parity
checks and debugging, and never mix its numbers into a comparison with vLLM
numbers. Tier budgets are defined for the vLLM default, so the HF quick parity
run at 176 seconds reports `within_budget=no` against the 60-second budget by
design. Engine configuration is part of the comparison key: changing
`gpu_memory_utilization` or `max_model_len` shifts item-level results even at a
fixed seed.

## Instrument predictiveness

The synthetic noisy-oracle ladder from `validate_suite.py` at seed `0`
rank-correlates at Spearman 1.000 for every tier pair.

| tier | eps=0.0 | eps=0.25 | eps=0.5 | eps=0.75 | eps=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| quick | 1.000 | 0.675 | 0.463 | 0.250 | 0.013 |
| medium | 1.000 | 0.738 | 0.492 | 0.274 | 0.024 |
| slow | 1.000 | 0.747 | 0.510 | 0.263 | 0.016 |
| deep | 1.000 | 0.747 | 0.467 | 0.247 | 0.016 |

Real floored-model per-family ranks remain tie-dominated and unstable across
tiers. Measured family-rank Spearman correlations were `quick|medium` 0.46,
`quick|slow` 0.60, `quick|deep` 0.65, `medium|slow` 0.72, `medium|deep` 0.57,
and `slow|deep` 0.49. Trust quick for relative progress during training, then
confirm conclusions on slow/deep.

## Reproduce

```bash
PY=/home/ericflo/Development/small-model-experimentation/.venv/bin/python
cd benchmarks/menagerie
$PY run.py --tier quick --seed <fresh>              # backend defaults to qwen_vllm
$PY run.py --tier deep  --seed <fresh>              # ~20 min
$PY run.py --tier quick --backend qwen --seed <s>   # HF deterministic parity oracle (~4x slower)
python3 run.py --estimate --tier all                 # CPU token-math projection
```

Use a fresh seed for every evaluation event. Compare only runs with matched
seed, tier, backend, and engine configuration. If `.venv-vllm` is missing,
build it per [`docs/vllm_inference.md`](../../../docs/vllm_inference.md) or fall back
to `--backend qwen`:

```bash
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python --torch-backend=cu129 requirements-vllm.lock.txt
```

Baseline artifacts:

- `results/quick_qwen35_base_think1024_seed31337.json`
- `results/medium_qwen35_base_think2048_seed31337.json`
- `results/slow_qwen35_base_think2048_seed31337.json`
- `results/deep_qwen35_base_think4096_seed31337.json`
- `results/quick_qwen35_base_think1024_seed31337_hf.json`
