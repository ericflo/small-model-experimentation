# Truth-Audited Universal Curriculum Pilot Report

## Summary

Neither preregistered integration geometry installed a universal feature. Continuing
the mature C53 `blend` adapter for one epoch on 800 designed-only rows raised fresh
synthetic exact accuracy from 0.500 to 0.692 and reduced cap contacts from 10/26 to
1/26. On held-out quick@1,024, it remained well above base (+0.1406 aggregate) but
regressed three families and lost 0.1385 to `blend`. From-base co-training on designed
plus broad replay also reached 0.692 local accuracy, but failed its prospective parse
and cap gates, so its benchmark remained sealed. The parent factorial is negative.

## Research Program Fit

- Program: `agentic_breadth_installation`.
- Question: can correct human-designed executable procedures add the capability core
  that self-harvest/replay curricula leave behind?
- Prior anchors: C14, C49, C53, C54, C56, and C59.

## Method

The inherited v1 experiment was audited before GPU work. It contained 16 contradictory
induction traces, at least 33 behaviorally collapsed two-step rules, a dead smoke
command, a fail-open shell chain, a firewall-invalid direct benchmark path, and no
evidence that its advertised training run had started.

The replacement deterministic generator emits 13 executable lesson types over six
abstract surfaces. Induction rows must be query-identifiable across every
probe-consistent composition, contain a genuine dead end, and differ from every
primitive on a deterministic witness bank. Six unit tests cover byte determinism across
hash seeds, truth gates, depth, mix size, and smoke breadth.

The first arm warm-started immutable `blend` and trained on the frozen 800-row fast
corpus for one epoch at learning rate `5e-5`, rank 32, alpha 64, effective batch 8,
max length 2,048, and think loss weight 0.2. All 800 rows encoded; zero were skipped.
Training and adapter identity are authenticated in
`runs/training/blend_then_designed_fast.json`.

The second arm trained from base on the union of those 800 rows and the frozen 2,240-row
C53 replay corpus at the same learning rate, rank, alpha, effective batch, and think
weight, with max length 4,096. A batch-2 first-step CUDA residency failure was preserved;
batch 1 / accumulation 8 kept the exact dose and effective batch. It completed 380
steps, 3,040/3,040 rows, zero skips, and finite loss 1.366.

## Infrastructure amendment

Native quick@8,192 cannot satisfy this host's public suite budget: the permitted
estimator reports 321.9 expected / 440.3 worst seconds against a 60-second gate. Two
scoreless gateway failures and one interrupted diagnostic occurred before any benchmark
value was exposed. Protocol amendment 001 freezes quick@1,024 and medium@2,048, the
highest power-of-two caps whose estimates remain within their tier gates. All arms use
explicitly serialized/merged composites and the same `qwen_vllm` backend.

## Results

### Fresh synthetic seed 88001

| arm | accuracy | parse rate | cap contacts | mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| `blend` | 0.500 | 0.615 | 10/26 | 763.7 |
| sequential designed continuation | 0.692 | 0.962 | 1/26 | 231.1 |

The candidate improved execution, optimization, probe choice, tracing, and verification,
but remained wrong on both induction and repair examples and incorrectly abstained on
both feasible routing examples.

### Aggregate-only quick@1,024 seed 78131

| family | base | `blend` | candidate | candidate − base | candidate − `blend` |
| --- | ---: | ---: | ---: | ---: | ---: |
| chronicle | 0.1250 | 0.6250 | 0.8750 | +0.7500 | +0.2500 |
| lockpick | 0.0000 | 0.5000 | 0.0000 | +0.0000 | -0.5000 |
| menders | 0.0000 | 0.0000 | 0.0208 | +0.0208 | +0.0208 |
| mirage | 0.1250 | 0.6250 | 0.2500 | +0.1250 | -0.3750 |
| rites | 0.1250 | 0.2500 | 0.0000 | -0.1250 | -0.2500 |
| siftstack | 0.0000 | 0.5000 | 0.7500 | +0.7500 | +0.2500 |
| sirens | 0.3750 | 0.5000 | 0.5000 | +0.1250 | +0.0000 |
| stockade | 0.1250 | 0.4581 | 0.0000 | -0.1250 | -0.4581 |
| toolsmith | 0.6667 | 1.0000 | 0.6771 | +0.0104 | -0.3229 |
| warren | 0.1250 | 0.0000 | 0.0000 | -0.1250 | +0.0000 |
| **aggregate** | **0.1667** | **0.4458** | **0.3073** | **+0.1406** | **-0.1385** |

The candidate has six positive, one zero, and three negative family deltas versus base.
It fails both the no-negative-family pilot gate and the strong-control comparison.

### From-base union local gate, seed 88002

| arm | accuracy | parse rate | cap contacts | mean generated tokens |
| --- | ---: | ---: | ---: | ---: |
| from-base designed plus replay | 0.692 | 0.846 | 4/26 | 394.4 |

The arm passed accuracy and feasible-route abstention checks but failed the frozen
parse-rate and cap-contact checks. Induction was 0/2, execution 0/2, and four answers
were unparsable. Per protocol amendment 002, no merge or benchmark event ran and seed
78132 remains unconsumed.

## Controls and validity

- No benchmark source, item, transcript, verifier detail, or raw child stream was read
  or retained. Only the gateway's aggregate/public-family schema crossed the boundary.
- Base/control/candidate used the same seed, quick tier, 1,024 cap, vLLM backend,
  composite config/tokenizer/template serialization, and benchmark source inventory.
- Every output was within the public suite budget. The event summary SHA-256 is
  `d01ff14a6ef13a6503dfa586b445531d51b88cabe0cedcfe834cd4c4a13f87ca`.
- The first result was not used to rewrite the from-base arm. It was evaluated only on
  its prospectively frozen local gate; any new integration geometry is in
  `qwen35_4b_universal_replay_anchor`.

## Interpretation

Designed executable supervision is not inert: it made large, coherent moves on unseen
surfaces and two held-out axes. But a fixed trace/template distribution at full
continuation rate overwrote broad policy components faster than it installed the hard
procedures. Local accuracy gains therefore cannot serve as a proxy for universal
transfer. From-base co-training did not reproduce the sequential arm's concise emission
behavior despite broad replay. The remaining discriminating test is mature warm-start
replay anchoring: if it retains `blend` while keeping the new axis gains, the bottleneck
was integration; if not, the designed content or its dose is too local.

## Next experiments

1. In the result-separated successor, test low-rate mature warm start with replay in
   every optimizer window against a matched replay-only refresh.
2. Benchmark only locally retained candidates; require strict positive family deltas.
3. Confirm a winner on independent quick seeds, medium@2,048, paired uncertainty, and
   matched-compute sampling before updating a shared claim.
