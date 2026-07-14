# Protocol Amendment 001: Hardware-Admissible Thinking Budgets

## Timing and contamination status

This amendment was frozen before any benchmark score was emitted. Quick seed 78131
produced two scoreless, raw-suppressed gateway failures and one interrupted diagnostic;
no aggregate or per-family value crossed the firewall. The corpus, adapters, arms,
backend, seed, promotion rule, and benchmark suite are unchanged.

## Reason

The original preregistration said “canonical tier budget” without first running the
suite's permitted `--estimate` preflight on the restored RTX 4090. The estimator shows:

| tier and thinking cap | expected seconds | worst seconds | tier gate | status |
| --- | ---: | ---: | ---: | --- |
| quick @ 8,192 | 321.9 | 440.3 | 60 | over |
| quick @ 2,048 | 82.4 | 112.6 | 60 | over |
| quick @ 1,024 | 42.5 | 58.0 | 60 | within |
| medium @ 8,192 | 734.0 | 1,103.4 | 300 | over |
| medium @ 4,096 | 370.8 | 557.2 | 300 | over |
| medium @ 2,048 | 189.2 | 284.2 | 300 | within |

The observed scoreless quick attempts ran until the gateway rejected the event; the
trusted gateway correctly refuses any private result whose `within_budget` field is not
true. Repeating native-budget events cannot produce admissible evidence on this host.

## Frozen correction

- Fast pilot and independent quick replications use an explicit 1,024-token thinking
  cap: the highest power-of-two cap that passes the quick gate.
- Medium confirmation uses an explicit 2,048-token cap: the highest power-of-two cap
  that passes the medium gate.
- Every paired arm uses the same tier, cap, seed, merged-checkpoint path, and
  `qwen_vllm` backend.
- Gateway output now records `think_budget`; the firewall regression test proves the
  override reaches the private runner without exposing item-level material.
- Native 8,192 confirmation remains a future hardware replication, not evidence this
  host can validly produce.

This is an infrastructure repair selected entirely from public estimator metadata, not
an adaptive response to benchmark scores.
