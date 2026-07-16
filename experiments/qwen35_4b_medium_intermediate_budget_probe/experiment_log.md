# Medium Intermediate Budget Probe Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened as the tb8192 stop's preregistered successor: think budget 4,096
  on fresh sealed seed 78,153, same four published composites, same
  post-review readings (scoped movement booleans; fail-closed
  implementation-signature contrast vs the pinned tb1024/78,150 summary),
  same base-first stop contract — strengthened: a second
  BUDGET_GATE_STOP closes the thinking-budget lever entirely for paired
  medium events.
- No model event has run; nothing trains in this cell.

## 2026-07-15 — The event: second stop; the lever closes

- CI green on the freeze; the event opened the ledger and ran base first;
  the gateway refused it with `budget_gate_failed` at tb4096, exactly as
  at tb8192. Zero treated arms ran; seed 78,153 spent.
- Per the frozen consequence the thinking-budget lever is closed ENTIRELY
  for paired medium events. The complete answer cost two seeds and two
  single-arm refusals with nothing exposed.
