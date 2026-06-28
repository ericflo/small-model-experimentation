# Experiment Log

## 2026-06-28

- Created standalone live strategy-discovery package.
- Copied calibration records, held-out baseline records, task cases, and sandbox utilities into this package.
- Planned a smoke run before full held-out evaluation so parsing/execution failures are caught before spending full generation budget.
- Ran a 6-task smoke test, found the first discovered strategy card mentioned pandas, and tightened the discovery prompt/sanitizer to require plain Python list transforms.
- Ran the full 50-task held-out evaluation with `max_discovered=2` and `max_repairs=1`.
- Final held-out result: direct JSON 21/50, discovered first-visible 22/50, discovered shape-triggered 23/50, discovered oracle union 23/50, included baseline first-visible 28/50, included baseline oracle 29/50.
- Primary gate result: discovered strategies added 0 visible-correct tasks beyond the included baseline oracle. The strategy-discovery arm is therefore a clean negative on held-out coverage expansion.
