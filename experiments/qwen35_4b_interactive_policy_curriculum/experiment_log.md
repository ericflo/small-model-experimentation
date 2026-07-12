# Interactive policy curriculum: oracle DAgger to execution-reward RL Experiment Log

## 2026-07-11 — intake, preregistration, and CPU smoke

- Routed to `agentic_breadth_installation`; closest duplicate is C53's static
  gauntlet frontier, and the novelty is live visited-state supervision plus
  complete-trajectory reward.
- Copied the C53 firewall-clean gym into a self-contained follow-up experiment.
- Froze five incremental training families, three incremental transfer
  families, disjoint seed namespaces, terminal-only reward, DAgger/RL gates,
  matched controls, and conditional Menagerie rules.
- Adversarial design review caught five material hazards before GPU spend:
  indexed imported oracles were not state-aware; validity shaping would repeat
  C50; raw entropy would repeat C52; transfer replay could invalidate holdouts;
  and injected close tokens were not policy actions. All are fixed in code or
  preregistration.
- CPU smoke passed: 6 curriculum tests, 13 vLLM wrapper tests, and all 14 gym
  family selftests. No result-bearing GPU stage or Menagerie event has run.

## 2026-07-11 — merge validation and collection repairs

- A one-step LoRA smoke produced a zero delta because the scheduler spent its
  only step in warmup. A two-step smoke produced nonzero LoRA-B weights, a
  changed merged-shard hash, and changed model output. The runner now validates
  local merged checkpoints recursively back to the pinned official revision.
- Fixed two result-threatening harness issues before the registered run:
  visited states are deduplicated before the expert-demo quota is applied, and
  paired evaluation requires distinct named checkpoint paths rather than
  silently accepting one model twice.
- Expanded the frozen smoke to 8 curriculum tests and 14 runner tests. These
  are implementation checks, not result evidence.

## 2026-07-11/12 — reached DAgger run and mechanism-gate stop

- Regenerated the C53 incumbent from 2,240 committed rows: 2,117 encoded at
  the frozen 2,048-token window, 123 skipped, 333 optimizer steps, and a
  receipt-verified merged shard (`9450848e9d5d...`).
- Collected 400 incumbent trajectories over 200 fresh train-family episodes.
  The final 2,270-row curriculum contains 1,386 unique visited-state
  corrections, 203 expert rows (12.8% of incremental data), and 681 C53 replay
  rows. No transfer-family row leaked into training or replay.
- Trained the DAgger warm start for the registered 1.5 epochs: all 2,270 rows
  encoded, 213 optimizer steps, finite loss, and a distinct receipt-verified
  merged shard (`842e3aa40e5e...`).
- The paired proxy gate failed decisively. Train-family macro changed
  0.6048→0.3517 (−0.2531; paired-bootstrap 95% CI
  [−0.2954, −0.2103]); untouched-family macro changed 0.6850→0.3519
  (−0.3331; CI [−0.3804, −0.2869]). Atom retention remained inside its
  guard (−0.0215), parsing stayed 1.000, and natural closure improved.
- Forensics localized the damage to semantic decision pivots. Only 55/2,270
  targets were `VERIFY`; the trained policy emitted zero `RUN` actions across
  all 600 loomfix evaluation turns and one across 600 untouched patchwheel
  turns. It learned the trace/closure surface while erasing verify/commit
  boundaries.
- This is not the registered mechanical-repair case: parsing, truncation,
  experts, checkpoint application, and atom retention were healthy. The gate
  therefore cancelled RL, matched controls, and Menagerie. Zero benchmark
  seeds were consumed.
