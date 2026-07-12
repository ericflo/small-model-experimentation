# Verifier-conditioned recovery banking curriculum — experiment log

## 2026-07-12 — intake and design

- Re-read the program index, scorecards, claim ledger, synthesis, model playbook, compute/vLLM guidance, and C54 predecessor artifacts.
- Chose conditional transition balancing over another think-token FTPO round: it directly addresses the observed `failed_test→no changed patch` collapse while retaining executable outcome selection.
- Created ten fresh two-defect procedural repository families. Initial and partial workspaces must fail both visible and hidden tests; the oracle must pass both.
- Registered three warm-start arms, an external scaffold control, matched-compute sampling, family-held-out transfer, confirmation, normal-loop retention, locality, and exploratory entropy/varentropy diagnostics.
- CPU fixture and bank smoke passed after strengthening private partial-state checks.
- GPU integration smoke established that the local C54 composite can be QLoRA-trained, explicitly merged, and loaded by the pinned vLLM runner.
- Integration smoke caught and fixed a binary-file search crash caused by test-created `__pycache__`; the public repository tools now expose only source/README files.

No result-bearing harvest, training, or evaluation had run when the preregistration was frozen.

## 2026-07-12 — full run

- Harvest: 58/72 tasks covered (80.6%); all family coverage gates passed. Patch minimization admitted 57 tasks and 399 rows/arm at 100% replay.
- All arms trained for 120 steps from the immutable apex warm start. `happy_action` and `recovery_action` had total merge-delta norms 28.03 and 29.17. `recovery_reason` reached 37.78.
- The nominal 5% plan mass was not a 5% realized dose. At step 10, recovery action logged loss/gradient 12.52/1.80; recovery reason logged 43.61/42.12. The plans were much more surprising than the action targets, so clipping made early updates plan-dominated.
- Calibration recovery success: base 0.4833, happy 0.8167, recovery action 0.8500, recovery reason 0.9167. The frozen selector chose reason and passed its mechanism gate.
- Registered reason locality failed: drift 0.3031 > 0.15, unrelated entropy −0.1058 nats. Transfer and Menagerie stopped unopened.
- Exploratory controls: happy drift 0.0833 and recovery action 0.0982, both locality-pass; action-only entropy +0.0060. The plan span, not verifier-conditioned action learning, caused collateral.
- Seam audit: all target action starts were rank 1 before training. Unnatural plan starts at inspect→patch, patch-ok→verify, start→inspect, and pass→commit were ranks ~8,404/~1,163/~135/3 and were driven to rank 1 with near-zero entropy by reason training.
- Decision: preserve this result as a locality-gated negative. Use a new experiment for locality-first interpolation; do not reinterpret action-only or a scaled reason checkpoint inside this frozen run.
