# Experiment log

## Preregistration and implementation

- Re-audited the program state, with C12/C22, C52, C53, C54, the evaluation-only FTPO round-2 repository harness, and the failed interactive-policy curriculum as closest anchors.
- Selected executable replay compression plus operator-balanced compact banking; entropy/varentropy are routing diagnostics only.
- Created ten procedural repository families: six train/search families and four family-disjoint transfer families.
- Verified on CPU that every family starts visible/hidden broken and becomes visible/hidden correct under its host-only oracle.
- Implemented constrained real filesystem tools, answer-region JSON parsing, terminal hidden grading, replay patch deletion, canonical trace reconstruction, operator balancing, and firewall checks.
- Froze seeds, arms, doses, gates, and the conditional benchmark license before result-bearing generation.
- After preregistration commit `462f6274`, the first GPU smoke stopped before model load because the current runner template lacked its older local-checkpoint field. Added exact Qwen3.5-4B local-checkpoint support plus an architecture fingerprint test; no scientific seed or output was consumed by the failed attempt.
- The corrected GPU smoke loaded the merged C53 checkpoint under vLLM 0.24 and repaired 5/6 one-task-per-family repositories within four turns (implementation evidence only). Its full trajectories remain external and firewall-clean.
