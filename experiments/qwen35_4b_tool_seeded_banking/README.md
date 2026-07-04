# Qwen3.5-4B Tool-Seeded Banking: does tool-search + banking cross the depth-3 wall?

## Research Program
- Program: `structured_execution_and_compilers`
- Program question: the C21 positive control — does seeding banking with tool-found depth-3 solutions cross the depth-3 wall self-banking couldn't?
- Prior anchors: C21 (self-banking can't climb), C12 (tool-search cracks depth-3), C19 (depth-3 representation is a thread).

## Question
C21: self-banking gives depth-3 coverage 0.00 (base samples ~0 depth-3 to bank). Does harvesting depth-3 via an interpreter-backed explorer, then banking, install depth-3?

## Hypothesis
Pre-registered (`reports/prereg.md`, hardened by `reports/design_review.md`): P1 explorer solves >=90% depth-3; P2 banked_tool depth-3 think cov >= 0.15 & >= base+0.10 & >=5 distinct; P3 generalization; P4 deployable + next rung.

## Setup
- Model: Qwen3.5-4B (only permitted model). No external model — explorer is interpreter brute-search over families' own 16-op DSL (CPU).
- Harvest depth-3 (130/130 solved) + C21's exact depth-1+2 pairs = 260 pairs {1:47,2:83,3:130}. Bank -> banked_tool.
- Eval: frozen PAIRED held-out set (behavioral function-signature dedup), n=40/depth 2/3/4; think (primary) + no-think (deployable).

## Run
Smoke: `python scripts/tool_harvest.py --smoke`
Full: `python scripts/tool_harvest.py --n-depth3 130 && bash runs/launch.sh && python scripts/analyze.py`

## Results
**CROSSED-BUT-WEAK.** Depth-3 think coverage 0.00 (0/40) -> 0.125 (5/40 distinct novel tasks) — significant vs the 0/40 floor where C21 gave exactly 0 (tools explore, banking installs). But weak/test-time-dominated: no-think depth-3 0.025, greedy@1 0.00, vs depth-2 deployable greedy@1 0.15. Depth-4 stayed 0. See `reports/report.md`, `analysis/tool_seeded_banking.png`.

## Interpretation
Validates the C21 recipe (explorer + installer, both required) but reveals the installer's efficacy decays with depth (echoes C19/C20). Each rung must be seeded; extendable with diminishing efficiency.

## Knowledgebase Update
- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C22)
- Claim ledger updated: C22 added

## Artifacts
- `scripts/tool_harvest.py` (CPU explorer + combine), `scripts/train_lora.py`, `scripts/eval_ladder.py` (frozen paired + behavioral dedup), `scripts/analyze.py`, `scripts/common.py`
- `data/{train,tool_depth3,eval_frozen,train_tasks}.jsonl`, `runs/eval_{base,banked}_{think,nt}.json`, `runs/verdict.json`
- `analysis/tool_seeded_banking.png`, `reports/{prereg,report,design_review}.md`
- `runs/banked_tool_adapter/` — adapter (~180MB, moved out of repo; regenerable)
