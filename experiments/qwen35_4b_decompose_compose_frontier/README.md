# Qwen3.5-4B Decompose-and-Compose Frontier

## Research Program

- Program: `structured_execution_and_compilers`. Mission: **extend the fixed Qwen3.5-4B frontier** — no
  teacher, no scaling, no contaminated data ([[unearth-latent-capability-mission]]).
- Attacks the open problem C11/M4 posed: self-training saturates at the model's sampling frontier
  (depth-3 uncrackable). Can the model give itself **serial depth** it lacks in one forward pass by
  composing its own reliable primitives through the interpreter?

## Question

The depth-3 wall may be a serial-composition wall: the frozen 4B reliably does depth-1/2 sub-steps but
can't compose 3 in one shot. A **decompose-and-compose search** turns a depth-3 task into sequential
depth-1 decisions — at each node the 4B ranks the next primitive (letter-logit read of current-state →
target), the interpreter executes it to materialize the intermediate state, recurse/backtrack over the 23
primitives. Two things to establish, both with sharp controls:

1. **Does model guidance beat matched-budget BRUTE-FORCE enumeration?** With 23 primitives, blind search
   already cracks depth-3, so guidance only *elicits latent capability* if it solves more per
   interpreter-call. We report the full solve-rate-vs-call-budget curve, not just accuracy.
2. **Does it crack depth-3 that MONOLITHIC sampling can't** (the frontier win) — and if so, does banking
   the found solutions (QLoRA-SFT) lift monolithic single-shot depth-3 (the bound M4 couldn't break)?

The planner-wall risk: the model never sees intermediate states, so per-step proposal from I/O may be as
weak as monolithic writing — the wall may just move to the planner. The brute-force comparison IS that test.

## Setup

- Fresh procedurally-generated tasks (10 visible + 8 hidden I/O examples; more visible than M1-M4 to
  constrain found pipelines to generalize). Held-out seed 777. Graded on **hidden** examples.
- `src/decompose_lib.py`: letter-logit primitive ranking + beam search + brute baseline. Reuses the
  substrate generator (`gen_tasks.py`), sandbox (`code_env.py`), runtime (`gen_lib.py`), trainer.

## Run

```bash
../../.venv/bin/python scripts/run_search.py --per-depth 40 --depths 2 3 --beam 50 --top-p 8   # search
../../.venv/bin/python analysis/search_curve.py                                                 # figure
# if depth-3 solutions found: bank them
../../.venv/bin/python scripts/train_lora.py --train data/found_solutions.jsonl --out runs/frontier_adapter
```

## Results

Full write-up in [reports/report.md](reports/report.md).

- **Search cracks the frontier:** hidden-generalizing depth-3 solve rate — monolithic 0.125 → decompose
  **0.40+ (3.4×)**. But held to the brute-force bar, the model's guidance buys **efficiency, not coverage**:
  guided solves with ~2.5× fewer interpreter calls and wins at low budget, but plateaus (planner-wall) while
  brute-force enumeration matches/beats it. The crack is the composition-structure + interpreter.
- **Banking extends the frontier into the weights:** QLoRA-SFT on 327 search-found solutions (no teacher)
  lifts monolithic held-out pass@5 0.125 → 0.237 (+0.112, ~2.6 SE), **depth-3 pass@5 4×** (0.025→0.10),
  greedy@1 +0.05 — the bound M4 could not break. **Replicated** (fresh harvest seed: greedy identical,
  pass@5 0.263, depth-3 pass@5 0.175).
- **Retro-audit (behavioral min-depth):** 40% of nominal depth-3 tasks are behaviorally depth ≤2
  (shallower-equivalent compositions). Re-sliced: decompose solved **16/16 collapsed but only 4/24 (17%)
  true depth-3**; monolithic true depth-3 = 0/24 (and 0 across the whole corpus). The frontier extension is
  real but far more modest than nominal numbers suggest; banking eval is ~30% collapsed at d3 (caveat).

**Answer to C11's open problem:** the frontier extends without a teacher via *tool-augmented search
(composition + interpreter) → harvest frontier-exceeding solutions → bank them*. Modest but real — and the
min-depth audit shows every prior "depth-3" number in the arc was inflated by shallower-equivalent tasks.
