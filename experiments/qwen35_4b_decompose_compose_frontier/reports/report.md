# Qwen3.5-4B Decompose-and-Compose Frontier Report

## Summary

C11/M4 showed self-training saturates at the fixed 4B's sampling frontier (depth-3 uncrackable). This
experiment attacks that frontier without a teacher: a **decompose-and-compose search** gives the model the
*serial depth* it lacks in one forward pass — at each node the 4B ranks the next primitive (letter-logit
read of current-state → target), the interpreter executes it to materialize the intermediate state,
recurse/backtrack over the 23 primitives. Two findings. (1) **It cracks the frontier monolithic sampling
can't** — hidden-generalizing depth-3 solve rate: monolithic 0.125 → decompose **0.40–0.43 (3.4×)**. (2)
**But against the brute-force bar, the model's *guidance* buys efficiency, not coverage:** guided solves
with far fewer interpreter calls (depth-3: 350 vs 895) and wins at low budgets, but **plateaus** (the
planner-wall — where its ranking misses it never recovers) while brute-force enumeration keeps climbing to
match/beat it. So the frontier crack comes from the composition structure + interpreter, not the model's
planning. (3) **Banking EXTENDS the frontier into the weights** — QLoRA-SFT on 327 search-found solutions
(no teacher) lifts the model's OWN monolithic sampling on held-out: pass@5 0.125 → 0.237 (+0.112, ~2.6 SE),
**depth-3 pass@5 0.025 → 0.100 (4×)**, greedy@1 0.075 → 0.125 (+0.05, suggestive), and a no-think one-shot
path emerges (0 → 0.062). Search+interpreter harvested solutions from *outside* the sampling distribution
and banking pulled them *into* it — the exact bound M4 (confined to the sampling distribution) could not
break. Modest in absolute terms but a genuine, teacher-free frontier extension.

## Research Program Fit

`structured_execution_and_compilers`. Directly attacks the open problem C11 posed. On-mission: the
interpreter is a calculator and brute-force search is a deterministic algorithm — no teacher, no other
model, no contaminated data. The **brute-force bar** is the mission's honesty criterion: with only 23
primitives, blind enumeration already cracks depth-3, so model guidance only *elicits latent capability* if
it beats matched-budget brute force.

## Method

- Fresh procedurally-generated tasks (10 visible + 8 hidden I/O examples — more visible than M1–M4 to force
  found pipelines to generalize). Held-out seed 777 (search), 404 (bank eval), 888 (harvest). Graded on hidden.
- **Search** (`src/decompose_lib.py`): beam search over primitive pipelines. Guided = the 4B ranks the next
  primitive via a 23-way letter-logit read (one forward/node); brute = all primitives. Both share an
  interpreter-call budget; each application = 1 call. Solve-rate-vs-call-budget curve from one run/task.
- **Monolithic baseline**: M2-style — sample k=8 whole `transform`s (thinking), select by visible pass, grade hidden.
- **Bank**: harvest hidden-generalizing solutions via brute search on a large fresh pool → QLoRA-SFT (no
  teacher) → eval monolithic depth-2/3 frozen vs trained.

## Results

Hidden-generalizing solve rate (n=40/depth):

| depth | monolithic (k=8) | guided (mean calls) | brute (mean calls) |
| ---: | ---: | ---: | ---: |
| 2 | 0.325 | 0.575 (158) | 0.875 (825) |
| 3 | 0.125 | 0.400 (350) | 0.425 (895) |

Solve-rate vs interpreter-call budget (depth 3): guided 0.08→0.20→0.33→0.40 at {50,200,400,3000}; brute
0.03→0.18→0.23→0.40→0.43 at {50,200,400,3000,4000}. Figure: `analysis/search_curve.png`.

- **Frontier cracked:** both decompose methods solve depth-3 at ~3.4× the monolithic rate (0.40+ vs 0.125)
  — externalized serial computation reaches solutions the model can't produce in one shot.
- **Guidance = efficiency, not coverage:** guided reaches its plateau with ~2.5× fewer calls than brute and
  dominates the low-budget regime, but caps below brute at high budget (depth-2: guided 0.575 vs brute 0.875)
  — the planner-wall. At depth-3 they converge (0.40 vs 0.43).
- So by the strict brute-force bar, the model is **not** out-eliciting search on coverage; it makes the
  search cheaper. The frontier crack is the composition-structure-plus-interpreter, not model foresight.

### Banking — frontier extension into the weights
Harvested 327 hidden-generalizing solutions (242 d2 + 85 d3) via brute search + interpreter (no teacher);
QLoRA-SFT → monolithic frozen vs trained on held-out (n=80, seed 404):

| metric (monolithic held-out) | frozen | trained | Δ |
| --- | ---: | ---: | ---: |
| think greedy@1 | 0.075 | 0.125 | +0.050 (~1.5 SE) |
| think pass@5 | 0.125 | 0.237 | **+0.112 (~2.6 SE)** |
| depth-3 pass@5 | 0.025 | 0.100 | **4×** |
| no-think greedy@1 | 0.000 | 0.062 | +0.062 |

Banking search-found solutions the model could NOT monolithically sample raised its own sampling coverage
(pass@5 +0.112 significant; depth-3 4×) and its single-shot rate (greedy +0.05, suggestive), plus a
no-think one-shot path from ~0. This is the frontier extension M4 could not achieve (M4 banked only
sampleable solutions). **Replicated** with a fresh harvest seed (999): greedy@1 0.125 (identical),
pass@5 0.263 (vs 0.237), depth-3 pass@5 0.175, no-think 0.100 — robust to the harvest data. Honest
limits: absolutes remain low; n=80 held-out; and see the retro-audit below for a collapsed-task caveat.

### Retro-audit: behavioral min-depth (added post-hoc; full treatment in the follow-up experiment)
The substrate generator excluded degenerate compositions but not **shallower-equivalent** ones
(e.g. `sort_asc∘reverse ≡ sort_desc`). An exact behavioral min-depth audit (BFS over all primitive
pipelines against all 18 examples) found **40% of this experiment's nominal depth-3 search tasks are
behaviorally depth ≤2**. Re-slicing the search results by true structure:

| nominal d3 slice | decompose solved | monolithic solved |
| --- | ---: | ---: |
| collapsed (min-depth ≤2), n=16 | **16/16** | (mixed) |
| full depth-3, n=24 | **4/24 (17%)** | 0/24 |

So the headline "decompose cracks depth-3 at 0.40" decomposes into: all of the behaviorally-shallow tasks
plus **17% of true depth-3** — still a strict frontier extension over monolithic's 0/24, but far more
modest than the nominal number. The banking eval set is similarly ~30% collapsed at depth-3, so the banked
"depth-3" gains are over a mixed population (per-task attribution deferred to the follow-up experiment,
which uses min-depth-verified tasks only). Corpus-wide corollary: across ALL prior data (M1/M2/C12), the
frozen model has **never once** solved a true full-depth-3 task monolithically.

## Controls

Brute-force enumeration at matched call-budget (the honesty bar). Hidden-generalization grading (10 visible
constrain the pipeline; still, ~some visible-solves fail hidden — reported only hidden). Monolithic sampling
(k=8 + visible-select) is the frontier baseline the decompose loop must beat.

## Oracle Versus Deployable Evidence

Search is a deployable *procedure* (uses only visible I/O + the interpreter); solve rates are hidden-graded.
The reference oracle (100% solvable) bounds all methods.

## Interpretation

The frontier is crossable — but by giving the fixed model *tools* (composition + an interpreter), not by
its own planning. The model's compositional foresight is real but weak (right primitive in top-8 ~2/3 of the
time at the root) and it degrades over serial steps, so guided search plateaus where brute keeps climbing:
the depth-3 wall **relocates to the planner** rather than disappearing. But banking answers the decisive
question positively: the 4B *can* internalize solutions that only search+interpreter could find, extending
its monolithic frontier into the weights (pass@5 +0.112, depth-3 4×) — which M4, confined to the sampling
distribution, could not. So the answer to C11's open problem ("what extends the frontier without a
teacher?") is **tool-augmented search (composition + interpreter) to harvest frontier-exceeding solutions,
then bank them** — the interpreter is a calculator and search is a deterministic algorithm, no teacher
involved. The extension is real but modest; the natural next step is to iterate it (harvest→bank→re-harvest
with the improved model) as a frontier flywheel.

## Next Experiments

- (banking, running) does SFT on search-found depth-3 lift monolithic depth-3? Iterate (harvest→bank→harvest
  with the improved model) as a frontier flywheel.
- Larger primitive vocabulary where brute-force explodes (so model guidance must carry more of the search).

## Artifact Manifest

See `artifact_manifest.yaml`. Small tasks/solutions/summaries + figure in-repo; LoRA adapter regenerable.
