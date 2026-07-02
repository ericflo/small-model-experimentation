# Qwen3.5-4B Depth-Wall Anatomy Experiment Log

## Design

Insight-first: decompose the "depth-3 wall" (center of the C11/C12 arc) into measurable parts with
pre-registered predictions (reports/prereg.md). Phase 0 = behavioral min-depth audit of ALL existing
substrate tasks (CPU, exact BFS). Phase 1 = verified-depth factorial grid, depth d x destructive-ops k.
Phase 2 = discriminator (bare vs plan-given vs intermediates-shown) separating hypothesis identification
from execution.

## Phase 0 results (predictions P0a/P0b CONFIRMED)

40% of nominal-d3 tasks behaviorally collapse to <=d2 (M1 6/15, C12 16/40). Monolithic TRUE depth-3
solves across the ENTIRE corpus: 0 -- every recorded d3 solve was a collapsed task. C12 decompose:
16/16 collapsed vs 4/24 (17%) true-d3. Destruction signal survives the collapse control (M2 true-d2:
k=0 6/8, k>=1 0/8). Retro-corrections to C12 committed (f6c2ca7, CI green).

## Phase 1

Grid generation: 425/425 verified tasks (17 cells, oracle 100%); generation cost ~35 min CPU (BFS
rejection; transparent cells reject often, e.g. negate-negate collapses). GPU run in progress.

## Phase 2

Pending (runs after grid; discriminator on grid task subset).

## Phase 1 results (grid, 425 verified tasks -- P1/P2 REFUTED, steeper law found)

pass@6: d1 0.88(k0)/0.72(k1); d2 0.16/0.04/0.08; d3-d5 ~0.00 EVERYWHERE (2 solves in 275 tasks at d>=3).
My pre-registered destruction hypothesis DIED: k=0 deep tasks are NOT solvable (P1 refuted -- predicted
>=0.4 at d4, got 0.00) and destruction count barely matters at fixed depth (P2 refuted; logistic coefs
-3.24 transparent vs -3.87 destructive; depth-only AIC 112.7 vs two-param 111.9). The motivating M2
signal (k=0 true-d2 6/8) was primitive-mix luck at n=8.

REPLACEMENT LAW: odds of solving fall ~30x per composed op (~ -3.5 logits/op), uniform across op types;
blind guessing over the 63-op space costs ~63x/op -> the model identifies each additional composed op at
only ~2x better than chance. The wall on genuinely novel compositions is at depth TWO. False-passes ~nil
(3/156). Planner first-op rank poor everywhere (median 7-13/23), no destruction effect.

Grid runtime note: sampled phase 6253s (2500 gens at budget 512) -- longer than projected (verified d>=3
tasks never terminate early + long thinking). Generation (BFS-verified) 35 min CPU.

## Phase 2 results (discriminator -- P7 CONFIRMED strongest form, P8 REFUTED)

pass@4: plan_given 0.90-1.00 at EVERY cell (d2-d4, k0/k2) vs bare 0.00-0.10 vs intermediates-shown
0.00-0.30. The wall contains ZERO execution deficit -- it is 100% hypothesis identification, and even
full observability of intermediate states barely rescues (the model cannot SEGMENT chains into the
depth-1 identifications it does at 0.88). Convergent constant: the grid's ~2x-over-chance/op law equals
C12's ~2x guided-vs-brute efficiency, measured independently. Headline: the fixed 4B is a reliable
compiler starved of hypothesis search. Runtime: 2725s for 1440 gens.
