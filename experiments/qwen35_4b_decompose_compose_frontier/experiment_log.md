# Qwen3.5-4B Decompose-and-Compose Frontier Experiment Log

## Design

Attacks C11/M4's open problem: self-training saturates at the sampling frontier (depth-3 uncrackable).
Give the fixed 4B serial depth via a decompose-and-compose search (rank next primitive -> execute ->
materialize intermediate state -> recurse), measured against BOTH monolithic sampling (the frontier
baseline) and matched-budget brute-force enumeration (the honesty bar). Then bank found solutions.

## Build notes

- First proposal design (parse the model's generated next-op) FAILED -- the model reasons verbosely and gets
  truncated. Switched to a 23-way letter-logit read (like the C10 verifier): one forward/node, ranked
  primitives, params enumerated for the top-p. Fast + reliable.
- 10 visible examples (vs 6 in M1-M4) to constrain found pipelines to generalize to hidden.

## Search results (see runs/search_summary.json, analysis/search_curve.png)

Hidden-generalizing solve rate (n=40/depth): d2 monolithic 0.325 / guided 0.575 (158 calls) / brute 0.875
(825 calls); d3 monolithic 0.125 / guided 0.400 (350 calls) / brute 0.425 (895 calls). Root ranking: true
first-op in model top-8 for 2/3 of tasks.

FRONTIER CRACKED but by STRUCTURE not planning: decompose solves depth-3 at ~3.4x monolithic (0.40+ vs
0.125). Model guidance = call-EFFICIENCY (2.5x fewer calls, wins low-budget) but PLATEAUS (planner-wall);
brute-force enumeration matches/beats it at high budget. By the brute-force bar the model does not
out-elicit search on coverage. Harvested 327 hidden-gen solutions (242 d2 + 85 d3) for banking.

## Banking results (see runs/eval_frozen.json vs eval_trained.json)

QLoRA-SFT on 327 search-found solutions (no teacher) -> monolithic held-out (n=80, seed 404), frozen->trained:
think_greedy@1 0.075->0.125 (+0.05, ~1.5 SE); think_pass@5 0.125->0.237 (+0.112, ~2.6 SE); depth-3 pass@5
0.025->0.100 (4x); nothink_greedy 0.000->0.062. FRONTIER EXTENDED INTO THE WEIGHTS: banking solutions the
model couldn't monolithically sample raised its own sampling coverage -- the bound M4 could not break.
Modest absolutes; greedy gain suggestive, pass@5 gain significant. Replication (harvest seed 999) running
via scripts/repl_bank.sh. Adapters (~170MB) gitignored + removed before commit.

## Replication + retro-audit (post-close additions)

- Banking REPLICATED with fresh harvest seed 999 (327 solutions): greedy@1 0.125 (identical to seed-888),
  pass@5 0.263 (vs 0.237), depth-3 pass@5 0.175, no-think 0.100. Robust to harvest data.
- BEHAVIORAL MIN-DEPTH RETRO-AUDIT (exact BFS over all pipelines vs all 18 examples): the generator did not
  exclude shallower-equivalent compositions; 40% of nominal-d3 search tasks are behaviorally depth<=2.
  Re-sliced: decompose solved 16/16 collapsed but only 4/24 (17%) TRUE depth-3; monolithic true-d3 0/24 --
  and 0 across the entire corpus (M1/M2/C12). Banking eval ~30% collapsed at d3 (mixed-population caveat).
  Full treatment + verified-depth substrate in the follow-up experiment (depth-wall anatomy).
