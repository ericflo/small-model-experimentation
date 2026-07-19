# RFT harvesting + ablation scripts (claim C60)

Diagnostic + native-trace-harvest scripts for the WHY-think collapse finding.
Corpora/adapters are ephemeral large-artifacts (not committed); these reproduce them.

- `rft_lib.py` — parse synthetic problems (entry point + asserts), execution filter
  (reuses the fitness harness `code_env`), think/answer split.
- `build_problems.py --offset O --count N` — draw a disjoint slice of the 40k why_think
  pool as runner-input problems + a meta sidecar.
- `sample.sh OFFSET COUNT K TAG` — sample the base via the pinned vLLM runner
  (`--n K --temperature 0.8 --thinking budget 8192`), then filter → native RFT corpus.
- `filter_build.py` — strip_think → extract_candidate_code → execute vs asserts → keep
  passers as native think + native clean code rows (w_think can be 1.0 since native).
- `train_rft.sh` — train two native-think arms (w_think 1.0 / 0.2) + merge + eval.
- `ablation_2x2.sh` / `ablation_cleanfree.sh` — the 2×2 (synthetic-think × #WHY) arms.

Base for all arms: the frozen `base_reserialized` composite. Recipe matches the ladder
(lr 1e-5, r32/a64, bs1 ga8, max-len 4096, seed 95201, 1 epoch). See report §Results.
