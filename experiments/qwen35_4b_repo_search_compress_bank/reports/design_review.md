# Adversarial design review

Self-review was required because the active operating policy forbids sub-agent delegation. This review occurred before any result-bearing model generation, training, or evaluation.

## Verdict

**Sound with fixed safeguards.** The raw idea—bank successful coding trajectories—would repeat the corpus's DAgger failure. The accepted design banks only a replay-minimized causal core and gives rare verification/commit operators equal loss mass.

## Main attacks and dispositions

1. **This is full-trajectory imitation with a new name.** Successful traces still contain wandering, failed patches, and accidental success. **Fix:** retain only source-changing patches, greedily delete replay-unnecessary patches, reconstruct a canonical trace, and require a fresh visible+hidden+submit replay.
2. **Patch tokens dominate again.** Exact code replacements can contain far more tokens than `test` or `submit`, even when row counts are balanced. **Fix:** equalize operator row loss mass and gate realized VERIFY/COMMIT behavior. The receipt separately reports row counts and weighted mass; token-mass diagnostics are required before interpreting training.
3. **Compact prose may be decorative.** The plans may add no causal value beyond action SFT. **Fix:** `action_only` uses identical states, actions, weights, steps, and optimizer budget, with only plan-span loss removed.
4. **Hidden tests become an oracle teacher.** Repeatedly inspecting private failures could leak edge cases. **Fix:** the model never receives hidden output or source; hidden checks run only at terminal scoring/compression replay and return booleans to host logic. The bank firewall rejects private field names or executable substrings.
5. **Training-family generalization is fake.** Procedural resampling can preserve templates. **Fix:** the primary gate uses four wholly different held-out algorithms, then repeats on new seeds. Train-family items are retention only.
6. **Eight serial turns simply outspend the baseline.** **Fix:** the required baseline gets two independent four-turn trajectories with the same eight calls and 6,144 reserved sampled tokens/task. Sample-more success is the union across both trajectories.
7. **A stronger search teacher smuggles in another model.** **Fix:** C53 is a checkpoint of the one permitted model and supplies only its own tool actions; executable verification, not a teacher label, selects traces. The candidate still trains from the pinned base.
8. **The C54 replay confounds the repository increment.** **Fix:** regenerate an identical apex-only control with the same trainer/seed/step budget; compare compact and action-only only against that control.
9. **Same steps do not mean same FLOPs.** Repository patches make batches longer. **Disposition:** fixed optimizer steps are the primary train-budget match; record target tokens, padded tokens, wall time, and peak memory. Compact versus action-only—the mechanism contrast—has identical sequence lengths, which is the decisive comparison.
10. **The locality threshold vetoes real learning.** Coding behavior should change on coding contexts. **Fix:** locality uses frozen unrelated contexts, not repository tasks, and compares compact to the identically regenerated apex rather than base. The 0.15 ceiling is a collateral guard, not a demand for zero change.
11. **Menagerie is used for iterative tuning.** **Fix:** its seeds are not assigned until both transfer blocks pass; one frozen candidate receives two paired quick/medium events. No result details or benchmark source enter the experiment.
12. **Claim collision with the active Pareto experiment.** **Fix:** no claim ID is reserved. Any result is framed as coding-curriculum transfer; Pareto consolidation remains owned by `qwen35_4b_pareto_policy_integration`.

## Residual risks accepted

- The repositories are deliberately small. Family-disjoint transfer and Menagerie arbitration bound the claim; success does not establish production SWE-bench competence.
- Greedy patch deletion finds a locally minimal patch list, not necessarily a globally shortest proof. Executable validity matters more than minimality optimality.
- Templated compact plans may teach a protocol more than deep algorithmic reasoning. The action-only control and family-disjoint transfer identify whether that protocol contributes useful capability.
- A single registered repository dose may be underpowered. A null remains a dose-specific result, but evaluation-driven retuning is not permitted in this experiment.
