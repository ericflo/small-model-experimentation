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
- Corrected smoke scoring to require both visible and private tests: the valid count is 4/6; one apparent fifth success passed private edge cases while regressing the visible suite. Added a regression test for this exact failure.
- Replaced equal row-count weighting with exact tokenizer-level action-token mass equality and separately equal compact-plan mass. The action-only control now retains the identical teacher-forced compact text and removes only its plan-span gradient.
- Froze 48 unrelated non-coding contexts for apex→compact centered-logit locality and implemented the full matched-step trainer, explicit composite merge, paired repository analysis, and gate-stopping continuation path while the registered harvest ran.

## Registered harvest and bank

- Search teacher covered 129/144 tasks (89.6%; per-family range 58.3–100%) from 576 trajectories and 1,225,314 sampled tokens. There were 376 conjunctive visible+private successes, 370 explicit post-patch verifications, and 351 commits after a pass.
- Replay minimization admitted all 129 covered tasks. Seven source trajectories needed two patch calls, but per-file initial→final collapse reduced every canonical trace to one patch; all canonical visible/private/submission replays passed.
- Built 516 rows: exactly 129 each for INSPECT/PATCH/VERIFY/COMMIT. Exact weighted action-token mass is 36,110 per operator and compact-plan mass 3,125.8 per operator. Longest target sequence is 879 tokens.
- Tokenizer preflight encoded all 4,669 C54 rows and all 516 repository rows with zero skips. Apex replay is exactly 2.0 dataset epochs at 584 steps; both candidates are exactly 1.8011 union epochs at the same 584 steps.
- Every registered pre-training gate passed; training authorized. Menagerie remains sealed.

## Training feasibility recovery

- Apex replay with batch 4 × accumulation 4 stopped at optimizer step 52 on a 3,193-token batch: the 9.54 GiB logits allocation exceeded 9.09 GiB free. Loss/gradients were finite, but no adapter or checkpoint was saved.
- Froze the compute-equivalent recovery before rerun: batch 2 × accumulation 8, effective batch 16, 584 steps, 9,344 examples, three apex padding duplicates, and exactly two control epochs. Enabled expandable CUDA segments. All arms share the corrected geometry.
- The 2 × 8 fallback completed three unsaved steps but measured 19–23 s/step. Root cause was dense cross-entropy's full 248k-vocabulary FP32 temporary, not forward activations. Restored the original 4 × 4 geometry with exact gradient-checkpointed 128-position loss chunks; CPU loss/gradient equivalence passes at 1e-6.
- A two-step canary on the eight longest 3,193-token rows passed at batch four, peaking at 48,375,846,912 CUDA bytes. This directly covers the formerly failing allocation path; the smoke adapter is not eligible for evaluation.

## Result-bearing training and necessary gate

- Apex replay completed 584/584 optimizer steps in 10,007 seconds; compact completed 584/584 in 9,633 seconds. Both used exact checkpointed full-vocabulary cross-entropy, effective batch 16, and approximately 48.58 GB peak CUDA memory. All 128 LoRA modules were nonzero and explicitly merged for both arms.
- Trained-family deep evaluation: apex 40/48, compact 48/48 (+16.7pp; paired CI [+6.3,+27.1]). Compact used the exact four-operator canonical path on all 48 tasks, with zero invalid calls and 402 versus 1,643 mean sampled tokens.
- Family-disjoint transfer: apex 49/72, compact 25/72 (−33.3pp; CI [−44.4,−22.2]). Matched two-by-four-turn apex sampling scored 38/72, still +18.1pp over compact. Compact invalid calls rose 9.3%→26.0%, verification retention fell 1.00→0.88, and median unrelated-context centered-logit drift was 0.386 versus the 0.15 ceiling.
- Transition audit: after a failed visible test, apex patched again on 24/26 transitions; compact did so on 0/48. After a pass, commit remained intact (64/65 apex, 22/22 compact). All 18 recursive-overlay compact trajectories repeated the identical already-rejected patch. Exact operator marginals preserved the happy path but not verifier-conditioned recovery.
- The primary gate stopped action-only training, confirmation, and Menagerie. Zero benchmark seeds were assigned or consumed. The action-only cancellation leaves plan-gradient attribution unresolved by design.
- Fixed a staged-run footgun found on the negative path: locality correctly wrote a failing receipt and returned status 4, but the orchestrator previously raised before consolidated analysis. Expected gate status is now explicitly accepted, summarized, and returned cleanly; unexpected nonzero statuses still raise, with a regression test.
