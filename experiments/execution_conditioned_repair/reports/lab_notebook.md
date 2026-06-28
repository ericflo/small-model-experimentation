# Lab Notebook: Execution-Conditioned Repair LoRA

## 2026-06-19 Environment Setup

- Read the experiment brief from `/root/.codex/attachments/67be2ef6-a2f4-424d-a411-f0479b10c076/pasted-text-1.txt`.
- Hardware: NVIDIA RTX 6000 Ada, 48GB VRAM; BF16 supported.
- Installed missing Python prerequisites: `transformers`, `accelerate`, `datasets`, `peft`, `trl`, `bitsandbytes`, plotting and test utilities.
- Verified `bitsandbytes` CUDA support.
- Verified primary model config:
  - `Qwen/Qwen3-4B-Instruct-2507`
  - revision `cdbee75f17c01a7cc42f958dc650907174af0554`
  - model type `qwen3`, 36 layers, hidden size 2560.
- Installed Docker and started `dockerd` with bridge networking disabled, then with `vfs` storage.
- Docker preflight failed at container layer registration with `unshare: operation not permitted`.
- Conclusion: official SWE-smith/SWE-bench Docker-backed execution is not valid in this runner until kernel/container privileges change.

## Synthetic Pilot Dataset v1

- Built a local executable Python repair suite with 13 tasks and 64 wrong-patch episodes.
- Splits:
  - Train: 50 episodes.
  - Synthetic held-out: 14 episodes.
- Wrong-patch variants:
  - near miss
  - wrong localization
  - syntax error
  - import error
  - visible-test overfit
- Failure classes:
  - assertion: 13
  - import: 13
  - syntax: 13
  - visible-pass-hidden-fail: 25
- Correctness audit:
  - Recomputed every `target_next_diff` from current wrong-patched files to clean files.
  - 50/50 train target diffs applied and passed hidden tests.
  - 14/14 validation target diffs applied and passed hidden tests.

## Frozen-Model Sampling Smoke

- Loaded `Qwen/Qwen3-4B-Instruct-2507` in 4-bit successfully.
- Sampled first-patch outputs on two synthetic tasks.
- Initial patch extraction failed on standard `index ...` diff headers; fixed extractor.
- Model often emits plausible final-fix diffs against the original buggy file, but with bad hunk counts or against the wrong tree.
- Added `git apply --recount` so count-mismatched hunks can apply when context is otherwise valid.

## Frozen-Model Wrong-Patch Stream

- Ran `scripts/sample_wrong_patches.py` over all 13 local tasks with one sample per task.
- Output: `data/frozen_wrong_patches_qwen3_local.jsonl`.
- Breakdown:
  - 13 total records.
  - 6 applied model patches.
  - 7 rejected/apply-error model patches.
  - 10 hidden-failing records usable as repair episodes.
  - 3 records already passed hidden tests and are not useful first-failure repair examples.
- Filtered usable failures to `data/frozen_wrong_patches_qwen3_local_failures.jsonl`.
- Correctness audit:
  - 10/10 filtered target diffs apply.
  - 10/10 filtered target diffs pass hidden tests.

This stream is small but more faithful to the brief than the scripted wrong-patch variants because `W_i` comes from the frozen Qwen3 model.

## Baseline and Adapter Runs

All adapters used:

- base model: `Qwen/Qwen3-4B-Instruct-2507`
- revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- 4-bit NF4 QLoRA
- rank 32, alpha 64
- target modules: Qwen attention projections and MLP projections
- assistant diff tokens only in loss

Initial 3-epoch results on held-out synthetic v1:

| Condition | repair_after_first_failure@1 | patch_apply_rate | syntax_valid_rate |
|---|---:|---:|---:|
| Frozen second attempt | 0.00 | 0.071 | 0.071 |
| Final-patch SFT | 0.00 | 0.429 | 0.286 |
| Failure-conditioned no-trace SFT | 0.00 | 0.571 | 0.571 |
| Failure-conditioned trace SFT | 0.00 | 0.714 | 0.643 |
| Failure-conditioned shuffled-trace SFT | 0.00 | 0.643 | 0.643 |

Interpretation:

- SFT improved diff format/application substantially.
- It did not improve semantic repair on held-out tasks.
- Normal trace beat shuffled/no-trace on patch application but not on hidden-test repair.
- No scientific success claim is supported.

## Undertraining Diagnostic

- Trace adapter after 3 epochs repaired only 3/20 train episodes.
- Shorter generation (`max_new_tokens=256`) did not improve this.
- Trained a stronger trace adapter v2:
  - 12 epochs
  - gradient accumulation 4
  - learning rate 2e-4
  - LoRA dropout 0.0
- v2 repaired 19/20 train examples, proving the pipeline can learn executable corrective diffs.
- v2 repaired 0/14 held-out synthetic examples.

Interpretation:

- The initial failure was partly undertraining, but the held-out failure remains after memorization.
- Current synthetic v1 data does not support transfer beyond memorized task patterns.
- The next experimental step should expand data diversity and task count, not further tune on the same 50 episodes.

## Trace Ablations

Ran full held-out synthetic trace ablations for the 3-epoch core trace adapter:

| Prompt mode | repair_after_first_failure@1 | patch_apply_rate |
|---|---:|---:|
| normal trace | 0.000 | 0.714 |
| no trace | 0.000 | 0.857 |
| wrong patch only | 0.000 | 0.286 |
| trace only | 0.071 | 0.571 |
| gold-file context removed | 0.000 | 0.143 |

Interpretation:

- The required evidence pattern `normal trace > no trace`, `normal trace > shuffled trace`, and `normal trace > wrong-patch-only` is absent.
- Trace-only producing one success while normal trace produced zero is likely noise on a 14-record slice, not evidence of useful trace grounding.
- The current positive effect is mostly patch applicability/formatting, not execution-conditioned semantic repair.
- The wrong-patch-only row above was rerun after fixing the stricter control definition to remove repository file context.

## Current Hypotheses

1. The core prompt/training/evaluation machinery is functional.
2. The local synthetic v1 train set is too small and too template-specific for held-out task transfer.
3. The model is learning patch formatting and current-tree anchoring before learning robust repair semantics.
4. Execution traces may help formatting/anchoring but there is no evidence yet that trace content causes repair success.
5. Official real-task transfer cannot be measured in this runner until Docker/SWE-bench execution works.

## Next Actions

1. Build a larger synthetic v2 dataset with many more train tasks per bug family and separate held-out families.
2. Generate at least some wrong patches from the frozen model and filter for valid first-failure episodes.
3. Re-run the C/D/E/F comparison on v2 with enough data to test transfer.
4. Only after positive synthetic repair exists, spend compute on SWE-bench-style real slices.

## 2026-06-20 Synthetic v2 Expansion

- Added `scripts/build_repair_dataset_v2.py`.
- Built a larger executable synthetic dataset under `data/v2`.
- v2 manifest:
  - 66 tasks.
  - 327 repair episodes.
  - 240 train episodes.
  - 60 IID validation episodes from trained bug families.
  - 27 held-out-family validation episodes from `path_norm` and `tie_breaking`.
  - 3 skipped episodes where a wrong patch already passed hidden tests.
- Verified target corrective diffs:
  - 240/240 train target diffs applied and passed hidden tests.
  - 60/60 IID validation target diffs applied and passed hidden tests.
  - 27/27 held-out-family validation target diffs applied and passed hidden tests.
- Trained four v2 adapters using the same base revision and QLoRA recipe:
  - `models/v2_final_patch_sft_lora`
  - `models/v2_failure_conditioned_no_trace_lora`
  - `models/v2_failure_conditioned_trace_lora`
  - `models/v2_failure_conditioned_shuffled_trace_lora`

v2 IID synthetic executable results:

| Condition | repair_after_first_failure@1 | patch_apply_rate | syntax_valid_rate | successes |
|---|---:|---:|---:|---:|
| Frozen second attempt | 0.100 | 0.183 | 0.167 | 6/60 |
| Final-patch SFT | 0.183 | 0.383 | 0.383 | 11/60 |
| Failure-conditioned no-trace SFT | 1.000 | 1.000 | 1.000 | 60/60 |
| Failure-conditioned trace SFT | 1.000 | 1.000 | 1.000 | 60/60 |
| Failure-conditioned shuffled-trace SFT | 1.000 | 1.000 | 1.000 | 60/60 |

v2 held-out-family executable results:

| Condition | repair_after_first_failure@1 | patch_apply_rate | syntax_valid_rate | successes |
|---|---:|---:|---:|---:|
| Frozen second attempt | 0.000 | 0.185 | 0.148 | 0/27 |
| Final-patch SFT | 0.000 | 0.370 | 0.185 | 0/27 |
| Failure-conditioned no-trace SFT | 0.000 | 0.444 | 0.444 | 0/27 |
| Failure-conditioned trace SFT | 0.000 | 0.444 | 0.444 | 0/27 |
| Failure-conditioned shuffled-trace SFT | 0.000 | 0.444 | 0.444 | 0/27 |

v2 held-out-family best-of-3 sampled results (`temperature=0.2`, `top_p=0.95`):

| Condition | repair_after_first_failure@1 | repair_after_first_failure@3 | patch_apply_rate | syntax_valid_rate | successes |
|---|---:|---:|---:|---:|---:|
| Frozen second attempt | 0.000 | 0.000 | 0.185 | 0.185 | 0/27 |
| Final-patch SFT | 0.000 | 0.000 | 0.407 | 0.185 | 0/27 |
| Failure-conditioned no-trace SFT | 0.000 | 0.000 | 0.444 | 0.444 | 0/27 |
| Failure-conditioned trace SFT | 0.000 | 0.000 | 0.444 | 0.444 | 0/27 |
| Failure-conditioned shuffled-trace SFT | 0.000 | 0.000 | 0.444 | 0.444 | 0/27 |

Held-out-family trace ablation for `models/v2_failure_conditioned_trace_lora`:

| Prompt mode | repair_after_first_failure@1 | patch_apply_rate | syntax_valid_rate |
|---|---:|---:|---:|
| normal trace | 0.000 | 0.444 | 0.444 |
| no trace | 0.000 | 0.556 | 0.556 |
| wrong patch only | 0.000 | 0.185 | 0.185 |
| trace only | 0.000 | 0.926 | 0.926 |
| gold-file context removed | 0.000 | 0.185 | 0.185 |

Interpretation:

- v2 proves the training/eval pipeline can produce strong same-family synthetic repair behavior.
- The same success does not transfer to held-out bug families.
- The trace controls still do not show the required evidence pattern.
- The normal trace adapter is not better than no-trace, shuffled-trace, or wrong-patch-only controls on hidden repair.
- The result is a negative transfer result, not a positive execution-conditioned repair result.

Control audit:

- Found and fixed a prompt-control bug: `wrong_patch_only` had incorrectly preserved repository file context, making it equivalent to no-trace.
- Updated `src/repair_experiment/prompts.py` so `wrong_patch_only` removes repository file context and blanks trace output while preserving the current wrong diff.
- Reran `reports/v2_trace_ablation_family_holdout_results.json` after the fix.
- The corrected wrong-patch-only control still repaired 0/27 hidden cases, and its patch-apply rate dropped from 0.556 to 0.185.

## 2026-06-20 Report Generation

- Regenerated the paper-style report at `reports/execution_conditioned_repair_paper.md`.
- Also refreshed `reports/transfer_gap_report.md` to the same content for compatibility with the earlier report path.
- Generated v2 figures:
  - `figures/v2_iid_repair_rate.png`
  - `figures/v2_family_holdout_repair_rate.png`
- Official SWE-bench-style Docker execution remains blocked by `unshare: operation not permitted` during Docker layer registration, recorded in `reports/swebench_preflight.json`.

## 2026-06-20 Coding-Specialist Ablation

Question:

- Is the Qwen3-4B held-out-family collapse specific to the primary base model, or does a coding-specialist base show the same behavior?

Setup:

- Model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Same v2 train/eval splits and QLoRA recipe as the primary runs.
- Trained the C/D/E/F coder adapter set:
  - `models/coder_v2_final_patch_sft_lora`
  - `models/coder_v2_failure_conditioned_no_trace_lora`
  - `models/coder_v2_failure_conditioned_trace_lora`
  - `models/coder_v2_failure_conditioned_shuffled_trace_lora`

Training checks:

- Coder final-patch adapter final IID eval loss: `7.366e-05`.
- Coder no-trace adapter final IID eval loss: `3.829e-04`.
- Coder trace adapter final IID eval loss: `1.235e-04`.
- Coder shuffled-trace adapter final IID eval loss: `1.363e-04`.

Deterministic executable results:

| Split | Condition | repair_after_first_failure@1 | patch_apply_rate | syntax_valid_rate | successes |
|---|---|---:|---:|---:|---:|
| IID | Coder final-patch SFT | 0.283 | 0.600 | 0.533 | 17/60 |
| IID | Coder no-trace repair SFT | 1.000 | 1.000 | 1.000 | 60/60 |
| IID | Coder trace repair SFT | 1.000 | 1.000 | 1.000 | 60/60 |
| IID | Coder shuffled-trace repair SFT | 1.000 | 1.000 | 1.000 | 60/60 |
| Held-out family | Coder final-patch SFT | 0.000 | 0.000 | 0.000 | 0/27 |
| Held-out family | Coder no-trace repair SFT | 0.074 | 0.667 | 0.667 | 2/27 |
| Held-out family | Coder trace repair SFT | 0.111 | 0.667 | 0.667 | 3/27 |
| Held-out family | Coder shuffled-trace repair SFT | 0.000 | 0.370 | 0.370 | 0/27 |

Interpretation:

- The coding-specialist base reproduces the same-family synthetic sanity result.
- Unlike the primary Qwen3-4B trace adapter, it finds 3 held-out-family repairs, so the broadest "zero transfer" claim is model-dependent.
- The trace result has only a weak edge over controls: 3/27 for trace, 2/27 for no-trace, 0/27 for shuffled-trace.
- This is suggestive enough to report as a secondary ablation, but too small for a strong causal trace claim.
- The result remains a weak held-out-family transfer signal, not a SWE-bench transfer result.

## 2026-06-20 Direct Non-Docker SWE-bench Probe

Motivation:

- Official SWE-bench Docker execution is still blocked by `unshare: operation not permitted`.
- I built a direct local pytest probe for one SWE-bench Verified task to get at least some real-task evidence outside Docker.
- This is not the official harness and should not be treated as a replacement for full Verified evaluation.

Task:

- Instance: `pallets__flask-5014`.
- Repo: `pallets/flask`.
- Base commit: `7ee9ceb71e868944a46e1ff00b506772a53a4f1d`.
- Test: `tests/test_blueprints.py::test_empty_name_not_allowed`.
- Context file shown to the model: `src/flask/blueprints.py`.

Harness validation:

- Base + official test patch failed as expected.
- Base + official test patch + official gold patch passed.
- Local runner caveats:
  - Uses direct pytest, not Docker.
  - Uses a manually validated Flask dependency profile under Python 3.12.
  - Forces `PYTHONPATH` to the current worktree `src` directory so each worktree imports its own source.

Qwen3 primary direct-probe results:

| Condition | repair_after_first_failure@1 | end_to_end_resolved@2 |
|---|---:|---:|
| Frozen first patch | n/a | 0/1 |
| Frozen second attempt | 0/1 | 0/1 |
| Final-patch SFT | 0/1 | 0/1 |
| No-trace repair SFT | 0/1 | 0/1 |
| Trace repair SFT | 0/1 | 0/1 |
| Shuffled-trace repair SFT | 0/1 | 0/1 |

Observed failure mode:

- The frozen first patch recognized the semantic fix, adding a `ValueError` for empty blueprint names, but inserted the check at the wrong location and duplicated it.
- The first patch did not apply cleanly.
- The trace repair adapter generated another plausible local edit, but it was still anchored to the wrong hunk and did not apply.

Interpretation:

- This one-task real probe gives real-task negative evidence consistent with the synthetic held-out-family result.
- Direct non-Docker real gain for trace over frozen second attempt is `0.0`.
- Using the v2 IID synthetic gain of `0.9`, the direct-probe transfer ratio is `0.0`.
- Because this is one task and not the official Docker harness, it is reported as a probe, not as the definitive SWE-bench result.

## 2026-06-20 Second Direct Non-Docker SWE-bench Probe

Motivation:

- Add a second validated real task using the same non-Docker direct pytest runner.
- Test whether the Flask negative result was an idiosyncratic failure of one repository layout.
- Generalize `scripts/eval_repair_swebench_direct.py` to per-repository profiles instead of hardcoding Flask.

Task:

- Instance: `psf__requests-5414`.
- Repo: `psf/requests`.
- Base commit: `39d0fdd9096f7dceccbc8f82e1eda7dd64717a8e`.
- Test: `tests/test_requests.py::TestRequests::test_invalid_url[InvalidURL-http://.example.com]`.
- Context file shown to the model: `requests/models.py`.

Harness validation:

- Base + official test patch failed as expected with `urllib3.exceptions.LocationParseError: Failed to parse: '.example.com', label empty or too long`.
- Base + official test patch + official gold patch passed.
- Local runner caveats:
  - Uses direct pytest, not Docker.
  - Uses a manually validated Requests dependency profile under Python 3.12.
  - Forces `PYTHONPATH` to the current worktree root so each worktree imports its own source.

Qwen3 primary direct-probe results:

| Condition | repair_after_first_failure@1 | end_to_end_resolved@2 |
|---|---:|---:|
| Frozen first patch | n/a | 0/1 |
| Frozen second attempt | 0/1 | 0/1 |
| Final-patch SFT | 0/1 | 0/1 |
| No-trace repair SFT | 0/1 | 0/1 |
| Trace repair SFT | 0/1 | 0/1 |
| Shuffled-trace repair SFT | 0/1 | 0/1 |

Observed failure mode:

- The frozen first patch applied, but it added an `idna.IDNAError` handler in the non-ASCII-host branch instead of changing the ASCII leading-dot guard.
- The visible test still failed with `LocationParseError`.
- Every second attempt emitted the same stale first-attempt diff, which then failed to apply on top of the wrong-patched tree.

Updated direct-probe aggregate:

- Validated direct probes: 2 (`pallets__flask-5014`, `psf__requests-5414`).
- Frozen second-attempt repair rate: 0/2.
- Trace repair SFT repair rate: 0/2.
- Direct non-Docker real gain for trace over frozen second attempt: `0.0`.
- Using the v2 IID synthetic gain of `0.9`, the direct-probe transfer ratio is `0.0`.
- The aggregate remains negative evidence only; official Docker SWE-bench execution is still blocked by the environment.

## 2026-06-20 Direct Runner Correction and Third Probe

Evaluation correction:

- While inspecting `psf__requests-6028`, I found that the direct runner evaluated second attempts after a patch-apply failure as `bad_first_patch + second_patch`.
- That is too strict for an apply-error failure: if the first patch does not apply, the repository remains at the original tree, so the second attempt should be evaluated as a replacement patch conditioned on the failed diff and apply-error trace.
- I changed `scripts/eval_repair_swebench_direct.py` so:
  - if the first patch applied, repair is evaluated on top of the wrong-patched tree;
  - if the first patch did not apply, repair is evaluated as a standalone replacement patch on the original tree.
- I reran affected direct probes `pallets__flask-5014` and `psf__requests-6028`; both remained 0/1 for every Qwen3 condition.

Additional preflights:

- `psf__requests-6028`: base failed and gold passed, usable.
- `psf__requests-2931`: base failed but gold did not pass under the direct local profile, so excluded.
- `psf__requests-1142`: local install failed on Python 3.12 because old vendored urllib3 imports `MutableMapping` from `collections`, so excluded.
- `psf__requests-1724`: local install failed on Python 3.12 because old vendored urllib3 imports `MutableMapping` from `collections`, so excluded.
- `psf__requests-1766`: local install failed on Python 3.12 because old vendored urllib3 cannot import `ssl.match_hostname` and lacks the backport, so excluded.
- `psf__requests-1921`: local install failed on Python 3.12 for the same vendored `ssl_match_hostname` issue, so excluded.
- `psf__requests-2317`: local install failed on Python 3.12 because old vendored urllib3 imports `Mapping` and `MutableMapping` from `collections`, so excluded.

Third task:

- Instance: `psf__requests-6028`.
- Repo: `psf/requests`.
- Base commit: `0192aac24123735b3eaf9b08df46429bb770c283`.
- Tests:
  - `tests/test_utils.py::test_prepend_scheme_if_needed[http://user:pass@example.com/path?query-http://user:pass@example.com/path?query]`
  - `tests/test_utils.py::test_prepend_scheme_if_needed[http://user@example.com/path?query-http://user@example.com/path?query]`
- Context file shown to the model: `requests/utils.py`.

Qwen3 primary direct-probe results:

| Condition | repair_after_first_failure@1 | end_to_end_resolved@2 |
|---|---:|---:|
| Frozen first patch | n/a | 0/1 |
| Frozen second attempt | 0/1 | 0/1 |
| Final-patch SFT | 0/1 | 0/1 |
| No-trace repair SFT | 0/1 | 0/1 |
| Trace repair SFT | 0/1 | 0/1 |
| Shuffled-trace repair SFT | 0/1 | 0/1 |

Observed failure mode:

- The frozen first patch omitted file headers, so it failed before tests.
- Trace repair generated a file-scoped diff, but it edited `resolve_proxies` instead of the correct `prepend_scheme_if_needed` behavior and failed to apply.

Updated direct-probe aggregate:

- Validated direct probes: 3 (`pallets__flask-5014`, `psf__requests-5414`, `psf__requests-6028`).
- Frozen second-attempt repair rate: 0/3.
- Trace repair SFT repair rate: 0/3.
- Direct non-Docker real gain for trace over frozen second attempt: `0.0`.
- Using the v2 IID synthetic gain of `0.9`, the direct-probe transfer ratio is `0.0`.
- Official Docker SWE-bench execution remains blocked by `unshare: operation not permitted`; the direct probes are negative supporting evidence, not an official SWE-bench score.
