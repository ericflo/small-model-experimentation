# Experiment Log

## 2026-06-26

- Created standalone verified algorithm retrieval + adaptation experiment package.
- Primary question: can retrieval of verified training algorithms plus Qwen adaptation recover held-out tasks missed by direct sampling?
- Primary metric: zero-to-one lift on tasks where direct K=8 sampling has no hidden-correct candidate.
- Planned arms:
  - direct base sampling baseline;
  - literal retrieved-code copy/rename baseline;
  - semantic retrieval adaptation;
  - random retrieval adaptation control;
  - shuffled retrieval adaptation control;
  - oracle over generated retrieval candidates as headroom.
- Guardrails:
  - hidden tests are evaluation only;
  - public tests are the only deployable evidence given to adaptation prompts;
  - random and shuffled retrieval controls must underperform semantic retrieval for a positive read.

### Setup

- Added local direct-sampling baseline pool: `data/base_sample_more_k8_records.jsonl`.
- Built verified algorithm library from MBPP train references:
  - loaded 374 train records;
  - kept 364 verified algorithms;
  - dropped 10 unverified references.
- Built retrieval plan for the eight direct-sampling misses:
  - tasks 15, 16, 20, 21, 24, 25, 26, 31;
  - top-k semantic retrieval = 3;
  - random and shuffled controls matched at top-k = 3.

### Iteration Notes

- First ran `copy_semantic`, a no-model literal copy/rename baseline.
- The first copy/rename attempt exposed a regex escaping bug in `copy_rename_code`; patched and reran.
- Then ran three Qwen adaptation arms in one model-load session:
  - semantic retrieval adaptation;
  - random retrieval adaptation;
  - shuffled retrieval adaptation.

### Results

| arm | retrieval coverage | zero-to-one | visible-pass hidden-wrong | parse/task | tokens |
|---|---:|---:|---:|---:|---:|
| retrieval_copy_rename_top3 | 12.5% | 1/8 | 2/3 | 3.00 | 0 |
| retrieval_adapt_semantic_top3 | 37.5% | 3/8 | 4/7 | 2.62 | 7699 |
| retrieval_adapt_random_top3 | 0.0% | 0/8 | 5/5 | 2.75 | 7982 |
| retrieval_adapt_shuffled_top3 | 12.5% | 1/8 | 6/7 | 2.88 | 7881 |

Combined with direct sample-more:

| arm | combined coverage | recovered tasks | forward tokens |
|---|---:|---|---:|
| base_sample_more | 66.7% | [] | 45406 |
| base_plus_retrieval_copy_rename_top3 | 70.8% | [20] | 45406 |
| base_plus_retrieval_adapt_semantic_top3 | 79.2% | [15, 20, 25] | 53105 |
| base_plus_retrieval_adapt_random_top3 | 66.7% | [] | 53388 |
| base_plus_retrieval_adapt_shuffled_top3 | 70.8% | [20] | 53287 |

### Readout

- Semantic retrieval adaptation passes the pilot gate: 3/8 direct misses recovered versus 0/8 random and 1/8 shuffled.
- Task 20 is not strong semantic evidence because copy/rename and shuffled retrieval also recovered it.
- Tasks 15 and 25 are the semantic-specific lift.
- Main failure mode: high visible-pass hidden-wrong rate, so retrieval adaptation needs a verifier/reranker or counterexample generation before commit.
