# Real-Repo Agentic Instrument Report

## Summary

Built a measurable replacement for this program's saturated eval: **200 execution-verified
stub-a-function tasks across 15 real OSS Python libraries**, 138 train / 62 held-out, firewalled by
repo, scored per-test against each task's `fail_to_pass` set with the repo's baseline as a regression
guard. The verifier is the repository's own pytest.

The raw-base pi baseline — a control this program has never had — is measuring now.

## Research Program Fit

`agentic_breadth_installation` closed its prior cell with a deployment result and an instrument
problem. Execution-selected best-of-N reaches 0.818 on the 11-task synthetic holdout (C63) and 0.909 on
toolz (C64), with **11/11 holdout tasks solvable**, so per-run noise is the size of the effects now
being chased. Four LoRA edits of the warm-start each regressed deployment (C62). Nothing queued behind
that — harvest-and-retrain, RLVR with a world-model auxiliary loss, confidence-gated selection — is
measurable on the old eval, and "the warm-start improved pi deployment" was inferred from a TRL-env
engagement contrast rather than measured against the base.

## Method

Each candidate repo is cloned at a recorded SHA, given its own test venv, and **baselined per test** in
a throwaway copy. A task replaces one function body with `raise NotImplementedError`; it is admitted
only if stubbing it in a copy makes tests that pass in the baseline **fail**. Episodes run through
pi-coding-agent with cwd set to a fresh copy, scored by `env_util.score_task`.

Reward: 1.0 when every `fail_to_pass` test passes and nothing in `pass_to_pass` regressed; partial
credit is the fraction of broken tests fixed, halved if anything regressed; 0.0 if the stub is still
present (engagement gate).

Firewall: by REPO, with each repo's side fixed in `repos.py` before any task was generated or any
number measured. `split_tasks.py` asserts no repo appears on both sides.

## Results

| | count |
|---|---|
| candidate libraries | 24 |
| usable after baselining | 18 |
| libraries yielding tasks | 15 |
| **train tasks** | **138** (9 repos) |
| **held-out tasks** | **62** (6 repos) |
| candidates validated | 310 (241 admitted) |

Held-out strata: 23 small / 30 medium / 9 large by body size; median 9 body lines. Reject reasons:
25 not test-covered, 21 suite unrunnable, 23 stub failed.

End-to-end validation on two held-out tasks with the **raw base**: `bidict.popitem` solved first try
(reward 1.0); `BidictBase.equals_order_sensitive` reached 25/27 target tests with zero regressions
before timing out (reward 0.563, `pi_exit 124`). That second episode reproduces the C62/C63 termination
failure mode independently on a new substrate — the model edits, passes most tests, and fails to finish.

## Controls

- **Two-side task validation**: the suite passes in an untouched copy, and stubbing the target breaks
  specific named tests. This is also the detector for import leakage — a leaking repo yields *zero*
  tasks rather than free rewards.
- **`pass_to_pass` regression guard** makes "edit the test instead of the code" unrewardable.
- **Repo-level firewall**, pre-registered in `repos.py`.
- **Closed-book memorization probe** (planned, not yet run): can the model emit the function from name
  and docstring alone with no repo access? These are public libraries, so a solve rate on that probe
  bounds how much of any score is recall rather than agentic capability.

## Oracle Versus Deployable Evidence

Every number is deployable-side: episodes run in the real agent scaffold, and rewards come from the
repository's own tests. There are no oracle arms. The one caveat to state plainly is contamination —
these are public repos, so absolute solve rates may include memorised implementations; the firewall
protects *training* claims, and the memorization probe is what would bound the absolute numbers.

## Interpretation

Three design decisions were each load-bearing, and the naive alternative would have produced confident
nonsense in a different direction:

1. **Whole-suite-green is the wrong admission gate.** It discarded 11 of 24 libraries over one or two
   environment-dependent failures whose outcome is identical before and after an agent's edit — while
   silently capping every episode at partial credit. Per-test sets exclude them by construction.
2. **The editable-install trap gives free rewards.** For `src/` layouts an editable install resolves
   imports to the *original* checkout, so the stub has no effect and every episode scores 1.0. Guarded
   by uninstalling the distribution, pointing `PYTHONPATH` at the copy, and the stub-breaks-tests gate.
3. **`-v` cannot read a pytest-xdist suite.** `wcwidth` baselined at "0 passed" until `-rA` was added;
   that fix recovered 8 libraries.

Generation and evaluation share one code path, because this program's three worst failures were harness
mismatches — which is why changing the test command requires `--rebaseline`.

## Next Experiments

1. Finish the raw-base held-out baseline (in flight), then the merged SFT warm-start on the same tasks:
   the first direct measurement of what that fine-tune bought in pi.
2. Execution-selected best-of-k on both arms — the "sample more" bar any training must beat (C63).
3. The closed-book memorization probe, to bound contamination in the absolute numbers.
4. Line 1 (successor cell): harvest execution-verified trajectories from the 138 train tasks, train ONE
   warm-start from base rather than editing the existing policy (0/4 base rate, C62), then GRPO with an
   ECHO-style auxiliary loss on environment-observation tokens.

## Artifact Manifest

`data/` holds the reproduction path and is committed: `repo_manifest.json` (resolved SHAs, install
strategy, dependency repairs, baseline counts), `repo_baselines.json` (exact passing/failing node ids
per repo), `tasks_all.json`, `tasks_split.json`. Repo checkouts, per-episode records and pi trajectories
live under `large_artifacts/` — see `artifact_manifest.yaml`.
