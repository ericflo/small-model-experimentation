# Real-Repo Agentic Instrument

Build a broad, firewalled real-codebase agentic-coding instrument for Qwen3.5-4B in pi-coding-agent —
and measure the raw-base baseline this program has never had.

**Status:** in-progress · since 2026-07-25 · task generation and baseline measurement under way

## Research Program

- Program: `agentic_breadth_installation` (cognitive-core coding sub-program).
- Program question: can real-codebase agentic coding capability be INSTALLED into Qwen3.5-4B on one
  24GB GPU, and can any installed gain beat matched-compute sampling?
- Prior anchors: C63 (execution-selected best-of-N is the deployable lift: 0.606 → 0.818 with zero
  training), C64 (the 4B already does real-repo agentic coding: 0.70 single-shot / 0.91 selected on
  toolz via pi), C62 (four LoRA edits of that warm-start each REGRESSED deployment), C60 (authored
  think traces crater a near-ceiling coder; only the model's own harvested traces are retention-safe).

## Question

**Can we measure an agentic-coding improvement at all?** Everything queued behind this experiment —
harvest-and-retrain, RLVR with a world-model auxiliary loss, confidence-gated selection — is
unmeasurable on the current eval, and one required control is missing entirely.

## Hypothesis

Not a capability hypothesis. Two instrument claims, each falsifiable:

1. A **broad, firewalled** real-repo task set (many libraries, difficulty-stratified, held-out repos
   never trained on) has enough headroom and resolution to detect a ≥0.05 change in deployed solve
   rate — unlike the 11-task holdout, which sits at 0.818 selected with 11/11 tasks solvable.
2. The **raw base** measured in pi is materially below the SFT warm-start. If it is *not*, then the
   program's deployment baseline was never the warm-start's achievement and every subsequent
   attribution needs revisiting.

## Setup

- Model: `Qwen/Qwen3.5-4B` only (pinned revision), served through vLLM to pi-coding-agent.
- Task source: 24 candidate pure-Python OSS libraries (`scripts/repos.py`) cloned at recorded SHAs.
  A task = one function body replaced by `raise NotImplementedError`; the verifier is **the
  repository's own pytest** — the property that made C64 credible.
- Train/eval split: **repo-level firewall**, each repo's side fixed in `repos.py` *before* any task was
  generated or any number measured. Held-out tasks live in libraries whose code, conventions, and test
  style are absent from any training corpus. (Splitting within a repo would leak: harvesting `funcy`
  teaches its idioms, its test layout, and often the neighbouring functions its held-out functions
  call.) `split_tasks.py` asserts no repo appears on both sides.
- Baseline: raw base through pi. Controls: merged SFT warm-start; execution-selected best-of-k on both
  arms (the "sample more" bar any training must beat, per C63); a closed-book memorization probe.
- Primary metric: single-shot solve rate and execution-selected best-of-k on the held-out split.
- Oracle-only: none. Partial reward is reported but the headline is binary solve.
- Hidden-label boundary: `test`-group repos are eval-only, forever; harvesting or training on them
  invalidates the instrument.

## Scoring: per-test, not whole-suite

Each task carries the `fail_to_pass` set its stub breaks; each repo carries its baseline passing set,
whose remainder is `pass_to_pass`. Reward 1.0 = all broken tests pass **and** nothing regressed;
partial credit is the fraction of broken tests fixed, halved if anything regressed.

**Requiring a globally green suite is wrong** — it dropped 11 of 24 repos over one or two
environment-dependent failures (a package-metadata check; a test wanting >6 GB). Those tests fail
identically before and after an agent's edit, so they carry zero information about the agent, yet they
would have silently capped every episode at partial credit. Per-test sets exclude them by construction,
and `pass_to_pass` makes "edit the test instead of the code" unrewardable. Dependency *installation* is
fair game (environment); editing a repo's test code never is — a suite we repaired is no longer "the
repository's own pytest".

## Three traps this harness is built to avoid

**The editable-install trap gives free rewards.** `uv pip install -e .` + copy-per-episode is silently
broken for `src/` layouts: the editable `.pth` resolves imports to the ORIGINAL checkout, so pytest in
the episode's copy imports unstubbed source, every test passes, and every episode scores 1.0 — an
instrument reporting a perfect agent that never wrote a line. Layered mitigation: install editable once
to resolve deps then **uninstall the distribution**; point `PYTHONPATH` at the copy's own roots; and
admit a task only if stubbing it in a copy actually **breaks** tests, so an import leak yields *zero
tasks* instead of free reward.

**`-v` alone cannot read a suite that uses pytest-xdist.** With `-n auto` in a repo's own addopts the
per-test progress lines vanish and the parser sees nothing — `wcwidth` baselined at "0 passed" until
`-rA` was added. The `-rA` summary section is authoritative in every execution mode.

**Generation and evaluation must share one code path.** Both call `env_util.prepare_copy` /
`run_tests` / `score_task`. This program's three worst measurement failures were harness mismatches
(real-repo ~0.00 vs pi 0.70; synthetic 0.486 vs 0.810; seven "absent" tasks that were a dict-default
bug), so a baseline captured with a different command than the episode is scored with is the same class
of bug — which is why changing the test command requires `--rebaseline`.

## Operational contribution: the WSL crash cause, established

This experiment's first runs killed the WSL VM, which forced the diagnosis the program had been missing
through seven prior deaths (`docs/wsl_stability.md`, rewritten). Replicated signature:

```
Out of memory: Killed process (python) total-vm:32964064kB, anon-rss:15671808kB
oom-kill: ... global_oom, task_memcg=/init.scope
init.scope: Failed with result 'oom-kill'   [15.1G mem peak, 15.8G swap peak]
```

One python process reaches ~31 GB of anonymous memory (15.3 GB resident + 16.1 GB swapped), saturating
the 16 GB VM cap *and* 16 GB swap; the kernel's **global** OOM condemns `/init.scope`, which under WSL2
contains all of WSL. Not host-memory exhaustion (falsified at crash #7 with 7.3 GB host free) and not
GPU-specific (crash #8 had zero GPU work). Two mechanisms produce it, both live in this repo:
unbounded third-party test allocation, and — the general one — `subprocess.run(..., stdout=PIPE)`
buffering a child's entire output in the **parent's** RAM, which is how every driver here captured
output, including pi episode runners that stream `--mode json` events for a 600 s wall.

Verified fixes: run pipelines in a memory-limited cgroup scope (`scripts/guard.sh` — a deliberate 3 GB
runaway at `MemoryMax=1G` is killed with the VM surviving); spool child output to disk and stream it
(`env_util.run_spooled`); cap untrusted children with `ulimit -v` (confirmed honoured by dash); kill
process *groups*, since a killed pytest's children keep printing.

## Run

Smoke (two repos, a handful of candidates, no GPU):

```bash
scripts/guard.sh .venv/bin/python scripts/fetch_repos.py --only toolz,mergedeep --jobs 1
scripts/guard.sh .venv/bin/python scripts/gen_tasks.py --only toolz --jobs 2
```

Full:

```bash
scripts/guard.sh .venv/bin/python scripts/fetch_repos.py --jobs 2      # -> data/repo_*.json
scripts/guard.sh .venv/bin/python scripts/gen_tasks.py --jobs 6        # -> data/tasks_all.json
.venv/bin/python scripts/split_tasks.py                                # -> data/tasks_split.json
scripts/serve.sh Qwen/Qwen3.5-4B                                       # readiness-gated vLLM
.venv/bin/python scripts/pi_episode_repo.py --label base --split test --k 3
```

Every heavy step runs under `scripts/guard.sh`; every runner checkpoints and resumes from its own
partial output (this box has died eight times and no completed episode has ever been lost).

## Results

Pending. Report single-shot, execution-selected best-of-k, and mean partial reward per split and
stratum, with the raw-base vs warm-start contrast as the headline.

## Interpretation

Pending.

## Knowledgebase Update

- Program evidence updated: pending
- Program backlog updated: pending
- Claim ledger updated: pending (an instrument cell may close without a capability claim; the
  raw-base-vs-warm-start contrast is claim-bearing if it lands)

## Artifacts

- `scripts/` — repo registry, fetch/baseline, task generation, splits, serving, pi episode runner
- `data/` — `repo_manifest.json`, `repo_baselines.json`, `tasks_all.json`, `tasks_split.json`
  (small and committed: this is the reproduction path)
- `large_artifacts/qwen35_4b_realrepo_agentic_instrument/` — episode records, pi trajectories
  (retained deliberately as harvest substrate for the Line-1 successor), serve/run logs
- `reports/artifact_manifest.yaml`
