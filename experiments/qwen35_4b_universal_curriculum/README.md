# Qwen3.5-4B: Installing Universal Features via Designed Synthetic Curricula

**Status:** in-progress · since 2026-07-13 · first transfer measurement (base vs. `universal1` on the held-out menagerie) is running; Results/Interpretation land when it returns.

This is the working experiment for the doctrine in `docs/installing_universal_features.md`
(at the repo root). The bet: we can *install* a generic capability in the fixed 4B by
**designing and synthesizing** training text (not harvesting the model's own outputs,
not narrating oracle answers), teaching an abstract circuit on surfaces **deliberately
disjoint from the eval**, and proving the feature is *universal* by **transfer** to a
held-out benchmark we never train on or read.

See [the doctrine](../../docs/installing_universal_features.md) for the full write-up.

## Research Program

- Program: `posttraining_and_adaptation` (capability installation), with
  `structured_execution_and_compilers` and `test_time_reasoning_budget` as the
  circuits under test.
- Program question: can supervised curriculum design install a *generic* reasoning
  circuit in the fixed 4B — one that lifts a held-out benchmark it was never shaped
  to — rather than only memorizing the eval's surface?
- Prior anchors: **C11** (harvest is coverage-bounded), **C39** (the 4B is an
  executor/retriever of pretrained structure, not an inducer), **C56** (oracle-answer
  SFT trains to 1.0 but deploys near 0 on composed induction; *exploration* — a real
  executable procedure — did install and transfer), **C59** (serial compute crosses
  the induction wall only via reasoning *content*, so *designed content* is the
  lever). The retracted "+0.32 capacity boundary" (C54–C56, softened) is the negative
  result this program is built to overturn: those walls were declared from slow,
  sparse curriculum search, not a proof of impossibility.

## Question

Does a generic, multi-skill, surface-agnostic curriculum — trained on abstract
surfaces the menagerie does not use — lift the **held-out menagerie** aggregate,
demonstrating that the installed circuit binds to *structure*, not tokens?

## Hypothesis

If the circuits we teach (compose/decompose, execute-a-stated-procedure,
check-constraints) are genuinely abstract, then rendering them over five disjoint
surfaces (digits, letters, roman numerals, invented words, greek syllables) and
varying everything the feature must be invariant to (alphabet, length, order,
vocabulary) should install a *surface-independent* feature. A universal feature
lifts *every* menagerie axis that shares the circuit; a memorized shape would lift
none of a benchmark it never saw. Transfer is therefore the proof, and the training
data looking nothing like the eval is the point — not a handicap.

## Setup

- Model: Qwen3.5-4B only (hard rule), pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; QLoRA think-channel SFT, co-trained
  from base (no warm-start).
- Dataset/task source: `scripts/gen_curriculum.py` synthesizes
  `data/sft_universal.jsonl` (CPU, seconds, truth-blind — values computed from the
  operators we wrote, never sampled from the model). 1400 rows, three skills:
  - **INDUCT** (600): infer a hidden *composed* rule from probe examples by
    decomposition *search* — fix a candidate step-1, compute intermediates, find the
    step-2 that maps them (includes a worked dead-end). Reuses the proven
    `generic_curriculum.comp_lesson` pedagogy.
  - **EXECUTE** (400): apply a *stated* multi-step procedure to an input, carrying
    the result forward step by step.
  - **SELECT** (400): pick the item satisfying a conjunction of stated constraints,
    checking every constraint explicitly.

  The generator also ships three more generic circuits the search explores — **TRACE**
  (multi-hop pointer dereference / navigation), **VERIFY** (check a candidate result
  against a stated procedure, the meta-skill behind the P(True) judge, C46), and
  **COUNT** (tally under a predicate) — selectable via `--mix`.
- Train/eval split: train is 100% synthetic abstract-surface curriculum; eval is the
  menagerie, which shares **no** surface, family, or generator with train.
- Baseline: base Qwen3.5-4B, same eval, same seed (paired arms).
- Controls: the menagerie firewall guarantees train/eval disjointness (we never read
  family contents). Same-surface performance would only be an upper-bound control and
  is not the claim.
- Primary metric: **menagerie transfer** — aggregate score delta (adapter − base) on
  menagerie-quick at think-budget 1024 via the HF `qwen` backend (adapter applied
  directly; no 9 GB merge — vLLM runtime-LoRA is a silent no-op on this model, C49).
- Oracle-only metrics: none — the transfer metric is the deployable metric.
- Hidden-label boundary: eval scores come only from the menagerie's own verifier;
  training never sees a menagerie item.

## Run

Smoke (synthesize a tiny curriculum, CPU, seconds):

```bash
python scripts/gen_curriculum.py --n-induct 20 --n-execute 10 --n-select 10 --out /tmp/sft_universal_smoke.jsonl
```

Single config (synthesize → fast-train `universal1` → paired transfer eval):

```bash
python scripts/gen_curriculum.py                    # writes data/sft_universal.jsonl (v1 mix)
bash scripts/train_eval_chain.sh                    # train + base-vs-adapter menagerie-quick@1024
```

Search over curriculum designs (the doctrine's fast loop — the real intent):

```bash
python scripts/search.py                            # runs scripts/search_configs.json, logs the transfer leaderboard
```

`scripts/gen_curriculum.py --mix "induct=600,execute=400,trace=300,verify=300"` selects any
mix of the six generic skills (induct, execute, select, trace, verify, count); the default
`--mix` reproduces v1 byte-for-byte. `search.py` fast-trains + transfer-evals each config in
`search_configs.json` and ranks them by menagerie transfer delta.

## Results

_First transfer measurement (base vs. `universal1`, menagerie-quick@1024, seed
59001) in flight — numbers land here when the chain completes. Deployable evidence
(transfer to the held-out menagerie) is the only evidence; there is no oracle track._

## Interpretation

_Pending the first result. The design question this loop then chases: which skills /
surfaces / lesson framings carry the transfer, and what "lifts all boats the most"._

## Knowledgebase Update

- Program evidence updated: pending first result.
- Program backlog updated: pending first result.
- Claim ledger updated: pending first result (a new claim on universal-feature
  transfer if it survives confirmation at 8192 + medium).

## Artifacts

- `scripts/gen_curriculum.py` — the multi-skill synthetic curriculum generator
  (six generic skills; `--mix` selects any subset/dose).
- `scripts/search.py` + `scripts/search_configs.json` — the curriculum-design search
  driver and its starter config grid (the doctrine's fast loop).
- `scripts/train_eval_chain.sh` — train one config (`universal1`) then paired transfer eval.
- `data/sft_universal.jsonl` — the synthesized v1 curriculum (checked in).
- Transfer-result JSONs land in `../qwen35_4b_gauntlet_frontier/runs/menagerie/`
  (that experiment owns `bench.py`); the training log is written to `runs/train.log`
  at run time (self-created by the runner).
- `reports/artifact_manifest.yaml` — external/omitted artifacts + reproduction.
