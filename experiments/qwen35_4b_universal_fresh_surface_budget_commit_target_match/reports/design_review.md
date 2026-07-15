# Adversarial Design Review

Five independent adversarial reviewers examined the frozen model-free construction
before any GPU event: contamination/firewall/freshness, statistical design,
exposure-matching code, lifecycle/receipt code, and a truth/shortcut audit of the
treatment corpora. Every mechanical claim below was verified by rerunning code, not
by reading it.

## Verdicts by lens

- Contamination/firewall/freshness: PASS. Ban-list coverage re-scanned directly on
  the frozen corpora (zero hits); fresh pools disjoint from the six originals and
  the nonce pattern; the 104 gate messages have zero canonical overlap against all
  three streams, both corpora, the replay pool, 13 regenerated prior local seeds,
  and — beyond the receipt — every data jsonl in the universal line; seeds
  77116/55117/51/88013/78143 appear nowhere else; no script touches `benchmarks/`
  content; the gate input is oracle-free.
- Truth/shortcut audit: PASS. A fully independent re-verifier re-derived all 320
  supervised answers from prompt text alone (including a complete op-pair
  identifiability search for every induction row): 320/320 correct. No prompt-side
  budget shortcut exceeds 0.75 (majority 0.675); every exhaustion row's decoy is
  mechanically present; narration never contradicts the answer; sentinels BUDGET
  and INSUFFICIENT are disjoint and contract-bound.
- Exposure matching: PASS with one MAJOR (fixed). Block sums, disjointness,
  1,280-line position alignment, zero skips, and byte-identical MILP re-solve were
  all independently reconfirmed from raw indices.
- Statistical design: FAIL → remediated. One BLOCKER and two MAJORs, below.
- Lifecycle/receipts: FAIL → remediated. The same BLOCKER and four MAJORs, below.

## Blockers and majors, with remediations (all applied before this freeze)

1. BLOCKER — `run.py --stage benchmark` invoked `run_benchmark.py` with only
   `--seed`, which argparse rejects; the stage could never execute and `run.py` is
   hash-pinned by the local design receipt. FIX: the stage now passes the full
   frozen argument set (name/tier/think-budget/candidate/four `--model` specs) with
   the candidate taken from the promotion receipt.
2. MAJOR — `run_benchmark.py` had no git preflight and did not require the
   promotion receipt committed. FIX: the event now requires a clean pushed `main`
   and the promotion receipt byte-identical at HEAD.
3. MAJOR — the benchmark authenticated only `merge_receipt.json` self-hashes, never
   weights. FIX: every arm's `model.safetensors` is now recomputed and bound —
   external arms against frozen pins (base `b654e033...`, parent `7ab4c419...`),
   this experiment's arms against their committed merge receipts; observed hashes
   are recorded in the event summary.
4. MAJOR — the recovery promotion-writer (`check_local.py --out`) emitted a receipt
   missing fields the benchmark hard-requires, permanently sealing a legitimate
   promotion. FIX: both writers now emit the identical schema.
5. MAJOR — `PUBLISHED_ARM_HASHES` pins were fail-open (silently skipped when
   unfilled). FIX: an unfilled pin now aborts whenever the receipt is relied on as
   a committed prerequisite.
6. MAJOR — the trainer/encoder bytes were not bound at train time although
   pin-filling commits occur between arm stages. FIX: `train_trial.py` now requires
   `sha256(train_think.py)` to equal the exposure receipt's `encoder_sha256`.
7. MAJOR — the preregistration overclaimed the banned-vocabulary scope and did not
   state the near-zero power of the quick-tier every-family gate or the composite
   promotion's relationship to arm D's headline question. FIX: the preregistration
   now states the audit's exact scope and carries frozen prospective power
   statements fixing how each negative outcome may be described.

## Accepted risks (noted, prospectively bounded)

- Quick-tier every-family-vs-base strictness is one-sided conservative; a pass is
  extremely strong, a fail is expected even under the hypothesis (now frozen in the
  preregistration's interpretation limits).
- The tiebreak can in principle promote an arm worse on parsed/abstention counts
  inside the bars; prespecified and low probability.
- `optimizer_steps` is asserted by construction; rows and skips are log-parsed.
- CI "green" remains operator-verified, as in every predecessor.
- The exhaustion decoy always sits at allowance+1, so "termination installed" is
  bounded to contract-following behavior within this distribution.
- Budget exhaust rows have systematically longer thinks (narrating all checks); a
  target-side length cue, semantically faithful, cannot affect grading.
- vLLM Mamba-cache auto re-exec could waste the unrepeatable local event;
  mitigated operationally by keeping the GPU single-tenant during the event.

**Verdict:** `PASS_EXPENSIVE_RUN`.
