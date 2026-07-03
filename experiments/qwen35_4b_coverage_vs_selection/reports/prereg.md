# Pre-registration: Anatomy of the generation wall — coverage vs. selection

Logged 2026-07-03, before any data. C13/C16 established that the fixed 4B's compositional wall is
**generation** (identifying a novel composition), not execution (the model is a universal compiler). But
"generation" conflates two failures: the correct program is **never proposed** (COVERAGE deficit) vs.
**proposed-but-not-selected** (SELECTION deficit). This decomposition decides which lever can cross the
wall — and whether *cleverer access* (the mission) can beat *sample more* (raw compute).

## Setup

Fresh, verified-depth, collapse-rejected tasks (families `list` and `register` from the crossfamily
harness), depths 1–4, n=20/depth/family, new seed (held out from all prior runs). For each task, draw
**K=32** bare-identification samples (I/O→`transform`, thinking on, budget 512, repo-standard sampling
T=0.6/top_p=0.95 — i.e. the same distribution "sample more" uses). Each sample is executed against the
**visible** examples (8) and the **hidden** examples (8) and recorded as (passes_visible, passes_hidden).

Selectors (none may see hidden labels):
- **first@1** — first sample, no selection (deployable single-shot baseline).
- **coverage@k** — oracle ceiling: any of the first k samples passes hidden (unbiased estimator). This IS
  sample-more's success rate at compute k.
- **vfilter@k** — keep samples passing ALL visible examples; pick by majority output-behavior (ties→first);
  grade hidden. "Sample + execute-filter": tool-based, no hidden labels.
- **vfilter-oracle@k** — among visible-passers, credit if ANY is hidden-correct (upper bound of any
  selection over the execution-consistent set).
- **mverify@k** — among visible-passers, rank by the model's OWN verifier (C10-style: given I/O + candidate,
  judge whether it is the intended rule that generalizes), pick top-1; grade hidden. The pure-elicitation,
  tool-free-of-hidden selector.

## Predictions (locked)

- **P1 (coverage rises above single-shot):** for list, coverage@32 ≥ 3× first@1 at depth 2, and
  coverage@32 > 0.10 at depth 3 (the right program IS drawn given enough samples even where single-shot ≈ 0).
- **P2 (a coverage wall exists at some depth):** there is a depth d\* where coverage@32 itself falls < 0.10
  for list — beyond d\*, sample-more is futile (true coverage deficit). Prediction: d\* = 4 for list
  (coverage@32 still > 0.10 at depth 3, < 0.10 at depth 4).
- **P3 (a selection gap exists):** at the depths where coverage@32 > 0.15, vfilter < coverage@32
  (execution-consistent candidates diverge on hidden; majority-behavior filtering leaves capability on the
  table — since vfilter-oracle over visible-passers equals coverage@32, the gap coverage−vfilter is the
  selection loss).
- **P4 (verifier elicits selection):** mverify > (random-among-visible-passers baseline) at depth 2–3 —
  the model's verifier picks hidden-correct candidates above chance among execution-consistent ones. STRONG
  form: mverify ≥ vfilter (the tool-free verifier matches or beats majority execution-filter).

## Decision mapping (the deliverable is a per-depth MAP of the wall)

- **SELECTION-BOUND** at depths where coverage@64 ≫ first@1 and a deployable selector (vfilter/mverify)
  recovers a large share of coverage ⇒ the lever is a better selector; *verify≫generate pays off; a selector
  beats sample-more-at-k=1* (on-mission win). Report how much of the oracle ceiling each selector recovers.
- **COVERAGE-BOUND** at depths where coverage@64 ≈ 0 ⇒ the program is never proposed; sample-more is futile;
  only tool-enumeration (C12) or banking new capability (C11) crosses. Honest confirmation of the
  sample-more ceiling for that regime.
- The **crossover depth** d\* (selection-bound → coverage-bound) is the key result: it tells a deployment
  system exactly when to switch from "sample + select" to "enumerate with tools."

## Controls / honesty

- Verified-depth tasks (nominal = real depth) so coverage isn't inflated by shallow-equivalents.
- coverage@k via the unbiased 1−C(n−c,k)/C(n,k) estimator, not a point count.
- mverify is scored only over visible-passers and compared to the *random-among-visible-passers* null and to
  vfilter-oracle (the achievable ceiling), so a "good verifier" claim is relative to what selection could
  possibly achieve, not to chance overall.
- register (nonzero deep-identification floor per C16) vs list (floors to 0) contrasts whether that floor is
  coverage or selection.
