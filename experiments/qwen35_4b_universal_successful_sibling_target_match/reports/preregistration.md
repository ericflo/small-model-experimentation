# Preregistration

## Question and causal contrast

Can complete, short, verifier-correct trajectories already sampled by the deployed parent install transferable decisions on tasks where its greedy policy fails? The sole causal contrast is 52 policy-supported successful-sibling rows versus disjoint replay rows, added to the same aligned replay core under exact exposure and training geometry.

The closest result selected the same kind of fresh failures but substituted hand-authored oracle restarts. It improved closure while losing semantic competence. This successor changes teacher support, not the local gate or the standard of evidence.

## Fixed identity

- Model: only `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: `qwen35_4b_universal_on_policy_prefix_repair_token_match` arm `replay_after_close`, deployed only as its authenticated explicit merged composite.
- Parent merged weight SHA-256: `7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e`.
- Runtime vLLM LoRA is forbidden.
- Seeds: construction 77,115; greedy 66,115; sibling sampling 66,116; selection 55,115; training 49; fresh local 88,011; conditional aggregate 78,141.

## Fresh procedural substrate

Construction seed 77,115 creates 624 executable truth-audited tasks, exactly 48 for each of `induct`, `execute`, `select`, `trace`, `verify`, `count`, `repair`, `optimize`, `abstain`, `state`, `order`, `probe`, and `route`. The public runner input contains only `id`, user messages, and public metadata; it excludes answers, reference thoughts, audits, and truth flags.

Canonical-message freshness must be zero against the two closest predecessor collection/local sources and generated local seeds 88,000–88,011. A failure stops before model load.

## Event 1: greedy failure collection

Run the authenticated parent once on all 624 tasks through the pinned experiment-local vLLM runner: natural thinking, greedy `n=1`, seed 66,115, 1,024 generation tokens, model length 4,096, and fixed engine geometry. Hard failure means any cap contact, missing answer, or wrong exact answer. Correct but verbose rows are not failures and cannot enter sibling sampling.

After the event is published green, a model-free stage freezes every hard failure. At least four hard failures must exist in every skill; otherwise the experiment closes `STOP_INSUFFICIENT_GREEDY_FAILURES`. The derived sibling input contains the original messages and public failure reasons but no answer or other oracle field.

## Event 2: sibling collection

Only the separately committed hard-failure prompts are sampled. The same parent, runner, model length, generation cap, and engine geometry are used with natural thinking, `n=16`, seed 66,116, temperature 0.6, top-p 0.95, and top-k 20. No additional or adaptive sampling event is allowed in this result directory.

A sibling qualifies only if all conditions hold:

- it comes from this exact event and parent;
- thinking naturally closes exactly once;
- generation stops naturally without truncation;
- its post-close tail is exactly the executable answer plus model EOS;
- its exact answer matches procedural truth;
- thinking uses at most 768 tokens.

For each task, select the qualified sibling with the fewest sampled tokens, then fewest thinking tokens, then a seeded hash tie-break. Within each skill, select the four shortest task winners with a seeded hash tie-break. Fewer than four qualified tasks in any skill closes `STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS`. There is no quota borrowing, threshold change, oracle-trace substitution, hand repair, or fallback to a greedy-correct task.

## Conditional exact-exposure training freeze

Only a 52-row balanced sibling source may open stream construction. Copy the same self-contained 2,240-row replay pool and 200-row aligned replay core used by the terminal predecessor. Prospectively target:

- 320 rows per arm;
- 200 byte-identical position-aligned replay rows;
- candidate variable block: 52 sibling trajectories plus 68 replay fillers;
- control variable block: 120 disjoint replay rows;
- exact equality on total forward tokens, loss-bearing target tokens, and absolute loss mass;
- zero tokenizer skips;
- one epoch, batch size one, gradient accumulation eight, 40 optimizer updates, LR `1e-5`, thought/close weights `0.2/0.2`, training seed 49;
- independent warm starts from the same published parent.

An exact solver failure is terminal. Do not pad, duplicate, truncate, mask, rewrite, or reweight sampled trajectories after seeing feasibility. A second adversarial compute review must be committed and both workflows green before replay-control training.

## Fresh local promotion

Materialize seed 88,011 only after paired training. Run parent, replay, and candidate through identical explicit-composite vLLM geometry on 26 new tasks, two per skill. Preserve the predecessor’s strict gate:

- candidate correct at least 17/26;
- parsed at least 24/26;
- cap contacts at most two;
- feasible-route abstentions at most one;
- `execute`, `induct`, and `probe` each correct at least one of two;
- candidate total correct strictly greater than parent and replay;
- candidate execute+induct+probe correct strictly greater than parent and replay.

No tie promotes and no threshold changes after output.

## Conditional held-out transfer and sample-more debt

Only a local pass may open one aggregate-only quick event at seed 78,141 through the trusted same-backend gateway. Candidate must strictly improve aggregate and every reported public family score versus the deployed parent, and must strictly beat active replay aggregate. A pass is a pilot, not a universal claim.

Before any universal-installation claim, a fresh result-separated confirmation must use independent quick seeds, the higher tier, paired uncertainty, and a matched-compute same-backend sample-more comparator. The sample-more comparison cannot be waived or inferred transitively from this collection event.

## Mandatory checkpoint order

1. Publish design on `main`; run full checks, rebase, push, and verify Validate Repository and Publish Research Site green.
2. Run only `collect-greedy`; preserve and publish all output.
3. Run only model-free `prepare-siblings`; preserve a pass or terminal stop and publish it.
4. Run only `collect-siblings`; preserve and publish all output.
5. Run only model-free `select-siblings`; preserve a pass or terminal stop and publish it.
6. Give every later stream, training, merge, local, and conditional aggregate stage its own checked, rebased, pushed, two-workflow-green checkpoint.

Every model or training event requires clean synchronized `main` and a committed prerequisite receipt. Rebase conflicts are resolved by regenerating derived artifacts from the combined source tree, never by choosing a stale generated side.
