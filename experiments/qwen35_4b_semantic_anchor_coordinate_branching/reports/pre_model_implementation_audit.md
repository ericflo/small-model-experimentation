# Pre-Model Implementation Audit

Completed after implementation and before any model call. This audit authorizes
only outcome-blind live-bf16 model smoke and, after it passes, the full 880-row
numeric calibration. It does not authorize mechanics outcomes.

## Boundary and data firewall

1. Only `Qwen/Qwen3.5-4B` at the pinned revision can load.
2. The implementation boundary hashes runner, model operations, coordinates,
   branch geometry, task data, every test, this audit, and canonical scientific
   config semantics with self-referential boundary blocks removed.
3. Design validation checks both local and committed bytes at an ancestor commit.
4. Model stages open only `mechanics_public.jsonl`; the exact five-field schema
   excludes correctness, first operation, hidden examples, and target pipeline.
5. Public data and frozen lens hashes are part of the scientific boundary.
6. The independent `label_map` seed now controls only diagnostic labels; a test
   proves changing it cannot change task behaviors, aliases, or pipelines.
7. All 76 task behaviors remain unique and disjoint from 1,046 ancestors.
8. Benchmark contents are never read or imported.

## Token and prefix contracts

9. Prefix generation consumes public prompt fields only.
10. Exactly one shared 512-token prefix is generated per mechanics task.
11. Natural close, EOS, or short generation fails without reseeding/substitution.
12. Native prefix generation uses the ordinary cached path; every anchor
    capture, patch, and readout uses padding-free cache-free full recomputation.
13. Exact prefix token IDs, seed, length, and SHA-256 are written and later reused.
14. Live `<think>`/`</think>` IDs must equal configuration.
15. All 24 leading-space concept IDs must be unique and equal frozen lens IDs.
16. Anchor IDs are assembled arithmetically, never found by last occurrence.
17. Piecewise scaffold tokenization must equal whole-string tokenization and
    decode exactly.
18. Source and donor have the same position and sequence length within a probe.
19. Every anchor context has one open, one later close, and no EOS.
20. Direct and consequence suffixes are hashed in position receipts.

## Clean donors and intervention semantics

21. Every `(task, probe, alias)` clean state is captured before patching.
22. Desired layer values come only from that layer's clean donor trajectory.
23. Desired tensors and dictionaries are cloned to CPU on patcher construction.
24. Clean donor tensors are hashed before/after every patched stage.
25. Full donor overwrites the entire block output at layers 4--8.
26. Donor J replaces all 24 normalized coordinates at rtol `1e-5`.
27. Mean J averages all 12 clean donor coordinate vectors independent of target.
28. Additive J is the exact prior centered alpha-one bank with replicated RMS
    norm anchors; no donor rescaling or sweep exists.
29. Logit-lens uses all 24 frozen concept unembedding columns in frozen order.
30. Wrong donor is a bijective non-source derangement and is never target/source.
31. Every patcher has an exact-once application counter and rejects repetition.
32. Realized deltas use cloned preassignment fp32 values, fixing the prior view-
    alias receipt footgun.
33. Coordinate, additive, logit, wrong, and non-J rows require finite nonzero
    realized deltas at every layer. Full-donor later-layer zero deltas are
    permitted only because an earlier full-state overwrite can make the later
    clean state already equal; each full arm must have at least one nonzero layer
    and exact-once hooks everywhere.

## Numeric controls and receipts

34. Non-J A/B seeds include task, probe, source, target, arm, layer, and draw.
35. Both controls pair to the same sequential donor-J forward's realized norm.
36. Live bf16 repair sees geometry only and enforces <=`1e-5` norm error and
    <=`0.01` projection into the complete 24-J span.
37. Representative smoke covers both probes, both controls, all five layers,
    and every patcher type without retaining logits/probabilities.
38. Full calibration requires exactly `4*11*2*2*5 = 880` valid numeric rows.
39. Full calibration also exercises full/J/mean/additive/wrong/logit patchers,
    exact hook counts, donor immutability, suffix causality, and token positions.
40. Receipts explicitly set outcomes, correctness, hidden fields, logits,
    probabilities, and confirmation to false/unopened.
41. Writes are atomic and failed numeric controls cannot leave score artifacts.
42. Calibration task-zero exact prefix IDs must equal model-smoke IDs.

## Mechanics lock

43. Mechanics remains unavailable until a later pushed boundary hashes model-
    smoke summary/prefixes and every full-calibration summary/row/position/
    prefix/intervention artifact.
44. Mechanics reruns all 880 numeric controls and every intervention geometry;
    they must match calibration before any probability file is written.
45. Mechanics uses only public `alias -> operation -> result -> randomized label`
    expectations and records task-label fields as unloaded.
46. The pure decision evaluator checks exact 880 outcome identities, numeric and
    parse gates, wrong-donor specificity, alias/label/task breadth, and separate
    additive transport.
47. Only complete randomized consequence transport can open continuation.

## Verification receipt

- Fifteen implementation/data/decision tests pass without bytecode writes.
- Python compilation passes.
- Model-smoke invocation currently fails before model construction because the
  implementation boundary is deliberately pending.
- No GPU/model call has occurred in this experiment.

## Authorization

After corrected data/design and implementation bytes are committed, pushed, and
hash-anchored, authorize one outcome-blind model-smoke run. Do not run full
calibration unless smoke passes. Do not run mechanics unless full calibration is
separately audited, committed, pushed, and locked.
