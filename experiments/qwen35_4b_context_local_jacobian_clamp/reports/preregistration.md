# Preregistration: Context-Local Jacobian Clamp

Frozen before any result-bearing model call.

## 1. Scientific question

The parent experiment showed late next-token writing without mapped-consequence
transport. This experiment tests the strongest mechanism-preserving correction:
edit a concept at the token position where it is explicitly represented, keep
that coordinate fixed to a counterfactual donor trajectory across several
layers, and measure a later consequence whose token identity never enters the
intervention.

## 2. Model and backend

- Only `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Transformers 5.13.0, torch 2.11.0, bf16, SDPA, CUDA, `use_cache=False`.
- Exact intervention calls use batch size one. A pre-result equivalence check
  compares batch-one and equal-length batched clean logits; intervention results
  remain batch-one regardless.
- No vLLM/HF arm mixing and no benchmark data.

## 3. Fresh data and frozen splits

Use 24 concepts already tokenizer-audited as one leading-space token and digits
0–9 audited as one bare token after a prefix-owned space. Each item samples eight
concepts and a one-to-one assignment to eight distinct digits. Source, target,
and wrong concepts and digits are distinct.

- lens fit: 48 prompts, disjoint from all evaluation mappings;
- full-donor band selection: 24 items;
- untouched confirmation: 48 items.

The split seed, mapping rows, prompt strings, token IDs, selected-token indices,
and SHA-256 receipts are written by CPU smoke. Lens and evaluation prompts share
the task grammar but not table assignments or source/target tuples.

## 4. Prompt pair and causal site

Every item has a shared prefix ending in one selected concept:

```text
Lookup table:
... eight concept = digit rows ...
Selected key: <source>
```

Two suffixes follow:

- direct: `Repeat the selected key exactly. Key:`
- consequence: `Return its one-digit table value. Value: `

The patched position is the final occurrence of the source token in the shared
prefix. Source and counterfactual target prompts must have equal token length and
the same selected-token index. Direct and consequence clean activations at that
position must agree to numerical tolerance because the architecture is causal.

## 5. Targeted context-local J dictionary

For every concept token `c`, fit at every source layer 4–28 the average pullback

`d[l,c] = mean_prompt d logit(c at direct report) / d h[l, selected_position]`.

The 48 fit prompts are expanded in concept microbatches. Each normalized
direction matrix is `V_l in R^(d_model x 24)`. Coordinate reads use a fixed SVD
pseudoinverse with relative cutoff `1e-5`. Record singular values, effective
rank, and condition number. The lens is eligible only if every layer is finite,
nonzero, and has full effective rank 24. No consequence digit or digit logit
is used in fitting.

## 6. Clean donor states and interventions

For each evaluation item, run clean source, target-donor, and wrong-donor prompts
and capture `h[l,p]` at every fitted layer. A primary J clamp at layer `l` sets
all 24 source coordinates to the fixed clean target-donor coordinates:

`h' = h + (c_target_donor - c_current) V_l^T`, with alpha exactly 1.

At every subsequent layer in the selected band the desired coordinates are the
clean donor values for that layer, not coordinates produced by an earlier patch.
This is a set-to-donor trajectory clamp, not repeated swapping.

## 7. Band selection without J peeking

Candidate five-layer bands are `[4..8]`, `[8..12]`, `[12..16]`, `[16..20]`,
`[20..24]`, and `[24..28]`. On the 24-item selection split only, replace the
entire selected-position activation with the clean target-donor activation at
every layer in each band. Select the earliest band with:

- clean source accuracy at least 0.80 on direct and consequence prompts;
- full-donor direct target rate at least 0.60;
- full-donor consequence target rate at least 0.50;
- full-donor consequence target rate at least 0.30 above a wrong-donor-to-target
  rate;
- parse rate at least 0.95.

If none passes, freeze `NO_CAUSAL_SITE` and stop. J outcomes are not computed on
the selection split and cannot select the band.

## 8. Untouched confirmation arms

At the selected band, evaluate each confirmation item in both direct and
consequence prompts under:

1. clean source baseline;
2. full target-donor activation clamp (positive control);
3. all-24 J coordinate clamp (primary);
4. random vectors orthogonal to the entire J dictionary, matched to the primary
   J delta norm separately for each item and layer;
5. wrong-donor all-24 J clamp;
6. source/target two-coordinate J clamp;
7. all-24 logit-lens coordinate clamp.

The random vectors use a fixed seed. Their realized norm must match the primary
J delta within relative error `1e-5` at every item/layer. A failed match
invalidates the affected comparison rather than being silently tolerated.

## 9. Frozen G0 decision

Clean confirmation requires source accuracy >=0.80 and parse rate >=0.95 for
both prompt types. The full-donor positive control must reproduce direct target
rate >=0.60 and consequence target rate >=0.50.

Primary J transport passes only if all hold on the 48 untouched items:

- direct target shift over baseline >=0.20;
- consequence target shift over baseline >=0.15;
- consequence J target rate minus norm-matched random >=0.10;
- consequence J target rate minus wrong-donor-to-target rate >=0.10;
- wrong-donor's own mapped digit rises by >=0.10 over baseline, establishing
  donor specificity rather than generic damage;
- parse-rate drop <=0.05;
- paired bootstrap 95% lower bound for J-minus-random consequence success >0,
  10,000 resamples with seed 2026071216.

Labels:

- `NO_CAUSAL_SITE`: full activation donor fails selection;
- `DONOR_ONLY`: full donor passes but primary J transport fails;
- `DIRECT_ONLY`: primary J changes the key report but fails consequence;
- `J_TRANSPORT`: every primary consequence criterion passes;
- `INVALID_CONTROL`: tokenizer, causal invariance, rank, or norm-match contract
  fails.

No threshold, band, arm hierarchy, or label changes after result inspection.

## 10. Diagnostic-only context-local gradients

After the band and confirmation results are frozen, compute on a fixed 12-item
slice the gradients of direct and consequence target-minus-source margins with
respect to the selected-position activation. Report cosine alignment with the
actual J delta and first-order predicted versus observed margin change. These
gradients cannot select a layer/band, choose an item, scale a patch, or enter the
primary intervention. They are oracle diagnostics, not capability evidence.

## 11. Scope and continuation rule

Every counterfactual intervention knows the target concept and is oracle-only.
Even `J_TRANSPORT` establishes a causal mechanism, not deployable improvement.
Only that label licenses a separate experiment on native `<think>...</think>`
states. That later experiment must learn a non-oracle state/action rule and beat
the frozen model plus matched-compute sampling on fresh held-out tasks.
