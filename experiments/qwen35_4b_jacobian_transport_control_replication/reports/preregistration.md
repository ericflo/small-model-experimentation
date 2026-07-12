# Preregistration: Quantization-Aware Jacobian Transport Replication

Frozen before any model call in this experiment.

## 1. Evidence being replicated

The parent produced 48/48 direct and 48/48 mapped-consequence target outcomes
under an all-24 J clamp at layers 4–8, versus 0/48 logit lens and 0/48 random;
wrong-donor J produced its own consequence 48/48. It is not valid evidence
because one random row exceeded its realized-norm tolerance and post-bf16 random
deltas were not tightly span-orthogonal.

## 2. Frozen carryover

- Model/revision: only `Qwen/Qwen3.5-4B` at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: Transformers 5.13.0, torch 2.11.0, bf16 SDPA, batch one,
  `use_cache=False`.
- Lens: exact copied parent artifact SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Band: layers 4,5,6,7,8. Alpha: 1. No layer or scale selection.
- Prompt/task grammar and 24 one-token concepts are unchanged.

## 3. New data

- Numeric-control calibration: 24 fresh mappings, balanced one source per
  concept. Outcomes are discarded and never written.
- Untouched confirmation: 48 fresh mappings, balanced two sources per concept.
- Every eight-key table has a fresh one-to-one assignment to eight distinct
  digits; source, target, and wrong labels/digits differ.
- Mapping fingerprints must be disjoint from both parent experiments as well as
  within this experiment. Parent public data artifacts may be read only for
  fingerprints and the frozen lens; no benchmark content is involved.

## 4. Primary J intervention

At the selected-key token and each frozen band layer, set all 24 normalized J
coordinates to the clean target-donor coordinates. Desired values are fixed
clean states for that layer; repeated-patch state never defines the target.
No digit token, digit unembedding, or consequence margin gradient enters.

## 5. Quantization-aware random controls

Two independent controls `random_a` and `random_b` are generated for every item,
prompt kind, and layer.

1. Draw 32 continuous vectors from fixed seeds and project each orthogonal to
   the complete normalized J dictionary.
2. Scale each to the primary J delta norm.
3. In the live layer hook, simulate bf16 application, remove the realized
   J-span component, renormalize, and feed the correction back into the requested
   vector with damping 0.5 for at most 512 iterations.
4. For the best candidates, binary-search scale for 64 steps and choose the
   first candidate satisfying both post-bf16 constraints; ties use candidate
   index, never logits or output identity.

Every realized layer delta must satisfy:

- relative norm error <=1e-5;
- J-span projection norm / delta norm <=0.01.

The 1% projection-norm ceiling bounds J-span energy to at most 1e-4 while
remaining feasible under bf16 quantization. Same-subspace specificity is tested
separately by wrong-donor J, so the orthogonal arm is not the only causal control.

## 6. Calibration firewall

On all 24 calibration mappings, construct primary J deltas and both random arms
for direct and consequence prompts, but discard logits immediately. Write only
token/position contracts, delta norms, norm errors, projection fractions,
iterations, and chosen candidate indices. Calibration passes only if all
24×2×2×5 = 480 realized random layer deltas meet both thresholds. Otherwise
freeze `CONTROL_UNREACHABLE` and do not open confirmation.

No optimizer hyperparameter changes after calibration. The parameters above are
the only allowed configuration.

## 7. Confirmation arms

On 48 untouched mappings at the fixed band:

1. source baseline;
2. full target-donor activation clamp;
3. all-24 J target-donor clamp (primary);
4. random_a;
5. random_b;
6. all-24 J wrong-donor clamp;
7. source/target pair J clamp;
8. all-24 concept logit-lens clamp.

Every arm uses batch-one full recomputation. Direct key and mapped digit are both
scored; mapped digit is primary.

## 8. Frozen decision

Instrumentation requires one-token contracts, equal source/donor positions and
lengths, exact causal suffix invariance, calibration pass, and every confirmation
random layer satisfying both numeric thresholds.

`REPLICATED_J_TRANSPORT` requires on confirmation:

- clean source accuracy >=0.80 and parse >=0.95 for both prompt kinds;
- full donor direct target >=0.60 and consequence target >=0.50;
- J direct and consequence target shifts >=0.20 and >=0.15;
- J consequence target rate minus the larger of random_a/random_b >=0.10;
- J consequence target rate minus wrong-donor-to-target >=0.10;
- wrong donor's own digit shift >=0.10;
- consequence parse drop <=0.05;
- paired 10,000-resample lower bound for J minus each random arm >0;
- all numeric control constraints pass.

Other labels:

- `CONTROL_UNREACHABLE`: calibration cannot meet numeric constraints;
- `INVALID_CONTROL`: confirmation instrumentation or control fails;
- `NO_REPLICATION`: valid controls but causal endpoint fails;
- `DIRECT_ONLY`: direct shift passes while consequence fails.

## 9. Scope

The intervention knows the target concept. A valid result confirms an oracle
mechanism only. It licenses a new experiment on native thought states; that work
must learn a non-oracle rule and beat frozen plus matched sampling on fresh
held-out exact tasks before any capability claim.
