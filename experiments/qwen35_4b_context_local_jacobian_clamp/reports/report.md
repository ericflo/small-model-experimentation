# Context-local J clamping shows perfect semantic transport, but the exact control gate invalidates the result

## Verdict

**Frozen terminal label: `INVALID_CONTROL`.** The all-24 targeted J clamp at the
earlier selected-key token redirected both a direct key report and a separately
computed arbitrary table consequence on 48/48 untouched items. Every specificity
control pointed the same way. However, one of 96 random-control rows missed the
registered realized perturbation-norm tolerance: 1.155e-5 relative error versus
the 1e-5 maximum. The experiment therefore does not receive `J_TRANSPORT`.

This is not a null. It is strong provisional evidence that the corrected early,
context-local coordinates behave differently from the parent's late
answer-position token motor. It requires an independent fresh replication with
a quantization-aware control that is both norm-matched and orthogonal after bf16
application.

## Design

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: Transformers 5.13.0, torch 2.11.0, bf16 SDPA,
  `use_cache=False`, unpadded batch one for every scientific call.
- Data: 48 lens prompts, 24 band-selection mappings, and 48 untouched
  confirmation mappings; every eight-key table used a fresh one-to-one random
  assignment to distinct digits.
- Lens: 24 direct-concept logit pullbacks from the future report to the earlier
  `Selected key:` token, fitted at layers 4–28. Every layer had effective rank
  24; condition numbers were 3.28–6.43.
- Intervention: at every layer of a fixed five-layer band, set clean source
  coordinates to clean counterfactual donor coordinates. Alpha was fixed at one.
- Answer firewall: no digit covector, digit unembedding, or consequence gradient
  constructed or selected the intervention.
- Oracle scope: target identity and donor states were supplied. This is a causal
  mechanism study, not deployable capability evidence.

The immutable design was pushed at commit
`c1f06c035404bde62303439daa66dba3c1f026f9`; the adversarial review preceded all
scientific calls.

## Full-activation donor site gate

Clean selection accuracy was 24/24 for both direct and consequence prompts.
Full-state target donors and wrong donors produced a sharp layer profile:

| band | target donor: direct target | target donor: consequence target | wrong donor: own consequence |
| --- | ---: | ---: | ---: |
| 4–8 | 24/24 | 24/24 | 24/24 |
| 8–12 | 24/24 | 24/24 | 24/24 |
| 12–16 | 24/24 | 24/24 | 24/24 |
| 16–20 | 24/24 | 24/24 | 24/24 |
| 20–24 | 1/24 | 0/24 | 0/24 |
| 24–28 | 0/24 | 0/24 | 0/24 |

The frozen earliest-passing rule selected band 4–8. This establishes that the
selected token's early trajectory is a causally sufficient state site and that
late replacement arrives after the downstream computation has consumed it.

## Untouched confirmation

All rates are on 48 fresh mappings at band 4–8:

| condition | direct target | mapped target digit | donor's own mapped digit | parse |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0/48 | 0/48 | 0/48 | 48/48 |
| full target donor | 48/48 | 48/48 | — | 48/48 |
| all-24 J clamp | **48/48** | **48/48** | — | 48/48 |
| source/target pair J clamp | 48/48 | 47/48 | — | 48/48 |
| wrong-donor all-24 J | 0/48 target | 0/48 target | **48/48 wrong** | 48/48 |
| concept logit-lens clamp | 0/48 | 0/48 | 0/48 | 48/48 |
| norm-matched span-orthogonal random | 0/48 | 0/48 | 0/48 | 48/48 |

The result is not a near-argmax artifact. Mean target-minus-source margin moved:

| condition | direct margin | consequence margin |
| --- | ---: | ---: |
| baseline | -11.88 | -9.19 |
| full target donor | +11.97 | +9.41 |
| all-24 J clamp | +11.22 | +8.55 |
| pair J clamp | +10.83 | +8.12 |
| logit lens | -11.77 | -9.19 |
| random | -11.88 | -9.21 |

The all-24 J clamp used about half the aggregate perturbation norm of full-state
replacement (mean 2.09 versus 4.01 across the five layers). Wrong-donor J
replacement was equally specific: every item recomputed the wrong donor's own
digit rather than merely suppressing the source.

## Frozen gate audit

- clean accuracy and parse >=0.80/0.95: pass (1.00/1.00);
- full-donor direct/consequence >=0.60/0.50: pass (1.00/1.00);
- direct J target shift >=0.20: pass (+1.00);
- consequence J target shift >=0.15: pass (+1.00);
- consequence J minus random >=0.10: pass (+1.00);
- consequence J minus wrong-target >=0.10: pass (+1.00);
- wrong donor's own digit shift >=0.10: pass (+1.00);
- paired-bootstrap lower bound >0: pass ([1.00, 1.00]);
- parse drop <=0.05: pass (0.00);
- every random row realized-norm error <=1e-5: **fail**.

Exactly one row failed: `confirm-0046`, consequence prompt, error
`1.1551673259871693e-5`. It remained source-correct and did not select the
target, but the frozen rule is about control validity, not whether the failure
was convenient. The requested random vectors had maximum J-span projection
2.34e-7; bf16 addition and residual rounding left maximum realized projection
0.0571. This second diagnostic strengthens the case for a quantization-aware
realized-orthogonal replication.

## Interpretation

The direct parent separated next-token writability from transport: a layer-24
coordinate changed the spoken concept 75% but changed a mapped digit 0%. This
follow-up moved the intervention to the actual antecedent token, fitted the
pullback through a future concept report, and clamped an early trajectory. Under
that correction, the coordinate edit propagated through an unseen arbitrary
mapping perfectly.

The wrong-donor and pair-only results are especially diagnostic. A generic
adversarial disruption would not make every wrong donor produce its own lookup
value, and an output-token motor should not let two concept coordinates change a
different digit token on 47/48 items. The ordinary concept logit lens remained
inert despite the same donor-coordinate operation. Thus the provisional best
explanation is that early context-local J coordinates track a causally consumed
concept state, whereas the parent's late averaged coordinate tracked output
motor control.

The formal conclusion remains invalid, not positive. One cannot repair a frozen
control threshold after seeing a perfect effect. The correct continuation is a
new result-bearing experiment with fresh mappings, a fixed band and lens, and an
in-hook optimizer that constrains the **realized** random delta's norm and J-span
projection simultaneously.

## Capability scope

Nothing here installs or autonomously elicits capability. The intervention is
given the target concept and uses clean target-donor coordinates. Even a valid
replication would establish an oracle causal mechanism. Native-thought work is
still ineligible until the control replication passes; a later deployable method
must infer when and what to edit without the answer and beat frozen plus
matched-compute sampling on fresh held-out tasks.

## Compute and artifacts

- Full lens fit: 27.7 seconds, 11.08 GB peak, 3.08 MB committed artifact.
- Donor gate: 44.2 seconds, 8.44 GB peak, 624 rows.
- Confirmation: 268.4 seconds, 8.44 GB peak, 672 rows; exact-norm control search
  dominated wall time.
- No training, adapter, benchmark, target-digit gradient, or native-thinking
  continuation was run.
- All row-level outputs and failure receipts are committed; see
  `reports/artifact_manifest.yaml`.
