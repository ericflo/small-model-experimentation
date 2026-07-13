# Post-Model-Smoke 002 Audit: Valid Numeric Failure

## Verdict

`LIVE_CONTROL_REPAIR_REQUIRED`. The corrected receipt is valid and outcome-
blind. It does not authorize mechanics.

## Results

- Exact model/revision/lens, 386-token prompt, 32-token live prefix, 12 branches,
  and one application at each layer passed.
- All J/non-J logits were finite; no probabilities, choices, target-selection
  metrics, correct aliases, or task outcomes were recorded.
- J requested-to-realized norm error ranged up to 4.16% under naïve bf16 add.
- Non-J realized norm differed from paired J by up to 3.39%, failing 1e-5.
- Non-J J-span projection reached 2.96%, failing 1% at layer 8.
- Diagnostic realized Gram error reached 5.83% and zero-sum residue 0.015625.

## Frozen-gate interpretation

The preregistration freezes exact zero-sum/rank/full-Gram construction for the
float branch bank and explicitly freezes two live-bf16 gates: paired non-J norm
error <=1e-5 and complete-J-span projection <=0.01. The initial runner
incorrectly elevated live J-request fidelity, realized zero-sum, and realized
Gram to additional hard gates. They remain reported adversarial diagnostics,
but the existing CPU receipt is the registered exact bank-geometry gate.

This interpretation does not rescue smoke 002: its two explicit live gates both
fail. The next repair keeps 1e-5/1% unchanged and uses only geometry to iteratively
correct each fixed non-J branch against its paired realized J norm.

## Authorized repair

- retain the fixed exact-Gram non-J branch as the starting request;
- after bf16 addition, project out the complete J span and correct toward the
  paired realized J norm;
- iterate at the preregistered 512-step/0.5-damping budget inherited from the
  independent transport control repair;
- freeze a row as soon as both live gates pass;
- record iterations and 12/12 row pass counts per layer; and
- re-anchor changed code/config/tests before smoke 003.

No task label, branch target outcome, or answer probability may enter repair.
