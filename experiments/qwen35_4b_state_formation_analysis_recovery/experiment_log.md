# State-Formation Analysis Recovery Experiment Log

## Scaffold

Created as a separate recovery experiment after the immutable source-v11 producer completed all
three LoRA joint training cells and seed 7411 trigger evaluation. Before any scientific values were
inspected, an independent production `_load_evaluation` reopen reproduced:

```text
RuntimeError: expected path is not lexical-canonical: .../experiments/
qwen35_4b_state_formation_capacity_adjudication/../../large_artifacts/
qwen35_4b_state_formation_capacity_adjudication/lora_joint_seed7411
```

The error occurs before evaluation rows are consumed. Editing the producer would invalidate its
source-bound artifacts, so the recovery is additive and separately reviewed.

## 2026-07-13 design

- Kept the producer at exact source-contract v11, SHA-256
  `5a8ed26ddb9446c728191ca8e7849ae44cff92a700e24b237dac522cf4286666`.
- Pinned exact `analysis.py` SHA-256
  `876888987d816fe29ae93fde0053fb91ed58301d16bf429d5cabaa809c23a2b0`.
- Rejected copying or modifying scientific summarizers.
- Authorized one temporary helper replacement that recognizes only the registered external prefix
  and clean descendants, then calls the original repository/no-symlink checker.
- Required canonical equivalence, unrelated-alias and traversal rejection, `finally` restoration,
  a source-bound smoke, original lock/source snapshot, a STARTED receipt, and an immutable output
  sidecar.
- Kept all result values, benchmark paths, and sealed contrast rows unopened during design/smoke.

## 2026-07-13 frozen smoke

The focused recovery suite passed 10/10. The non-result smoke then returned
`RECOVERY_SMOKE_PASS`:

- file SHA-256: `02f6f9275f9c30fddb2f49d4b061237e4e11985d92569642d60cd107b05243f7`
- receipt identity: `30353be5429d4987509715cfa56a6187f24a80ab353b9b774908071de7ed2f8f`
- recovery source contract: `6ab26016b3de397307c7c8def9c685315b6660370ee98af1a757da11fe1ee94b`
- exact producer source v11 and analyzer hashes matched
- canonical equivalence, defect reproduction, unrelated-alias rejection, and prefix-traversal
  rejection all passed
- result rows, benchmark paths, sealed contrast rows, and scientific analysis calls: zero

Any change to the frozen config, design review, runner, source, or tests now invalidates this smoke.
The next licensed action is publication and green CI, followed by LoRA trigger evaluation seeds 7412
and 7413. Running a recovery analysis phase remains prohibited until that three-seed matrix is
complete.
