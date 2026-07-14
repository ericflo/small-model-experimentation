# State-Formation Branch Authorization Recovery Experiment Log

## 2026-07-14 — downstream defect discovered fail-closed

- The authoritative recovered producer analysis emitted `LORA_JOINT_MISS_CONTROLS_REQUIRED` and was
  published at commit `719c8e5a`; both repository workflows passed.
- The exact registered source-v11 full-rank seed-7411 G0 command then stopped at
  `branch_authorization` before model load. Downstream authorization re-runs the immutable analyzer's
  evidence validator, which reaches the same nonlexical registered external prefix.
- Producer v11 durably wrote byte-identical, inode-distinct canonical/mirror failure receipts at
  SHA-256 `47305826…2c71`, identity `070c23af…aa24`. They report zero completed checks, no model load,
  no training/evaluation, zero benchmark/sealed access, and no authority.
- Source v11 refuses to overwrite the pair and requires archival. Direct retry is prohibited.

## 2026-07-14 — additive recovery design

- Created a separate experiment because the analysis recovery is already frozen and result-bearing.
- Pinned producer v11 source/config/implementation/CLI/GPU-runner/analyzer, the authoritative LoRA
  receipt, the first recovery sidecar, and the failed G0 pair.
- Kept exact producer CLI execution and source-snapshot validation. The only temporary runtime change
  is `src.analysis._canonical_expected_path` on the module object actually used by downstream branch
  validation.
- Required real downstream authorization smoke, alias/traversal/restoration controls, immutable
  invocation receipts, and commit-backed failure archival before verified retirement.

## 2026-07-14 — downstream smoke and failure archive pass

- The focused recovery suite passed 14/14.
- Real downstream smoke reproduced the original v11 rejection, then recomputed the exact
  `LORA_JOINT_MISS_CONTROLS_REQUIRED` receipt through the seam. All six controls pass. File SHA-256
  is `8bf5bb36…6849`, receipt identity `d1135ea2…49b5`, and frozen recovery source contract
  `55d0a455…56f3`.
- Smoke loaded no model, started no training/evaluation, opened no benchmark or sealed contrast, and
  changed no scientific interpretation.
- The archive copied exact failure bytes `47305826…2c71` to an inode-distinct recovery path and
  emitted receipt SHA-256 `4fcccea3…45ed`, identity `ff478d40…0ec3`. Both source receipts remain
  present, byte-identical, and inode-distinct; the archive authorizes no retry by itself.

## Current authorization

Only archive-checkpoint documentation, validation, commit, push, and workflow verification are
authorized. After both workflows pass, retire exactly the canonical/mirror failure paths using that
full commit SHA, then publish and validate the retirement before any model-bearing retry.

## 2026-07-14 — green archive checkpoint proves and retires exact source pair

- Archive commit `bdedabf4ea7d153d21a9b3a0cbe03cad66c8b6b2` passed Validate Repository and
  Publish Research Site.
- Retirement reopened the archive copy, archive receipt, producer canonical, and producer mirror
  from that exact Git commit and required byte equality before mutation.
- STARTED receipt SHA-256 is `90cf8dd1…380e`. Only the two producer failure paths were unlinked under
  held-inode verification; the recovery archive remains at exact SHA-256 `47305826…2c71`.
- Terminal retirement receipt SHA-256 is `6e4c8ee3…53ad`, identity `c9abdc59…eae7`. It records zero
  model/training/evaluation/benchmark/sealed access and authorizes recovered producer retry only
  after retirement publication.

## Current authorization

Only retirement documentation, validation, commit, push, and workflow verification are authorized.
After both workflows are green, retry the exact full-rank seed-7411 G0 through the frozen wrapper.
No positive control or result training is authorized until that G0 passes and is published.
