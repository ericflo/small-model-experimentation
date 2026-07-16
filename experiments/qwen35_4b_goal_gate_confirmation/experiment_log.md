# Goal-Gate Confirmation Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened as the mandatory successor to the recorded 10/10 at seed 78,154:
  three independent sealed medium seeds (78,155/78,156/78,157), two
  authenticated arms (base, hygiene_explore), K-seed write-ahead ledger,
  ordered confirmation verdict (CONFIRMED / AGGREGATE_ONLY /
  NOT_REPLICATED) with the discovery seed reported but never counted,
  per-seed fragility readings on the menders/warren margins,
  implementation-signature equality anchored to the discovery event.
- No model event has run; nothing trains in this cell.

## 2026-07-15 — Standalone-reproducibility retrofit (owner directive)

- Retrofitted the new standalone gate (AGENTS.md / docs/quality_gates.md,
  commits 02284dc3/1b4248a0) before the freeze: the hygiene_explore
  composite's complete model-reproduction package now lives in this cell.
  `data/lineage/` carries byte-identical copies of the six ordered SFT
  datasets (stage 3 preserved as
  `stage03_close_xi__targeted_standard.jsonl` — the source filename does
  not match the arm name) plus `lineage_manifest.json` (fixed seeds
  42/43/44/47/51/55, full hyperparameters incl. stage 3's targeted close
  overrides and stages 1-2's missing close channel, per-stage produced
  shas as verification aids, final merge onto the raw HF base).
  `scripts/lineage_trainers/` carries the three trainer variants
  (400e4b85… / 10b4914c… / 0cfb126f…) and `scripts/merge_adapter.py`
  (cb9af8b4…) byte-identically. The frozen `blend` root adapter (weights
  ad2ef4fa…, no committed creation receipt — a hard provenance boundary,
  documented as such) is vendored into this cell's own
  `large_artifacts/…/lineage_root/blend` (~181 MiB, six files
  hash-pinned). `scripts/rebuild_lineage.py` replays stages 1→6 plus the
  merge with per-stage sha verification; its `--verify-inputs` mode
  (no GPU) is wired into `run.py --smoke`, and the design receipt now
  pins the whole package (regenerated; sha changed accordingly). The
  rebuilt merge is verified on `model.safetensors` (e2112344…) and the
  content files — the published tree sha additionally covers a merge
  receipt embedding a machine-local absolute adapter path, recorded
  honestly in the manifest.
- Still no model event; the GPU rebuild path has not been executed.

## 2026-07-15 — Standalone retrofit (owner directive) before freeze

- The owner's standalone-reproducibility directive landed mid-build and
  this cell is the first to comply: six lineage datasets copied in
  byte-identically with a fixed-seed manifest, three trainer variants and
  the merger vendored into scripts/, rebuild_lineage.py with a no-GPU
  verify-inputs mode wired into smoke, and the C53-era root adapter
  vendored into this cell's own artifact storage with its provenance
  boundary (no committed creation receipt) stated plainly in the
  preregistration.
- Lineage fact established by the receipts walk: every stage and every
  merge uses the raw official Qwen3.5-4B revision as base; the lineage is
  carried entirely by one warm-started LoRA adapter; the root predates
  committed receipts. 123 tests green; smoke green; receipt regenerated
  (3864e812…) and --check byte-identical twice.

## 2026-07-15 — Adversarial review: the verdict chain hardened pre-freeze

- Three lenses; the lineage package audited clean against the source
  receipts (zero findings). Two MAJORs confirmed in the K-seed machinery
  and fixed: the verdict inputs are now provenance-anchored end-to-end
  (receipt shas pinned into closed ledger records; the readout refuses
  any break in the sealed chain; smoke verifies receipt hashes), and the
  summary-write/closed-append crash window closes via byte-equal
  reconciliation with a documented recovery. Four minors fixed including
  the honest 216-combination verdict-partition enumeration.
- 146 tests green; smoke green; receipt 66c19b24… --check twice;
  PASS_BENCHMARK_EVENT granted.

## 2026-07-15 — The three-seed event and closure

- CI green on the freeze; the six runs executed in the frozen seed-major
  order; every closed ledger record carries both receipt pins; the
  readout verified the full provenance chain before rendering.
- Verdict AGGREGATE_ONLY: aggregate strict wins on all three seeds
  (0.3287/0.3737/0.3837 vs 0.0586/0.1122/0.0982); goal gate 1/3 (78,157
  swept 10/10; 78,155 read 9/10 blocked by a menders 0-margin tie with
  warren WON at +0.267; 78,156 read 8/10 blocked by menders and warren
  ties; zero losses anywhere).
- Position: two full sweeps across four independent sealed seeds;
  demonstrated, not confirmed at the frozen 2/3 bar; menders is the
  single binding family (0.0 margin on every failing seed). The
  dose-scale intake aims at a precisely-known target.
