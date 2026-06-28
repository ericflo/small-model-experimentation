# Qwen Context-Conditioned Trace Verifier Experiment Log

## 2026-06-23

Objective: train and evaluate a context-conditioned learned verifier for local
candidate traces emitted by a frozen Qwen-attached hidden VM compiler. The
verifier uses candidate execution traces, compiler-local features, and compact
Qwen hidden-state summaries from the prompt. Reports must be standalone.

Initial setup:

- Created standalone experiment directory.
- Copied the frozen mixed-domain trace compiler into this experiment's own
  large-artifact area:
  `/workspace/large_artifacts/qwen_context_trace_verifier/checkpoints/fixed_mixed_vm_trace_compiler_s512`.
- Seeded local source from the mixed-domain verifier implementation and prepared
  to add context features, hard-negative contrastive loss, and optional paired
  training.
- Patched the context-conditioned reranker path so context normalization
  statistics are returned, saved, and used during evaluation.
- Switched the primary positive label from full trajectory exactness to final
  answer correctness, keeping state-exact and program-exact metrics as
  diagnostics. This aligns the experiment with complete-program candidate
  selection from executable answer labels.
- Ran `smoke_context_trace_reranker`, a tiny Qwen-backed end-to-end smoke test.
  It loaded the fixed `Qwen/Qwen3-4B` hidden-VM compiler checkpoint, generated
  candidate groups, trained the reranker for two epochs, wrote run artifacts,
  and saved a small verifier checkpoint. Fresh paired length-3 base, learned,
  and pair-rerank accuracy were all 66.7%; oracle was 100.0%. This validates
  the pipeline and confirms there is answer-correct candidate headroom even at
  smoke scale.
- Ran `pilot_context_trace_reranker_s96` with answer-correct positives, length-6
  fresh splits, length-8/10 hard splits, top-k 3, and two-edit candidate
  neighborhoods. Validation improved from 60.4% base to 64.6% learned, and hard
  standard improved from 54.2% to 56.2%, but fresh paired fell from 65.6% base
  to 57.8% learned; pair reranking recovered to 64.1%. Interpretation:
  answer-correct labels are dense, averaging 29.4 positives per 111 candidates
  on fresh paired, and are not sharp enough by themselves for robust top-1
  selection.
- Ran matched state-exact and oracle-selector pilots. State-exact training and
  single-oracle training mostly selected the base program everywhere, matching
  base accuracy on fresh and hard splits. Interpretation: sharper labels alone
  are insufficient when base-correct groups dominate the training signal.
- Ran `pilot_context_trace_reranker_s96_repairfocus`, which used single-oracle
  positives with base-positive groups downweighted and repairable groups
  upweighted. This also mostly selected the base program. Interpretation:
  weighting alone did not expose a strong repair preference under the current
  feature and trace representation.
- Ran `main_context_trace_reranker_s384_answer`, the main answer-label run with
  384 training prompts, 128 validation prompts, 128 fresh/hard evaluation
  prompts, 96 paraphrase pairs, length-6 in-distribution evaluation, and
  length-8/10 extrapolation. Best validation epoch was 14. The learned selector
  improved validation from 59.4% to 60.2%, fresh standard length-6 from 68.8%
  to 70.3%, and fresh paraphrase length-6 from 57.8% to 60.2%. It degraded
  fresh paired length-6 from 57.3% to 55.2%, hard standard length-8 from 50.0%
  to 46.9%, hard paraphrase length-8 from 38.3% to 35.9%, harder standard
  length-10 from 40.6% to 35.2%, and harder paraphrase length-10 from 17.2% to
  14.1%. Oracle answer-correct candidate availability stayed high, ranging from
  91.4% to 100.0% across the main evaluation splits.
- Generated aggregate CSVs, six figures, a standalone Markdown report, and a
  standalone HTML report with `src/analyze_qwen_context_trace_verifier.py`.
  The core conclusion is diagnostic: local executable candidate search has
  enough headroom, but answer-only verifier training is underdetermined and does
  not learn robust program selection for paired or longer-chain prompts.
