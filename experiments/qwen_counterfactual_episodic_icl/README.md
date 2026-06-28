# Counterfactual Episodic ICL Posttraining

This standalone experiment tests whether answer-only LoRA posttraining on
counterfactual few-shot episodes improves Qwen's ability to infer a text
transformation from support examples.

The training data is synthetic and deliberately counterfactual: the same
input style can require incompatible outputs depending on the examples in
the prompt. Public benchmark examples are used only for evaluation.

## Main Run

- Run: `main_v2`
- Model: `Qwen/Qwen3-4B`
- Adapter: `/workspace/large_artifacts/qwen_counterfactual_episodic_icl/checkpoints/main_v2/adapter`
- Synthetic held-out counterfactual full-task exact: base `25.0%`, adapter `91.7%`, adapter with shuffled support `31.7%`.
- Public text-transformation full-task exact: base `23.3%`, adapter `56.7%`, adapter with shuffled support `16.7%`.

## Artifacts

- Report: `reports/qwen_counterfactual_episodic_icl_report.md`
- HTML report: `reports/qwen_counterfactual_episodic_icl_report.html`
- Source: `src/qwen_counterfactual_episodic_icl.py`
- Run CSVs: `runs/main_v2/`
- Figures: `analysis/figures/`
- Large artifacts: `/workspace/large_artifacts/qwen_counterfactual_episodic_icl`
