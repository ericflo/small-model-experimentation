# Qwen Support-Contrastive Meta-ICL

Standalone experiment testing whether a support-contrastive LoRA objective makes Qwen3-4B bind text-transformation answers to the provided support examples, rather than only adapting to the prompt format or task family.

Large artifacts, checkpoints, and the public benchmark mirror live under:

`/workspace/large_artifacts/qwen_support_contrastive_meta_icl`

## Planned Arms

- `contrastive_cf`: counterfactual synthetic episodes with positive CE plus support-corruption margin losses.
- `ce_cf`: same counterfactual synthetic episodes with positive CE only.
- `ce_ordinary`: ordinary synthetic few-shot episodes with positive CE only.
- `ce_shuffled_labels`: counterfactual synthetic episodes with shuffled support labels during CE training.

All arms use a fixed synthetic/public evaluation seed. The final report is generated as Markdown and HTML with charts.

## Final Read

The support-contrastive arm improves public strict task consistency over base and produces the cleanest support dependence:

- Base public full-task exact: `17.8%`.
- Support-contrastive public full-task exact: `50.4% ± 1.3%` over three seeds.
- Support-contrastive public corrupted-support controls: shuffled `7.4%`, no-support `0.0%`.

The margin objective is not the best raw-accuracy recipe in this run:

- CE-only counterfactual control: `53.3%`.
- CE-only ordinary control: `55.6%`.
- CE-only shuffled-label control: `48.9%`, with shuffled-support performance also `48.9%`.

The result is therefore a clean tradeoff: contrastive training buys causal support binding, but at this margin/weight it sacrifices several points of public normal-support accuracy relative to CE-only tuning.

Reports:

- Markdown: `reports/qwen_support_contrastive_meta_icl_report.md`
- HTML: `reports/qwen_support_contrastive_meta_icl_report.html`
