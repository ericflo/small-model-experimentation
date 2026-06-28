# Qwen Typed Bytecode Expert Iteration Summary

Primary run: `main_typed_bytecode_ei_s384_u4096`

## Headline

- Fresh paired seed direct accuracy: 61.5%
- Fresh paired expert-iteration direct accuracy: 73.0% (+11.5 pp)
- Fresh paired full-supervised direct accuracy: 99.6%
- Fresh paired expert-iteration search accuracy: 87.3%
- Fresh paired full-supervised search accuracy: 100.0%
- Hard-composition seed/expert/full direct: 45.9% / 53.9% / 80.7%

## Final Expert Round

- Answer-verified targets found: 93.4%
- Changed-target rate among found targets: 5.2%
- Mean local candidates per prompt: 240.9
- Candidate valid rate: 63.8%

## Frozen-Qwen Attached Pilot

- Fresh paired seed direct accuracy: 17.6%
- Fresh paired expert-iteration direct accuracy: 50.4% (+32.8 pp)
- Fresh paired full-supervised direct accuracy: 94.5%
- Fresh paired expert-iteration search accuracy: 74.2%

## Interpretation

The typed stack-machine ABI made dense bytecode supervision highly learnable. Full supervised training reached near-ceiling fresh accuracy and preserved strong hard-composition transfer. Answer-verified expert iteration also improved the deployable compiler, but it did not close the full supervised gap. The remaining bottleneck is target quality: final-answer verification produces many useful targets, but it is weaker than direct bytecode traces and can admit semantically accidental programs.
