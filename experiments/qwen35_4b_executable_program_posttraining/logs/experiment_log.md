# Experiment Log

## 2026-06-22

- Created standalone experiment directory.
- Selected official base model `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Confirmed the model loads through `AutoModelForImageTextToText` and can generate from a text-only prompt.
- Designed the experiment around executable DSL repair with visible-test reranking.
- Initial trace adapter completed. Full held-out trace evaluation showed strong modulo and tuple transfer but complete length+contains hidden failure.
- Inspected failed length+contains generations. The model repeatedly emitted `(count_eq text needle)` forms instead of the required conjunction, indicating missing conjunction support in the supervised DSL distribution.
- Added training-only conjunction families while keeping the 240-record training budget fixed for the next iteration.
