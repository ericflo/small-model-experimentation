# Qwen3.5-4B verified-macro exact CUDA-graph rerun log

## 2026-07-10 — experiment split and protocol freeze

- Created a new experiment attached to `operator_and_skill_inventories`; named
  `qwen35_4b_verified_macro_capacity_fit_rerun` as the closest near-duplicate.
- Copied only frozen tasks, demonstrations, libraries, prompt identities, macro DSL, and model
  harness. Exact hashes and the no-output inheritance boundary are in
  `data/source_provenance.json`.
- Derived one local vLLM runner from predecessor SHA
  `fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782`. The new runner accepts an
  explicit capture-size tuple, requires its maximum to equal `max_num_seqs`, records vLLM's
  resolved compilation config, and fails unless sizes match under a full-decode graph mode. Frozen
  derived SHA: `3a98eb8da787054aded56a1ec3fd040ee2edaacc7d0694b4aec5a0309488774a`.
- Registered 49k shapes `[1,2,4,8,16,19]` with max-seqs 19 and 61k shapes
  `[1,2,4,8,15]` with max-seqs 15. Both remain subject to the independent live-KV fit gate.
- Preserved one-engine/one-phase execution, K4 nonpromotion, content-blind first-adequate selection,
  receipt-last storage, full-history verification, and three-pass analysis.
- Assigned a new external artifact namespace and forbade both predecessor roots.
- Added model-free tests for exact mapping, truncated resolved-list rejection, engine-config
  invariants, storage/state-machine behavior, parser/domain behavior, and hidden-content nonaccess.
- No GPU process or model call was launched.

## 2026-07-10 — model-free verification

- All 42 experiment-local CPU tests passed under the uv-managed vLLM environment, including
  explicit rejection of resolved `NONE` and piecewise-only CUDA-graph modes.
- `uv pip check --python .venv-vllm/bin/python` passed for all 189 installed packages.
- `scripts/run.py --validate` passed with record hashes
  `bd66aa64942f9e57e1fe55ae716c154ea1231480d6163f1811a07828ba364907` (base) and
  `c5a6cd00d9600b7a63c8e2c132e202b25da30f30af299afb3735a8f5525d9e86`
  (designed ceiling).
- The validated exact capture mappings are 49,152 → `[1,2,4,8,16,19]` and 61,440 →
  `[1,2,4,8,15]`; their maxima exactly equal max-seqs 19 and 15.
- The frozen protocol binding at this verification point is
  `9d2692c6acad35d3b7ab56ddf368c9974c1ddaf6e0a06997b01015c0de397158`.
- No GPU process or model call was launched by either validation command.

## 2026-07-10 — independent prelaunch GO

- Before engine construction, a separate read-only reviewer checked the frozen binding
  `9d2692c6acad35d3b7ab56ddf368c9974c1ddaf6e0a06997b01015c0de397158`, exact-width resolution
  assertions, live block-rounded KV gate, fresh artifact namespace, receipt-last state machine,
  content-blind first-adequate rule, K4 nonpromotion, and stop conditions.
- The independent verdict was **GO** for only the registered fresh 49,152-token K=4 probe. It did
  not authorize automatic rung advancement, K=12 generation, decoded inspection, or scoring.

## 2026-07-10 — exact-capture 49k K=4 result

- The live capacity audit passed: 996,864 exposed KV tokens, 528-token blocks, 963,072 tokens of
  block-rounded demand at max-seqs 19, and 33,792 tokens of remaining margin.
- The constructed engine resolved full-decode CUDA graphs exactly at
  `[1,2,4,8,16,19]`; width 19 was covered as preregistered.
- Receipt SHA-256
  `61da6f616365bf080e97f341bd0c2305b889998c4d161c61e63d06e5dfb5923c` commits 48/48 completed
  K4 samples. The external four-file tree is 30,653,162 bytes with SHA-256
  `654d44119fc46fe83428c154680ee502073c00022ccb9cdb0922c1896ca20685`.
- All 48 samples contacted the reasoning boundary and were force-closed. The content-blind audit
  found 38 periodic loops, 10 unresolved cap contacts, and six answer-limit contacts. Rates of
  79.17%, 20.83%, and 12.50% fail all three registered thresholds, rejecting the rung.
- Generation sampled 2,363,163 tokens in 4,809.081014 seconds, or 491.395964 sampled tokens/s.
  Late JIT warnings remain included in elapsed time. This is 4.1636% faster than the closest
  implicit-capture capacity-fit probe and 16.2114% slower than the overcommitted max-seqs-64
  diagnostic; the comparisons are descriptive across changed scheduler protocols.
- No decoded or scored content was inspected. At this checkpoint the 61k K4 probe was authorized;
  its later terminal result is recorded below. No K12 matrix, semantic analysis, or macro claim was
  authorized.

## 2026-07-10 — exact-capture 61k K=4 terminal result

- Ran one separately fresh 61,440-token K=4 probe after the 49k rejection. The live capacity gate
  passed with 997,888 KV tokens, 528-token blocks, 950,400 block-rounded tokens required at
  max-seqs 15, and a 47,488-token margin.
- The constructed engine resolved full-decode CUDA graphs exactly at `[1,2,4,8,15]`, covering the
  registered active width 15.
- Receipt SHA-256
  `8f00535a773e347ec4f90a48eb6b00960935d7010c3500af7ccdf57fedd6f2e1` commits 48/48 completed
  K4 samples. Across both probes the external eight-file tree is 70,191,578 bytes with SHA-256
  `4aa311ab579c301f2b2d7383591e3e68ed66035d184bbaaa2659d59bd95542d3`.
- All 48 samples contacted the reasoning boundary and were force-closed. The content-blind audit
  found 40 periodic loops, eight unresolved cap contacts, and four answer-limit contacts. Rates of
  83.33%, 16.67%, and 8.33% fail all three registered thresholds.
- Generation sampled 2,951,995 tokens in 7,422.885983 seconds, or 397.688312 sampled tokens/s,
  after 100.293580 seconds of model loading.
- Terminal selection SHA-256
  `acbaf7cdb84ee5633e4f86b0716360c382f2262eb50e79ca92ce02b6e157fb07` records `pass=false` and
  `selected_thinking_budget=null`. No K12 arm, decoded inspection, semantic analysis, or macro
  result is authorized. The setup negative is preserved rather than erased or promoted.
