# State-Carry Versus State-Bag Counterfactual

**Status: implementation-ready, CPU-validated, no Qwen model call yet. This is not a result.**

## Research Program

- Primary program: `structured_execution_and_compilers`
- Program question: what internal structure turns brittle shallow computation into reliable longer-horizon execution?
- Closest near-duplicate: [`qwen_fastweight_hook`](../qwen_fastweight_hook/reports/latent_fastweight_qwen_paper.md), a negative thin-adapter recurrence result.
- Constructive anchors: [`latent_executor`](../latent_executor/reports/latent_executor_paper.md), [`sampled_query_filter_executor`](../sampled_query_filter_executor/reports/sampled_query_filter_executor_paper.md), [C44](../../knowledge/claims/index.md#c44-the-forward-pass-induction-wall-is-a-serial-compute-limit-not-a-knowledge-limit-reasoning-sft-induces-held-out-rules-perfectly-via-generation-100-but-at-chance-in-one-forward-pass-001----the-cot-is-100-load-bearing), and [C54](../../knowledge/claims/index.md#c54-tier-pareto-frontier-novel-serial-compute-mechanisms-length-penalized-compression-advantage--skin-shuffle-decisively-clear-the-032-medium-menagerie-bar-for-the-first-time-0345-all-events-but-no-single-qwen35-4b-model-clears-quick-and-medium-together-by-any-method-training-capacity-data-interpolation-or-weight-space-model-soup--the-two-tiers-occupy-a-non-convex-pareto-frontier-and-compete-for-the-fixed-models-representational-budget).

## Question

When parameters, data, optimization, readout, and transformer-layer FLOPs are matched, does organizing repeated Qwen computation as one serially carried state produce a representation that is more capable than aggregating the same number of independent shallow states?

This experiment is deliberately not satisfied by “looping helps.” A positive requires State-Carry to beat a separately trained State-Bag twin, improve when test-time recurrence exceeds the trained `K=4` horizon, expose increasingly sufficient intermediate state, and behave causally under state swaps.

## Architecture

`Qwen/Qwen3.5-4B` is split at complete native hybrid motifs:

```text
Prelude P: layers  0..11
Loop R:    layers 12..19  (two [GDN,GDN,GDN,attention] motifs)
Coda C:    layers 20..31
```

Eight existing-vocabulary `<|fim_pad|>` tokens form a causal state bottleneck before the natural-language query. The untouched first `P→R→C` pass has recurrence LoRA disabled; therefore `K=1` must reproduce the standard model's logits on the identical token sequence.

For extra calls:

```text
State-Carry: z_t = R(prompt_memory, z_{t-1}, step=t)
State-Bag:   b_t = R(prompt_memory, z_1,     step=t)
```

Both arms use the same tied Qwen layers, loop-only LoRA, sinusoidal step signal, damping, state initializer, last-plus-mean aggregator, auxiliary state heads, data order, seeds, and number of calls. Non-state activations are reset after every extra call; only the state slots can cross recurrence depth.

See [`docs/architecture.md`](docs/architecture.md) for the exact forward, [`docs/research_handoff.md`](docs/research_handoff.md) for the reasoning behind every choice, and [`reports/implementation_review.md`](reports/implementation_review.md) for the final pre-GPU audit.

## Substrate

Every item is a fresh, exactly executed finite world with randomly skinned node names. The hidden state is `(node, phase, checksum)`; the next edge and state update depend on the preceding state. Two transition families and depths 1–4 are trained; depths 5–12, a third transition family, a held-out rendering, and their conjunction are untouched evaluations.

The workspace tokens occur before `Query:`. The recurrent state knows the world and requested transition count but not whether it will later be asked for the terminal node or checksum. Shared auxiliary heads query node, phase, and checksum after every iteration, pressuring one state to remain jointly sufficient rather than answer-specific.

No file under `benchmarks/` is read or imported.

## Primary Arms and Controls

1. Separately trained continuous-state **Carry** and **Bag** twins.
2. A trained Carry checkpoint evaluated with its edge cut into Bag mode.
3. Counterfactual donor-state swaps in pairs sharing the same world and rule.
4. A pre-registered mixed continuous/semantic-echo carry/bag pair, opened only by the interface branch condition.
5. A standard autoregressive explicit-state-trace LoRA trained on the same procedural rows, followed by matched-layer-token sample-more evaluation.
6. A static `K=1` LoRA diagnostic; it cannot establish recurrent depth.

## Run Order

From the repository root, rebuild the pinned `.venv` first. Then:

```bash
# CPU contracts; already passes in this commit.
python3 experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --smoke
python3 -m unittest discover -s experiments/qwen35_4b_state_carry_vs_state_bag/tests -v

# Deterministic full data and live Qwen mechanics gate.
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --stage prepare-data
.venv/bin/python experiments/qwen35_4b_state_carry_vs_state_bag/scripts/run.py --stage model-smoke
```

Do not launch training unless `runs/model_smoke/receipt.json` says `MODEL_SMOKE_PASS`. The precise pilot, promotion, multiseed, echo, and sample-more commands are in [`docs/gpu_runbook.md`](docs/gpu_runbook.md).

## Primary Metrics

- paired State-Carry minus State-Bag accuracy at `K=semantic depth`, depths 5–12;
- machine-verified initialization and training-compute equality for every seed pair;
- paired Carry gain from trained `K=4` to unseen `K=semantic depth`;
- exact node/phase/checksum trajectory accuracy;
- donor-following versus recipient-preserving outcomes under matched state swaps;
- quick/overthinking retention at `K=1,4,8,12`;
- matched-compute explicit-CoT majority accuracy and oracle `pass@N`.

The fail-closed verdict ladder is defined in [`reports/preregistration.md`](reports/preregistration.md) and implemented in `src/analysis.py`.

## Artifacts

- Source and tests are committed here.
- Generated full JSONL is deterministic and gitignored under `data/generated/`; its manifest and hashes are produced at runtime.
- Adapters, loop state, and checkpoint receipts live under `large_artifacts/qwen35_4b_state_carry_vs_state_bag/` and are declared in [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).
- Small evaluation rows, summaries, and paired analysis remain under `runs/` and `analysis/` unless size requires manifesting them.
