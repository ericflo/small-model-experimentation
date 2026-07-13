# State-Carry Versus State-Bag Counterfactual

**Status:** finished

**Terminal LoRA status: `PILOT_MECHANISM_MISS`.** The fixed-source, independently seeded
rank-32 LoRA pilot was valid and complete, but it failed the preregistered deep-state-formation
gate. Confirmation and sample-more were therefore not run. This result mandates a fresh
full-rank extra-R-delta successor; it does not close the serial-state question.

## Research Program

- Primary program: `structured_execution_and_compilers`
- Program question: what internal structure turns brittle shallow computation into reliable longer-horizon execution?
- Closest near-duplicate: [`qwen_fastweight_hook`](../qwen_fastweight_hook/reports/latent_fastweight_qwen_paper.md), a negative thin-adapter recurrence result.
- Constructive anchors: [`latent_executor`](../latent_executor/reports/latent_executor_paper.md), [`sampled_query_filter_executor`](../sampled_query_filter_executor/reports/sampled_query_filter_executor_paper.md), [C44](../../knowledge/claims/index.md#c44-the-forward-pass-induction-wall-is-a-serial-compute-limit-not-a-knowledge-limit-reasoning-sft-induces-held-out-rules-perfectly-via-generation-100-but-at-chance-in-one-forward-pass-001----the-cot-is-100-load-bearing), and [C54](../../knowledge/claims/index.md#c54-tier-pareto-frontier-novel-serial-compute-mechanisms-length-penalized-compression-advantage--skin-shuffle-decisively-clear-the-032-medium-menagerie-bar-for-the-first-time-0345-all-events-but-no-single-qwen35-4b-model-clears-quick-and-medium-together-by-any-method-training-capacity-data-interpolation-or-weight-space-model-soup--the-two-tiers-occupy-a-non-convex-pareto-frontier-and-compete-for-the-fixed-models-representational-budget).

## Question

When parameters, data, optimization, readout, and decoder-layer token applications are matched, does organizing repeated Qwen computation as one serially carried state produce a representation that is more capable than aggregating the same number of independent shallow states?

This experiment is deliberately not satisfied by “looping helps.” A positive requires State-Carry to beat a separately trained State-Bag twin, improve when test-time recurrence exceeds the trained `K=4` horizon, accurately track the registered joint state, transfer to the joint family+surface holdout, lose its benefit when the carry edge is cut, and behave causally under state swaps.

## Pilot Result

The source-bound seed-7401 Carry/Bag pilot completed all registered cells and diagnostics with
matched initialization, data order, prompt tokens, decoder-layer-token applications, and exact
post-checkpoint K=1 parity. The analyzer emitted `PILOT_MECHANISM_MISS` because Carry's mean joint
node+phase+checksum step accuracy was `0.0045948`, far below the `0.40` promotion threshold; node
step accuracy was `0.0641912`. This is a valid failure to form the registered deep joint state, not
a mechanics, data-integrity, or infeasible-gate stop.

The answer-level signs were insufficient to override that failure. Matched-depth Carry minus Bag
was `+0.04296875` on 256 tasks (pilot 95% interval `[-0.0078125, 0.09375]`), and unseen-K Carry
minus K=4 was `+0.01171875` (`[-0.03515625, 0.05859375]`). Both query strata were positive and the
joint holdout diagnostic was `+0.05078125` (`[0.0078125, 0.09765625]`), but donor following changed
by only `+0.0078125` under swaps (`[-0.0234375, 0.0390625]`) and remained below recipient
preservation by `0.0546875`.

No seeds 7411–7413 were trained or evaluated. The same-checkpoint edge cut and the explicit-CoT
sample-more comparator were not licensed. These missing stages are consequences of the registered
pilot stop, not missing evidence for a full-run claim.

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

Both arms use the same tied Qwen layers, loop-only LoRA, sinusoidal step signal, damping, state initializer, last-plus-mean aggregator, auxiliary state heads, ordered training rows, seeds, and number of calls. Non-state activations are reset after every extra call; only the state slots can cross recurrence depth. Checkpoints receipt the exact ordered-row digest, runtime-source contract, environment lock, phase, and fixed final step.

See [`docs/architecture.md`](docs/architecture.md) for the exact forward, [`docs/research_handoff.md`](docs/research_handoff.md) for the reasoning behind every choice, and [`reports/implementation_review.md`](reports/implementation_review.md) for the final pre-GPU audit.

## Substrate

Every item is a fresh, exactly executed finite world with randomly skinned node names. The hidden state is `(node, phase, checksum)`; the next edge and state update depend on the preceding state. Two transition families and depths 1–4 are trained. A disjoint pilot seed and structurally firewalled pilot splits gate promotion; the three confirmatory seeds and all depth-5–12, third-family, held-out-rendering, and joint-holdout rows remain untouched until confirmation. Node/checksum queries are balanced within each family×template×depth cell.

The workspace tokens occur before `Query:`. The recurrent state knows the world and requested transition count but not whether it will later be asked for the terminal node or checksum. Shared auxiliary heads query node, phase, and checksum after every iteration, pressuring one state to remain jointly sufficient rather than answer-specific.

No file under `benchmarks/` is read or imported.

## Primary Arms and Controls

1. Separately trained continuous-state **Carry** and **Bag** twins.
2. A trained Carry checkpoint evaluated with its edge cut into Bag mode.
3. Bidirectional counterfactual donor-state swaps in geometry-matched pairs sharing the same world, initial node, rule, and answer interface.
4. A standard autoregressive explicit-state-trace LoRA trained on the same procedural rows, followed by matched-layer-token sample-more evaluation with frozen sampling/allocation parameters and termination/interface validity gates.

The earlier outcome-dependent mixed semantic-echo branch was removed during this review. It is a distinct architecture with missing shuffled/wrong-task controls and, under the repository lifecycle, must be a fresh successor experiment if continuous-state results license it.

Rank-32 LoRA remains the first adaptation because it touches every linear projection in both repeated
motifs while preserving a cheap, exactly disabled K=1 path; the carried workspace itself remains
full-width and receives dense joint-state supervision. A valid negative that fails to establish deep
state formation does not settle whether LoRA was too restrictive. Preregistration section 10 mandates
creating and executing a fresh zero-initialized full-rank extra-R-delta successor in that case,
preserving the same model and exact base path. Mechanics/data-integrity failures and mathematically
invalid gates require repair rather than a capacity test. A readable-but-unused state or a
sample-more-only loss does not trigger full rank because LoRA has then already formed the deeper
representation; the former instead licenses the separately controlled interface successor.

## Terminal Disposition

Do not advance this LoRA experiment to G2, edge-cut confirmation, or G4. Preserve its valid negative
and the earlier invalidated analysis-dispatch attempt. The next authorized capacity test is a new
experiment directory using zero-initialized full-rank weight deltas on layers 12–19 only during extra
R applications, with the ordinary first pass, coda, exact K=1 path, Carry/Bag counterfactual, pilot
firewall, and causal gates held fixed.

## Primary Metrics

- paired State-Carry minus State-Bag accuracy at `K=semantic depth`, depths 5–12, with a crossed task×training-seed bootstrap;
- machine-verified initialization and training-compute equality for every seed pair;
- paired Carry gain from trained `K=4` to unseen `K=semantic depth`;
- exact joint node+phase+checksum trajectory accuracy;
- same-checkpoint intact-minus-edge-cut accuracy with positive evidence in every seed;
- bidirectional pre/post donor-following under geometry-matched state swaps;
- joint family+surface holdout non-reversal and both node/checksum query strata;
- quick/overthinking retention at `K=1,4,8,12`;
- matched-budget explicit-CoT majority accuracy and oracle `pass@N`, with natural-close reporting and preregistered parse/cap-contact gates.

The fail-closed verdict ladder is defined in [`reports/preregistration.md`](reports/preregistration.md) and implemented in `src/analysis.py`.

## Artifacts

- Source and tests are committed here.
- Generated full JSONL is deterministic across Python hash seeds and gitignored under `data/generated/`; its manifest binds the generator/source contract and hashes are produced at runtime.
- Realized pilot adapters, loop state, checkpoint/run identities, and hashes live under `large_artifacts/qwen35_4b_state_carry_vs_state_bag/pilot_{carry,bag}_seed7401/` and are declared in [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).
- Small evaluation rows, summaries, and paired analysis remain under `runs/` and `analysis/` unless size requires manifesting them.
