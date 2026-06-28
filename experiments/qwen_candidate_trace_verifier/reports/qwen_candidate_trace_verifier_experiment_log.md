# Qwen Candidate-Trace Verifier Experiment Log

## Objective

Train a non-oracle verifier that reads each candidate program's execution trace
and chooses a local repair of a fixed Qwen-compiled modular-arithmetic program.

The verifier is trained from offline trajectory labels, but at test time it sees
only candidate-local information: copied slots, edit locations, compiler
confidence, candidate states, base states, and differentiable-executor support.

## Success Criteria

- Create a standalone experiment directory with source, run metadata, analysis,
  reports, and a checkpoint manifest.
- Keep large artifacts under `large_artifacts/`.
- Evaluate base compiler selection, trace-verifier selection, paired consistency
  reranking, and oracle trajectory selection.
- Run smoke, pilot, and main configurations rather than one-shotting the final
  experiment.

## Runs

### Smoke

`smoke_trace_verifier`

- Used a top-2/one-edit candidate neighborhood, tiny data, and a one-layer trace
  transformer.
- Verified checkpoint loading, trace construction, training, metric writing, and
  checkpoint writing.

### Pilot: Two-Layer Trace Verifier

`pilot_trace_verifier_s128`

- Full top-3/two-edit neighborhood.
- 128 verifier-training examples and 64 validation examples.
- Two-layer trace transformer.
- Validation improved from 34.4% base to 53.1%, with an 84.4% oracle ceiling.
- Fresh paired improved from 30.5% base to 40.6%; paired reranking reached 44.5%.

### Pilot: Deeper Trace Verifier

`pilot_trace_deep_s192`

- 192 verifier-training examples and 96 validation examples.
- Three-layer trace transformer.
- Validation improved from 32.3% base to 54.2%, with an 84.4% oracle ceiling.
- Fresh paired improved from 30.5% base to 44.5%.
- This configuration was selected for the main run because it gave the strongest
  single-prompt trace-verifier result.

### Main

`main_trace_verifier_s512`

- 512 verifier-training examples and 128 validation examples.
- 256 fresh standard examples, 256 fresh paraphrase examples, and 256 paired
  latent programs rendered twice.
- Top-3/two-edit candidate neighborhood: 1,299 candidates per length-24 example.
- Three-layer trace transformer, model width 128, four attention heads, 18
  epochs, selected by validation trace-verifier executor accuracy.

Main fresh results:

| Split | Base | Trace verifier | Pair rerank | Oracle |
|---|---:|---:|---:|---:|
| Standard L24 | 28.5% | 50.4% | n/a | 90.6% |
| Paraphrase L24 | 28.5% | 55.5% | n/a | 86.7% |
| Paired L24 | 30.3% | 53.7% | 56.2% | 88.1% |

Fresh paired details:

| Metric | Base | Trace verifier | Pair rerank | Oracle |
|---|---:|---:|---:|---:|
| Executor accuracy | 30.3% | 53.7% | 56.2% | 88.1% |
| Program exact | 30.3% | 53.7% | 56.2% | 87.7% |
| State prefix fraction | 58.6% | 76.4% | 78.0% | 90.7% |
| Pair both-correct | 28.1% | 39.8% | 52.3% | 85.9% |
| Pair state consistency | 71.1% | 58.2% | 87.1% | 92.6% |

## Interpretation

The trace verifier materially improves non-oracle repair selection. It recovers
40.5% of the base-to-oracle gap on fresh paired length-24 programs.

Single-prompt trace verification increases exact accuracy but lowers paired
state consistency, because the two prompt renderings can choose different local
repairs. Pair reranking restores much of that consistency and improves paired
both-correct accuracy.

The oracle ceiling remains far above the learned verifier, so candidate search is
not the limiting factor. Selection quality is still the main bottleneck.

## Artifacts

- Small files: `experiments/qwen_candidate_trace_verifier/`
- Large checkpoints: `large_artifacts/qwen_candidate_trace_verifier/checkpoints/`
