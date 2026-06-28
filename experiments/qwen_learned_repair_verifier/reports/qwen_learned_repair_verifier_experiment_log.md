# Qwen Learned Repair Verifier Experiment Log

## Objective

Train a small learned verifier/reranker that chooses among local repairs of a
fixed Qwen-compiled modular-arithmetic program. The verifier is trained with
offline labels from exact state trajectories, but at test time it receives only
non-oracle candidate features.

## Success Criteria

- Build a standalone experiment directory with source, runs, analysis, reports,
  and a checkpoint manifest.
- Keep large model artifacts under `large_artifacts/`.
- Evaluate base compiler selection, learned verifier selection, paired
  consistency reranking, and oracle state-verifier selection on fresh length-24
  standard, paraphrase, and paired splits.
- Treat the oracle verifier as a ceiling, not as a deployable method.

## Runs

### Smoke

`smoke_learned_verifier`

- Used tiny verifier datasets with top-2/one-edit repair.
- Verified Qwen checkpoint loading, candidate generation, verifier training,
  metric writing, and checkpoint writing.
- The low oracle ceiling was expected because the smoke used a deliberately
  restricted repair neighborhood.

### Pilot: Fragment-Mixed Train/Val

`pilot_learned_verifier_s128`

- Used the full top-3/two-edit search.
- Train and validation prompts used per-fragment mixed renderings.
- Result: train/validation oracle ceilings were only 17.2% and 9.4%, while clean
  fresh standard/paraphrase ceilings were near 90%.
- Interpretation: fragment-mixed renderings were outside the useful compiler
  distribution. The dataset construction was patched to train and validate on
  clean standard/paraphrase renderings.

### Pilot: Clean Standard/Paraphrase

`pilot_clean_learned_verifier_s128`

- Used 128 train examples and 64 validation examples.
- Validation improved from 34.4% base to 43.8% learned, with an 84.4% oracle
  ceiling.
- Fresh standard improved from 29.7% to 40.6%.
- Fresh paraphrase and paired gains were small, so the next pilot tested data
  balance and richer features.

### Pilot: Paraphrase-Weighted

`pilot_para_weighted_learned_verifier_s192`

- Overweighted paraphrase renderings in verifier training.
- Fresh standard improved from 29.7% to 43.8%.
- Fresh paraphrase improved from 20.3% to 26.6%.
- Validation was weaker than the balanced clean pilot, so paraphrase weighting
  was not used for the main run.

### Pilot: Rich Candidate Features

`pilot_rich_features_s128`

- Added non-oracle features for edit values, operation mix, argument-change
  magnitudes, candidate argument statistics, and base/candidate operation mix.
- Validation improved from 34.4% base to 51.6% learned.
- Fresh standard improved from 29.7% to 50.0%.
- Fresh paired improved from 30.5% to 45.3%.
- This configuration was selected for the main run.

### Main

`main_rich_learned_verifier_s512`

- 512 train examples, 128 validation examples.
- 256 fresh standard examples, 256 fresh paraphrase examples, and 256 paired
  latent programs rendered twice.
- Top-3/two-edit candidate neighborhood, 1,299 candidates per length-24 example.
- Verifier width 192, 18 epochs, selected by validation learned executor
  accuracy.

Main fresh results:

| Split | Base | Learned | Pair rerank | Oracle |
|---|---:|---:|---:|---:|
| Standard L24 | 28.5% | 44.1% | n/a | 90.6% |
| Paraphrase L24 | 28.5% | 48.0% | n/a | 86.7% |
| Paired L24 | 30.3% | 47.3% | 51.0% | 88.1% |

Paired details:

| Metric | Base | Learned | Pair rerank | Oracle |
|---|---:|---:|---:|---:|
| Executor accuracy | 30.3% | 47.3% | 51.0% | 88.1% |
| Program exact | 30.3% | 47.3% | 51.0% | 87.7% |
| State prefix fraction | 58.6% | 71.2% | 73.4% | 90.7% |
| Pair both-correct | 28.1% | 34.4% | 46.5% | 85.9% |
| Pair state consistency | 71.1% | 55.1% | 82.8% | 92.6% |

## Interpretation

The learned verifier substantially improves over the fixed compiler, but it does
not close most of the oracle gap. The result is a positive but partial conversion
of oracle repair headroom into a non-oracle mechanism.

The paired consistency reranker is useful when two renderings of the same latent
program are available: it raises paired executor accuracy from 47.3% to 51.0% and
paired both-correct accuracy from 34.4% to 46.5%.

## Artifacts

- Small files: `experiments/qwen_learned_repair_verifier/`
- Large checkpoints: `large_artifacts/qwen_learned_repair_verifier/checkpoints/`
