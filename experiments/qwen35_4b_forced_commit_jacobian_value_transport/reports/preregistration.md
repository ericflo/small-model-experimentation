# Preregistration: Forced-Commit Jacobian Value Transport

Frozen before any model call. CPU generation, exhaustive task checks, lens-hash
validation, gate arithmetic, and unit tests may precede the immutable design
commit. No model output informed these rules.

## 1. Scientific ladder

This experiment separates four statements:

1. **Forced-policy seam:** a fixed budget controller that injects `</think>` on
   cap contact yields parseable answers with correctness headroom.
2. **Value-readable:** disjoint forced-policy continuation success is
   held-out-by-task decodable from J coordinates at a thought-prefix endpoint.
3. **Value-causal:** changing one learned scalar J coordinate improves fresh
   forced-policy continuations beyond identity and geometry controls.
4. **Deployable capability:** a non-oracle policy improves untouched tasks over
   frozen and matched-compute sampling.

Statements 1--3 are eligible here. Statement 4 requires a new experiment.

## 2. Counterfactual-policy boundary

The prior natural-close selector found 0/48 closes through 1,024. This design
does not weaken or reinterpret that result. At a fixed cap `B`, the deployed
policy is:

1. run native thought until natural `</think>`, EOS, or `B` thought-generation
   steps;
2. if natural close occurs by `B`, keep the natural answer;
3. if the trace contacts `B`, append exactly one close token (ID 248069);
4. generate at most 16 answer tokens under the same sampling policy.

Every injected-close row is explicitly `counterfactual_to_natural_close=true`.
Its legitimacy comes from using the same action in calibration, value labeling,
causal evaluation, and any later deployment. No result may be called a natural
seam or autonomous commit. C51's counterfactual-state warning remains binding.

## 3. Fixed model, lens, and backend

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Frozen 24-coordinate band: layers 4, 5, 6, 7, 8.
- Transformers bf16 SDPA, unpadded batch one, KV cache enabled.
- Temperature 0.6, top-p 0.95, top-k 20, all values explicit.
- Prompt cap 768, total sequence cap 2048, answer cap 16.
- Every generation audit must show one full prefill followed only by one-token
  cached forwards. vLLM and cache-free samples are excluded.

Activation/value/causal replays use one exact `prompt + thought prefix +
</think>` prefill. Features and patches are evaluated at the last thought token,
immediately before the injected close. Including the close in the same prefill
matches the live numerical sequence length while causal masking prevents it from
supplying semantic future information to the endpoint.

## 4. Fresh exact-depth substrate

Generate 96 fresh tasks: 16 seam selection, 16 seam confirmation, 32 value fit,
and 32 causal confirmation. Each has eight visible and eight hidden examples
from the inherited depth-two list DSL and fixed one-token alias map.

For every visible set, exhaustive enumeration requires:

- every matching depth-two pipeline shares one first-operation type; and
- no concrete depth-one operation matches all visible examples.

`negate` may appear second/distractor but is excluded as a target because its
first-step compositions are reorderable. First-operation counts differ by at
most one per split. All fingerprints are unique and have zero overlap with
`qwen35_4b_jacobian_value_transport`,
`qwen35_4b_native_thought_jacobian_value_transport`, and
`qwen35_4b_native_thought_seam_budget_ladder`. No `benchmarks/` content is read.

## 5. Seam selection

On 16 tasks, sample three native traces each to the maximum cap 1,024 with stable
trace seeds. For each same sampled path, evaluate policy caps 256, 512, and 1,024:

- if a natural close was already emitted, use its natural answer;
- otherwise replay exactly the first `B` thought tokens, append close, and use a
  cap/task/trace-specific answer seed.

The 144 policy rows are paired views of 48 traces, not independent traces.
Select the smallest cap passing all:

- policy parse rate >=0.90;
- forced-only parse rate >=0.90;
- forced-commit rate >=0.50;
- policy exact success in [0.05,0.95];
- at least six tasks with both correct and incorrect policy traces; and
- answer-cap contact <=0.05.

Malformed EOS-before-cap rows are parse failures and cannot be force-replayed.
If no cap passes, terminal `FORCED_COMMIT_SEAM_FAIL`; later stages stay sealed.

## 6. Seam confirmation

Open only the selected cap on 16 untouched tasks with three traces and disjoint
trace/answer seeds. Repeat every selection threshold unchanged. Verify the
complete selection trace and policy hashes before loading the model.

Passing yields `FORCED_COMMIT_SEAM_REPLICATED`; failure yields
`FORCED_COMMIT_SEAM_NOT_REPLICATED`. A larger cap cannot rescue confirmation.
Both gates are mathematically reachable: 48 rows, at least 24 forced rows, six
mixed tasks, and a feasible 0.5 success assignment.

## 7. Forced-policy prefix value labels

Only a replicated seam opens 32 fresh value-fit tasks. Generate three traces per
task to selected cap `B*`. For each trace that remains open at the relevant
checkpoint, retain exact thought prefixes at `0.5 B*` and `1.0 B*` (both integer
for all candidate caps). Natural/EOS prefixes that terminate before a checkpoint
are excluded and reported.

From each exact prefix, append close and sample three disjoint-seed answer
continuations. Define

`V(prefix) = exact correct forced-policy continuations / 3`.

Whole-trace correctness is never copied onto prefix tokens. Seam, value,
causal-selection, and causal-evaluation seeds are disjoint.

## 8. Value features and held-out evaluation

Replay `prompt + exact prefix + close` in one batch-one prefill. At the final
thought token, read all 24 frozen J coordinates at layers 4--8 and concatenate
them. Standardize using training folds only. Fit L2 logistic/ridge value readout
with four GroupKFold splits by task; train-fold task and checkpoint-fraction
means are removed without test leakage.

Primary metric: task-macro pairwise AUC, comparing higher versus lower `V`
within the same task and checkpoint fraction. Comparators:

- correct-alias J-coordinate activity;
- first-answer-token entropy/logprob;
- prefix length/fraction;
- full residual linear probe (diagnostic only); and
- labels shuffled within task/fraction.

G1 requires at least 16 mixed tasks, 128 scored prefixes, J pairwise AUC >=0.65,
J minus correct-alias activity >=0.03, and shuffled AUC within 0.05 of chance.
The full residual probe cannot rescue J failure. On pass, freeze normalization,
beta, shuffled beta, logit comparator, and training 0.90 score quantile.

## 9. Scalar J intervention

Let `z_l` be the 24 J coordinates and `beta_l` the frozen value-readout slice at
layer `l`. For recipient score `s` and target score `s*`, set

`z'_l = z_l + (s* - s) beta_l / sum_j ||beta_j||^2`.

At the live final thought token during the exact prefix-plus-close prefill,
clamp layers 4--8 to these clean desired coordinates. The modified cache then
generates the answer. This transfers one global scalar; it does not copy the
donor's full J state or answer token.

## 10. Exact post-bf16 controls

Before causal outcomes, calibrate on at least 16 forced seam-confirmation
prefixes. For every recipient and live prefix length, construct `random_a` and
`random_b` from 32 fixed continuous starts using the replicated quantization-
aware correction plus exact bf16 lattice solver. Candidate selection sees only
activation, dictionary, target norm, and geometry errors.

Every realized layer delta must have relative norm error <=1e-5 and J-span
projection fraction <=0.01. No control basis is reused across different live
prefixes/lengths. Failure yields `CONTROL_UNREACHABLE`; thresholds cannot relax.

## 11. Causal pairs and arms

On 32 untouched tasks, generate forced-policy prefixes at the frozen fractions
and estimate `V` with three causal-selection continuations. Within the same task
and fraction, choose the deterministic largest high/low gap with low <=1/3 and
high >=2/3. Require at least 16 pairs. Ties use stable IDs. Evaluation uses two
new shared answer seeds per pair.

Arms:

1. unpatched low prefix;
2. scalar J value clamp (primary);
3. full-24-J same-task high donor;
4. correct-alias-versus-best-competitor J clamp;
5. within-task/fraction shuffled value axis;
6. exact random A;
7. exact random B;
8. separately fitted logit-lens value axis;
9. full raw same-task high activation;
10. J-span component of the high-minus-low donor delta;
11. per-layer norm-matched non-J donor remainder;
12. value-fit high-minus-low ActAdd; and
13. full-J donor from another task with a different correct alias.

All magnitude controls match the realized primary scalar delta per layer. Every
confirmation random row repeats both numeric gates. Direct alias-coordinate
movement, parse, answer length, natural/forced branch, and token costs are
diagnostics; exact alias success is primary.

## 12. Frozen causal decision

Require at least 16 pairs. `FORCED_VALUE_CAUSAL` requires:

- scalar-clamp paired success uplift >=0.10 over replay baseline;
- paired 10,000-resample 95% lower bound >0;
- scalar minus worse exact-random arm >=0.08;
- scalar minus shuffled-axis >=0.08;
- scalar minus matched non-J >=0.05;
- parse-rate drop <=0.05; and
- all design, seam, value, numeric, cache, position, and completeness contracts.

If full-J/correct-alias controls work but scalar fails, use
`IDENTITY_NOT_VALUE`. Other labels are `VALUE_NOT_DECODABLE` and
`DECODED_NOT_CAUSAL`. No secondary arm can rescue the primary.

## 13. Oracle and capability boundary

The seam policy itself is deployable after selection/confirmation, but gold
outcomes fit value beta and select high/low donors. Therefore even
`FORCED_VALUE_CAUSAL` is oracle mechanism evidence. Report total thought/answer
tokens and matched additional baseline coverage, but do not call it capability.

Only a separate non-oracle controller may use frozen prefix features without
test labels. It must replicate a contamination-free held-out gain over frozen
Qwen, the strongest relevant controls, and matched-compute sampling, while
ruling out close-token, format, extra-token, and backend artifacts.

## 14. Stage and artifact discipline

Immutable design -> outcome-blind model smoke -> seam selection -> untouched
seam confirmation -> value fit -> numeric calibration -> frozen pair receipt ->
one causal confirmation. Every stage hash-unlocks the next. Incomplete rows are
buffered and cannot create a summary. Until audited implementations land,
`prefix-value`, `control-calibration`, and `causal-confirmation` must fail closed
rather than emit placeholders.

Preserve every terminal negative, update owning program ledgers/synthesis, run
`make check`, synchronize before shared-index/claim work and every push, and
inspect CI. No claim ID is reserved while the claim re-grade remains open.
