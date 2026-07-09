# Preregistration: verified macro invention

Status: draft frozen before the first scored vLLM run. Constants live in
`configs/default.yaml`; the committed analyzer, not hand calculation, produces verdicts.

## Question

Can reusable composite operators, derived only from prior verified programs, improve
visible-only selected accuracy on fresh behaviorally true-depth-5 tasks beyond matched-compute sampling over
the original primitives?

## Experimental unit and splits

The unit is one procedurally generated list-transformation task. Each task has disjoint
visible, hidden-grade, and unlabeled-probe inputs. Programs and macro expansions use a
bounded, parameter-free DSL with exact local execution.

- `reuse`: behaviorally minimum-depth-verified base programs formed from recurring motifs in
  combinations absent from the macro-construction corpus. No macro equals a full evaluation
  program.
- `no_reuse`: a primitive-multiset permutation of a paired reuse task at the same verified
  depth, explicitly excluding the three evaluation-recurrent motifs. Train-only decoy motifs
  may remain, so generic macro usefulness is not removed by construction.
- `smoke`: a separate seed and task set; never included in the full estimate.

Before any model call, reject exact full-program overlap and equal behavioral signatures
between macro-construction and evaluation programs. Record contiguous-window overlap because
subprogram reuse is the treatment, not leakage.

## Libraries and arms

All composite entries expand to two or three base primitives and are execution-equivalent to
their expansion by construction.

1. `base`: base primitives only.
2. `mined`: top supported non-redundant subsequences mined deterministically from the frozen
   train-only proposal view.
3. `mined_hint`: the identical mined subsequences are highlighted in the prompt, but the model
   must emit their expanded base primitives; this arm receives
   the search prior without a callable macro representation.
4. `qwen_ranked`: unique Qwen-proposed subsequences from the identical proposal view, filtered only
   by syntax, exact expansion verification, and train support; ranking is frozen before eval.
5. `random_0` ... `random_4`: five frozen non-selected libraries matched to the mined library
   on count, expansion length, and training-support bins. A separate five-draw placebo ensemble
   is matched to `qwen_ranked` if its profile differs.
6. `designed_ceiling`: generator-known motifs, reported as a non-discovery ceiling.

The solver sees base primitives plus the arm's frozen library. Macro aliases are standardized
(`M0`, `M1`, ...) and definitions show exact expansions, preventing descriptive-name quality
from confounding library content. Qwen-generated names are recorded but not used by the solver.
Every arm receives the same train-only demonstration tasks, in the same order, with programs
expanded to base primitives. No arm-specific demonstration is selected because it contains that
arm's motifs. `mined` and `mined_hint` show identical neutral aliases and definitions; they differ
only in whether aliases are legal output actions. Their contrast isolates callable surface/action
compression rather than highlighted prior information.

## Model and inference

The only model is `Qwen/Qwen3.5-4B` at the repository-pinned revision. Every proposal and
solver completion uses this experiment's copied `src/vllm_runner.py` under `.venv-vllm`.
There is no Transformers inference arm and no backend mixing. Full prompts are batched and
sampling uses vLLM `n`, not a Python loop.

Solver output is a strict `PROGRAM: OP | OP | ...` surface. Programs are expanded to base
primitives before all execution and scoring. Sampling channel, budget, temperature, top-p,
answer cap, and parent run seed are identical across arms. The vLLM runner derives effective
per-record seeds from the required arm-qualified record id, so arms are deterministic but do not
use common random numbers. The base arm is generated to a larger K so
its prefix curve can be compared at the macro arms' actual token costs.

## Metrics

Primary deployable metric:

- selected hidden-all accuracy: choose the earliest sampled, syntactically valid candidate
  with maximal visible-example score; ties use sample order. Abstention is failure. Hidden
  labels never influence selection.

Secondary metrics:

- hidden oracle coverage@K (oracle-only, never called deployable);
- parse rate, valid-program rate, visible-pass rate, abstention, and false-visible-pass rate;
- macro-use rate among selected/correct candidates;
- expanded primitive depth and surface-call depth;
- library motif recall, support, compression coverage, and train/eval overlap audits;
- sampled tokens, unique prompt tokens, logical model-input tokens, generation wall time, and
  interpreter calls. Qwen macro-invention cost is reported once and amortized per eval task.

## Confirmatory contrasts and decision rules

The sole primary contrast is `mined - base` on the pooled `reuse` split at K=12. Report a
paired task bootstrap 95% interval and point difference. The macro mechanism clears only if:

1. point improvement is at least +0.10 and the paired interval lower bound is above zero;
2. the improvement remains positive against the base prefix selected at a no-smaller sampled
   plus unique-prompt token budget;
3. at least half of the treatment-only correct selections actually use a macro; and
4. `mined - base` on `no_reuse` is no larger than half the reuse improvement.

`mined_hint` distinguishes mechanisms. A callable-chunking verdict additionally requires
`mined - mined_hint >= +0.05` with a paired interval lower bound above zero. If it ties
`mined`, highlighting a verified search prior is sufficient and callable chunking is not
established. `mined_hint - base` estimates the highlighted-prior effect directly.

Learned recurrence additionally requires `mined - random_mean >= +0.05` with a positive
task-and-library-draw interval. The complete callable-abstraction verdict is the conjunction of
the base, hint, and random gates; no favorable single contrast substitutes for the others.

Qwen-specific value is a separate secondary verdict. `qwen_ranked` clears only if it contains
exactly the frozen target of eight unique supported verified entries, exceeds its independently
matched random ensemble by at least +0.05 with a positive interval, finishes within -0.05 of
`mined`, and at least two correct selections are uniquely enabled by Qwen-exclusive entries.
Otherwise the result is ranking-only recovery or proposal-construction failure, not invention.

All other arm and slice comparisons are descriptive. Significant-versus-nonsignificant arm
comparisons are not used as evidence of a difference.

## Gates and branches

CPU gate, before GPU:

- all evaluation tasks pass behavioral min-depth and split-manifest checks;
- no macro is a full evaluation program;
- designed macros reduce median surface depth by at least two calls on `reuse` and by no more
  than 0.5 calls on `no_reuse`.

Smoke gate, before full vLLM generation:

- at least 0.50 solver parse rate across base and designed-ceiling smoke arms;
- at least 0.50 parse rate within each of those arms, at least two valid macro-using designed
  candidates, answer truncation below 0.05, and designed is not below base oracle coverage on the
  smoke set. Failure means repair the interface on a new smoke set and refreeze; never tune on
  full evaluation outputs.

If any relevant full arm force-closes above 0.50, run every corresponding comparison arm at the
precommitted 1536-token budget on a frozen 20-task subset before making a mechanism claim, or label
the result budget-confounded.

Branches:

- If `designed_ceiling` has no oracle advantage in the full run, the experiment diagnoses an
  unusable macro interface; do not interpret mined/Qwen differences as abstraction quality.
- If oracle coverage rises but deployable accuracy does not, selection is the bottleneck.
- If random ties mined, shorter syntax/additional inventory is sufficient and semantic reuse
  is not supported.
- If mined wins but Qwen fails the proposal gate, verified tool mining works but model-driven
  invention does not.
- If reuse and no-reuse improve equally, reject the recurring-abstraction mechanism.

## Known scope

This tests reuse of composite abstractions in a fresh procedural DSL, not universal concept
formation. A positive result may be a systems capability from verified tools plus Qwen rather
than a new capability stored in the weights. That distinction will remain explicit.
