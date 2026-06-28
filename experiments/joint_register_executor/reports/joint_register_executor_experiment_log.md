# Joint Register Executor Experiment Log

## Objective

Test whether a latent recurrent runtime can execute two-register modular programs that include cross-register operations. The key change is that the hidden state is a categorical distribution over joint register pairs `(A, B)`, rather than separate marginal distributions for `A` and `B`.

The experiment is standalone. It asks whether exact algorithmic accuracy rises when the recurrent step budget `K` reaches the program length `L`, including on lengths longer than the training horizon.

## Hypothesis

If a joint latent state is necessary for cross-register programs, then:

1. A joint-state recurrent executor should learn exact execution on programs containing both constant and cross-register operations.
2. Accuracy should be poor when `K < L` and high when `K >= L`.
3. The threshold pattern should hold for held-out lengths beyond training.
4. A marginal-state recurrent control should struggle because it loses correlations between `A` and `B`.
5. A static one-shot compiler should not show the same K-dependent length generalization.

## Task

Programs operate over two registers modulo `p`.

Operations:

- `A=A+c`
- `A=A-c`
- `B=B+c`
- `B=B-c`
- `A=A+B`
- `B=B+A`
- `A=A-B`
- `B=B-A`

Each generated example contains random initial registers, a random program, exact trace targets after every prefix, and an exact final pair target.

## Models

### Joint Recurrent Executor

The hidden state is a categorical distribution over all `(A, B)` pairs. Each recurrent step applies a learned op-conditioned transition matrix:

- constant operations use the instruction argument as the transition selector
- `A` cross-register operations use current `B` as the selector
- `B` cross-register operations use current `A` as the selector

This keeps the transition table compact at `8 * p * p * p` parameters while retaining joint correlations.

### Marginal Recurrent Control

The control uses the same transition parameterization and recurrent schedule, but it stores only separate distributions over `A` and `B`. It can model marginal effects but not correlations induced by cross-register operations.

### Static Control

The static control embeds the whole program and initial state, then predicts the final pair in one shot.

## Planned Sequence

1. Smoke test at small modulus and a few steps to catch shape/runtime errors.
2. Pilot joint-state run at small modulus to see whether the mechanism learns.
3. Matched marginal and static controls.
4. Main joint-state run at larger modulus with train lengths 1-8 and eval lengths 4/8/12/16/24.
5. Analyze threshold curves and write a standalone report.

## Smoke Test

Run: `../runs/smoke_joint_mod7`

Configuration:

- mode: joint
- modulus: 7
- operation family: full
- train length max: 3
- train steps: 2
- eval lengths: 2, 3

Result: the script compiled, trained, evaluated, wrote metrics, and saved checkpoints under `../../../large_artifacts/joint_register_executor/checkpoints/smoke_joint_mod7`.

Interpretation: shape logic, trace generation, recurrent updates, metrics, and separated checkpoint writing are functional. The run is intentionally too short and small to test learning.

## Pilot 1: Joint Executor, Full Operations, Modulus 11

Run: `../runs/pilot_joint_mod11`

Configuration:

- mode: joint
- modulus: 11
- operation family: full
- train lengths: 1-6
- eval lengths: 3, 6, 9, 12
- recurrent budgets: `K=0,1,2,3,6,9,12`
- eval examples: 1,024 per length
- training steps: 300

Result: the joint executor learned rapidly. By step 100 it already showed the intended threshold curve:

- `L=3`: 100% exact pair accuracy at `K>=3`
- `L=6`: 100% exact pair accuracy at `K>=6`
- `L=9`: 100% exact pair accuracy at `K>=9`
- `L=12`: 100% exact pair accuracy at `K=12`

For `K < L`, pair accuracy stayed near chance, with partial-register accuracy rising as more prefixes were executed.

Step 300 reproduced the same threshold pattern exactly. This is the first positive evidence that the joint latent state can learn reusable transitions for cross-register programs and length-generalize beyond the training horizon.

## Control 1: Marginal Executor, Full Operations, Modulus 11

Run: `../runs/control_marginal_mod11`

Configuration matched Pilot 1 except `mode=marginal`.

Result: the marginal executor also reached 100% exact pair accuracy at `K>=L`, including held-out lengths 9 and 12.

Interpretation: this control changed the experimental diagnosis. With an exact initial pair, the latent state is always a point mass. A marginal state can carry a point-mass `A` and point-mass `B` separately, so cross-register operations do not actually require a joint distribution. The current task proves recurrent transition execution, but it does not isolate the need for joint latent state.

Next step: revise the task to use a correlated belief state. The model will receive an initial relation such as `B=A+d` with `A` unknown, so the hidden state must track a line of possible `(A,B)` pairs. The target becomes the final belief distribution after executing the program. A joint executor can represent this distribution; a marginal executor loses the correlation because both marginals are uniform.

## Belief-State Task Revision

The revised task initializes a correlated belief state:

```text
B = A + d (mod p), with A unknown
```

The support contains `p` possible `(A,B)` pairs. Programs then transform this support through the same full operation set. The target at each recurrent step is the exact final support distribution, not one sampled pair.

Metrics:

- `top1_on_support`: whether the model's highest-probability pair lies inside the correct final support.
- `target_mass`: total probability assigned to the correct final support.
- `target_nll`: mean negative log probability across support elements.

For modulus 11, a factorized uniform joint gives about `1/11 = 9.1%` target mass.

## Pilot 2: Joint Belief Executor, Modulus 11

Run: `../runs/pilot_belief_joint_mod11`

Configuration:

- mode: joint
- task: belief_line
- modulus: 11
- operation family: full
- train lengths: 1-6
- eval lengths: 3, 6, 9, 12
- recurrent budgets: `K=0,1,2,3,6,9,12`
- eval examples: 1,024 per length
- training steps: 300

Result at step 300:

- `L=3`: `K=3` reached 100% top-1-on-support and 97.5% target mass.
- `L=6`: `K=6` reached 100% top-1-on-support and 95.2% target mass.
- `L=9`: `K=9` reached 100% top-1-on-support and 92.9% target mass.
- `L=12`: `K=12` reached 100% top-1-on-support and 90.6% target mass.

For `K<L`, target mass stayed near the 9.1% factorized/uniform baseline except for partial-prefix cases.

Interpretation: this is the first clean positive result for correlated latent belief execution. The joint state tracks the support and the K threshold generalizes beyond the training horizon.

## Control 2: Marginal Belief Executor, Modulus 11

Run: `../runs/control_belief_marginal_mod11`

Configuration matched Pilot 2 except `mode=marginal`.

Result: the marginal control stayed flat throughout training. At step 300, target mass was about 9.1% for every length and every K, with target NLL around 4.796.

Interpretation: this is the expected representational failure. The marginal state sees uniform `A` and uniform `B` but cannot store the relation between them.

## Control 3: Static Belief Compiler, Modulus 11

Run: `../runs/control_belief_static_mod11`

Configuration matched Pilot 2 except `mode=static`; the static model receives the relation parameter and program, then predicts the final support in one shot.

Result at step 300:

- `L=3`: 25.3% top-1-on-support and 18.4% target mass.
- `L=6`: 11.0% top-1-on-support and 10.1% target mass.
- `L=9`: 9.9% top-1-on-support and 9.1% target mass.
- `L=12`: 10.1% top-1-on-support and 9.1% target mass.

Interpretation: the static model learns some short-length signal but does not length-generalize. It does not reproduce the recurrent threshold behavior.

## Main Run: Joint Belief Executor, Modulus 31

Run: `../runs/main_belief_joint_mod31`

Configuration:

- mode: joint
- task: belief_line
- modulus: 31
- operation family: full
- train lengths: 1-8
- eval lengths: 4, 8, 12, 16, 24
- recurrent budgets: `K=0,1,2,4,8,12,16,24`
- eval examples: 1,024 per length
- training steps: 500

Result at step 500:

- `L=4`: `K=4` reached 100% top-1-on-support and 97.8% target mass.
- `L=8`: `K=8` reached 100% top-1-on-support and 95.7% target mass.
- `L=12`: `K=12` reached 100% top-1-on-support and 93.6% target mass.
- `L=16`: `K=16` reached 100% top-1-on-support and 91.6% target mass.
- `L=24`: `K=24` reached 100% top-1-on-support and 87.4% target mass.

For `K<L`, target mass stayed near the `1/31 = 3.2%` baseline except when a partial prefix overlapped the final support. This is the main positive result.

## Scaled Control: Marginal Belief Executor, Modulus 31

Run: `../runs/control_belief_marginal_mod31`

Configuration matched the main run except `mode=marginal` and 200 training steps.

Result: target mass stayed at 3.2% for every length and every K. Target NLL stayed around 6.87, matching the uniform joint baseline.

Interpretation: the marginal recurrent state cannot represent the correlated line support.

## Scaled Control: Static Belief Compiler, Modulus 31

Run: `../runs/control_belief_static_mod31`

Configuration matched the main run except `mode=static` and 300 training steps.

Result at step 300:

- `L=4`: 10.8% top-1-on-support and 7.9% target mass.
- `L=8`: 3.8% top-1-on-support and 3.5% target mass.
- `L=12`: 3.0% top-1-on-support and 3.2% target mass.
- `L=16`: 2.5% top-1-on-support and 3.2% target mass.
- `L=24`: 3.3% top-1-on-support and 3.2% target mass.

Interpretation: the static compiler did not solve the scaled belief task and did not length-generalize beyond the training horizon.
