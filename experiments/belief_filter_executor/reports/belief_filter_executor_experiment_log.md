# Belief Filter Executor Experiment Log

## Objective

Test whether a latent recurrent runtime can maintain, transform, and condition a correlated belief state over two modular registers.

The hidden state is a distribution over `(A, B)` pairs. Programs contain arithmetic updates and observation/filter instructions. The experiment measures whether additional recurrent steps improve exact belief-state tracking when the step budget `K` reaches program length `L`.

## Hypothesis

If the runtime has a usable joint belief state, then:

1. It should learn arithmetic transitions and observation filters from dense belief supervision.
2. Target-support mass should be low when `K < L` and high when `K >= L`.
3. The threshold should generalize to held-out program lengths longer than training.
4. A marginal recurrent control should fail because it cannot represent correlations between `A` and `B`.
5. A static compiler should not reproduce the same length-generalizing K threshold.

## Task

Initial belief:

```text
B = A + d (mod p), with A unknown
```

Programs use:

- arithmetic operations over `A` and `B`
- observation filters of the form `A % m = r`
- observation filters of the form `B % m = r`

Observation residues are sampled from the current support so the target support is never empty.

## Models

### Joint Filter Executor

The primary model stores a categorical distribution over all `(A,B)` pairs. At each recurrent step, it applies the next learned arithmetic transition or learned observation likelihood, then normalizes the belief state.

### Marginal Filter Control

The marginal control stores separate distributions over `A` and `B`. It can condition each marginal but cannot preserve pairwise correlations.

### Static Compiler Control

The static control receives the initial relation and whole program, then predicts the final belief support in one shot.

## Planned Sequence

1. Smoke test at tiny modulus.
2. Pilot joint filter executor at small modulus.
3. Matched marginal and static controls at small modulus.
4. Main scaled run at modulus 31.
5. Matched scaled controls.
6. Aggregate metrics, generate figures, and write a standalone report.

## Smoke Test

Run: `../runs/smoke_joint_mod7`

Configuration:

- mode: joint
- modulus: 7
- observation modulus: 3
- observation probability: 0.4
- train length max: 3
- train steps: 2
- eval lengths: 2, 3

Result: the script compiled, trained, evaluated, wrote metrics, and saved checkpoints under `../../../large_artifacts/belief_filter_executor/checkpoints/smoke_joint_mod7`.

Interpretation: arithmetic transitions, observation filters, belief targets, evaluation metrics, and separated checkpoint writing are functional. The run is intentionally too short to test learning.

## Pilot 1: Joint Filter Executor, Modulus 11

Run: `../runs/pilot_joint_mod11`

Configuration:

- mode: joint
- modulus: 11
- observation modulus: 4
- observation probability: 0.3
- train lengths: 1-6
- eval lengths: 3, 6, 9, 12
- recurrent budgets: `K=0,1,2,3,6,9,12`
- eval examples: 1,024 per length
- training steps: 300

Result at step 300:

- `L=3`: `K=3` reached 100% top-1-on-support and 96.2% target mass.
- `L=6`: `K=6` reached 100% top-1-on-support and 94.1% target mass.
- `L=9`: `K=9` reached 100% top-1-on-support and 93.0% target mass.
- `L=12`: `K=12` reached 100% top-1-on-support and 92.2% target mass.

For `K<L`, target mass stayed far below the sufficient-K values. This is the first positive result that the joint runtime can combine arithmetic transitions and observation filters.

## Control 1: Marginal Filter Executor, Modulus 11

Run: `../runs/control_marginal_mod11`

Configuration matched Pilot 1 except `mode=marginal`.

Result at step 300:

- `L=3`: best sufficient-K target mass 10.7%.
- `L=6`: best sufficient-K target mass 10.2%.
- `L=9`: best sufficient-K target mass 8.9%.
- `L=12`: best sufficient-K target mass 7.9%.

Interpretation: the marginal model can learn some filtering behavior, but it cannot reconstruct the joint support. It remains far below the joint executor's 92-96% target mass.

## Control 2: Static Filter Compiler, Modulus 11

Run: `../runs/control_static_mod11`

Configuration matched Pilot 1 except `mode=static`.

Result at step 300:

- `L=3`: 17.5% target mass.
- `L=6`: 8.2% target mass.
- `L=9`: 3.4% target mass.
- `L=12`: 2.3% target mass.

Interpretation: the static compiler learns some short-length signal but does not solve longer held-out programs.

## Main Run: Joint Filter Executor, Modulus 31

Run: `../runs/main_joint_mod31`

Configuration:

- mode: joint
- modulus: 31
- observation modulus: 5
- observation probability: 0.3
- train lengths: 1-8
- eval lengths: 4, 8, 12, 16, 24
- recurrent budgets: `K=0,1,2,4,8,12,16,24`
- eval examples: 1,024 per length
- training steps: 500

Result at step 500:

- `L=4`: `K=4` reached 100% top-1-on-support and 96.2% target mass.
- `L=8`: `K=8` reached 100% top-1-on-support and 93.9% target mass.
- `L=12`: `K=12` reached 100% top-1-on-support and 92.4% target mass.
- `L=16`: `K=16` reached 100% top-1-on-support and 91.7% target mass.
- `L=24`: `K=24` reached 100% top-1-on-support and 91.3% target mass.

For `K<L`, target mass stayed near zero on long programs, with only partial-prefix gains. This is the main positive result.

## Scaled Control: Marginal Filter Executor, Modulus 31

Run: `../runs/control_marginal_mod31`

Configuration matched the main run except `mode=marginal` and 300 training steps.

Result at step 300:

- `L=4`: best sufficient-K target mass 3.9%.
- `L=8`: best sufficient-K target mass 2.9%.
- `L=12`: best sufficient-K target mass 2.3%.
- `L=16`: best sufficient-K target mass 1.7%.
- `L=24`: best sufficient-K target mass 1.4%.

Interpretation: the marginal model learns weak filtering signals but cannot preserve the joint support.

## Scaled Control: Static Filter Compiler, Modulus 31

Run: `../runs/control_static_mod31`

Configuration matched the main run except `mode=static` and 300 training steps.

Result at step 300:

- `L=4`: 5.4% target mass.
- `L=8`: 2.1% target mass.
- `L=12`: 0.7% target mass.
- `L=16`: 0.3% target mass.
- `L=24`: 0.2% target mass.

Interpretation: the static compiler does not solve the scaled filtered belief task and does not length-generalize.
