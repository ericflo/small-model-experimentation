# Backlog

## Next Experiments

- Compare uncertainty, disagreement, information gain, and learned policies on the same acquisition pool.
- Generate expected-output-free tests where candidates can be compared by agreement or invariants.
- Train family-aware acquisition policies for date/time, numeric, table, and code tasks.
- Measure acquisition value under strict token or tool budgets.
- Feed acquired evidence into selector training rather than direct prompting only.

## Required Controls

- Random acquisition.
- Input-diversity acquisition.
- Fixed-order acquisition.
- Oracle among tested acquisitions, clearly labeled.

## Stop Conditions

Do not count an acquisition policy as successful if it helps only because it leaks hidden labels or increases budget without improving selection quality per unit evidence.
