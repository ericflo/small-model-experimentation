# Backlog

## Next Experiments

- Build a common tool-state schema for direct answer, executable program, verifier result, repair history, and budget.
- Compare learned controllers with simple heuristics on identical pools.
- Train STOP/MORE policies under visible-only labels and evaluate under hidden labels.
- Stress process policies with noisy tools and misleading visible examples.
- Measure whether controllers transfer from table tasks to code or text transformations.

## Required Controls

- Fixed budget baseline.
- Always-stop and always-continue policies.
- Random action policy.
- Oracle policy ceiling.

## Stop Conditions

Do not claim process control progress unless the learned policy beats simple budget heuristics under the same evidence constraints.
