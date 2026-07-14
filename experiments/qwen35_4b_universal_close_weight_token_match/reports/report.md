# Close-Weighted Universal Commit Seam Report

**Status:** all three arms trained; paired local evaluation pending

## Design result

The outcome-free construction is feasible and frozen. All three continuations start
from the authenticated designed160 parent and receive 320 rows, 286,814 forward
tokens, 40 optimizer steps, and zero skipped rows. `standard_xi` and `close_xi` use
the same SHA-256 `12fc613b...14f00` stream. The treatment changes only assigned
autonomous-close weights on 80 fresh execute/induct rows; active replay exactly
matches their total forward-token exposure.

## Model results

All three arms completed exactly 40 updates with zero skips. Replay, standard, and
close train losses were 0.4477, 0.6882, and 0.6822; wrapper wall times were 303.44,
302.15, and 287.13 seconds; weights hashes were `ca5601cd...59d78`,
`271569fd...3569c`, and `16e9dc75...3c179`. These are authenticated training
receipts, not behavioral evidence. Fresh local seed 88,006 has not been evaluated,
and conditional aggregate seed 78,136 remains sealed.

## Interpretation boundary

Construction feasibility is not evidence for the hypothesis. Results will be added
only after all three registered arms have completed and the paired local receipt has
been written. A local or quick-pilot pass will remain below the universal-claim bar
until independently confirmed against matched-compute sampling.
