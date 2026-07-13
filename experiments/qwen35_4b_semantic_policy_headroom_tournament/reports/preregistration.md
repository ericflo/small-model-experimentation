# Preregistration: semantic-policy headroom tournament

Frozen before any Qwen output.

## Objective

Find verifier-conditioned semantic-policy axes with replicated, nontrivial
failed-test headroom in the learned transaction parent before any further
training. This experiment produces eligibility evidence only.

## Model and runtime

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact merged parent weight SHA-256
  `1cf5fbca317808d6d00225f5cd533c82c7e1602b2b2e5e2da8f4307b01941ba3`.
- Copied vLLM 0.24 backend for both blocks; greedy; 512 think + 512 answer;
  one trajectory × six calls per recovery case.
- Official processes freeze Python hash seed and bytecode writes.

## Substrate

Inferred axes are negative quantity, non-integer quantity, and blank resource.
Each appears as bundle mapping, record dictionary, and tuple sequence. The
issue states the valid input domain and ordinary unknown/insufficient `False`
policy but does not state the malformed exception; public visible tests and
failed-test output reveal it. Three explicit controls state one exception per
axis verbatim.

Headroom A uses seed 88200; B uses 88300. Each contains 12 families × three
repositories × rejected/failed states = 72 cases. Public-content digests must
be unique within each block and disjoint across blocks. Initial/partial must
fail both executable suites and oracle must pass both for every repository.

## Frozen qualification

For each block, compute success by family and scenario from case-level receipts.
An inferred axis is eligible only if:

- failed-test macro success across its three shapes is between .15 and .80,
  inclusive, in both blocks;
- at least two of three shape-family failed-test rates are individually inside
  .15–.80 in both blocks.

The overall qualification additionally requires explicit-control failed-test
success ≥.85 in both blocks; invalid-action and answer-cap rates ≤.05 in both;
and at least one eligible inferred axis. Report rejected-patch rates
diagnostically but do not select on them.

No threshold or family adaptation occurs between A and B. All axes are
preserved. A passing axis licenses a new, separately preregistered curriculum
using disjoint skins/seeds; it does not license training in this directory.

## Firewall and stopping

Only public repository state enters prompts. Hidden tests/results and repairs
stay host-side. Nothing under `benchmarks/` may be read/imported. Menagerie is
hard-disabled. Stop labels:

- `INSTRUMENT_FAIL` if explicit/interface/content gates fail;
- `NO_QUALIFIED_AXIS` if no axis replicates inside the band;
- `HEADROOM_AXIS_QUALIFIED` if all global gates pass and ≥1 axis qualifies.
