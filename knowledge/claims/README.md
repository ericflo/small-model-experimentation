# Claims Ledger

Claims are hand-curated statements that future experiments can support, weaken, or overturn. They should link to evidence and name their status.

The structured source of truth is [claim_ledger.json](claim_ledger.json). Generated navigation lives in [index.md](index.md) and [index.csv](index.csv). Edit the JSON ledger, then run `make check`.

Allowed statuses:

- `Confirmed`: directly supported by result-bearing experiments.
- `Promising`: supported by pilots or partial evidence.
- `Negative`: tested and failed under recorded conditions.
- `Open`: plausible and not adequately tested.
- `Retired`: contradicted enough that it should not guide new work without new evidence.

Use one narrative file per substantial claim cluster when the ledger grows, but keep every durable claim represented in `claim_ledger.json` so validation can check programs and evidence references.
