# Experiment Log

## 2026-06-28

- Created standalone experiment package for Foofah program-strategy portfolio search.
- Protocol: deterministic family split; search candidate executable-program prompt variants on train/dev families; freeze selected portfolio; evaluate on held-out families.
- Primary comparison: direct JSON versus frozen selected program portfolio, with direct-miss recoveries, direct-correct losses, visible-program oracle, commit precision, and forward-token accounting.
- Smoke run started with eight prompt variants. It was stopped after two cases because the full eight-variant sweep was too slow for the intended staged protocol. The smoke also showed `shape_first`, `aggregation_grouping`, and `transpose_restructure` were repair-heavy or non-incremental on the first two cases, so the pilot candidate set was pruned to five variants: `verified_structural`, `row_column_rule`, `split_fold_unpivot`, `header_aware`, and `cell_parser`.
- Fixed a selector tie-sort bug found by the tiny smoke output and added `--max-cases-per-family` for family-balanced pilots.
- Train pilot completed on one case per train family: direct 9/30, consensus_2 10/30, first-visible 8/30, direct-or-program oracle 12/30.
- Dev pilot completed on one case per dev family: direct 4/10, consensus_2 5/10, direct-or-program oracle 5/10. The frozen selected portfolio uses all five pruned variants with `consensus_2`.
- Held-out test completed on all 50 cases from 10 test families: direct 21/50, selected `consensus_2` 24/50, first-visible 28/50, direct-or-program oracle 29/50.
- Generated final report and figures under `reports/`.
