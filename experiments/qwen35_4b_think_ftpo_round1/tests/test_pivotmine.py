"""Selftests for outcome-conditioned pivot mining."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pivotmine  # noqa: E402


def _seqs(prefix, branches):
    """branches: list of (branch_token, tail, count, won)."""
    sequences, successes = [], []
    for tok, tail, count, won in branches:
        for i in range(count):
            sequences.append(list(prefix) + [tok] + [t + i for t in tail])
            successes.append(won)
    return sequences, successes


def test_clean_pivot_detected():
    prefix = list(range(100, 120))  # depth 20 shared prefix
    sequences, successes = _seqs(prefix, [
        (7, [1, 2, 3], 4, True),    # winning branch
        (9, [4, 5, 6], 4, False),   # losing branch
    ])
    rows = pivotmine.mine_pivots(sequences, successes)
    assert len(rows) == 1
    row = rows[0]
    assert row.depth == 20
    assert row.rejected_id == 9
    assert row.chosen_ids == (7,)
    assert row.gap == 1.0
    assert row.prefix == tuple(prefix)


def test_min_depth_excludes_early_divergence():
    prefix = list(range(100, 105))  # depth 5 < 16
    sequences, successes = _seqs(prefix, [
        (7, [1, 2], 4, True),
        (9, [4, 5], 4, False),
    ])
    assert pivotmine.mine_pivots(sequences, successes) == []


def test_small_gap_excluded():
    prefix = list(range(100, 120))
    # 2/4 vs 1/4 success: gap 0.25 < 0.5
    sequences, successes = [], []
    for i, won in enumerate([True, True, False, False]):
        sequences.append(prefix + [7, 50 + i])
        successes.append(won)
    for i, won in enumerate([True, False, False, False]):
        sequences.append(prefix + [9, 70 + i])
        successes.append(won)
    assert pivotmine.mine_pivots(sequences, successes) == []


def test_single_rollout_branch_ineligible():
    prefix = list(range(100, 120))
    sequences, successes = _seqs(prefix, [
        (7, [1], 1, True),   # only one rollout — below min_branch_rollouts
        (9, [4], 4, False),
    ])
    assert pivotmine.mine_pivots(sequences, successes) == []


def test_multiple_chosen_branches():
    prefix = list(range(100, 120))
    sequences, successes = _seqs(prefix, [
        (7, [1, 2], 2, True),
        (8, [3, 4], 2, True),
        (9, [5, 6], 3, False),
    ])
    rows = pivotmine.mine_pivots(sequences, successes)
    assert len(rows) == 1
    assert rows[0].rejected_id == 9
    assert rows[0].chosen_ids == (7, 8)
    assert rows[0].n_chosen == 4


def test_max_nodes_and_ordering():
    # Two nested pivots: shallow with gap 1.0, deeper with gap 1.0 -> tie on
    # gap resolves toward the DEEPER node first.
    prefix = list(range(100, 120))
    sequences, successes = [], []
    # Winning side splits again deeper with its own clean pivot.
    for i in range(2):
        sequences.append(prefix + [7] + list(range(200, 210)) + [31, 40 + i])
        successes.append(True)
    for i in range(2):
        sequences.append(prefix + [7] + list(range(200, 210)) + [33, 60 + i])
        successes.append(False)
    for i in range(4):
        sequences.append(prefix + [9, 80 + i])
        successes.append(False)
    rows = pivotmine.mine_pivots(sequences, successes, max_nodes=2)
    assert len(rows) == 2
    assert rows[0].depth > rows[1].depth  # deeper tie-break first
    assert rows[1].rejected_id == 9
    assert rows[0].rejected_id == 33


def test_all_same_outcome_yields_nothing():
    prefix = list(range(100, 120))
    sequences, successes = _seqs(prefix, [
        (7, [1, 2], 4, True),
        (9, [4, 5], 4, True),
    ])
    assert pivotmine.mine_pivots(sequences, successes) == []


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
