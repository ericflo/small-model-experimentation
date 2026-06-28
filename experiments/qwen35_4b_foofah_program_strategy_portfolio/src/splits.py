from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def family_splits(rows: list[dict[str, Any]], seed: int = 271828) -> dict[str, list[str]]:
    families = sorted({row["family"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(families)
    n = len(families)
    train_n = int(round(0.6 * n))
    dev_n = int(round(0.2 * n))
    return {
        "train": sorted(families[:train_n]),
        "dev": sorted(families[train_n : train_n + dev_n]),
        "test": sorted(families[train_n + dev_n :]),
    }


def split_for_family(splits: dict[str, list[str]], family: str) -> str:
    for split, families in splits.items():
        if family in families:
            return split
    raise KeyError(f"family {family!r} is not present in splits")


def rows_by_split(rows: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[split_for_family(splits, row["family"])].append(row)
    return dict(grouped)

