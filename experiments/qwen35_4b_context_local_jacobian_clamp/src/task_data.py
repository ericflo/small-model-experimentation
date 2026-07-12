"""Fresh prompt-local lookup tasks; no benchmark imports or reads."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any


CONCEPTS = (
    "cat", "dog", "horse", "tiger", "apple", "lemon", "river", "ocean",
    "silver", "gold", "circle", "square", "winter", "summer", "music", "dance",
    "bread", "glass", "stone", "cloud", "green", "purple", "north", "south",
)
DIGITS = tuple(str(value) for value in range(10))


def shared_prefix(item: dict[str, Any], *, selected: str | None = None) -> str:
    chosen = item["source"] if selected is None else selected
    rows = "\n".join(f"{row['concept']} = {row['digit']}" for row in item["mapping"])
    return f"Lookup table:\n{rows}\nSelected key: {chosen}"


def direct_prompt(item: dict[str, Any], *, selected: str | None = None) -> str:
    return shared_prefix(item, selected=selected) + "\nRepeat the selected key exactly. Key:"


def consequence_prompt(item: dict[str, Any], *, selected: str | None = None) -> str:
    return shared_prefix(item, selected=selected) + "\nReturn its one-digit table value. Value: "


def digit_for(item: dict[str, Any], concept: str) -> str:
    matches = [row["digit"] for row in item["mapping"] if row["concept"] == concept]
    if len(matches) != 1:
        raise ValueError(f"concept {concept!r} is not unique in mapping")
    return matches[0]


def fingerprint(item: dict[str, Any]) -> str:
    payload = {
        "mapping": item["mapping"],
        "source": item["source"],
        "target": item["target"],
        "wrong": item["wrong"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _make_item(rng: random.Random, *, item_id: str, source: str) -> dict[str, Any]:
    if source not in CONCEPTS:
        raise ValueError(f"unknown source concept: {source}")
    other = [concept for concept in CONCEPTS if concept != source]
    target, wrong = rng.sample(other, 2)
    required = {source, target, wrong}
    fillers = rng.sample([concept for concept in other if concept not in required], 5)
    concepts = [source, target, wrong, *fillers]
    rng.shuffle(concepts)
    digits = rng.sample(DIGITS, len(concepts))
    mapping = [
        {"concept": concept, "digit": digit}
        for concept, digit in zip(concepts, digits, strict=True)
    ]
    item = {
        "item_id": item_id,
        "mapping": mapping,
        "source": source,
        "target": target,
        "wrong": wrong,
    }
    item.update({
        "source_digit": digit_for(item, source),
        "target_digit": digit_for(item, target),
        "wrong_digit": digit_for(item, wrong),
    })
    validate_item(item)
    return item


def _balanced_items(*, count: int, seed: int, prefix: str) -> list[dict[str, Any]]:
    if count % len(CONCEPTS) != 0:
        raise ValueError("split count must be a multiple of the 24-concept dictionary")
    rng = random.Random(seed)
    sources = list(CONCEPTS) * (count // len(CONCEPTS))
    for start in range(0, count, len(CONCEPTS)):
        block = sources[start : start + len(CONCEPTS)]
        rng.shuffle(block)
        sources[start : start + len(CONCEPTS)] = block
    return [
        _make_item(rng, item_id=f"{prefix}-{index:04d}", source=source)
        for index, source in enumerate(sources)
    ]


def generate_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    seeds = config["seeds"]
    splits = {
        "lens_fit": _balanced_items(
            count=int(data["lens_fit_items"]), seed=int(seeds["lens"]), prefix="lens"
        ),
        "band_selection": _balanced_items(
            count=int(data["band_selection_items"]),
            seed=int(seeds["selection"]),
            prefix="select",
        ),
        "confirmation": _balanced_items(
            count=int(data["confirmation_items"]),
            seed=int(seeds["confirmation"]),
            prefix="confirm",
        ),
    }
    validate_splits(splits)
    return splits


def validate_item(item: dict[str, Any]) -> None:
    mapping = item["mapping"]
    concepts = [row["concept"] for row in mapping]
    digits = [row["digit"] for row in mapping]
    if len(mapping) != 8 or len(set(concepts)) != 8 or len(set(digits)) != 8:
        raise ValueError("lookup table must be a one-to-one eight-row mapping")
    if not set(concepts).issubset(CONCEPTS) or not set(digits).issubset(DIGITS):
        raise ValueError("lookup table contains an unregistered token")
    identities = (item["source"], item["target"], item["wrong"])
    if len(set(identities)) != 3 or not set(identities).issubset(concepts):
        raise ValueError("source, target, and wrong donor must be distinct table concepts")
    expected_digits = tuple(digit_for(item, concept) for concept in identities)
    if len(set(expected_digits)) != 3:
        raise ValueError("source, target, and wrong donor digits must be distinct")
    if expected_digits != (item["source_digit"], item["target_digit"], item["wrong_digit"]):
        raise ValueError("stored digit labels disagree with table")


def validate_splits(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for name, rows in splits.items():
        counts = {concept: 0 for concept in CONCEPTS}
        for row in rows:
            validate_item(row)
            if row["item_id"] in seen_ids:
                raise ValueError(f"duplicate item id across splits: {row['item_id']}")
            seen_ids.add(row["item_id"])
            key = fingerprint(row)
            if key in seen_fingerprints:
                raise ValueError(f"duplicate mapping tuple across splits: {name}/{row['item_id']}")
            seen_fingerprints.add(key)
            counts[row["source"]] += 1
        if len(set(counts.values())) != 1:
            raise ValueError(f"source concepts are not balanced in split {name}")
