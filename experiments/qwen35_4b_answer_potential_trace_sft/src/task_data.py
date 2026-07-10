"""Procedural split construction and verifier-equivalent answer targets.

No code below imports repository benchmarks.  Family imports are restricted
to the experiment-local copy, and held families live behind an explicit
evaluation-only loader.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from gym import base
from gym.families import TRAIN_FAMILIES, load as load_train_family
from io_utils import canonical_json, write_json, write_jsonl

HELDOUT_FAMILIES = ("brinework", "spindle")
MAX_EQUIVALENT_ANSWERS = 64

SPLIT_SPECS: dict[str, dict[str, Any]] = {
    "calibration": {
        "seed": 61001,
        "families": TRAIN_FAMILIES,
        "levels": (1, 2),
        "total": 64,
    },
    "train": {
        "seed": 62001,
        "families": TRAIN_FAMILIES,
        "levels": (1, 2),
        "per_family_level": 30,
    },
    "iid_eval": {
        "seed": 63001,
        "families": TRAIN_FAMILIES,
        "levels": (1, 2),
        "per_family_level": 20,
    },
    "held_family_eval": {
        "seed": 64001,
        "families": HELDOUT_FAMILIES,
        "levels": (1, 2),
        "per_family_level": 25,
        "heldout": True,
    },
    "hard_eval": {
        "seed": 65001,
        "families": TRAIN_FAMILIES,
        "levels": (3,),
        "per_family_level": 10,
    },
}


def prompt_digest(prompt: str) -> str:
    normalized = prompt.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _heldout_loader(name: str):
    # The import is deliberately inside this evaluation-only function.  A
    # training-stage process can import task_data without importing held code.
    from gym.heldout_families import load

    return load(name)


def load_family(name: str, *, heldout: bool = False):
    return _heldout_loader(name) if heldout else load_train_family(name)


def oracle_content(module: Any, item: dict[str, Any]) -> str:
    answer = base.extract_answer(module.oracle_atom(item))
    if not answer:
        raise ValueError(f"oracle emitted no ANSWER content for {item['id']}")
    return answer.strip()


def _stallwright_variants(item: dict[str, Any]) -> list[str]:
    """Enumerate canonical optimal ID sets, not formatting permutations."""
    gold = item["gold"]
    stalls = gold["stalls"]
    variants: list[str] = []
    for mask in range(1, 1 << len(stalls)):
        ids = [stalls[index]["id"] for index in range(len(stalls)) if mask & (1 << index)]
        rendered = ", ".join(ids)
        if math.isclose(
            load_train_family("stallwright").score_atom(item, f"ANSWER: {rendered}"),
            1.0,
        ):
            variants.append(rendered)
    return sorted(set(variants))


def _glyphgate_variants(module: Any, item: dict[str, Any]) -> list[str]:
    gold = item["gold"]
    if gold["mode"] == "predict":
        return [gold["output"]]
    target = module._parse_glyphs(gold["target"], gold["length"])  # noqa: SLF001
    if target is None:
        raise ValueError(f"invalid generated glyph target for {item['id']}")
    return sorted(module._fmt(value) for value in module._preimages(gold["rule"], target))  # noqa: SLF001


def _burrowmaze_variants(item: dict[str, Any]) -> list[str]:
    gold = item["gold"]
    if gold["kind"] != "route":
        return [str(gold["value"])]
    opposite = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
    }
    directions = tuple(opposite)
    adjacency: dict[str, dict[str, str]] = {}
    for a, direction, b in gold["edges"]:
        adjacency.setdefault(a, {})[direction] = b
        adjacency.setdefault(b, {})[opposite[direction]] = a
    variants: list[str] = []
    for steps in itertools.product(directions, repeat=gold["length"]):
        node = gold["src"]
        for direction in steps:
            node = adjacency.get(node, {}).get(direction, "")
            if not node:
                break
        if node == gold["dst"]:
            variants.append(", ".join(steps))
    return sorted(variants)


def equivalent_answers(module: Any, item: dict[str, Any]) -> tuple[list[str], str | None]:
    """Return finite canonical verifier-equivalent contents and exclusion reason.

    The verifier may accept infinitely many harmless formatting variants.  We
    enumerate semantic alternatives in one registered canonical rendering.
    Stallwright is excluded even when its optimal set is unique because its
    prompt explicitly makes arbitrary permutations and names valid, producing
    a combinatorial formatting event rather than a clean short target.
    """
    family = item["family"]
    canonical = oracle_content(module, item)
    if family == "stallwright":
        variants = _stallwright_variants(item)
        reason = f"combinatorial_order_and_alias_equivalence:{len(variants)}_optimal_sets"
        return variants[:MAX_EQUIVALENT_ANSWERS], reason
    if family == "glyphgate":
        variants = _glyphgate_variants(module, item)
    elif family == "loomfix" and item["gold"]["kind"] == "loc":
        variants = [str(value) for value in item["gold"]["fixable"]]
    elif family == "ferrier":
        variants = list(item["gold"]["valid"])
    elif family == "burrowmaze":
        variants = _burrowmaze_variants(item)
    else:
        variants = [canonical]
    variants = sorted({value.strip() for value in variants if value.strip()})
    if canonical not in variants:
        variants.insert(0, canonical)
    invalid = [
        value for value in variants if module.score_atom(item, f"ANSWER: {value}") < 1.0
    ]
    if invalid:
        raise ValueError(f"invalid equivalent answer(s) for {item['id']}: {invalid[:3]}")
    if not variants:
        return [], "no_finite_equivalent_rendering"
    if len(variants) > MAX_EQUIVALENT_ANSWERS:
        return variants[:MAX_EQUIVALENT_ANSWERS], f"equivalence_set_too_large:{len(variants)}"
    return variants, None


def answer_shape(answer: str) -> str:
    value = answer.strip()
    if re.fullmatch(r"-?\d+", value):
        return "integer"
    if "(" in value and value.endswith(")"):
        return "call"
    if "," in value:
        return "list"
    if value.count("-") >= 2:
        return "dash_sequence"
    return "word"


def procedural_decoy(module: Any, item: dict[str, Any], canonical: str) -> str | None:
    candidates: list[str] = []
    parsed_int = base.canon_int(canonical)
    if re.fullmatch(r"-?\d+", canonical.strip()) and parsed_int is not None:
        candidates.extend((str(parsed_int + 1), str(parsed_int - 1), "0"))
    else:
        candidates.extend(("impossible", "none", "north", "za-ke-ro"))
    for candidate in candidates:
        if candidate != canonical and module.score_atom(item, f"ANSWER: {candidate}") == 0.0:
            return candidate
    return None


def _decorate_item(
    item: dict[str, Any], *, split: str, seed: int, heldout: bool
) -> dict[str, Any]:
    module = load_family(item["family"], heldout=heldout)
    canonical = oracle_content(module, item)
    variants, excluded_reason = equivalent_answers(module, item)
    if module.score_atom(item, f"ANSWER: {canonical}") != 1.0:
        raise ValueError(f"oracle answer fails verifier for {item['id']}")
    return {
        **item,
        "split": split,
        "generator_seed": seed,
        "prompt_sha256": prompt_digest(item["prompt"]),
        "canonical_answer": canonical,
        "answer_variants": variants,
        "answer_shape": answer_shape(canonical),
        "potential_scorable": excluded_reason is None,
        "potential_exclusion_reason": excluded_reason,
        "procedural_decoy": procedural_decoy(module, item, canonical),
    }


def generate_split(name: str) -> list[dict[str, Any]]:
    if name not in SPLIT_SPECS:
        raise KeyError(f"unknown split: {name!r}")
    spec = SPLIT_SPECS[name]
    heldout = bool(spec.get("heldout", False))
    cells = list(itertools.product(spec["families"], spec["levels"]))
    counts: dict[tuple[str, int], int]
    if "total" in spec:
        base_count, remainder = divmod(spec["total"], len(cells))
        counts = {
            cell: base_count + (index < remainder)
            for index, cell in enumerate(cells)
        }
    else:
        counts = {cell: int(spec["per_family_level"]) for cell in cells}
    rows: list[dict[str, Any]] = []
    for family, level in cells:
        module = load_family(family, heldout=heldout)
        generated = module.gen_atoms(spec["seed"], level, counts[(family, level)])
        rows.extend(
            _decorate_item(item, split=name, seed=spec["seed"], heldout=heldout)
            for item in generated
        )
    if len(rows) != spec.get("total", sum(counts.values())):
        raise AssertionError(f"split {name} generated wrong row count")
    return rows


def audit_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    dimensions = {
        "id": lambda row: row["id"],
        "prompt": lambda row: row["prompt"],
        "prompt_sha256": lambda row: row["prompt_sha256"],
        "family_generator_seed": lambda row: (row["family"], row["generator_seed"]),
    }
    collisions: dict[str, list[dict[str, Any]]] = {}
    names = list(splits)
    for dimension, getter in dimensions.items():
        found: list[dict[str, Any]] = []
        for left_index, left in enumerate(names):
            left_values = {canonical_json(getter(row)) for row in splits[left]}
            for right in names[left_index + 1 :]:
                right_values = {canonical_json(getter(row)) for row in splits[right]}
                overlap = sorted(left_values & right_values)
                if overlap:
                    found.append({"left": left, "right": right, "examples": overlap[:5]})
        collisions[dimension] = found
    # Seed overlap is prohibited across splits but repeats within a split are expected.
    passed = not any(collisions.values())
    if not passed:
        raise ValueError(f"cross-split firewall collision: {json.dumps(collisions, indent=2)}")
    return {
        "passed": True,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "scorable_counts": {
            name: sum(bool(row["potential_scorable"]) for row in rows)
            for name, rows in splits.items()
        },
        "family_counts": {
            name: dict(sorted(Counter(row["family"] for row in rows).items()))
            for name, rows in splits.items()
        },
        "collisions": collisions,
    }


def build_all(output_dir: Path) -> dict[str, Any]:
    splits = {name: generate_split(name) for name in SPLIT_SPECS}
    audit = audit_splits(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "split_specs": SPLIT_SPECS,
        "audit": audit,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
