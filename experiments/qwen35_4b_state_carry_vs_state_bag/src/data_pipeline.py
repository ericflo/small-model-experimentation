"""Deterministic data construction and contamination receipts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import SOURCE_CONTRACT_VERSION, source_contract_sha256
from .substrate import generate_counterfactual_pair, generate_example, verify_example


def data_contract_sha256(
    config: Mapping[str, Any], *, source_digest: str | None = None
) -> str:
    source_digest = source_digest or source_contract_sha256()
    payload = {
        "experiment_id": config["experiment_id"],
        "substrate": config["substrate"],
        "state_token": config["architecture"]["state_token"],
        "state_slots": config["architecture"]["state_slots"],
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_digest,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # gzip.open embeds wall-clock time and the output filename, so two builds of
    # identical rows otherwise get different hashes. Freeze both header fields.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _generate_rows(
    *,
    count: int,
    seed: int,
    split: str,
    families: Sequence[str],
    templates: Sequence[str],
    depths: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    substrate = config["substrate"]
    architecture = config["architecture"]
    rows = []
    for index in range(count):
        item_seed = seed * 10_000_000 + index
        # Query kind is an explicit grid dimension, not an RNG side effect.
        # Adjacent items cover both queries for the same family/template/depth
        # cell, giving exact balance for every (necessarily even) split size.
        query_kind = ("node", "checksum")[index % 2]
        cell_index = index // 2
        rows.append(
            generate_example(
                seed=item_seed,
                split=split,
                family=families[cell_index % len(families)],
                template=templates[(cell_index // len(families)) % len(templates)],
                depth=int(
                    depths[
                        (cell_index // (len(families) * len(templates)))
                        % len(depths)
                    ]
                ),
                node_count=int(substrate["node_count"]),
                checksum_modulus=int(substrate["checksum_modulus"]),
                num_choices=int(substrate["num_choices"]),
                state_token=str(architecture["state_token"]),
                state_slots=int(architecture["state_slots"]),
                max_attempts=int(substrate["max_generation_attempts"]),
                query_kind=query_kind,
            )
        )
    return rows


def _generate_counterfactual_rows(
    *,
    pair_count: int,
    seed: int,
    split: str,
    families: Sequence[str],
    templates: Sequence[str],
    depths: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    substrate = config["substrate"]
    architecture = config["architecture"]
    rows: list[dict[str, Any]] = []
    for pair_index in range(pair_count):
        query_kind = ("node", "checksum")[pair_index % 2]
        cell_index = pair_index // 2
        family = families[cell_index % len(families)]
        template = templates[(cell_index // len(families)) % len(templates)]
        depth = depths[
            (cell_index // (len(families) * len(templates))) % len(depths)
        ]
        first, second = generate_counterfactual_pair(
            seed=int(seed) * 1_000_000 + pair_index,
            split=split,
            family=family,
            template=template,
            depth=int(depth),
            node_count=int(substrate["node_count"]),
            checksum_modulus=int(substrate["checksum_modulus"]),
            num_choices=int(substrate["num_choices"]),
            state_token=str(architecture["state_token"]),
            state_slots=int(architecture["state_slots"]),
            max_attempts=int(substrate["max_generation_attempts"]),
            query_kind=query_kind,
        )
        rows.extend((first, second))
    return rows


def _query_kind_grid(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    cells: dict[str, Counter[str]] = {}
    for row in rows:
        key = f"{row['family']}|{row['template']}|depth={int(row['depth'])}"
        cells.setdefault(key, Counter())[str(row["query_kind"])] += 1
    return {
        key: dict(sorted(counts.items()))
        for key, counts in sorted(cells.items())
    }


def build_datasets(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source_digest = source_contract_sha256()
    substrate = config["substrate"]
    architecture = config["architecture"]
    seeds = substrate["seeds"]
    train_families = list(substrate["train_families"])
    train_templates = list(substrate["train_templates"])
    train_depths = list(substrate["train_depths"])
    extrapolation = list(substrate["extrapolation_depths"])
    eval_n = int(substrate["evaluation_examples_per_split"])
    pilot_n = int(substrate["pilot_examples_per_split"])

    specs = {
        "train": (
            int(substrate["train_examples"]), seeds["train"], train_families, train_templates, train_depths
        ),
        "validation": (
            int(substrate["validation_examples"]),
            seeds["validation"],
            train_families,
            train_templates,
            train_depths,
        ),
        "pilot_validation": (
            int(substrate["pilot_validation_examples"]),
            seeds["pilot_validation"],
            train_families,
            train_templates,
            train_depths,
        ),
        "depth_extrapolation": (
            eval_n, seeds["depth"], train_families, train_templates, extrapolation
        ),
        "family_holdout": (
            eval_n,
            seeds["family"],
            [substrate["heldout_family"]],
            train_templates,
            train_depths + extrapolation,
        ),
        "template_holdout": (
            eval_n,
            seeds["template"],
            train_families,
            [substrate["heldout_template"]],
            train_depths + extrapolation,
        ),
        "joint_holdout": (
            eval_n,
            seeds["joint"],
            [substrate["heldout_family"]],
            [substrate["heldout_template"]],
            extrapolation,
        ),
        "pilot_depth": (
            pilot_n,
            seeds["pilot_depth"],
            train_families,
            train_templates,
            extrapolation,
        ),
        "pilot_joint": (
            pilot_n,
            seeds["pilot_joint"],
            [substrate["heldout_family"]],
            [substrate["heldout_template"]],
            extrapolation,
        ),
    }

    datasets: dict[str, list[dict[str, Any]]] = {}
    for split, (count, seed, families, templates, depths) in specs.items():
        datasets[split] = _generate_rows(
            count=int(count),
            seed=int(seed),
            split=split,
            families=list(families),
            templates=list(templates),
            depths=list(depths),
            config=config,
        )

    datasets["counterfactual"] = _generate_counterfactual_rows(
        pair_count=int(substrate["counterfactual_pairs"]),
        seed=int(seeds["counterfactual"]),
        split="counterfactual",
        families=train_families,
        templates=train_templates,
        depths=extrapolation,
        config=config,
    )
    datasets["pilot_counterfactual"] = _generate_counterfactual_rows(
        pair_count=int(substrate["pilot_counterfactual_pairs"]),
        seed=int(seeds["pilot_counterfactual"]),
        split="pilot_counterfactual",
        families=train_families,
        templates=train_templates,
        depths=extrapolation,
        config=config,
    )

    all_seen: dict[str, str] = {}
    files: dict[str, Any] = {}
    for split, rows in datasets.items():
        for row in rows:
            verify_example(row, str(architecture["state_token"]), int(architecture["state_slots"]))
            fingerprint = row["structural_fingerprint"]
            previous = all_seen.get(fingerprint)
            if previous is not None:
                raise AssertionError(f"structural duplicate crosses {previous} and {split}")
            all_seen[fingerprint] = split
        query_grid = _query_kind_grid(rows)
        for cell, counts in query_grid.items():
            if counts.get("node", 0) != counts.get("checksum", 0):
                raise AssertionError(f"query-kind imbalance in {split} cell {cell}")
        path = output_dir / f"{split}.jsonl.gz"
        _write_jsonl_gz(path, rows)
        files[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": _sha256(path),
            "families": dict(Counter(row["family"] for row in rows)),
            "templates": dict(Counter(row["template"] for row in rows)),
            "depths": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
            "query_kinds": dict(sorted(Counter(row["query_kind"] for row in rows).items())),
            "query_kind_grid": query_grid,
        }

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_contract_sha256": data_contract_sha256(config, source_digest=source_digest),
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_digest,
        "generator": "src.data_pipeline.build_datasets",
        "files": files,
        "cross_split_structural_duplicates": 0,
        "benchmark_files_read": 0,
        "state_token": architecture["state_token"],
        "state_slots": architecture["state_slots"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
