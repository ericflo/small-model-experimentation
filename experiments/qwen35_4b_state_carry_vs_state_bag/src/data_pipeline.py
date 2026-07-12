"""Deterministic data construction and contamination receipts."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .substrate import generate_counterfactual_pair, generate_example, verify_example


def data_contract_sha256(config: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": config["experiment_id"],
        "substrate": config["substrate"],
        "state_token": config["architecture"]["state_token"],
        "state_slots": config["architecture"]["state_slots"],
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
        rows.append(
            generate_example(
                seed=item_seed,
                split=split,
                family=families[index % len(families)],
                template=templates[(index // len(families)) % len(templates)],
                depth=int(depths[(index // (len(families) * len(templates))) % len(depths)]),
                node_count=int(substrate["node_count"]),
                checksum_modulus=int(substrate["checksum_modulus"]),
                num_choices=int(substrate["num_choices"]),
                state_token=str(architecture["state_token"]),
                state_slots=int(architecture["state_slots"]),
                max_attempts=int(substrate["max_generation_attempts"]),
            )
        )
    return rows


def build_datasets(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    substrate = config["substrate"]
    architecture = config["architecture"]
    seeds = substrate["seeds"]
    train_families = list(substrate["train_families"])
    train_templates = list(substrate["train_templates"])
    train_depths = list(substrate["train_depths"])
    extrapolation = list(substrate["extrapolation_depths"])
    eval_n = int(substrate["evaluation_examples_per_split"])

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
            int(seeds["template"]) + 17,
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

    counterfactual: list[dict[str, Any]] = []
    pair_count = int(substrate["counterfactual_pairs"])
    for pair_index in range(pair_count):
        depth = extrapolation[pair_index % len(extrapolation)]
        family = train_families[pair_index % len(train_families)]
        template = train_templates[(pair_index // len(train_families)) % len(train_templates)]
        first, second = generate_counterfactual_pair(
            seed=int(seeds["counterfactual"]) * 1_000_000 + pair_index,
            split="counterfactual",
            family=family,
            template=template,
            depth=int(depth),
            node_count=int(substrate["node_count"]),
            checksum_modulus=int(substrate["checksum_modulus"]),
            num_choices=int(substrate["num_choices"]),
            state_token=str(architecture["state_token"]),
            state_slots=int(architecture["state_slots"]),
            max_attempts=int(substrate["max_generation_attempts"]),
        )
        counterfactual.extend((first, second))
    datasets["counterfactual"] = counterfactual

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
        path = output_dir / f"{split}.jsonl.gz"
        _write_jsonl_gz(path, rows)
        files[split] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": _sha256(path),
            "families": dict(Counter(row["family"] for row in rows)),
            "templates": dict(Counter(row["template"] for row in rows)),
            "depths": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
        }

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_contract_sha256": data_contract_sha256(config),
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
