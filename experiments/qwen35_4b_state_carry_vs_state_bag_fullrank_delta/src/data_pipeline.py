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


def canonical_rows_receipt(path: str | Path) -> dict[str, Any]:
    """Content-address decompressed rows, including IDs, order, and all fields."""
    digest = hashlib.sha256()
    rows = 0
    for row in read_jsonl(path):
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8") + b"\n")
        rows += 1
    return {"rows": rows, "canonical_rows_sha256": digest.hexdigest()}


def _parent_data_parity_receipt(
    config: Mapping[str, Any],
    output_dir: Path,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove regenerated rows equal the frozen parent task, without parent imports."""
    contract = config["parent_data_contract"]
    if config.get("evidence_profile") != "confirmatory":
        return {
            "status": "NONCONFIRMATORY_PARITY_NOT_APPLICABLE",
            "parent_experiment_id": contract["parent_experiment_id"],
            "frozen_contract_match": False,
            "reason": "reduced smoke data intentionally has different split sizes",
        }
    expected = contract["splits"]
    actual = {
        split: canonical_rows_receipt(output_dir / metadata["path"])
        for split, metadata in sorted(files.items())
    }
    mismatches = {
        split: {"expected": expected.get(split), "actual": actual.get(split)}
        for split in sorted(set(expected) | set(actual))
        if expected.get(split) != actual.get(split)
    }
    if mismatches:
        raise RuntimeError(
            "regenerated data differs from the frozen parent row contract: "
            + json.dumps(mismatches, sort_keys=True)
        )

    parent_dir = (
        Path(__file__).resolve().parents[2]
        / str(contract["parent_experiment_id"])
        / "data"
        / "generated"
    )
    parent_manifest_path = parent_dir / "manifest.json"
    direct_match: bool | None = None
    parent_manifest_sha256: str | None = None
    if parent_manifest_path.is_file():
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        if parent_manifest.get("experiment_id") != contract["parent_experiment_id"]:
            raise RuntimeError("parent data manifest has the wrong experiment identity")
        parent_files = parent_manifest.get("files")
        if not isinstance(parent_files, Mapping) or set(parent_files) != set(expected):
            raise RuntimeError("parent data manifest split set differs from parity contract")
        parent_actual = {}
        for split, metadata in sorted(parent_files.items()):
            path = parent_dir / str(metadata["path"])
            if not path.is_file():
                raise RuntimeError(f"parent data artifact is missing for {split}: {path}")
            parent_actual[split] = canonical_rows_receipt(path)
        if parent_actual != actual:
            raise RuntimeError("regenerated rows differ from available parent artifacts")
        direct_match = True
        parent_manifest_sha256 = _sha256(parent_manifest_path)

    contract_digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "PARENT_DATA_PARITY_PASS",
        "parent_experiment_id": contract["parent_experiment_id"],
        "canonicalization": contract["canonicalization"],
        "frozen_contract_sha256": contract_digest,
        "frozen_contract_match": True,
        "parent_artifacts_available": parent_manifest_path.is_file(),
        "direct_parent_artifact_match": direct_match,
        "parent_manifest_sha256": parent_manifest_sha256,
        "splits": actual,
    }


def validate_parent_data_parity(
    config: Mapping[str, Any], output_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the load-bearing parent-row contract from current artifacts.

    A manifest receipt is not self-authenticating: file hashes and PASS strings
    can be updated together after a corpus changes.  Every model-bearing
    consumer therefore re-hashes the canonical decompressed rows and validates
    the complete frozen parity metadata before accepting prepared data.
    """
    contract = config["parent_data_contract"]
    expected_splits = contract["splits"]
    parity = manifest.get("parent_data_parity")
    if not isinstance(parity, Mapping):
        raise RuntimeError("prepared data has no parent-data parity receipt")
    expected_contract_digest = hashlib.sha256(
        json.dumps(expected_splits, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    required_metadata = {
        "status": "PARENT_DATA_PARITY_PASS",
        "parent_experiment_id": contract["parent_experiment_id"],
        "canonicalization": contract["canonicalization"],
        "frozen_contract_sha256": expected_contract_digest,
        "frozen_contract_match": True,
    }
    mismatches = {
        key: {"expected": value, "actual": parity.get(key)}
        for key, value in required_metadata.items()
        if parity.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "prepared parent-data parity metadata mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    recorded_splits = parity.get("splits")
    if not isinstance(recorded_splits, Mapping) or dict(recorded_splits) != dict(
        expected_splits
    ):
        raise RuntimeError("prepared parent-data parity split receipts are not frozen")

    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(expected_splits):
        raise RuntimeError("prepared parent-data parity split set is incomplete")
    actual = {
        split: canonical_rows_receipt(output_dir / str(files[split]["path"]))
        for split in sorted(files)
    }
    if actual != dict(expected_splits) or actual != dict(recorded_splits):
        raise RuntimeError("current prepared rows differ from the frozen parent contract")

    parent_artifacts_available = parity.get("parent_artifacts_available")
    direct_match = parity.get("direct_parent_artifact_match")
    parent_manifest_sha256 = parity.get("parent_manifest_sha256")
    if parent_artifacts_available is True:
        if direct_match is not True:
            raise RuntimeError("available parent artifacts lack a direct parity match")
        if parent_manifest_sha256 != config["parent_experiment"][
            "data_manifest_sha256"
        ]:
            raise RuntimeError("direct parent manifest identity mismatch")
    elif parent_artifacts_available is False:
        if direct_match is not None or parent_manifest_sha256 is not None:
            raise RuntimeError("absent parent artifacts have contradictory parity metadata")
    else:
        raise RuntimeError("parent artifact availability receipt is invalid")
    return dict(parity)


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
    manifest["parent_data_parity"] = _parent_data_parity_receipt(
        config, output_dir, files
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
