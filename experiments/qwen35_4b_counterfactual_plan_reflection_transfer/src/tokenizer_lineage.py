"""Exact public-file identity for the sole permitted Qwen tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PIN_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "pinned_tokenizer_structure.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_pinned_tokenizer() -> dict[str, Any]:
    value = json.loads(PIN_PATH.read_text())
    if (
        set(value) != {"schema_version", "model_id", "model_revision", "files"}
        or value["schema_version"] != 1
        or value["model_id"] != "Qwen/Qwen3.5-4B"
        or value["model_revision"]
        != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or not isinstance(value["files"], dict)
        or set(value["files"])
        != {
            "chat_template.jinja",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        }
        or any(
            set(details) != {"sha256", "size"}
            or not isinstance(details["size"], int)
            or details["size"] < 1
            or not isinstance(details["sha256"], str)
            or len(details["sha256"]) != 64
            for details in value["files"].values()
        )
    ):
        raise ValueError("pinned tokenizer structure schema changed")
    return value


def authenticate_tokenizer_snapshot(
    snapshot: Path | None = None, *, ensure_downloaded: bool = False
) -> dict[str, Any]:
    """Authenticate every file capable of changing fast-tokenizer text semantics."""
    pin = load_pinned_tokenizer()
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ValueError("huggingface_hub is required for tokenizer identity") from error
        snapshot = Path(
            snapshot_download(
                repo_id=pin["model_id"],
                revision=pin["model_revision"],
                allow_patterns=sorted(pin["files"]),
                local_files_only=not ensure_downloaded,
            )
        )
    observed: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(pin["files"].items()):
        path = snapshot / name
        if not path.is_file():
            raise ValueError(f"pinned tokenizer file is absent: {name}")
        details = {"sha256": _sha256_file(path), "size": path.stat().st_size}
        if details != expected:
            raise ValueError(f"pinned tokenizer file differs from exact revision: {name}")
        observed[name] = details
    return {
        "schema_version": 1,
        "model_id": pin["model_id"],
        "model_revision": pin["model_revision"],
        "files": observed,
        "files_sha256": _canonical_sha256(observed),
    }
