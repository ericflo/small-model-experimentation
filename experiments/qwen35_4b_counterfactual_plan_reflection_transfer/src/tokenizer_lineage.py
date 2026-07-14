"""Exact public-file identity for the sole permitted Qwen tokenizer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
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
        set(value)
        != {"schema_version", "model_id", "model_revision", "files", "absent_files"}
        or value["schema_version"] != 2
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
        or value["absent_files"]
        != ["added_tokens.json", "special_tokens_map.json"]
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
                allow_patterns=sorted({*pin["files"], *pin["absent_files"]}),
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
    for name in pin["absent_files"]:
        if (snapshot / name).exists() or (snapshot / name).is_symlink():
            raise ValueError(f"pinned tokenizer semantic file must be absent: {name}")
    return {
        "schema_version": 2,
        "model_id": pin["model_id"],
        "model_revision": pin["model_revision"],
        "files": observed,
        "absent_files": list(pin["absent_files"]),
        "files_sha256": _canonical_sha256(observed),
        "semantic_surface_sha256": _canonical_sha256(
            {"present": observed, "absent": pin["absent_files"]}
        ),
    }


def authenticate_closed_tokenizer_view(path: Path) -> dict[str, Any]:
    """Require an exact local directory containing only the pinned load surface."""
    path = path.resolve()
    pin = load_pinned_tokenizer()
    if not path.is_dir() or path.is_symlink():
        raise ValueError("closed tokenizer view is not a regular local directory")
    entries = {item.name for item in path.iterdir()}
    if entries != set(pin["files"]):
        raise ValueError("closed tokenizer view has a missing or extra semantic file")
    if any(not item.is_file() or item.is_symlink() for item in path.iterdir()):
        raise ValueError("closed tokenizer view contains a non-regular file")
    return authenticate_tokenizer_snapshot(path)


def ensure_closed_tokenizer_view(
    *,
    ensure_downloaded: bool = False,
    cache_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Materialize a content-addressed five-file view used by every tokenizer load."""
    pin = load_pinned_tokenizer()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ValueError("huggingface_hub is required for tokenizer identity") from error
    source = Path(
        snapshot_download(
            repo_id=pin["model_id"],
            revision=pin["model_revision"],
            allow_patterns=sorted({*pin["files"], *pin["absent_files"]}),
            local_files_only=not ensure_downloaded,
        )
    )
    commitment = authenticate_tokenizer_snapshot(source)
    base = (
        cache_root
        if cache_root is not None
        else Path(
            os.environ.get(
                "SME_AUTHENTICATED_TOKENIZER_CACHE",
                str(Path.home() / ".cache" / "small-model-experimentation" / "tokenizers"),
            )
        )
    ).resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = base / commitment["semantic_surface_sha256"]
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix="tokenizer-view-", dir=base))
        try:
            for name in sorted(pin["files"]):
                destination = temporary / name
                shutil.copyfile(source / name, destination)
                destination.chmod(0o444)
            authenticate_closed_tokenizer_view(temporary)
            temporary.chmod(0o555)
            try:
                temporary.rename(target)
            except FileExistsError:
                temporary.chmod(0o755)
                shutil.rmtree(temporary)
        except Exception:
            if temporary.exists():
                temporary.chmod(0o755)
                for item in temporary.iterdir():
                    item.chmod(0o644)
                shutil.rmtree(temporary)
            raise
    observed = authenticate_closed_tokenizer_view(target)
    if observed != commitment:
        raise ValueError("closed tokenizer view differs from authenticated source")
    return target, commitment
