"""Canonical source inventory for semantic control authorization."""

from __future__ import annotations

from pathlib import Path

from io_utils import canonical_hash, sha256_file


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]


def control_code_inventory() -> dict[str, object]:
    """Hash every experiment source file that can participate in control runs.

    The deliberately broad recursive inventory keeps authorization conservative:
    a control run cannot silently inherit an unbound helper through the runner,
    trainer, overlay builder, auditor, or any of their local dependencies.
    """

    paths = {
        *EXP.joinpath("scripts").rglob("*.py"),
        *EXP.joinpath("src").rglob("*.py"),
    }
    rows = []
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"control-code inventory contains an unsafe file: {path}")
        rows.append(
            {
                "path": path.relative_to(REPO).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError("control-code inventory is empty")
    return {
        "files": rows,
        "file_count": len(rows),
        "sha256": canonical_hash(rows),
    }
