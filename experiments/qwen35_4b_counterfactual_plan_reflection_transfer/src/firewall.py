"""Process-local read firewall for the repository's forbidden benchmark tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def install_benchmark_firewall(repo: Path) -> None:
    benchmark_root = os.path.realpath(repo / "benchmarks")

    def forbidden(value: object) -> bool:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.path.realpath(os.fsdecode(value))
        try:
            return os.path.commonpath((path, benchmark_root)) == benchmark_root
        except ValueError:
            return False

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args and forbidden(args[0]):
            raise PermissionError("benchmark read firewall: open denied")
        if event in {"os.listdir", "os.scandir"} and args and forbidden(args[0]):
            raise PermissionError(f"benchmark read firewall: {event} denied")

    sys.addaudithook(audit)
