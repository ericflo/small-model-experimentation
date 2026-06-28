#!/usr/bin/env python3
"""Create a new durable research program scaffold."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ROOT / "research_programs"
TEMPLATE = ROOT / "templates" / "research_program"
REGISTRY = PROGRAMS / "registry.yaml"
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def title_from_slug(slug: str) -> str:
    return " ".join(word.upper() if word in {"abi", "rag", "vm", "rl"} else word.capitalize() for word in slug.split("_"))


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def yaml_inline_list(values: list[str]) -> str:
    return "[" + ", ".join(yaml_scalar(value) for value in values) + "]"


def registry_ids() -> set[str]:
    if not REGISTRY.exists():
        return set()
    ids: set[str] = set()
    for line in REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            ids.add(stripped.split(":", 1)[1].strip().strip('"'))
    return ids


def write_program_files(program_dir: Path, program_id: str, title: str, focus: str, seed_experiments: list[str]) -> None:
    charter = (TEMPLATE / "charter.md").read_text(encoding="utf-8")
    charter = charter.replace("# Research Program Title", f"# {title}", 1)
    charter = charter.replace(
        "What durable line of inquiry does this program own?",
        focus or "TODO: state the durable line of inquiry this program owns.",
        1,
    )
    charter = charter.replace(
        "Why is this broader than one experiment and distinct from existing programs?",
        "TODO: explain why this is broad enough to host multiple experiments and distinct from existing programs.",
        1,
    )
    (program_dir / "charter.md").write_text(charter, encoding="utf-8")

    backlog = (TEMPLATE / "backlog.md").read_text(encoding="utf-8")
    (program_dir / "backlog.md").write_text(backlog, encoding="utf-8")

    evidence = (TEMPLATE / "evidence.md").read_text(encoding="utf-8")
    if seed_experiments:
        seed_lines = "\n".join(f"- `{experiment_id}`" for experiment_id in seed_experiments)
        evidence = evidence.replace("- Experiment:", seed_lines, 1)
    (program_dir / "evidence.md").write_text(evidence, encoding="utf-8")

    (program_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"This directory is the working surface for `{program_id}`.",
                "",
                "- [charter.md](charter.md)",
                "- [backlog.md](backlog.md)",
                "- [evidence.md](evidence.md)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def append_registry(program_id: str, title: str, focus: str, seed_tags: list[str], seed_experiments: list[str]) -> None:
    if not REGISTRY.exists():
        REGISTRY.write_text(
            "# Hand-curated durable research lines. Current experiments are seed evidence, not a closed taxonomy.\nprograms:\n",
            encoding="utf-8",
        )
    block = [
        f'  - id: "{program_id}"',
        f"    title: {yaml_scalar(title)}",
        f'    charter: "research_programs/{program_id}/charter.md"',
        f"    focus: {yaml_scalar(focus or 'TODO: define the program focus.')}",
        f"    seed_tags: {yaml_inline_list(seed_tags)}",
    ]
    if seed_experiments:
        block.append("    seed_experiments:")
        block.extend(f"      - {yaml_scalar(experiment_id)}" for experiment_id in seed_experiments)
    else:
        block.append("    seed_experiments: []")
    content = REGISTRY.read_text(encoding="utf-8", errors="replace")
    if content and not content.endswith("\n"):
        content += "\n"
    REGISTRY.write_text(content + "\n".join(block) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program_id", help="snake_case id, for example multimodal_small_models")
    parser.add_argument("--title", default="", help="human-readable program title")
    parser.add_argument("--focus", default="", help="one-sentence program focus")
    parser.add_argument("--seed-tag", action="append", default=[], help="tag that should route experiments into this program")
    parser.add_argument("--seed-experiment", action="append", default=[], help="existing experiment id that seeds this program")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    program_id = args.program_id.strip()
    if not SLUG_RE.match(program_id):
        print(f"invalid program id {program_id!r}; use lower snake_case", file=sys.stderr)
        return 2
    if program_id in registry_ids():
        print(f"program already exists in registry: {program_id}", file=sys.stderr)
        return 2

    program_dir = PROGRAMS / program_id
    if program_dir.exists():
        print(f"program directory already exists: {program_dir.relative_to(ROOT)}", file=sys.stderr)
        return 2

    PROGRAMS.mkdir(exist_ok=True)
    program_dir.mkdir()
    title = args.title.strip() or title_from_slug(program_id)
    focus = args.focus.strip()
    seed_tags = [tag.strip() for tag in args.seed_tag if tag.strip()]
    seed_experiments = [experiment_id.strip() for experiment_id in args.seed_experiment if experiment_id.strip()]

    write_program_files(program_dir, program_id, title, focus, seed_experiments)
    append_registry(program_id, title, focus, seed_tags, seed_experiments)

    print(f"created research program: research_programs/{program_id}")
    print("next: fill charter/backlog/evidence, then run make catalog && make validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
