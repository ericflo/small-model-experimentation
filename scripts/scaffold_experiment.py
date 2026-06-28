#!/usr/bin/env python3
"""Create a new self-contained experiment scaffold attached to programs."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PROGRAMS = ROOT / "research_programs"
REGISTRY = PROGRAMS / "registry.yaml"
TEMPLATE = ROOT / "templates" / "experiment"
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def title_from_slug(slug: str) -> str:
    return " ".join(word.upper() if word in {"abi", "rag", "vm", "rl"} else word.capitalize() for word in slug.split("_"))


def yaml_scalar(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def registry_ids() -> set[str]:
    if not REGISTRY.exists():
        return set()
    ids: set[str] = set()
    for line in REGISTRY.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            ids.add(stripped.split(":", 1)[1].strip().strip('"'))
    return ids


def write_metadata(exp_dir: Path, experiment_id: str, title: str, summary: str, tags: list[str], programs: list[str]) -> None:
    lines = [
        "# Generated starter metadata. Run make catalog after adding artifacts.",
        f"id: {yaml_scalar(experiment_id)}",
        f"title: {yaml_scalar(title)}",
        'source_track: "new"',
        f'path: "experiments/{experiment_id}"',
        f'primary_readme: "experiments/{experiment_id}/README.md"',
        f'primary_report: "experiments/{experiment_id}/reports/report.md"',
        f"summary: {yaml_scalar(summary or 'New experiment scaffold. Fill this before running full experiments.')}",
        "tags:",
    ]
    for tag in tags:
        lines.append(f"  - {yaml_scalar(tag)}")
    if not tags:
        lines.append('  - "experiment"')
    lines.append("research_programs:")
    for program_id in programs:
        lines.append(f"  - {yaml_scalar(program_id)}")
    lines.extend(
        [
            "top_level_dirs:",
            '  - "configs"',
            '  - "reports"',
            '  - "scripts"',
            '  - "src"',
            "file_counts: {}",
            "total_files: 0",
            "total_size_bytes: 0",
            'generated_on: ""',
            "",
        ]
    )
    (exp_dir / "metadata.yaml").write_text("\n".join(lines), encoding="utf-8")


def write_readme(exp_dir: Path, title: str, summary: str, programs: list[str]) -> None:
    readme = (TEMPLATE / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("# Experiment Title", f"# {title}", 1)
    program_lines = ", ".join(f"`{program_id}`" for program_id in programs)
    readme = readme.replace("- Program:", f"- Program: {program_lines}", 1)
    if summary:
        readme = readme.replace(f"# {title}\n", f"# {title}\n\n{summary}\n", 1)
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")


def write_report(exp_dir: Path, title: str) -> None:
    report_template = TEMPLATE / "reports" / "report.md"
    report = report_template.read_text(encoding="utf-8")
    report = report.replace("# Report", f"# {title} Report", 1)
    (exp_dir / "reports" / "report.md").write_text(report, encoding="utf-8")


def write_run_script(exp_dir: Path, experiment_id: str) -> None:
    run_py = exp_dir / "scripts" / "run.py"
    run_py.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                '"""Starter run harness for this experiment."""',
                "",
                "from __future__ import annotations",
                "",
                "import argparse",
                "import sys",
                "",
                "",
                "def main() -> int:",
                "    parser = argparse.ArgumentParser(description=__doc__)",
                '    parser.add_argument("--smoke", action="store_true", help="check the scaffold without running the full experiment")',
                "    args = parser.parse_args()",
                "",
                "    if args.smoke:",
                f'        print("smoke scaffold passed: {experiment_id}")',
                "        return 0",
                '    parser.error("implement the full experiment run before using this command")',
                "    return 2",
                "",
                "",
                'if __name__ == "__main__":',
                "    sys.exit(main())",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(run_py, 0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", help="snake_case id, for example qwen_new_control_probe")
    parser.add_argument("--program", action="append", required=True, help="research program id; repeat for multi-program work")
    parser.add_argument("--title", default="", help="human-readable experiment title")
    parser.add_argument("--summary", default="", help="short experiment summary")
    parser.add_argument("--tag", action="append", default=[], help="navigation tag; repeat as needed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id.strip()
    if not SLUG_RE.match(experiment_id):
        print(f"invalid experiment id {experiment_id!r}; use lower snake_case", file=sys.stderr)
        return 2

    known_programs = registry_ids()
    programs = []
    for program_id in args.program:
        program_id = program_id.strip()
        if not program_id:
            continue
        programs.append(program_id)
    unknown = sorted(set(programs) - known_programs)
    if unknown:
        print("unknown research program(s): " + ", ".join(unknown), file=sys.stderr)
        print("create the program first with scripts/scaffold_research_program.py", file=sys.stderr)
        return 2
    if not programs:
        print("at least one --program is required", file=sys.stderr)
        return 2

    exp_dir = EXPERIMENTS / experiment_id
    if exp_dir.exists():
        print(f"experiment already exists: experiments/{experiment_id}", file=sys.stderr)
        return 2

    for dirname in ["src", "scripts", "configs", "data", "runs", "analysis", "reports"]:
        (exp_dir / dirname).mkdir(parents=True, exist_ok=True)

    title = args.title.strip() or title_from_slug(experiment_id)
    summary = args.summary.strip()
    tags = [tag.strip() for tag in args.tag if tag.strip()]

    write_readme(exp_dir, title, summary, programs)
    write_metadata(exp_dir, experiment_id, title, summary, tags, programs)
    write_report(exp_dir, title)
    write_run_script(exp_dir, experiment_id)
    (exp_dir / "experiment_log.md").write_text(f"# {title} Experiment Log\n\n## Scaffold\n\nCreated as a new experiment scaffold.\n", encoding="utf-8")
    (exp_dir / "configs" / "default.yaml").write_text("# Experiment configuration goes here.\n", encoding="utf-8")
    (exp_dir / "src" / "README.md").write_text("# Source\n\nPut experiment-local code here.\n", encoding="utf-8")

    print(f"created experiment: experiments/{experiment_id}")
    print("next: implement the smoke path, update program evidence/backlog, then run make catalog && make validate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
