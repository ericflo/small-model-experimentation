#!/usr/bin/env python3
"""Build the static research website: a findings-first, multi-page reading surface.

Pages: home (latest findings feed), experiment explorer, one page per experiment
(rendered README/report/log, figure gallery, data files, repro), programs, claims,
future queue, and the knowledge notebook. Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True  # keep scripts/__pycache__ out of the working tree

import site_markdown
from site_markdown import LinkResolver, render_markdown, slugify, strip_inline

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
PROGRAMS_DIR = ROOT / "research_programs"
EXPERIMENTS_DIR = ROOT / "experiments"
TEMPLATE = ROOT / "templates" / "site"

GITHUB = "https://github.com/ericflo/small-model-experimentation"
SITE_NAME = "Small Model Experimentation"

FIGURE_EXTS = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp"}
FIGURE_MAX_BYTES = 8_000_000
DATA_FILE_RE = re.compile(r"(summary|results?|metrics|report|eval|scores?|pareto|table|baseline)", re.I)
DATA_DIRS = ("runs", "reports", "analysis")
DATA_MAX_BYTES = 200_000
DATA_MAX_FILES = 24
IMPORT_TRACK_LABEL = {"track-y": "imported · line Y", "track-z": "imported · line Z", "new": "new in this repo"}
IMPORT_TRACK_TITLE = {
    "track-y": "Imported 2026-06-28 from the predecessor working repo (research line Y). Run dates shown are recovered from records inside the experiment folder.",
    "track-z": "Imported 2026-06-28 from the predecessor working repo (research line Z). Run dates shown are recovered from records inside the experiment folder.",
    "new": "Created in this repository after the 2026-06-28 corpus import.",
}
RUN_SURFACE_NOTE = {
    "documented-command": "The exact run commands are recorded in this experiment's artifact manifest.",
    "documented-scripts": "Run steps are documented inside the experiment folder (README and scripts).",
    "scripts-undocumented": "Runnable scripts exist in the experiment folder, but the exact invocation was not written down.",
    "source-or-analysis": "This entry is source code or analysis only — there is no separate run to reproduce.",
}
QUEUE_STATUS_LABEL = {
    "ready-for-intake": ("protocol ready", "Fully specified — this proposal can be run next."),
    "program-seed": ("program seed", "Would open a new research program; the protocol still needs to be written."),
    "idea": ("idea", "Direction worth keeping; not yet specified as a runnable protocol."),
}

SLUG_ACRONYMS = {
    "abi": "ABI", "dpo": "DPO", "grpo": "GRPO", "sft": "SFT", "rl": "RL", "repl": "REPL",
    "icl": "ICL", "rag": "RAG", "mdp": "MDP", "ttt": "TTT", "lora": "LoRA", "moe": "MoE",
    "graphir": "GraphIR", "mbpp": "MBPP", "humaneval": "HumanEval", "gsm8k": "GSM8K",
    "opsd": "OPSD", "passk": "Pass@k", "dagger": "DAgger", "vs": "vs", "kv": "KV",
}
SLUG_SMALL_WORDS = {"and", "of", "in", "for", "with", "to", "the", "a", "an", "on"}


def prettify_slug(slug: str) -> str:
    """Readable display title for experiments whose catalog title is a raw slug."""
    words = slug.split("_")
    out: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if word == "qwen35" and index + 1 < len(words) and words[index + 1] == "4b":
            out.append("Qwen3.5-4B")
            index += 2
            continue
        if word in SLUG_ACRONYMS:
            out.append(SLUG_ACRONYMS[word])
        elif word in SLUG_SMALL_WORDS and out:
            out.append(word)
        else:
            out.append(word.capitalize())
        index += 1
    return " ".join(out)


_TITLE_SMALL = {"and", "or", "of", "for", "the", "to", "in", "a", "an", "on", "at", "by", "vs", "with", "from"}


def _nice_title(title: str) -> str:
    """Lowercase connective small-words in an already title-cased string
    ('Interpretability And Diagnostics' -> 'Interpretability and Diagnostics')."""
    words = title.split()
    return " ".join(
        w if i == 0 or w.lower() not in _TITLE_SMALL else w.lower() for i, w in enumerate(words)
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def png_size(path: Path) -> tuple[int, int] | None:
    """Width/height from a PNG IHDR header (no image libraries in CI)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(26)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    if 0 < width < 20000 and 0 < height < 20000:
        return width, height
    return None


def format_size(size: int) -> str:
    if size < 1000:
        return f"{size} B"
    if size < 1_000_000:
        value = size / 1000
        return f"{value:.1f} kB" if value < 10 else f"{value:.0f} kB"
    return f"{size / 1_000_000:.1f} MB"

RESULT_HEADING_RE = re.compile(
    r"^(final result snapshot|final results?|headline results?|results?( and analysis)?|key findings?|findings?|outcome|what we found|key numbers)\b",
    re.I,
)
SUMMARY_HEADING_RE = re.compile(r"^(summary|abstract|executive summary|tl;?dr)\b", re.I)
INTERP_HEADING_RE = re.compile(r"^(interpretation|conclusions?|takeaways?|discussion)\b", re.I)
BORING_README_HEADS = re.compile(r"^(layout|contents|large files|useful commands|reproduce|run|setup|environment)\b", re.I)

CLAIM_STATUS_ORDER = ["Confirmed", "Promising", "Open", "Negative", "Retired"]
CLAIM_STATUS_META = {
    "Confirmed": ("check", "Directly supported by result-bearing experiments"),
    "Promising": ("half", "Pilot or partial evidence"),
    "Open": ("open", "Plausible, not adequately tested"),
    "Negative": ("cross", "Tested and failed under recorded conditions"),
    "Retired": ("pause", "Contradicted; do not build on without new evidence"),
}
STATUS_ICON = {
    "check": "&#10003;",
    "half": "&#9681;",
    "open": "&#9675;",
    "cross": "&#10007;",
    "pause": "&#8856;",
}

esc = lambda value: html.escape(str(value), quote=True)  # noqa: E731


# --------------------------------------------------------------------------- io


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def split_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def parse_inline_list(value: str) -> list[str]:
    value = value.strip()
    if not value.startswith("[") or not value.endswith("]"):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]


def load_programs() -> list[dict[str, object]]:
    registry = PROGRAMS_DIR / "registry.yaml"
    programs: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    active_list: str | None = None
    for raw_line in read_text(registry).splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "programs:":
            continue
        if stripped.startswith("- id:"):
            if current:
                programs.append(current)
            current = {"id": stripped.split(":", 1)[1].strip().strip('"'), "title": "", "focus": "", "seed_tags": []}
            active_list = None
            continue
        if current is None:
            continue
        if stripped.startswith("- ") and active_list:
            current[active_list].append(stripped[2:].strip().strip('"'))  # type: ignore[union-attr]
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"seed_tags", "seed_experiments"}:
            parsed = parse_inline_list(value)
            current[key] = parsed
            active_list = key if not parsed else None
        else:
            current[key] = value.strip('"')
            active_list = None
    if current:
        programs.append(current)
    return programs


def git_dates() -> dict[str, dict[str, object]]:
    """One pass over history: per-experiment first/last activity and commit count.

    Skips metadata.yaml (bulk regeneration sweeps touch every experiment and
    would make every directory look freshly active).
    """
    try:
        out = subprocess.run(
            ["git", "log", "--format=D%ad %at", "--date=short", "--name-only", "--", "experiments"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {}
    dates: dict[str, dict[str, object]] = {}
    current_date = ""
    current_ts = 0
    seen_this_commit: set[str] = set()
    for line in out.splitlines():
        m = re.match(r"D(\d{4}-\d{2}-\d{2}) (\d+)$", line)
        if m:
            current_date, current_ts = m.group(1), int(m.group(2))
            seen_this_commit = set()
            continue
        if not line.startswith("experiments/"):
            continue
        parts = line.split("/")
        if len(parts) < 3 or parts[-1] == "metadata.yaml":
            continue
        exp_id = parts[1]
        entry = dates.setdefault(exp_id, {"first": current_date, "last": current_date,
                                          "last_ts": current_ts, "commits": 0})
        entry["first"] = current_date  # log is newest-first; last assignment wins
        if exp_id not in seen_this_commit:
            entry["commits"] = int(entry["commits"]) + 1
            seen_this_commit.add(exp_id)
    return dates


# ------------------------------------------------------------------- findings


def _sections(text: str) -> list[tuple[str, int, str]]:
    """Split markdown into (heading, level, body) tuples; body excludes subsections."""
    lines = text.splitlines()
    sections: list[tuple[str, int, list[str]]] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
        heading = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if heading and not in_fence:
            sections.append((heading.group(2), len(heading.group(1)), []))
        elif sections:
            sections[-1][2].append(line)
    return [(head, level, "\n".join(body)) for head, level, body in sections]


_EXCERPT_DROP_RE = re.compile(
    r"(?i)\b(?:table|figure|chart|plot|section)s?\s+(?:above|below)\b"
    r"|/(?:root|workspace)/|\.codex\b|\battachments?/"
    r"|\bprimary outputs?\b|\bintentionally outside\b|\blarge_artifacts/"
)
_PATH_SENTENCE_RE = re.compile(r"[\w./-]+\.(?:md|csv|json|png|yaml|yml|txt)[.,;]?")
_TABLE_POINTER = " (table on the experiment page)."
_SUMMARY_MARKER_RE = re.compile(r"^(?:Arc|Bottom line|Net|Takeaway|Overall|Conclusion|Verdict)\b\s*[:,—–-]")
_RESULT_VERB_RE = re.compile(
    r"(?i)\b(?:win|wins|won|beat|beats|fail|fails|failed|improv\w*|gain\w*|drop\w*|collaps\w*|regress\w*"
    r"|match\w*|refut\w*|confirm\w*|hold|holds|held|generaliz\w*|transfer\w*|outperform\w*|underperform\w*"
    r"|dominat\w*|null|no effect|does not|doesn.t|works|worked|solve[sd]?)\b"
)


def _result_score(sentence: str) -> int:
    score = 0
    if re.search(r"%|→|±|\bpp\b|@\d|pass@|\d\.\d", sentence):
        score += 2
    if re.search(r"\d", sentence):
        score += 1
    if _RESULT_VERB_RE.search(sentence):
        score += 1
    return score


def _flatten(markdown_body: str) -> str:
    """Markdown -> running prose. Skips fences/tables/headings; drops intro
    lines left dangling by a skipped table ("Headline results:")."""
    parts: list[str] = []
    in_fence = False

    def fix_dangler(kind: str = "") -> None:
        # the previous line introduced content we are about to skip: drop short
        # labels ("Headline results:"), keep real sentences and say where the
        # skipped content lives
        if not parts or not parts[-1].endswith(":"):
            return
        if len(parts[-1]) < 60:
            parts.pop()
        elif kind == "table":
            parts[-1] = parts[-1][:-1].rstrip() + _TABLE_POINTER
        else:
            parts[-1] = parts[-1][:-1].rstrip() + "."

    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            fix_dangler("code")
            continue
        if in_fence or not stripped:
            continue
        if stripped.startswith(("|", "#", "![", "<")):
            fix_dangler("table" if stripped.startswith("|") else "")
            continue
        stripped = re.sub(r"^(?:[-*+]|\d{1,3}[.)])\s+", "", stripped)
        plain = strip_inline(stripped)
        if plain:
            parts.append(plain)
    text = " ".join(parts).strip()
    # drop leading pointer phrases ("Full results in reports/report.md; figure x.png.")
    return re.sub(
        r"^(?:(?i:full|see|figures?|detailed|complete)\b[^.]{0,140}?\.(?:md|png|csv|json)\b[).;,\s]*)+",
        "",
        text,
    ).lstrip()


def _excerpt(markdown_body: str, limit: int) -> str:
    """Findings excerpt: prefer the author's own summary sentence, else start at
    the first result-bearing sentence; mark any truncation with an ellipsis."""
    text = _flatten(markdown_body)
    sentences = [
        s
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s and not _EXCERPT_DROP_RE.search(s) and not _PATH_SENTENCE_RE.fullmatch(s.strip())
    ]
    if not sentences:
        return ""
    for index, sentence in enumerate(sentences):
        if _SUMMARY_MARKER_RE.match(sentence) and len(sentence) >= 60:
            sentences = sentences[index:]
            break
    else:
        first_result = next((i for i, s in enumerate(sentences) if _result_score(s) >= 2), None)
        if first_result is not None and first_result > 1:
            sentences = sentences[first_result:]
    kept: list[str] = []
    used = 0
    truncated = False
    for sentence in sentences:
        if kept and used + len(sentence) + 1 > limit:
            truncated = True
            break
        kept.append(sentence)
        used += len(sentence) + 1
    while kept and kept[-1].endswith(":"):  # trailing lead-in with its content elsewhere
        kept.pop()
        truncated = True
    if not kept:
        return ""
    text = " ".join(kept)
    if len(text) > limit * 1.3:  # single overlong sentence: cut at a word boundary
        cut = text[:limit]
        text = (cut[: cut.rfind(" ")] if " " in cut else cut).rstrip(",;:")
        truncated = True
    return text + (" …" if truncated else "")


_VOCAB_WORD_RE = re.compile(r"[a-z0-9][a-z0-9@._/-]{2,29}")


def search_vocab(*texts: str, cap: int = 700) -> str:
    """Unique-word bag for token search: covers a whole document at a fraction
    of its size (search matches tokens independently, so order is irrelevant)."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for text in texts:
        for word in _VOCAB_WORD_RE.findall(text.lower()):
            if word in seen_set or word.replace(".", "").replace("-", "").replace("/", "").isdigit():
                continue
            seen_set.add(word)
            seen.append(word)
            if len(seen) >= cap:
                return " ".join(seen)
    return " ".join(seen)


def extract_finding(readme: str, report: str, fallback: str, limit: int = 760) -> tuple[str, str]:
    """Best-effort 'what we found' excerpt and its source label."""
    candidates: list[tuple[str, str]] = []
    for head, _, body in _sections(readme):
        if RESULT_HEADING_RE.match(head):
            candidates.append((body, f"README · {head}"))
    for head, _, body in _sections(readme):
        if SUMMARY_HEADING_RE.match(head):
            candidates.append((body, f"README · {head}"))
    for head, _, body in _sections(report):
        if SUMMARY_HEADING_RE.match(head) or RESULT_HEADING_RE.match(head):
            candidates.append((body, f"report · {head}"))
    for head, _, body in _sections(report):
        if INTERP_HEADING_RE.match(head):
            candidates.append((body, f"report · {head}"))
    for body, label in candidates:
        text = _excerpt(body, limit)
        if len(text) < 60 or text.lower().startswith(("pending", "tbd", "todo")):
            continue
        if text.endswith(_TABLE_POINTER.strip()) and "." not in text[: -len(_TABLE_POINTER)]:
            continue  # nothing but a table lead-in: the numbers live in the table
        return text, label
    for source, label in ((readme, "README"), (report, "report")):
        text = _excerpt(source, limit)
        if len(text) >= 60:
            return text, label
    return _excerpt(fallback, limit) or fallback[:limit], "catalog"


# ------------------------------------------------------------------ resolvers


class ExperimentResolver(LinkResolver):
    """Resolve links/images inside one markdown doc of one experiment."""

    def __init__(self, exp: dict, doc_repo_path: Path, prefix: str, roster: set[str], copied: set[str]):
        self.exp = exp
        self.doc_dir = doc_repo_path.parent
        self.exp_dir = EXPERIMENTS_DIR / str(exp["id"])
        self.prefix = prefix
        self.roster = roster
        self.copied = copied  # experiment-relative posix paths copied under files/

    def _repo_target(self, url: str) -> Path | None:
        if re.match(r"^[a-z][a-z0-9+.-]*:", url) or url.startswith(("/", "#")):
            return None
        clean = url.split("#", 1)[0].split("?", 1)[0]
        if not clean:
            return None
        try:
            resolved = (self.doc_dir / clean).resolve()
            resolved.relative_to(ROOT)
        except (ValueError, OSError):
            return None
        return resolved

    def _exp_rel(self, target: Path) -> str | None:
        try:
            return target.relative_to(self.exp_dir).as_posix()
        except ValueError:
            return None

    def image(self, url: str) -> str:
        target = self._repo_target(url)
        if target is None:
            return url if re.match(r"^https?:", url) else ""
        rel = self._exp_rel(target)
        if rel and rel in self.copied:
            return f"files/{rel}"
        return ""

    def link(self, url: str) -> str:
        target = self._repo_target(url)
        if target is None:
            return url
        rel = self._exp_rel(target)
        if rel is not None:
            if rel in self.copied:
                return f"files/{rel}"
            anchors = {
                str(self.exp.get("readme_rel", "")): "#readme",
                str(self.exp.get("report_rel", "")): "#report",
                str(self.exp.get("log_rel", "")): "#log",
            }
            if rel in anchors:
                return anchors[rel]
            return f"{GITHUB}/blob/main/{target.relative_to(ROOT).as_posix()}"
        try:
            other = target.relative_to(EXPERIMENTS_DIR)
            other_id = other.parts[0]
            if other_id in self.roster:
                if len(other.parts) == 1:
                    return f"{self.prefix}experiments/{other_id}/"
                return f"{GITHUB}/blob/main/{target.relative_to(ROOT).as_posix()}"
        except ValueError:
            pass
        kind = "tree" if target.is_dir() else "blob"
        return f"{GITHUB}/{kind}/main/{target.relative_to(ROOT).as_posix()}"

    def code_span_target(self, text: str) -> str:
        if not re.match(r"^[\w./~-]+\.(png|svg|jpe?g|gif|webp)$", text):
            return ""
        for base in (self.doc_dir, self.exp_dir):
            try:
                resolved = (base / text).resolve()
                rel = resolved.relative_to(self.exp_dir).as_posix()
            except (ValueError, OSError):
                continue
            if rel in self.copied:
                return f"files/{rel}"
        return ""

    def image_size(self, resolved: str) -> tuple[int, int] | None:
        if not resolved.startswith("files/"):
            return None
        return png_size(self.exp_dir / resolved[len("files/"):])


class KnowledgeResolver(LinkResolver):
    """Resolve links inside knowledge/ and research_programs/ docs to site pages."""

    def __init__(self, doc_repo_path: Path, prefix: str, roster: set[str], program_ids: set[str]):
        self.doc_dir = doc_repo_path.parent
        self.prefix = prefix
        self.roster = roster
        self.program_ids = program_ids

    def link(self, url: str) -> str:
        if re.match(r"^[a-z][a-z0-9+.-]*:", url) or url.startswith("#"):
            return url
        clean = url.split("#", 1)[0].split("?", 1)[0]
        if not clean:
            return url
        try:
            target = (self.doc_dir / clean).resolve()
            repo_rel = target.relative_to(ROOT).as_posix()
        except (ValueError, OSError):
            return url
        parts = repo_rel.split("/")
        if parts[0] == "experiments" and len(parts) > 1 and parts[1] in self.roster:
            return f"{self.prefix}experiments/{parts[1]}/"
        if parts[0] == "research_programs" and len(parts) > 1 and parts[1] in self.program_ids:
            return f"{self.prefix}programs/{parts[1]}/"
        if repo_rel.startswith("knowledge/claims"):
            return f"{self.prefix}claims/"
        notebook = {
            "knowledge/synthesis.md": "synthesis",
            "knowledge/research_roadmap.md": "roadmap",
            "knowledge/patterns.md": "patterns",
            "knowledge/open_questions.md": "open-questions",
            "knowledge/program_scorecards.md": "scorecards",
        }
        if repo_rel in notebook:
            return f"{self.prefix}notebook/{notebook[repo_rel]}/"
        kind = "tree" if target.is_dir() else "blob"
        return f"{GITHUB}/{kind}/main/{repo_rel}"

    def image(self, url: str) -> str:
        return url if re.match(r"^https?:", url) else ""


# ------------------------------------------------------------------- assembly


def discover_files(exp_dir: Path) -> tuple[list[Path], list[Path]]:
    figures: list[Path] = []
    data: list[Path] = []
    for path in sorted(exp_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(exp_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in FIGURE_EXTS and path.stat().st_size <= FIGURE_MAX_BYTES:
            figures.append(rel)
        elif (
            suffix in {".json", ".csv"}
            and rel.parts
            and rel.parts[0] in DATA_DIRS
            and DATA_FILE_RE.search(path.name)
            and path.stat().st_size <= DATA_MAX_BYTES
        ):
            data.append(rel)
    data.sort(key=lambda rel: (len(rel.parts), str(rel)))
    return figures, data[:DATA_MAX_FILES], max(0, len(data) - DATA_MAX_FILES)


def find_log(exp_dir: Path) -> Path | None:
    for candidate in (
        exp_dir / "experiment_log.md",
        exp_dir / "logs" / "experiment_log.md",
        exp_dir / "lab_notebook.md",
        exp_dir / "logs" / "lab_notebook.md",
    ):
        if candidate.exists():
            return candidate
    reports = exp_dir / "reports"
    if reports.exists():
        for pattern in ("*experiment_log*.md", "*lab_notebook*.md"):
            for candidate in sorted(reports.glob(pattern)):
                return candidate
    return None


def recorded_dates() -> dict[str, dict[str, str]]:
    """Run windows recovered from in-repo records (see extract_experiment_dates.py)."""
    path = KNOWLEDGE / "experiment_dates.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    entries = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


IMPORT_DATE = "2026-06-28"  # bulk corpus import; git dates on/before it are collapsed


def _run_window(recorded: dict, git_info: dict) -> dict:
    """Run window for an experiment. Prefer the curated dates entry; otherwise
    fall back to the metadata-sweep-filtered git window for POST-import dirs, so
    a new experiment always shows a real date even before the dates artifact is
    updated (dates never drift on the site)."""
    start = str(recorded.get("start", ""))
    end = str(recorded.get("end", ""))
    confidence = str(recorded.get("confidence", ""))
    if not start:
        git_first = str(git_info.get("first", ""))
        git_last = str(git_info.get("last", ""))
        if git_first and git_first > IMPORT_DATE:  # created after the import → git is real
            start, end, confidence = git_first, git_last, "high"
    return {"ran_start": start, "ran_end": end, "date_confidence": confidence}


def build_experiments(programs: list[dict]) -> list[dict]:
    catalog = read_csv(KNOWLEDGE / "experiment_catalog.csv")
    readiness = {row["id"]: row for row in read_csv(KNOWLEDGE / "experiment_readiness.csv")}
    dates = git_dates()
    recorded = recorded_dates()
    roster = {row["id"] for row in catalog}
    experiments: list[dict] = []
    for row in catalog:
        exp_id = row["id"]
        exp_dir = EXPERIMENTS_DIR / exp_id
        ready = readiness.get(exp_id, {})
        readme_path = ROOT / row["primary_readme"] if row["primary_readme"] else None
        report_path = ROOT / row["primary_report"] if row["primary_report"] else None
        log_path = find_log(exp_dir)
        readme_text = read_text(readme_path) if readme_path and readme_path.exists() else ""
        report_text = read_text(report_path) if report_path and report_path.exists() else ""
        finding, finding_source = extract_finding(readme_text, report_text, row["summary"])
        figures, data_files, data_dropped = discover_files(exp_dir)
        info = dates.get(exp_id, {})
        title = row["title"].strip()
        if not title or "_" in title or title == exp_id:
            title = prettify_slug(exp_id)
        exp = {
            "id": exp_id,
            "title": title,
            "summary": row["summary"],
            "track": row["source_track"],
            "tags": split_list(row["tags"]),
            "programs": split_list(row["research_programs"]),
            "path": row["path"],
            "readme_path": readme_path,
            "report_path": report_path,
            "log_path": log_path,
            "readme_rel": readme_path.relative_to(exp_dir).as_posix() if readme_path else "",
            "report_rel": report_path.relative_to(exp_dir).as_posix() if report_path and report_path.exists() else "",
            "log_rel": log_path.relative_to(exp_dir).as_posix() if log_path else "",
            "readme_text": readme_text,
            "report_text": report_text,
            "finding": finding,
            "finding_source": finding_source,
            "figures": figures,
            "data_files": data_files,
            "data_dropped": data_dropped,
            "first": str(info.get("first", "")),
            "last": str(info.get("last", "")),
            "last_ts": int(info.get("last_ts", 0) or 0),
            "commits": int(info.get("commits", 0) or 0),
            **_run_window(recorded.get(exp_id, {}), info),
            "run_surface": ready.get("run_surface", ""),
            "smoke_command": ready.get("smoke_command", ""),
            "anchor_ready": ready.get("anchor_ready", ""),
            "experiment_log_flag": ready.get("experiment_log", ""),
            "total_files": int(row.get("total_files") or 0),
            "total_size_bytes": int(row.get("total_size_bytes") or 0),
        }
        experiments.append(exp)
    for exp in experiments:
        # True chronology is the experiment's OWN run window (curated dates, else the
        # creation commit) — never git-last-touched. A sweeping mechanical commit (the
        # bulk artifact removal) rewrote 155 folders' last-touched date to one day, so
        # ordering on it kept bumping settled experiments back to the top.
        exp["when"] = exp["ran_end"] or exp["ran_start"] or exp["first"] or exp["last"]
        exp["recent"] = bool(exp["when"]) and exp["when"] > IMPORT_DATE
    experiments.sort(
        key=lambda exp: (exp["when"] or "0000-00-00", exp["ran_end"] or "", exp["ran_start"] or "", exp["first"] or "", exp["id"]),
        reverse=True,
    )
    known_programs = {str(program["id"]) for program in programs}
    for exp in experiments:
        exp["programs"] = [pid for pid in exp["programs"] if pid in known_programs] or exp["programs"]
    return experiments


# finished/in-progress lifecycle status (slot, label, tooltip)
EXPERIMENT_STATUS = {
    "finished": ("done", "Finished", "Concluded — a result is recorded (a win, a null, or a ruled-out negative)."),
    "in-progress": ("live", "In progress", "Running — set up and under way, no conclusion recorded yet."),
}
def load_experiment_status() -> dict[str, dict]:
    """Curated experiment lifecycle from knowledge/experiment_status.json.

    Finished is the DEFAULT — an experiment is in-progress ONLY if it has an
    explicit in-progress entry here. Status is deliberately NEVER inferred from
    report prose: preregistrations describe their planned "verdict / negative /
    pass-fail" and finished experiments say "we did not run <ablation>", so any
    regex guess silently mislabels in both directions (that was the old footgun).
    Explicit-only means a concluded experiment cannot stay stuck in-progress, and
    an in-progress entry must be dated + reasoned (enforced in CI by
    validate_repository.py) so it can be reviewed and expired.

    Returns {id: {"status", "since", "reason"}}.
    """
    path = KNOWLEDGE / "experiment_status.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    entries = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    out: dict[str, dict] = {}
    for key, value in entries.items():
        if isinstance(value, dict):
            status, since, reason = value.get("status"), value.get("since", ""), value.get("reason", "")
        else:
            status, since, reason = value, "", ""
        if status in EXPERIMENT_STATUS:
            out[str(key)] = {"status": status, "since": str(since or ""), "reason": str(reason or "")}
    return out


def full_command_from_manifest(exp_dir: Path) -> tuple[str, str, bool]:
    manifest = exp_dir / "reports" / "artifact_manifest.yaml"
    if not manifest.exists():
        return "", "", False
    commands = {"smoke_command": "", "full_command": ""}
    lines = read_text(manifest).splitlines()
    for index, line in enumerate(lines):
        match = re.match(r'^(\s*)(smoke_command|full_command):\s*(.*?)\s*$', line)
        if not match:
            continue
        indent, key, value = len(match.group(1)), match.group(2), match.group(3)
        if value in {">", ">-", "|", "|-"}:  # YAML block scalar: gather indented lines
            block: list[str] = []
            for follower in lines[index + 1:]:
                if not follower.strip():
                    break
                if len(follower) - len(follower.lstrip()) <= indent:
                    break
                block.append(follower.strip())
            joiner = " " if value.startswith(">") else "\n"
            commands[key] = joiner.join(block)
        else:
            commands[key] = value.strip('"').strip("'")
    return commands["smoke_command"], commands["full_command"], True


# ----------------------------------------------------------------- html bits


def strip_leading_h1(text: str) -> str:
    """Drop a document's opening H1: detail pages already carry a title."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if re.match(r"^#\s+", line):
            return "\n".join(lines[index + 1:])
        break
    return text


def load_shell() -> str:
    return read_text(TEMPLATE / "index.html")


def page_html(
    shell: str,
    *,
    title: str,
    description: str,
    prefix: str,
    active: str,
    content: str,
    generated: str,
    extra_head: str = "",
) -> str:
    nav_items = [
        ("experiments/", "Experiments", "experiments"),
        ("programs/", "Programs", "programs"),
        ("claims/", "Claims", "claims"),
        ("queue/", "Queue", "queue"),
        ("notebook/", "Notebook", "notebook"),
    ]
    current = ' aria-current="page"'
    nav = "".join(
        f'<a href="{prefix}{href}"{current if key == active else ""}>{label}</a>'
        for href, label, key in nav_items
    )
    out = shell.replace("%%TITLE%%", esc(title))
    out = out.replace("%%DESCRIPTION%%", esc(description))
    out = out.replace("%%ROOT%%", prefix)
    out = out.replace("%%NAV%%", nav)
    out = out.replace("%%GENERATED%%", esc(generated))
    out = out.replace("%%EXTRA_HEAD%%", extra_head)
    return out.replace("%%CONTENT%%", content)


def program_chip(pid: str, prefix: str, slots: dict[str, int], titles: dict[str, str]) -> str:
    slot = slots.get(pid, 0)
    label = titles.get(pid, pid.replace("_", " "))
    if pid not in titles:  # placeholder or candidate program: no page to link
        return (
            '<span class="chip prog pending" data-slot="0" title="Proposed program — no dedicated page yet">'
            f'<span class="dot" aria-hidden="true"></span>{esc(label)} · proposed</span>'
        )
    return (
        f'<a class="chip prog" data-slot="{slot}" href="{prefix}programs/{esc(pid)}/">'
        f'<span class="dot" aria-hidden="true"></span>{esc(label)}</a>'
    )


def status_chip(status: str) -> str:
    icon_key, _ = CLAIM_STATUS_META.get(status, ("open", ""))
    return (
        f'<span class="chip status status-{icon_key}"><span aria-hidden="true">{STATUS_ICON[icon_key]}</span>'
        f" {esc(status)}</span>"
    )


def exp_status_chip(status: str) -> str:
    """Finished / In progress lifecycle chip (distinct from a claim's status)."""
    slot, label, tip = EXPERIMENT_STATUS.get(status, EXPERIMENT_STATUS["finished"])
    return (
        f'<span class="chip xstatus x-{slot}" title="{esc(tip)}">'
        f'<span class="xdot" aria-hidden="true"></span>{esc(label)}</span>'
    )


def format_range(start: str, end: str) -> str:
    if not start or start == end:
        return end or start
    if start[:7] == end[:7]:  # same month: 2026-05-12 → 15
        return f"{start} → {end[8:]}"
    if start[:4] == end[:4]:  # same year: 2026-05-28 → 06-02
        return f"{start} → {end[5:]}"
    return f"{start} → {end}"


def date_label(exp: dict) -> str:
    if exp["ran_start"] or exp["ran_end"]:
        approx = "~" if exp["date_confidence"] == "low" else ""
        return approx + format_range(exp["ran_start"], exp["ran_end"])
    if not exp["last"]:
        return "in progress"
    if not exp["recent"]:
        return f"imported {exp['last']}"
    if exp["first"] and exp["first"] != exp["last"]:
        return format_range(exp["first"], exp["last"])
    return str(exp["last"])


def date_span(exp: dict) -> str:
    """Date label with a tooltip whenever the label needs interpretation."""
    title = ""
    if exp["ran_start"] or exp["ran_end"]:
        if exp["date_confidence"] == "low":
            title = "Approximate: inferred from a date-shaped random seed in the experiment's configs."
    elif exp["last"] and not exp["recent"]:
        title = (
            "No run record survives inside this experiment's files; "
            "this is the day it entered the repo with the corpus import."
        )
    attr = f' title="{esc(title)}"' if title else ""
    return f'<span class="card-date"{attr}>{esc(date_label(exp))}</span>'


def track_chip(track: str) -> str:
    """Provenance chip. New-in-repo work is the unmarked default: chip only for imports."""
    if track == "new" or not track:
        return ""
    label = IMPORT_TRACK_LABEL.get(track, track)
    title = IMPORT_TRACK_TITLE.get(track, "")
    return f'<span class="chip track track-{esc(track)}" title="{esc(title)}">{esc(label)}</span>'


def figure_url(exp_id: str, rel: Path | str, prefix: str) -> str:
    return f"{prefix}experiments/{exp_id}/files/{Path(rel).as_posix()}"


_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")
_CLAIM_REF_RE = re.compile(r"\b(C\d{1,2})\b")
_EXP_REF_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b")


def _transform_text_nodes(html_text: str, transform, skip_inline_code: bool) -> str:
    """Apply transform to text nodes outside <a> and <pre> (and optionally <code>)."""
    parts = _TAG_SPLIT_RE.split(html_text)
    anchor_depth = 0
    code_depth = 0
    pre_depth = 0
    for index, part in enumerate(parts):
        if part.startswith("<"):
            tag = part[1:].split(None, 1)[0].rstrip(">").lower()
            if tag == "a":
                anchor_depth += 1
            elif tag == "/a":
                anchor_depth = max(0, anchor_depth - 1)
            elif tag == "code":
                code_depth += 1
            elif tag == "/code":
                code_depth = max(0, code_depth - 1)
            elif tag == "pre":
                pre_depth += 1
            elif tag == "/pre":
                pre_depth = max(0, pre_depth - 1)
            continue
        if anchor_depth or pre_depth or (skip_inline_code and code_depth):
            continue
        parts[index] = transform(part)
    return "".join(parts)


def linkify_claims(html_text: str, prefix: str, claim_ids: set[str]) -> str:
    """Turn bare claim mentions ("claim C9") in rendered prose into links."""

    def transform(text: str) -> str:
        return _CLAIM_REF_RE.sub(
            lambda m: (
                f'<a href="{prefix}claims/#{m.group(1).lower()}">{m.group(1)}</a>'
                if m.group(1).lower() in claim_ids
                else m.group(1)
            ),
            text,
        )

    return _transform_text_nodes(html_text, transform, skip_inline_code=True)


def linkify_experiments(html_text: str, prefix: str, roster: set[str]) -> str:
    """Link bare experiment ids in prose (including inline code) to their pages."""

    def transform(text: str) -> str:
        return _EXP_REF_RE.sub(
            lambda m: (
                f'<a href="{prefix}experiments/{m.group(1)}/">{m.group(1)}</a>'
                if m.group(1) in roster
                else m.group(1)
            ),
            text,
        )

    return _transform_text_nodes(html_text, transform, skip_inline_code=False)


def activity_chart(experiments: list[dict]) -> str:
    """Run-activity histogram: one thin bar per day (or per week for long spans)."""
    dated = sorted(exp["when"] for exp in experiments if exp["when"])
    if len(set(dated)) < 4:
        return ""
    first_day = dt.date.fromisoformat(dated[0])
    last_day = dt.date.fromisoformat(dated[-1])
    daily = (last_day - first_day).days <= 60

    def bucket_of(day: dt.date) -> dt.date:
        return day if daily else day - dt.timedelta(days=day.weekday())

    step = dt.timedelta(days=1 if daily else 7)
    buckets: dict[dt.date, int] = {}
    cursor = bucket_of(first_day)
    while cursor <= bucket_of(last_day):
        buckets[cursor] = 0
        cursor += step
    for value in dated:
        buckets[bucket_of(dt.date.fromisoformat(value))] += 1

    unit, param = ("", "day") if daily else ("week of ", "week")
    entries = sorted(buckets.items())
    peak = max(count for _, count in entries) or 1
    plot_h, axis_h, gap = 96, 20, 2
    bar_w, pad = 18, 4
    width = len(entries) * (bar_w + gap) + pad * 2
    height = plot_h + axis_h + 18
    bars = []
    for index, (bucket, count) in enumerate(entries):
        x = pad + index * (bar_w + gap)
        bar_h = round(plot_h * count / peak) if count else 0
        y = 14 + plot_h - bar_h
        label = f"{unit}{bucket.isoformat()}: {count} experiment{'s' if count != 1 else ''} — click to filter"
        if count:
            bars.append(
                f'<a href="?{param}={bucket.isoformat()}" aria-label="{esc(label)}">'
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="2" fill="var(--slot-1)">'
                f"<title>{esc(label)}</title></rect></a>"
            )
        if count == peak:
            bars.append(
                f'<text x="{x + bar_w / 2}" y="{y - 4}" text-anchor="middle" font-size="11" fill="var(--ink-2)">{count}</text>'
            )
        month_start = bucket.day <= (1 if daily else 7)
        if index == 0 or (daily and bucket.weekday() == 0) or (not daily and month_start):
            tick = bucket.strftime("%b %d").replace(" 0", " ")
            bars.append(
                f'<text x="{x}" y="{14 + plot_h + 14}" font-size="10" fill="var(--muted)">{esc(tick)}</text>'
            )
    baseline = f'<line x1="0" x2="{width}" y1="{14 + plot_h}" y2="{14 + plot_h}" stroke="var(--baseline)" stroke-width="1"/>'
    rows = "".join(
        f"<tr><td>{unit}{bucket.isoformat()}</td><td style=\"text-align:right\">{count}</td></tr>"
        for bucket, count in entries
        if count
    )
    head = "Day" if daily else "Week"
    table = (
        '<details class="chart-table"><summary>Data table</summary><div class="table-wrap">'
        f'<table class="md-table"><thead><tr><th>{head}</th><th style="text-align:right">Experiments</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></details>"
    )
    caption = f"Experiments by the {'day' if daily else 'week'} they last ran — click a bar to filter the list"
    return (
        f'<figure class="activity-chart"><figcaption>{caption}</figcaption>'
        f'<svg viewBox="0 0 {width} {height}" role="group" aria-label="Experiments per {"day" if daily else "week"}, {entries[0][0]} to {entries[-1][0]}; each bar links to that period" '
        f'preserveAspectRatio="xMinYMid meet">{baseline}{"".join(bars)}</svg>{table}</figure>'
    )


# ------------------------------------------------------------- native charts


def load_viz() -> dict[str, list[dict]]:
    """Verified chart specs extracted from each experiment's own result files."""
    path = KNOWLEDGE / "experiment_viz.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    entries = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    out: dict[str, list[dict]] = {}
    for exp_id, entry in entries.items():
        charts = entry.get("charts", []) if isinstance(entry, dict) else []
        valid = [spec for spec in charts if _valid_spec(spec)]
        if valid:
            out[str(exp_id)] = valid
    return out


def load_briefs() -> dict[str, dict]:
    """Plain-language practitioner briefs (see knowledge/experiment_brief.json).

    Page-level fields render the verdict banner + brief card; each brief's
    `charts` list carries per-chart plain framing keyed by chart index. Absent
    or partial entries degrade gracefully — the page just shows less."""
    path = KNOWLEDGE / "experiment_brief.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    entries = payload.get("experiments", {}) if isinstance(payload, dict) else {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def _valid_spec(spec: object) -> bool:
    if not isinstance(spec, dict) or spec.get("kind") not in {"bar", "line", "heatmap", "scatter"}:
        return False
    kind = spec["kind"]
    if kind == "heatmap":
        xc, yc, vals = spec.get("x_categories"), spec.get("y_categories"), spec.get("values")
        if not (isinstance(xc, list) and xc and isinstance(yc, list) and yc):
            return False
        if not isinstance(vals, list) or len(vals) != len(yc):
            return False
        for row in vals:
            if not isinstance(row, list) or len(row) != len(xc):
                return False
            if not all(v is None or isinstance(v, (int, float)) for v in row):
                return False
        return True
    series = spec.get("series")
    if not isinstance(series, list) or not series:
        return False
    if kind == "bar":
        cats = spec.get("categories")
        if not isinstance(cats, list) or not cats:
            return False
        for entry in series:
            values = entry.get("values") if isinstance(entry, dict) else None
            if not isinstance(values, list) or len(values) != len(cats):
                return False
            if not all(isinstance(v, (int, float)) for v in values):
                return False
    else:  # line, scatter
        for entry in series:
            points = entry.get("points") if isinstance(entry, dict) else None
            if not isinstance(points, list) or not points:  # single points render as reference markers
                return False
            if not all(isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) for p in points):
                return False
    return True


def _fmt_val(value: float, y_format: str) -> str:
    if y_format in {"percent01", "percent100"}:
        pct = value * 100 if y_format == "percent01" else value
        text = f"{pct:.1f}".rstrip("0").rstrip(".")
        return f"{text}%"
    av = abs(value)
    # compact magnitude suffixes instead of scientific notation (2.34e+04 -> 23.4k)
    if av >= 1e6:
        return f"{value / 1e6:.4g}".rstrip("0").rstrip(".") + "M"
    if av >= 1e4:
        return f"{value / 1e3:.4g}".rstrip("0").rstrip(".") + "k"
    if value == int(value):
        return str(int(value))
    if 0 < av < 0.001:  # keep the tiniest values compact but not raw-exponent-ugly
        return f"{value:.1e}"
    return f"{value:.3g}"


def _nice_ticks(lo: float, hi: float) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    first = math.ceil(lo / step) * step
    ticks = []
    tick = first
    while tick <= hi + step * 0.01:
        ticks.append(round(tick, 10))
        tick += step
    return ticks


def _nice_axis(dmin: float, dmax: float, zero: bool = True) -> tuple[float, float, list[float]]:
    """A value axis rounded to nice steps so the last gridline lands AT or beyond
    the data. zero=True anchors at 0 (correct for bars, whose length encodes
    magnitude); zero=False pads a tight range around the data (for lines/scatter,
    so a shape sitting at 0.7–0.8 is not flattened against a forced 0 baseline)."""
    if zero:
        lo, hi = min(0.0, dmin), max(0.0, dmax)
    else:
        lo, hi = dmin, dmax
        if lo == hi:
            lo, hi = lo - abs(lo or 1) * 0.5, hi + abs(hi or 1) * 0.5
        pad = (hi - lo) * 0.1
        lo, hi = lo - pad, hi + pad
        if dmin >= 0 and lo < 0:  # never dip below zero for non-negative data
            lo = 0.0
    if lo == hi:
        return lo, lo + 1, [lo, lo + 1]
    raw = (hi - lo) / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag), 10 * mag)
    axis_lo = math.floor(lo / step) * step
    axis_hi = math.ceil(hi / step) * step
    if axis_hi <= axis_lo:
        axis_hi = axis_lo + step
    ticks, tick = [], axis_lo
    while tick <= axis_hi + step * 1e-6:
        ticks.append(round(tick, 10))
        tick += step
    return axis_lo, axis_hi, ticks


_SERIES_SLOTS = [1, 2, 3, 5, 8, 6, 7, 4]  # hue-separated order (slots 2 and 4 are both greens)


def _series_color(index: int) -> str:
    return f"var(--slot-{_SERIES_SLOTS[index % len(_SERIES_SLOTS)]})"


def _spec_values(spec: dict) -> list[float]:
    out: list[float] = []
    for entry in spec["series"]:
        if spec["kind"] == "bar":
            out.extend(entry["values"])
        else:
            out.extend(p[1] for p in entry["points"])
    return out


def _tw(text: str, fs: float) -> float:
    """Approximate rendered text width in px (system-ui, average glyph ~0.56em)."""
    return len(str(text)) * fs * 0.56


def _bar_svg(spec: dict, W: int, mini: bool) -> tuple[list[str], float]:
    """Bar chart. Auto-selects horizontal layout when category labels are long or
    numerous — the clean fix for the label overflow the old vertical bars suffered."""
    series = spec["series"]
    cats = [str(c) for c in spec.get("categories", [])]
    n_series, n_groups = len(series), len(cats)
    yf = spec.get("y_format", "number")
    vals = [v for e in series for v in e["values"]]
    lo, hi, ticks = _nice_axis(min(vals), max(vals))
    longest_cat = max((len(c) for c in cats), default=0)
    horizontal = (not mini) and (longest_cat > 10 or n_groups > 7 or n_groups * n_series > 20)
    parts: list[str] = []

    if horizontal:
        left = int(min(232, max(72, longest_cat * 6.5 + 10)))
        top, right = 12, 56
        band = int(max(24, min(48, n_series * 15 + 14)))
        plot_h = n_groups * band
        H = top + plot_h + 34
        plot_w = W - left - right

        def sx(v: float) -> float:
            return left + (v - lo) / (hi - lo) * plot_w
        for t in ticks:
            x = sx(t)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="var(--grid)"/>')
            parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 16:.1f}" text-anchor="middle" font-size="11" fill="var(--muted)">{esc(_fmt_val(t, yf))}</text>')
        x0 = sx(0)
        parts.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{top}" y2="{top + plot_h}" stroke="var(--baseline)" stroke-width="1.3"/>')
        bar_h = min(20.0, (band - 8) / n_series)
        for gi, cat in enumerate(cats):
            gy = top + gi * band
            ctxt = cat if len(cat) <= 34 else cat[:33] + "…"
            parts.append(f'<text x="{left - 8}" y="{gy + band / 2 + 4:.1f}" text-anchor="end" font-size="11.5" fill="var(--ink-2)"><title>{esc(cat)}</title>{esc(ctxt)}</text>')
            cluster_h = n_series * bar_h + (n_series - 1) * 2
            for si, e in enumerate(series):
                v = e["values"][gi]
                y = gy + (band - cluster_h) / 2 + si * (bar_h + 2)
                xa, xb = sx(min(0.0, v)), sx(max(0.0, v))
                w = max(1.2, xb - xa)
                lbl = f'{esc(e["label"])} · {esc(cat)}: {esc(_fmt_val(v, yf))}' if n_series > 1 else f'{esc(cat)}: {esc(_fmt_val(v, yf))}'
                parts.append(f'<rect x="{xa:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="2" fill="{_series_color(si)}" data-viz-pt data-viz-series="{si}" data-viz-label="{lbl}"/>')
                if bar_h >= 11:
                    # label hugs the OUTER end of the bar (right for +, left for -)
                    lx, anchor = (xa - 4, "end") if v < 0 else (xb + 4, "start")
                    parts.append(f'<text x="{lx:.1f}" y="{y + bar_h / 2 + 3.5:.1f}" text-anchor="{anchor}" font-size="10" font-weight="600" fill="var(--ink-2)" data-viz-series-label="{si}">{esc(_fmt_val(v, yf))}</text>')
        return parts, H

    # vertical bars
    top, left, right = 16, 46, 14
    group_w0 = (W - left - right) / max(n_groups, 1)
    rotate = (not mini) and any(_tw(c, 11) > group_w0 * 0.92 for c in cats)
    bottom = 58 if rotate else 30
    plot_h = 128 if mini else 240
    H = top + plot_h + bottom
    plot_w = W - left - right
    group_w = plot_w / max(n_groups, 1)

    def sy(v: float) -> float:
        return top + plot_h - (v - lo) / (hi - lo) * plot_h
    for t in ticks:
        y = sy(t)
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        if not mini:
            parts.append(f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{esc(_fmt_val(t, yf))}</text>')
    parts.append(f'<line x1="{left}" x2="{W - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="var(--baseline)" stroke-width="1.3"/>')
    gap = 2 if mini else 3
    bar_w = min(30.0 if mini else 46.0, max(6.0, (group_w * 0.76) / n_series - gap))
    cluster_w = n_series * bar_w + (n_series - 1) * gap
    show_val = (not mini) and bar_w >= 16 and n_groups * n_series <= 16
    for gi, cat in enumerate(cats):
        gx = left + gi * group_w + (group_w - cluster_w) / 2
        for si, e in enumerate(series):
            v = e["values"][gi]
            x = gx + si * (bar_w + gap)
            ya, yb = sy(max(0.0, v)), sy(min(0.0, v))
            h = max(1.2, yb - ya)
            lbl = f'{esc(e["label"])} · {esc(cat)}: {esc(_fmt_val(v, yf))}' if n_series > 1 else f'{esc(cat)}: {esc(_fmt_val(v, yf))}'
            parts.append(f'<rect x="{x:.1f}" y="{ya:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="2.5" fill="{_series_color(si)}" data-viz-pt data-viz-series="{si}" data-viz-label="{lbl}"/>')
            if show_val:
                # label sits just outside the bar end: above for +, below for -
                ly = yb + 12 if v < 0 else ya - 4
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="var(--ink-2)" data-viz-series-label="{si}">{esc(_fmt_val(v, yf))}</text>')
        cx = left + gi * group_w + group_w / 2
        if mini and n_groups > 4:
            continue
        if rotate:
            ctxt = cat if len(cat) <= 26 else cat[:25] + "…"
            title = f"<title>{esc(cat)}</title>" if len(cat) > 26 else ""
            parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + 14:.1f}" transform="rotate(-35 {cx:.1f} {top + plot_h + 14:.1f})" text-anchor="end" font-size="10.5" fill="var(--ink-2)">{title}{esc(ctxt)}</text>')
        else:
            max_chars = max(4, int(group_w / (5.4 if mini else 6.4)))
            ctxt = cat if len(cat) <= max_chars else cat[:max_chars - 1] + "…"
            title = f"<title>{esc(cat)}</title>" if len(cat) > max_chars else ""
            parts.append(f'<text x="{cx:.1f}" y="{top + plot_h + (13 if mini else 16):.1f}" text-anchor="middle" font-size="{10 if mini else 11}" fill="var(--ink-2)">{title}{esc(ctxt)}</text>')
    return parts, H


def _line_svg(spec: dict, W: int, mini: bool) -> tuple[list[str], float]:
    """Line chart. Few series get direct end-labels (dodged); many series drop the
    labels (an interactive legend identifies them) and cycle dash patterns."""
    series = spec["series"]
    yf = spec.get("y_format", "number")
    n = len(series)
    many = n > 4
    vals = [p[1] for e in series for p in e["points"]]
    lo, hi, y_ticks = _nice_axis(min(vals), max(vals), zero=False)
    xs = [p[0] for e in series for p in e["points"]]
    xlo, xhi = min(xs), max(xs)
    if xlo == xhi:
        xhi = xlo + 1
    top, left, bottom = 14, 46, 30
    if mini or many:
        right = 12
    else:
        right = int(min(150, max(64, max(len(str(e["label"])[:18]) for e in series) * 6.2 + 14)))
    plot_h = 128 if mini else 240
    H = top + plot_h + bottom
    plot_w = W - left - right

    def sy(v: float) -> float:
        return top + plot_h - (v - lo) / (hi - lo) * plot_h

    def sx(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * plot_w
    parts: list[str] = []
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        if not mini:
            parts.append(f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{esc(_fmt_val(t, yf))}</text>')
    if lo <= 0 <= hi:  # only draw a zero baseline when zero is on the (tight) axis
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="var(--baseline)" stroke-width="1.3"/>')
    if not mini:
        xt = _nice_ticks(xlo, xhi)[:7]
        for t in xt:
            parts.append(f'<text x="{sx(t):.1f}" y="{top + plot_h + 16:.1f}" text-anchor="middle" font-size="11" fill="var(--ink-2)">{esc(_fmt_val(t, "number"))}</text>')
    dashes = ["", "5 3", "2 3", "7 3 2 3"]
    ends: list[tuple[float, int, str]] = []
    for si, e in enumerate(series):
        pts = sorted(e["points"], key=lambda p: p[0])
        dash = f' stroke-dasharray="{dashes[(si // 8) % len(dashes)]}"' if many else ""
        if len(pts) > 1:
            path = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{_series_color(si)}" stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round"{dash} data-viz-series-line="{si}"/>')
        r = (3 if mini else 3.5) if len(pts) > 1 else (4.5 if mini else 5.5)
        for p in pts:
            lbl = f'{esc(e["label"])} · {esc(_fmt_val(p[0], "number"))}: {esc(_fmt_val(p[1], yf))}'
            parts.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="{r}" fill="{_series_color(si)}" data-viz-pt data-viz-series="{si}" data-viz-label="{lbl}"/>')
        lab = str(e["label"])
        ends.append((sy(pts[-1][1]), si, lab))
    if not many and not mini:
        ends.sort(key=lambda t: t[0])
        ys = [y for y, _, _ in ends]
        for i in range(1, len(ys)):
            if ys[i] < ys[i - 1] + 13:
                ys[i] = ys[i - 1] + 13
        overflow = ys[-1] - (top + plot_h - 2)
        if overflow > 0:
            ys = [y - overflow for y in ys]
        ys = [max(y, top + 6) for y in ys]
        for (y0, si, full), y in zip(ends, ys):
            txt = full if len(full) <= 18 else full[:17] + "…"
            title = f"<title>{esc(full)}</title>" if len(full) > 18 else ""
            parts.append(f'<text x="{W - right + 7:.1f}" y="{y + 3.5:.1f}" font-size="11" font-weight="600" fill="{_series_color(si)}" data-viz-series-label="{si}">{title}{esc(txt)}</text>')
    return parts, H


def _heatmap_svg(spec: dict, W: int, uid: str) -> tuple[list[str], float]:
    """Categorical grid, colored by a sequential value scale — a chart kind the
    old bar/line engine could not represent at all."""
    xcats = [str(c) for c in spec["x_categories"]]
    ycats = [str(c) for c in spec["y_categories"]]
    rows = spec["values"]
    yf = spec.get("value_format", spec.get("y_format", "number"))
    flat = [v for r in rows for v in r if isinstance(v, (int, float))]
    vlo, vhi = (min(flat), max(flat)) if flat else (0.0, 1.0)
    if vlo == vhi:
        vhi = vlo + 1
    longest_y = max((len(c) for c in ycats), default=0)
    left = int(min(200, max(60, longest_y * 6 + 8)))
    top, right = 12, 20
    cell_w = (W - left - right) / max(len(xcats), 1)
    longest_x = max((len(c) for c in xcats), default=0)
    rotate_x = any(_tw(c, 10) > cell_w for c in xcats)
    bottom = int(min(150, max(30, longest_x * 6))) if rotate_x else 30
    cell_h = min(46.0, max(20.0, cell_w * 0.7))
    H = top + len(ycats) * cell_h + bottom + 30

    def color(v: float) -> str:
        t = (v - vlo) / (vhi - vlo)
        return f"rgb({int(247 - 107 * t)},{int(249 - 129 * t)},{int(251 - 37 * t)})"
    parts: list[str] = [f'<defs><linearGradient id="hm-{esc(uid)}"><stop offset="0" stop-color="{color(vlo)}"/><stop offset="1" stop-color="{color(vhi)}"/></linearGradient></defs>']
    for yi, yc in enumerate(ycats):
        cy = top + yi * cell_h
        ytxt = yc if len(yc) <= 30 else yc[:29] + "…"
        parts.append(f'<text x="{left - 6}" y="{cy + cell_h / 2 + 4:.1f}" text-anchor="end" font-size="11" fill="var(--ink-2)"><title>{esc(yc)}</title>{esc(ytxt)}</text>')
        for xi in range(len(xcats)):
            v = rows[yi][xi] if (yi < len(rows) and xi < len(rows[yi])) else None
            cx = left + xi * cell_w
            if not isinstance(v, (int, float)):
                parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="var(--page)" stroke="var(--surface)"/>')
                continue
            lbl = f'{esc(xcats[xi])} · {esc(yc)}: {esc(_fmt_val(v, yf))}'
            parts.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color(v)}" stroke="var(--surface)" data-viz-pt data-viz-label="{lbl}"/>')
            if cell_w >= 30 and cell_h >= 18:
                ink = "#fff" if (v - vlo) / (vhi - vlo) > 0.62 else "var(--ink-2)"
                parts.append(f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 3.5:.1f}" text-anchor="middle" font-size="10" fill="{ink}">{esc(_fmt_val(v, yf))}</text>')
    yb = top + len(ycats) * cell_h
    for xi, xc in enumerate(xcats):
        cx = left + xi * cell_w + cell_w / 2
        if rotate_x:
            xtxt = xc if len(xc) <= 24 else xc[:23] + "…"
            parts.append(f'<text x="{cx:.1f}" y="{yb + 14:.1f}" transform="rotate(-40 {cx:.1f} {yb + 14:.1f})" text-anchor="end" font-size="10" fill="var(--ink-2)">{esc(xtxt)}</text>')
        else:
            parts.append(f'<text x="{cx:.1f}" y="{yb + 16:.1f}" text-anchor="middle" font-size="10.5" fill="var(--ink-2)">{esc(xc)}</text>')
    ly = yb + bottom + 6
    parts.append(f'<rect x="{left}" y="{ly}" width="140" height="10" fill="url(#hm-{esc(uid)})" stroke="var(--grid)"/>')
    parts.append(f'<text x="{left - 4}" y="{ly + 9}" text-anchor="end" font-size="10" fill="var(--muted)">{esc(_fmt_val(vlo, yf))}</text>')
    parts.append(f'<text x="{left + 146}" y="{ly + 9}" font-size="10" fill="var(--muted)">{esc(_fmt_val(vhi, yf))}</text>')
    return parts, H


def _scatter_svg(spec: dict, W: int, mini: bool) -> tuple[list[str], float]:
    """Point cloud with real axes; series distinguished by color."""
    series = spec["series"]
    yf = spec.get("y_format", "number")
    pts_all = [(p[0], p[1]) for e in series for p in e["points"]]
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    xlo, xhi = min(xs), max(xs)
    xpad = (xhi - xlo) * 0.06 or 1
    xlo, xhi = xlo - xpad, xhi + xpad
    ylo, yhi, y_ticks = _nice_axis(min(ys), max(ys), zero=False)
    top, left, right, bottom = 14, 48, 16, 30
    plot_h = 128 if mini else 240
    plot_w = W - left - right
    H = top + plot_h + bottom

    def sy(v: float) -> float:
        return top + plot_h - (v - ylo) / (yhi - ylo) * plot_h

    def sx(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * plot_w
    parts: list[str] = []
    for t in y_ticks:
        y = sy(t)
        parts.append(f'<line x1="{left}" x2="{W - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>')
        if not mini:
            parts.append(f'<text x="{left - 7}" y="{y + 3.5:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{esc(_fmt_val(t, yf))}</text>')
    for t in _nice_ticks(xlo, xhi)[:7]:
        if not mini:
            parts.append(f'<text x="{sx(t):.1f}" y="{top + plot_h + 16:.1f}" text-anchor="middle" font-size="11" fill="var(--ink-2)">{esc(_fmt_val(t, "number"))}</text>')
    for si, e in enumerate(series):
        for p in e["points"]:
            lbl = f'{esc(e["label"])} · ({esc(_fmt_val(p[0], "number"))}, {esc(_fmt_val(p[1], yf))})'
            parts.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="{3 if mini else 4}" fill="{_series_color(si)}" fill-opacity="0.78" data-viz-pt data-viz-series="{si}" data-viz-label="{lbl}"/>')
    return parts, H


def chart_svg(spec: dict, *, mini: bool = False, uid: str = "", plain_axes: bool = False) -> str:
    """Render one chart spec as a clean SVG. Bar auto-orients (horizontal for long
    labels), line switches to legend mode past 4 series, and heatmap/scatter are
    first-class kinds. plain_axes is accepted for API compatibility (axis titles
    are surfaced by chart_figure)."""
    kind = spec.get("kind")
    W = 360 if mini else 720
    if kind == "bar":
        parts, H = _bar_svg(spec, W, mini)
    elif kind == "line":
        parts, H = _line_svg(spec, W, mini)
    elif kind == "heatmap":
        parts, H = _heatmap_svg(spec, W, uid or "0")
    elif kind == "scatter":
        parts, H = _scatter_svg(spec, W, mini)
    else:
        return ""
    title = str(spec.get("title", ""))
    aria = f'{title}: {spec.get("note", "")}' if spec.get("note") else title
    return (
        f'<svg viewBox="0 0 {W} {int(round(H))}" role="img" aria-label="{esc(aria)}" '
        f'preserveAspectRatio="xMidYMid meet" class="viz-svg{" viz-mini" if mini else ""}">{"".join(parts)}</svg>'
    )


def _chart_table(spec: dict) -> str:
    """A data-table twin of the chart, so every number is reachable and sortable."""
    yf = spec.get("y_format", "number")
    kind = spec["kind"]
    if kind == "bar":
        head = "".join(f"<th>{esc(e['label'])}</th>" for e in spec["series"])
        rows = "".join(
            "<tr><td>" + esc(str(cat)) + "</td>"
            + "".join(f'<td style="text-align:right">{esc(_fmt_val(e["values"][ci], yf))}</td>' for e in spec["series"])
            + "</tr>"
            for ci, cat in enumerate(spec["categories"])
        )
        thead = f"<tr><th>{esc(spec.get('x_label') or 'condition')}</th>{head}</tr>"
    elif kind == "line" or kind == "scatter":
        xs = sorted({p[0] for e in spec["series"] for p in e["points"]})
        head = "".join(f"<th>{esc(e['label'])}</th>" for e in spec["series"])
        thead = f"<tr><th>{esc(spec.get('x_label') or 'x')}</th>{head}</tr>"
        rows = ""
        for x in xs:
            cells = ""
            for e in spec["series"]:
                match = next((p[1] for p in e["points"] if p[0] == x), None)
                cells += f'<td style="text-align:right">{esc(_fmt_val(match, yf)) if match is not None else "—"}</td>'
            rows += f"<tr><td>{esc(_fmt_val(x, 'number'))}</td>{cells}</tr>"
    elif kind == "heatmap":
        xcats = [str(c) for c in spec["x_categories"]]
        head = "".join(f"<th>{esc(c)}</th>" for c in xcats)
        thead = f"<tr><th></th>{head}</tr>"
        vf = spec.get("value_format", spec.get("y_format", "number"))
        rows = ""
        for yi, yc in enumerate(spec["y_categories"]):
            cells = "".join(
                f'<td style="text-align:right">{esc(_fmt_val(v, vf)) if isinstance(v, (int, float)) else "—"}</td>'
                for v in (spec["values"][yi] if yi < len(spec["values"]) else [])
            )
            rows += f"<tr><td>{esc(str(yc))}</td>{cells}</tr>"
    else:
        return ""
    return (
        '<details class="chart-table"><summary>Data table</summary><div class="table-wrap">'
        f'<table class="md-table" data-sortable><thead>{thead}</thead><tbody>{rows}</tbody></table></div></details>'
    )


def _chart_legend(spec: dict) -> str:
    """Interactive series legend — for multi-series bars and many-series lines
    (few-series lines carry direct end labels instead)."""
    kind = spec["kind"]
    series = spec.get("series") or []  # heatmap has no series (its color scale is the legend)
    need = (kind == "bar" and len(series) > 1) or (kind == "line" and len(series) > 4) or (kind == "scatter" and len(series) > 1)
    if not need:
        return ""
    chips = "".join(
        f'<button type="button" class="viz-key" data-viz-toggle="{si}" aria-pressed="true">'
        f'<span class="dot" style="background:{_series_color(si)}"></span>{esc(e["label"])}</button>'
        for si, e in enumerate(series)
    )
    return f'<div class="viz-legend">{chips}</div>'


def chart_figure(spec: dict, exp: dict, index: int, plain: dict | None = None) -> str:
    """Full chart block: caption, optional plain framing, svg, legend, data-table
    twin, source. Practitioner framing (plain) demotes the jargon title+note."""
    plain = plain or {}
    has_plain = bool(str(plain.get("chart_read", "")).strip() or str(plain.get("chart_plain_title", "")).strip())
    svg = chart_svg(spec, mini=False, uid=f'{exp["id"]}-{index}', plain_axes=has_plain)
    legend = _chart_legend(spec)
    table = _chart_table(spec)
    source = str(spec.get("source", ""))
    if "/" in source or source.endswith((".json", ".csv")):
        src_html = f'<a href="{GITHUB}/blob/main/experiments/{esc(exp["id"])}/{esc(source)}"><code>{esc(source)}</code></a>'
    else:
        src_html = esc(source)
    # axis line (only when no plain "how to read" panel already translates the axes)
    axes = ""
    if not has_plain and spec["kind"] in {"bar", "line", "scatter"}:
        yl, xl = str(spec.get("y_label", "")).strip(), str(spec.get("x_label", "")).strip()
        bits = []
        if yl:
            bits.append(f'<b>{esc(yl)}</b>' + (" ↑" if spec["kind"] != "bar" else ""))
        if xl:
            bits.append(f'{esc(xl)} →')
        if bits:
            axes = f'<p class="viz-axes muted">{" · ".join(bits)}</p>'

    plain_title = str(plain.get("chart_plain_title", "")).strip()
    read = str(plain.get("chart_read", "")).strip()
    takeaway = str(plain.get("chart_takeaway", "")).strip()
    tech_note = str(spec.get("note", "")).strip()

    if plain_title or read or takeaway:
        headline = plain_title or str(spec.get("title", ""))
        caption = f'<figcaption><strong>{esc(headline)}</strong></figcaption>'
        read_html = f'<div class="viz-read"><p class="kicker">How to read</p><p>{esc(read)}</p></div>' if read else ""
        take_html = f'<p class="viz-takeaway"><span>Takeaway →</span> {esc(takeaway)}</p>' if takeaway else ""
        tech_bits = [b for b in (str(spec.get("title", "")).strip() if plain_title else "", tech_note) if b]
        tech_html = (
            f'<details class="viz-technical"><summary>Technical framing</summary><p class="muted">{esc(" — ".join(tech_bits))}</p></details>'
            if tech_bits else ""
        )
        return (
            f'<figure class="viz-chart">{caption}{read_html}{svg}{take_html}{legend}{table}'
            f'<p class="viz-src muted">Numbers from {src_html}</p>{tech_html}</figure>'
        )
    note = f' <span class="muted">{esc(tech_note)}</span>' if tech_note else ""
    return (
        f'<figure class="viz-chart"><figcaption><strong>{esc(spec.get("title", ""))}</strong>{note}</figcaption>'
        f'{axes}{svg}{legend}{table}'
        f'<p class="viz-src muted">Numbers from {src_html}</p></figure>'
    )


def stat_tile(label: str, value: str, sub: str = "", href: str = "") -> str:
    body = (
        f'<span class="stat-label">{esc(label)}</span>'
        f'<span class="stat-value">{esc(value)}</span>'
        + (f'<span class="stat-sub">{esc(sub)}</span>' if sub else "")
    )
    if href:
        return f'<a class="stat-tile" href="{href}">{body}</a>'
    return f'<div class="stat-tile">{body}</div>'


def feed_card(
    exp: dict, prefix: str, slots: dict[str, int], titles: dict[str, str], *, big: bool, rich=None, chart: str = ""
) -> str:
    """Card visual is a native mini-chart of the experiment's own numbers —
    never a raw figure PNG (user preference: the data deserves better)."""
    url = f'{prefix}experiments/{esc(exp["id"])}/'
    thumb = ""
    if big and chart:
        thumb = f'<a class="card-thumb" href="{url}" tabindex="-1" aria-hidden="true">{chart}</a>'
    chips = "".join(program_chip(pid, prefix, slots, titles) for pid in exp["programs"][:3])
    if not big:
        chips += "".join(f'<span class="chip tag">{esc(tag)}</span>' for tag in exp["tags"][:3])
    meta_bits = [date_span(exp)]
    if exp.get("status") == "in-progress":  # finished is the norm; flag only the live frontier
        meta_bits.append(exp_status_chip(exp["status"]))
    meta_bits.append(track_chip(exp["track"]))
    if exp["figures"]:
        meta_bits.append(f'<span class="muted">{len(exp["figures"])} figure{"s" if len(exp["figures"]) != 1 else ""}</span>')
    summary = exp.get("card_summary") or exp["finding"]
    finding = rich(summary, prefix) if rich else esc(summary)
    # verdict headline: a tone-colored dot + the crisp standalone verdict tag,
    # the one-line "what we learned" a reader scans before the fuller answer.
    verdict = ""
    tag = exp.get("verdict_tag")
    if tag:
        tone = exp.get("outcome") or "neutral"
        verdict = (
            f'<p class="card-verdict tone-{tone}">'
            f'<span class="verdict-dot" aria-hidden="true"></span>'
            f'<span class="card-verdict-tag">{esc(tag)}</span></p>'
        )
    return (
        f'<article class="feed-card{" big" if big else ""}">{thumb}<div class="card-body">'
        f'<div class="card-meta">{"".join(meta_bits)}</div>'
        f'<h3><a href="{url}">{esc(exp["title"])}</a></h3>'
        f'{verdict}'
        f'<p class="card-finding">{finding}</p>'
        f'<div class="card-chips">{chips}</div>'
        f"</div></article>"
    )


# -------------------------------------------------------------------- pages


class SiteBuilder:
    def __init__(self, out_dir: Path):
        self.out = out_dir
        self.shell = load_shell()
        self.generated = dt.date.today().isoformat()
        self.programs = load_programs()
        self.program_titles = {str(p["id"]): _nice_title(str(p.get("title") or p["id"])) for p in self.programs}
        self.experiments = build_experiments(self.programs)
        self.roster = {exp["id"] for exp in self.experiments}
        self.by_id = {exp["id"]: exp for exp in self.experiments}
        ledger = read_json(KNOWLEDGE / "claims" / "claim_ledger.json")
        self.claims = list(ledger.get("claims", [])) if isinstance(ledger, dict) else []
        # Optional plain-language layer for claims (knowledge/claims/claim_plain.json):
        # a fact-checked, jargon-free rewrite of each claim's title/summary/implication
        # keyed by claim id. Falls back to the raw ledger fields when absent.
        plain_path = KNOWLEDGE / "claims" / "claim_plain.json"
        plain_claims = read_json(plain_path) if plain_path.exists() else {}
        self.claim_plain = plain_claims.get("claims", {}) if isinstance(plain_claims, dict) else {}
        queue = read_json(KNOWLEDGE / "future_experiment_queue.json")
        self.queue = [p for p in queue.get("proposals", []) if isinstance(p, dict)]
        self.candidate_programs = [c for c in queue.get("candidate_programs", []) if isinstance(c, dict)]
        counts = Counter(pid for exp in self.experiments for pid in exp["programs"])
        ranked = [pid for pid, _ in counts.most_common()]
        # every program gets a colored dot — cycle the 8 hue slots rather than
        # dropping the 9th+ programs to an indistinct grey.
        self.slots = {pid: (idx % 8) + 1 for idx, pid in enumerate(ranked)}
        self.claims_by_exp: dict[str, list[dict]] = defaultdict(list)
        for claim in self.claims:
            for evidence in claim.get("evidence", []):
                if evidence.get("kind") == "experiment":
                    self.claims_by_exp[str(evidence.get("id"))].append(claim)
        self.program_ids = {str(p["id"]) for p in self.programs}
        self.claim_anchor_ids = {slugify(str(claim.get("id"))) for claim in self.claims}
        self.viz = load_viz()
        self.briefs = load_briefs()
        self.status_map = load_experiment_status()
        for exp in self.experiments:
            # Finished is the DEFAULT. In-progress must be an explicit, dated,
            # reasoned declaration in knowledge/experiment_status.json — never
            # inferred from the report, because preregistrations and finished
            # ablations share the same "verdict / negative / not run" vocabulary,
            # so any prose guess silently mislabels in both directions. Explicit-
            # only means nothing rots stuck in-progress; CI (validate_repository.py)
            # keeps the in-progress list honest (dated, reasoned, non-stale).
            meta = self.status_map.get(exp["id"]) or {}
            exp["status"] = meta["status"] if meta.get("status") in EXPERIMENT_STATUS else "finished"
            exp["status_since"] = meta.get("since", "")
            exp["status_reason"] = meta.get("reason", "")
            brief = self.briefs.get(exp["id"], {})
            tone = str(brief.get("verdict_tone", "")).strip()
            exp["outcome"] = tone if tone in {"positive", "negative", "mixed", "neutral"} else ""
            # cards/feed lead with the plain-language, standalone brief answer — not the
            # jargon-dense finding excerpted from the report; the finding stays on the page.
            exp["card_summary"] = str(brief.get("plain_answer", "")).strip() or exp["finding"]
            exp["verdict_tag"] = str(brief.get("verdict_tag", "")).strip()
        self.dropped_notes: list[str] = []

    # ------------------------------------------------------------- utilities

    def write_page(self, rel: str, **kwargs) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page_html(self.shell, generated=self.generated, **kwargs), encoding="utf-8")

    def render_doc(self, text: str, resolver: LinkResolver, slug_prefix: str) -> site_markdown.RenderResult:
        return render_markdown(strip_leading_h1(text), resolver=resolver, slug_prefix=slug_prefix, heading_shift=1)

    def chip_row(self, exp: dict, prefix: str, limit: int | None = None) -> str:
        pids = exp["programs"]
        shown = pids if limit is None else pids[:limit]
        chips = "".join(program_chip(pid, prefix, self.slots, self.program_titles) for pid in shown)
        extra = len(pids) - len(shown)
        if extra > 0:
            chips += f'<span class="chip tag" title="{esc(", ".join(self.program_titles.get(p, p) for p in pids[limit:]))}">+{extra} more</span>'
        return chips

    def rich_text(self, text: str, prefix: str) -> str:
        """Escape plain text, then link claim ids and experiment slugs it mentions."""
        out = linkify_claims(esc(text), prefix, self.claim_anchor_ids)
        return linkify_experiments(out, prefix, self.roster)

    def practitioner_brief(self, exp: dict) -> str:
        """Verdict banner + plain-language brief card for the top of the page.

        Plain fields are escaped only (never claim/experiment linkified — the
        brief is deliberately jargon-free per the content ban-list)."""
        brief = self.briefs.get(exp["id"])
        if not brief:
            return ""
        tone = str(brief.get("verdict_tone", "neutral")).strip() or "neutral"
        if tone not in {"positive", "negative", "mixed", "neutral"}:
            tone = "neutral"
        tag = str(brief.get("verdict_tag", "")).strip()
        banner = (
            f'<div class="verdict-banner tone-{tone}"><span class="verdict-dot" aria-hidden="true"></span>'
            f'<span class="verdict-tag">{esc(tag)}</span></div>'
            if tag
            else ""
        )
        rows = []
        for label, field in (
            ("The one idea you need", "concept_primer"),
            ("The question", "plain_question"),
            ("What we found", "plain_answer"),
            ("Why it matters", "why_it_matters"),
        ):
            value = str(brief.get(field, "")).strip()
            if value:
                rows.append(f'<div class="brief-row"><p class="kicker">{label}</p><p>{esc(value)}</p></div>')
        kpis = ""
        numbers = brief.get("key_numbers", [])
        if isinstance(numbers, list):
            tiles = "".join(
                stat_tile(str(n.get("label", "")), str(n.get("value", "")), str(n.get("sub", "")))
                for n in numbers[:4]
                if isinstance(n, dict) and n.get("value")
            )
            if tiles:
                kpis = f'<div class="brief-kpis stat-row">{tiles}</div>'
        card = f'<div class="practitioner-brief">{"".join(rows)}{kpis}</div>' if (rows or kpis) else ""
        return banner + card

    def brief_chart_plain(self, exp_id: str) -> dict[int, dict]:
        brief = self.briefs.get(exp_id)
        out: dict[int, dict] = {}
        if brief and isinstance(brief.get("charts"), list):
            for chart in brief["charts"]:
                if isinstance(chart, dict) and isinstance(chart.get("index"), int):
                    out[chart["index"]] = chart
        return out

    def headline_chart(self, exp_id: str) -> str:
        specs = self.viz.get(exp_id, [])
        if not specs:
            return ""
        spec = next((s for s in specs if s.get("headline")), specs[0])
        return chart_svg(spec, mini=True, uid=f"{exp_id}-mini")

    # ------------------------------------------------------------ experiment

    def copy_experiment_files(self, exp: dict) -> set[str]:
        exp_dir = EXPERIMENTS_DIR / exp["id"]
        copied: set[str] = set()
        for rel in list(exp["figures"]) + list(exp["data_files"]):
            src = exp_dir / rel
            dest = self.out / "experiments" / exp["id"] / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            copied.add(Path(rel).as_posix())
        return copied

    def experiment_page(self, exp: dict, prev_exp: dict | None, next_exp: dict | None) -> None:
        prefix = "../../"
        exp_id = exp["id"]
        exp_dir = EXPERIMENTS_DIR / exp_id
        copied = self.copy_experiment_files(exp)

        sections: list[tuple[str, str]] = []
        side_toc: list[str] = []

        def add_section(anchor: str, heading: str, body: str, sub_toc: list[tuple[int, str, str]] | None = None) -> None:
            sections.append((anchor, f'<section id="{anchor}" class="doc-section"><h2>{heading}</h2>{body}</section>'))
            subs = ""
            if sub_toc:
                top = min((level for level, _, _ in sub_toc), default=0)
                items = "".join(
                    f'<li><a href="#{esc(slug)}">{esc(text)}</a></li>' for level, slug, text in sub_toc if level == top
                )
                if items:
                    subs = f"<ol>{items}</ol>"
            side_toc.append(f'<li><a href="#{anchor}">{heading.split("<")[0] if "<" in heading else heading}</a>{subs}</li>')

        viz_specs = self.viz.get(exp_id, [])
        if viz_specs:
            plain_charts = self.brief_chart_plain(exp_id)
            blocks = "".join(
                chart_figure(spec, exp, index, plain=plain_charts.get(index))
                for index, spec in enumerate(viz_specs)
            )
            add_section(
                "charts",
                f'Results at a glance <span class="count">{len(viz_specs)}</span>',
                f'<div class="viz-grid">{blocks}</div>',
            )

        if exp["readme_text"]:
            resolver = ExperimentResolver(exp, exp["readme_path"], prefix, self.roster, copied)
            readme = self.render_doc(exp["readme_text"], resolver, "readme-")
            # bare "C49"/experiment-slug mentions become links to their evidence
            html = linkify_experiments(linkify_claims(readme.html, prefix, self.claim_anchor_ids), prefix, self.roster)
            add_section("readme", "Overview", html)
        if exp["report_text"]:
            resolver = ExperimentResolver(exp, exp["report_path"], prefix, self.roster, copied)
            report = self.render_doc(exp["report_text"], resolver, "report-")
            gh = f"{GITHUB}/blob/main/{exp['report_path'].relative_to(ROOT).as_posix()}"
            html = linkify_experiments(linkify_claims(report.html, prefix, self.claim_anchor_ids), prefix, self.roster)
            body = f'<p class="doc-src muted">Rendered from <a href="{esc(gh)}">{esc(exp["report_rel"])}</a></p>{html}'
            add_section("report", "Report", body, report.toc)
        if exp["log_path"]:
            resolver = ExperimentResolver(exp, exp["log_path"], prefix, self.roster, copied)
            log_text = read_text(exp["log_path"])
            log = self.render_doc(log_text, resolver, "log-")
            log_dates = sorted(set(re.findall(r"^#{1,6}[^\n]*?\b(20\d{2}-\d{2}-\d{2})", log_text, re.M)))
            entry_count = len(re.findall(r"^#{2,6}\s", log_text, re.M))
            scent_bits = []
            if entry_count:
                scent_bits.append(f"{entry_count} entr{'ies' if entry_count != 1 else 'y'}")
            if log_dates:
                scent_bits.append(format_range(log_dates[0], log_dates[-1]))
            scent = f" ({', '.join(scent_bits)})" if scent_bits else ""
            log_html = linkify_experiments(linkify_claims(log.html, prefix, self.claim_anchor_ids), prefix, self.roster)
            body = f'<details class="log-details"><summary>Show the running log{esc(scent)}</summary>{log_html}</details>'
            heading = "Experiment log" + (f' <span class="count">{entry_count}</span>' if entry_count else "")
            add_section("log", heading, body)

        if exp["figures"]:
            cards = []
            for rel in exp["figures"]:
                rel_posix = Path(rel).as_posix()
                caption = Path(rel).stem.replace("_", " ")
                parent = Path(rel).parent.as_posix()
                where = f'<span class="muted"> · {esc(parent)}/</span>' if parent != "." else ""
                size = png_size(exp_dir / rel)
                dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
                cards.append(
                    f'<figure class="fig-card"><a class="fig-link" href="files/{esc(rel_posix)}">'
                    f'<img src="files/{esc(rel_posix)}" alt="{esc(caption)}"{dims} loading="lazy"></a>'
                    f"<figcaption>{esc(caption)}{where}</figcaption></figure>"
                )
            add_section("figures", f'Figures <span class="count">{len(cards)}</span>', f'<div class="fig-grid">{"".join(cards)}</div>')

        if exp["data_files"]:
            rows = []
            for rel in exp["data_files"]:
                rel_posix = Path(rel).as_posix()
                size = (exp_dir / rel).stat().st_size
                rows.append(
                    f'<li class="data-row"><button class="data-preview" type="button" data-src="files/{esc(rel_posix)}">preview</button>'
                    f'<a href="files/{esc(rel_posix)}"><code>{esc(rel_posix)}</code></a>'
                    f'<span class="muted">{format_size(size)}</span></li>'
                )
            more = ""
            if exp["data_dropped"]:
                more = (
                    f'<p class="muted">{exp["data_dropped"]} more result file{"s" if exp["data_dropped"] != 1 else ""} '
                    f'not shown here — <a href="{GITHUB}/tree/main/{esc(exp["path"])}">browse the full folder on GitHub</a>.</p>'
                )
            body = (
                '<p class="muted">Result tables and metrics copied from the experiment folder — preview inline or open the raw file.</p>'
                f'<ul class="data-list">{"".join(rows)}</ul>{more}'
                '<div class="data-viewer" hidden><div class="data-viewer-head"><code class="data-viewer-name"></code>'
                '<button class="data-viewer-close" type="button">close</button></div><div class="data-viewer-body"></div></div>'
            )
            add_section("data", f'Data files <span class="count">{len(exp["data_files"])}</span>', body)

        smoke, full, has_manifest = full_command_from_manifest(exp_dir)
        # readiness CSV's smoke_command is a yes/no availability flag, never a command
        if smoke.lower() in {"yes", "no", "none", ""}:
            smoke = ""
        if full.lower() in {"yes", "no", "none"}:
            full = ""
        repro_bits = []
        if smoke:
            repro_bits.append(f"<p>Smoke test</p><pre><code>{esc(smoke)}</code></pre>")
        if full:
            repro_bits.append(f"<p>Full run</p><pre><code>{esc(full)}</code></pre>")
        surface_note = RUN_SURFACE_NOTE.get(exp["run_surface"], "")
        if exp["run_surface"] == "documented-command" and not has_manifest:
            surface_note = "The run commands are documented inside the experiment folder (see the README)."
        if surface_note:
            repro_bits.append(f'<p class="muted" title="run surface: {esc(exp["run_surface"])}">{esc(surface_note)}</p>')
        if repro_bits:
            repro_bits.append(
                f'<p><a href="{GITHUB}/tree/main/{esc(exp["path"])}">Browse the experiment folder on GitHub ↗</a></p>'
            )
            add_section("repro", "Reproduce", "".join(repro_bits))

        related_bits = []
        cited = self.claims_by_exp.get(exp_id, [])
        if cited:
            items = "".join(
                f'<li>{status_chip(str(c.get("status")))} <a href="{prefix}claims/#{slugify(str(c.get("id")))}">'
                f'{esc(c.get("id"))} · {esc(c.get("title"))}</a></li>'
                for c in cited
            )
            related_bits.append(f"<h3>Cited by claims</h3><ul class=\"plain\">{items}</ul>")
        generic = {"qwen35", "4b", "qwen", "executor", "compiler", "experiment", "pilot", "probe", "ladder", "audit"}
        own_tokens = set(exp_id.split("_")) - generic
        own_tags = set(exp["tags"])

        def kinship(other: dict) -> tuple:
            shared_tokens = len((set(other["id"].split("_")) - generic) & own_tokens)
            shared_programs = len(set(other["programs"]) & set(exp["programs"]))
            shared_tags = len(set(other["tags"]) & own_tags)
            return (shared_tokens * 3 + shared_tags + shared_programs, other["when"] or "")

        candidates = [other for other in self.experiments if other["id"] != exp_id and kinship(other)[0] > 0]
        candidates.sort(key=kinship, reverse=True)
        if candidates:
            items = "".join(
                f'<li>{date_span(o)} <a href="{prefix}experiments/{esc(o["id"])}/">{esc(o["title"])}</a></li>'
                for o in candidates[:6]
            )
            related_bits.append(
                '<h3>Related experiments</h3>'
                '<p class="muted">Nearest by shared name, tags, and programs.</p>'
                f'<ul class="plain">{items}</ul>'
            )
        if related_bits:
            add_section("related", "Related", "".join(related_bits))

        pager = []
        if prev_exp:  # feed order is newest-first, so the previous entry is newer
            pager.append(
                f'<a class="pager prev" href="../{esc(prev_exp["id"])}/"><span class="pager-dir">← Newer</span>'
                f'<span>{esc(prev_exp["title"])}</span></a>'
            )
        if next_exp:
            pager.append(
                f'<a class="pager next" href="../{esc(next_exp["id"])}/"><span class="pager-dir">Older →</span>'
                f'<span>{esc(next_exp["title"])}</span></a>'
            )

        finding_block = ""
        if exp["finding"]:
            source = exp["finding_source"]
            section_name = source.split("·", 1)[1].strip() if "·" in source else ""
            if source.lower().startswith("readme"):
                anchor, origin_label = "#readme", "the Overview"
            elif source.lower().startswith("report"):
                anchor, origin_label = "#report", "the Report"
            else:
                anchor, origin_label = "", "the catalog summary"
            origin = f'<a href="{anchor}">{origin_label}</a>' if anchor else origin_label
            src_label = f"from {origin}" + (f" · “{esc(section_name)}”" if section_name else "")
            # never dead-end a truncated excerpt: link straight to the full section on-page
            more = (
                f' <a class="finding-more" href="{anchor}">Read the full result →</a>'
                if exp["finding"].rstrip().endswith("…") and anchor else ""
            )
            finding_block = (
                f'<div class="finding-callout"><p class="finding-label">In the author’s words <span class="muted">{src_label}</span></p>'
                f'<p>{self.rich_text(exp["finding"], prefix)}{more}</p></div>'
            )

        status_note = ""
        if exp["status"] == "in-progress" and exp.get("status_since"):
            reason = f' — {esc(exp["status_reason"])}' if exp.get("status_reason") else ""
            status_note = (
                f'<p class="status-note">In progress since '
                f'<time datetime="{esc(exp["status_since"])}">{esc(exp["status_since"])}</time>{reason}</p>'
            )
        header = (
            f'<nav class="crumbs"><a href="{prefix}">Home</a> / <a href="{prefix}experiments/">Experiments</a> / '
            f"<span>{esc(exp['title'])}</span></nav>"
            f"<h1>{esc(exp['title'])}</h1>"
            f'<div class="page-meta">{exp_status_chip(exp["status"])}{date_span(exp)}{track_chip(exp["track"])}'
            f"{self.chip_row(exp, prefix, limit=4)}"
            f'<a class="chip gh" href="{GITHUB}/tree/main/{esc(exp["path"])}">GitHub ↗</a></div>'
            f"{status_note}"
        )

        toc_list = f'<ol>{"".join(side_toc)}</ol>'
        mobile_toc = (
            f'<details class="mobile-toc"><summary>On this page</summary>{toc_list}</details>' if side_toc else ""
        )
        brief_block = self.practitioner_brief(exp)
        # With a plain brief, it becomes the headline; the jargon finding callout
        # is demoted and slots in just below the charts, above the full documents.
        top_block = brief_block or finding_block
        demoted_finding = ""
        if brief_block and finding_block:
            # the plain brief is the headline; collapse the report's raw technical
            # phrasing behind a details so it doesn't re-wall the page below the charts
            demoted_finding = (
                finding_block
                .replace('<div class="finding-callout">', '<details class="finding-callout demoted">', 1)
                .replace('<p class="finding-label">', '<summary class="finding-label">', 1)
                .replace("</p><p>", "</summary><p>", 1)
                .replace("</div>", "</details>", 1)
            )
        body_parts: list[str] = []
        placed_finding = False
        for anchor, body in sections:
            body_parts.append(body)
            if anchor == "charts" and demoted_finding:
                body_parts.append(demoted_finding)
                placed_finding = True
        if demoted_finding and not placed_finding:
            body_parts.insert(0, demoted_finding)
        content = (
            f'<div class="detail-layout"><div class="detail-main">{header}{top_block}{mobile_toc}'
            + "".join(body_parts)
            + (f'<div class="pager-row">{"".join(pager)}</div>' if pager else "")
            + "</div>"
            + f'<aside class="detail-side"><nav class="side-toc" aria-label="On this page"><p>On this page</p>{toc_list}</nav></aside></div>'
        )
        self.write_page(
            f"experiments/{exp_id}/index.html",
            title=f"{exp['title']} · {SITE_NAME}",
            description=exp["finding"][:300] or exp["summary"][:300],
            prefix=prefix,
            active="experiments",
            content=content,
        )

    # -------------------------------------------------------------- explorer

    def explorer_page(self) -> None:
        prefix = "../"
        options = "".join(
            f'<option value="{esc(pid)}">{esc(self.program_titles.get(pid, pid))}</option>'
            for pid in sorted(self.program_ids, key=lambda p: self.program_titles.get(p, p))
        )
        tracks = "".join(
            f'<option value="{esc(track)}">{esc(IMPORT_TRACK_LABEL.get(track, track))}</option>'
            for track in sorted({exp["track"] for exp in self.experiments})
        )
        cards = []
        for index, exp in enumerate(self.experiments):
            text = " ".join(
                [exp["id"], exp["title"], exp["finding"], " ".join(exp["tags"]), " ".join(exp["programs"])]
            ).lower()
            cards.append(
                f'<li class="explorer-item" data-programs="{esc(" ".join(exp["programs"]))}" data-track="{esc(exp["track"])}" '
                f'data-status="{esc(exp["status"])}" data-outcome="{esc(exp["outcome"])}" '
                f'data-date="{esc(exp["when"] or "")}" data-figs="{len(exp["figures"])}" data-title="{esc(exp["title"].lower())}" '
                f'data-rank="{index}" data-text="{esc(text)}">'
                + feed_card(exp, prefix, self.slots, self.program_titles, big=False, rich=self.rich_text)
                + "</li>"
            )
        n = len(self.experiments)
        n_finished = sum(1 for e in self.experiments if e["status"] == "finished")
        n_inprog = n - n_finished
        n_import = sum(1 for exp in self.experiments if exp["track"] != "new")

        def seg(value, label, count):
            sel = "true" if value == "finished" else "false"
            return (
                f'<button type="button" role="tab" class="seg-btn" data-status-tab="{value}" aria-selected="{sel}">'
                f'{label} <span class="seg-n">{count}</span></button>'
            )
        segmented = (
            '<div class="seg" role="tablist" aria-label="Experiment status">'
            + seg("finished", "Finished", n_finished)
            + seg("in-progress", "In progress", n_inprog)
            + seg("all", "All", n)
            + "</div>"
        )
        outcomes = (
            '<select id="explorer-outcome" aria-label="Outcome"><option value="">Any outcome</option>'
            '<option value="positive">Wins</option><option value="mixed">Mixed</option>'
            '<option value="negative">Ruled out</option><option value="neutral">Other</option></select>'
        )
        content = (
            '<header class="page-head"><h1>Experiments</h1>'
            f'<p class="lede">Every experiment is self-contained — its own question, code, data, and result. '
            f'{n} in all, newest first: <strong>{n_finished} finished</strong>, {n_inprog} in progress.</p>'
            f'<details class="explorer-provenance"><summary>About dates &amp; provenance</summary>'
            f'<p class="muted">Ordering follows each experiment’s own run window (recovered from records inside its '
            f'folder), not when a file was last touched. {n_import} experiments arrived in the 2026-06-28 bulk import from '
            'the predecessor working repo — its two parallel working tracks show here as provenance lines Y and Z.</p></details>'
            '</header>'
            + segmented
            + '<form id="explorer-controls" class="filter-row" onsubmit="return false">'
            '<input id="explorer-search" type="search" placeholder="Search experiments…" aria-label="Search experiments">'
            + outcomes
            + f'<select id="explorer-program" aria-label="Program"><option value="">All programs</option>{options}</select>'
            f'<select id="explorer-track" aria-label="Provenance"><option value="">Any provenance</option>{tracks}</select>'
            '<select id="explorer-sort" aria-label="Sort"><option value="date">Newest first</option>'
            '<option value="date-asc">Oldest first</option>'
            '<option value="title">Title A–Z</option><option value="figs">Most figures</option></select>'
            '<span id="explorer-count" class="result-count" aria-live="polite"></span>'
            '<button id="filter-reset" type="button" hidden>Reset ✕</button></form>'
            + activity_chart(self.experiments)
            + f'<ol id="explorer" class="explorer-list">{"".join(cards)}</ol>'
            '<p id="explorer-empty" class="empty-note" hidden>No experiments match these filters '
            '<span id="empty-scope" class="muted"></span>. '
            '<button id="explorer-clear" type="button">Clear filters</button></p>'
        )
        self.write_page(
            "experiments/index.html",
            title=f"Experiments · {SITE_NAME}",
            description="Every experiment in the corpus, newest first, with its finished/in-progress status and outcome.",
            prefix=prefix,
            active="experiments",
            content=content,
        )

    # -------------------------------------------------------------- programs

    def program_pages(self) -> None:
        prefix = "../"
        counts = Counter(pid for exp in self.experiments for pid in exp["programs"])
        claim_counts = Counter(pid for claim in self.claims for pid in claim.get("programs", []))
        queue_counts = Counter(pid for item in self.queue for pid in item.get("programs", []))
        cards = []
        for program in sorted(self.programs, key=lambda p: -counts[str(p["id"])]):
            pid = str(program["id"])
            slot = self.slots.get(pid, 0)
            cards.append(
                f'<a class="program-card" data-slot="{slot}" href="{esc(pid)}/">'
                f'<span class="dot" aria-hidden="true"></span><h3>{esc(self.program_titles.get(pid, program.get("title") or pid))}</h3>'
                f'<p>{esc(program.get("focus", ""))}</p>'
                f'<p class="muted">{_plural(counts[pid], "experiment")} · {_plural(claim_counts[pid], "claim")} · {queue_counts[pid]} queued</p></a>'
            )
        content = (
            '<header class="page-head"><h1>Research programs</h1>'
            '<p class="lede">Durable lines of inquiry. Each program page carries its charter, the evidence gathered so far, and every experiment that advanced it.</p></header>'
            f'<div class="program-grid">{"".join(cards)}</div>'
        )
        self.write_page(
            "programs/index.html",
            title=f"Programs · {SITE_NAME}",
            description="Research programs: charters, evidence, and their experiments.",
            prefix=prefix,
            active="programs",
            content=content,
        )

        scorecards = read_text(KNOWLEDGE / "program_scorecards.md") if (KNOWLEDGE / "program_scorecards.md").exists() else ""
        score_sections = {head: body for head, _, body in _sections(scorecards)}
        for program in self.programs:
            pid = str(program["id"])
            page_prefix = "../../"
            pdir = PROGRAMS_DIR / pid
            parts: list[str] = []
            title = str(program.get("title") or pid)
            for name, heading, collapsed in (
                ("evidence.md", "What we have learned", False),
                ("charter.md", "Charter", True),
                ("backlog.md", "Backlog", True),
            ):
                doc = pdir / name
                if not doc.exists():
                    continue
                resolver = KnowledgeResolver(doc, page_prefix, self.roster, self.program_ids)
                rendered = render_markdown(strip_leading_h1(read_text(doc)), resolver=resolver, slug_prefix=f"{name.split('.')[0]}-", heading_shift=1)
                doc_html = linkify_experiments(
                    linkify_claims(rendered.html, page_prefix, self.claim_anchor_ids), page_prefix, self.roster
                )
                if collapsed:
                    parts.append(
                        f'<section class="doc-section"><h2>{heading}</h2><details><summary>Show {name}</summary>{doc_html}</details></section>'
                    )
                else:
                    parts.append(f'<section class="doc-section"><h2>{heading}</h2>{doc_html}</section>')
            if title in score_sections:
                resolver = KnowledgeResolver(KNOWLEDGE / "program_scorecards.md", page_prefix, self.roster, self.program_ids)
                rendered = render_markdown(score_sections[title], resolver=resolver, slug_prefix="score-", heading_shift=2)
                parts.insert(1, f'<section class="doc-section"><h2>Scorecard</h2>{linkify_claims(rendered.html, page_prefix, self.claim_anchor_ids)}</section>')

            exps = [exp for exp in self.experiments if pid in exp["programs"]]
            rows = "".join(
                f'<li>{date_span(exp)} '
                f'<a href="{page_prefix}experiments/{esc(exp["id"])}/">{esc(exp["title"])}</a>'
                f'<p class="muted clamp">{esc((exp.get("card_summary") or exp["finding"])[:220])}</p></li>'
                for exp in exps
            )
            parts.append(
                f'<section class="doc-section"><h2>Experiments <span class="count">{len(exps)}</span></h2><ul class="plain exp-list">{rows}</ul></section>'
            )
            claims = [claim for claim in self.claims if pid in claim.get("programs", [])]
            if claims:
                items = "".join(
                    f'<li>{status_chip(str(c.get("status")))} <a href="{page_prefix}claims/#{slugify(str(c.get("id")))}">'
                    f'{esc(c.get("id"))} · {esc(c.get("title"))}</a></li>'
                    for c in claims
                )
                parts.append(f'<section class="doc-section"><h2>Claims</h2><ul class="plain">{items}</ul></section>')

            queued = [item for item in self.queue if pid in item.get("programs", [])]
            if queued:
                queue_items = "".join(
                    f'<li><span class="chip prio prio-{esc(str(item.get("priority", "")).lower())}" title="Queue priority: P0 = do next, P2 = later">{esc(item.get("priority", ""))}</span> '
                    f'<a href="{page_prefix}queue/#{esc(slugify(str(item.get("id") or item.get("title", ""))))}">'
                    f'{esc(item.get("title", ""))}</a></li>'
                    for item in queued
                )
                parts.append(
                    f'<section class="doc-section"><h2>Queued proposals <span class="count">{len(queued)}</span></h2>'
                    f'<ul class="plain">{queue_items}</ul></section>'
                )

            disp_title = _nice_title(title)
            header = (
                f'<nav class="crumbs"><a href="{page_prefix}">Home</a> / <a href="{page_prefix}programs/">Programs</a> / <span>{esc(disp_title)}</span></nav>'
                f"<h1>{esc(disp_title)}</h1><p class=\"lede\">{esc(program.get('focus', ''))}</p>"
            )
            self.write_page(
                f"programs/{pid}/index.html",
                title=f"{title} · {SITE_NAME}",
                description=str(program.get("focus", "")),
                prefix=page_prefix,
                active="programs",
                content=f'<div class="doc-page">{header}{"".join(parts)}</div>',
            )

    # ---------------------------------------------------------------- claims

    def claims_page(self) -> None:
        prefix = "../"
        counts = Counter(str(claim.get("status")) for claim in self.claims)
        filter_tiles = (
            f'<button type="button" class="claim-filter" data-claim-status="" aria-pressed="true">'
            f'<span class="stat-label">All</span><span class="stat-value">{len(self.claims)}</span></button>'
            + "".join(
                f'<button type="button" class="claim-filter" data-claim-status="{esc(status)}" aria-pressed="false" '
                f'title="{esc(CLAIM_STATUS_META[status][1])}"><span class="stat-label">{esc(status)}</span>'
                f'<span class="stat-value">{counts.get(status, 0)}</span></button>'
                for status in CLAIM_STATUS_ORDER if counts.get(status)
            )
        )
        groups: list[str] = []
        for status in CLAIM_STATUS_ORDER:
            cards = []
            for claim in self.claims:
                if str(claim.get("status")) != status:
                    continue
                cid = str(claim.get("id"))
                _p = self.claim_plain.get(cid)
                _p = _p if isinstance(_p, dict) else {}
                c_title = _p.get("title") or claim.get("title")
                c_summary = _p.get("summary") or claim.get("summary")
                c_implication = _p.get("implication") or claim.get("implication")
                c_next_tests = _p.get("next_tests") or claim.get("next_tests", [])
                c_avoid = _p.get("avoid") or claim.get("avoid", [])
                evidence_items = []
                for evidence in claim.get("evidence", []):
                    if evidence.get("kind") == "experiment" and str(evidence.get("id")) in self.roster:
                        eid = str(evidence.get("id"))
                        evidence_items.append(
                            f'<li>{date_span(self.by_id[eid])} '
                            f'<a href="{prefix}experiments/{esc(eid)}/">{esc(self.by_id[eid]["title"])}</a>'
                            f'<p class="muted clamp">{esc(self.by_id[eid]["finding"][:180])}</p></li>'
                        )
                    elif evidence.get("kind") == "doc":
                        path = str(evidence.get("path", ""))
                        evidence_items.append(f'<li><a href="{GITHUB}/blob/main/{esc(path)}">{esc(path)}</a></li>')
                    elif evidence.get("kind") == "program":
                        pid = str(evidence.get("id"))
                        evidence_items.append(f'<li><a href="{prefix}programs/{esc(pid)}/">{esc(self.program_titles.get(pid, pid))}</a></li>')
                extras = []
                if c_implication:
                    extras.append(f'<p class="claim-implication"><strong>Implication.</strong> {esc(c_implication)}</p>')
                if evidence_items:
                    extras.append(f'<details><summary>Evidence ({len(evidence_items)})</summary><ul class="plain">{"".join(evidence_items)}</ul></details>')
                next_tests = [str(t) for t in c_next_tests]
                if next_tests:
                    items = "".join(f"<li>{esc(t)}</li>" for t in next_tests)
                    extras.append(f'<details><summary>Next tests ({len(next_tests)})</summary><ul class="plain">{items}</ul></details>')
                avoid = [str(t) for t in c_avoid]
                if avoid:
                    items = "".join(f"<li>{esc(t)}</li>" for t in avoid)
                    extras.append(f'<details><summary>Avoid</summary><ul class="plain">{items}</ul></details>')
                chips = "".join(program_chip(pid, prefix, self.slots, self.program_titles) for pid in claim.get("programs", []))
                ctext = " ".join([cid, str(c_title or ""), str(c_summary or ""), str(c_implication or ""),
                                  " ".join(self.program_titles.get(p, p) for p in claim.get("programs", []))]).lower()
                cards.append(
                    f'<article class="claim-card" id="{esc(slugify(cid))}" data-status="{esc(status)}" data-text="{esc(ctext)}">'
                    f'<div class="card-meta">{status_chip(status)}<span class="claim-id">{esc(cid)}</span></div>'
                    f"<h3>{esc(c_title)}</h3><p>{esc(c_summary)}</p>"
                    f'{"".join(extras)}<div class="card-chips">{chips}</div></article>'
                )
            if cards:
                groups.append(f'<section class="claim-group"><h2>{esc(status)}</h2>{"".join(cards)}</section>')
        content = (
            '<header class="page-head"><h1>Claims under test</h1>'
            '<p class="lede">The shared, evidence-linked belief ledger. Every claim points at the experiments that support or challenge it.</p></header>'
            '<form id="claims-controls" class="filter-row" onsubmit="return false">'
            '<input id="claims-search" type="search" placeholder="Search claims…" aria-label="Search claims">'
            '<span id="claims-count" class="result-count" aria-live="polite"></span></form>'
            f'<div class="claim-filters" role="group" aria-label="Filter by status">{filter_tiles}</div>'
            f'<div id="claims-board">{"".join(groups)}</div>'
            '<p id="claims-empty" class="empty-note" hidden>No claims match.</p>'
        )
        self.write_page(
            "claims/index.html",
            title=f"Claims · {SITE_NAME}",
            description="Evidence-linked claims: confirmed, promising, open, negative, retired.",
            prefix=prefix,
            active="claims",
            content=content,
        )

    # ----------------------------------------------------------------- queue

    def queue_page(self) -> None:
        prefix = "../"
        by_priority: dict[str, list[dict]] = defaultdict(list)
        for item in self.queue:
            by_priority[str(item.get("priority", "P2"))].append(item)
        labels = {"P0": "Do next", "P1": "Soon", "P2": "Later"}
        columns = []
        for priority in ("P0", "P1", "P2"):
            items = by_priority.get(priority, [])
            cards = []
            for item in items:
                chips = "".join(program_chip(pid, prefix, self.slots, self.program_titles) for pid in item.get("programs", [])[:3])
                details = []
                for key, label in (
                    ("hypothesis", "Hypothesis"),
                    ("minimal_protocol", "Minimal protocol"),
                    ("success_signal", "Success signal"),
                    ("failure_signal", "Failure signal"),
                    ("avoid", "Avoid"),
                    ("next_step", "Next step"),
                ):
                    if item.get(key):
                        details.append(f"<p><strong>{label}.</strong> {esc(item[key])}</p>")
                artifacts = [str(a) for a in item.get("expected_artifacts", [])]
                if artifacts:
                    listing = "".join(f"<li><code>{esc(a)}</code></li>" for a in artifacts)
                    details.append(f"<p><strong>Expected artifacts.</strong></p><ul>{listing}</ul>")
                raw_status = str(item.get("status", ""))
                status_label, status_help = QUEUE_STATUS_LABEL.get(raw_status, (raw_status, ""))
                status_html = (
                    f'<span class="muted" title="{esc(status_help)}">{esc(status_label)}</span>'
                    if status_label
                    else ""
                )
                effort = str(item.get("effort", ""))
                effort_html = (
                    f'<span class="chip effort" title="Estimated experiment size (build + run)">effort: {esc(effort)}</span>'
                    if effort
                    else ""
                )
                card_id = slugify(str(item.get("id") or item.get("title", "")))
                cards.append(
                    f'<article class="queue-card" id="{esc(card_id)}"><div class="card-meta">'
                    f"{effort_html}{status_html}</div>"
                    f"<h3>{esc(item.get('title'))}</h3><p>{esc(item.get('question', ''))}</p>"
                    f'<details><summary>Protocol</summary>{"".join(details)}</details>'
                    f'<div class="card-chips">{chips}</div></article>'
                )
            columns.append(
                f'<section class="queue-col queue-{priority.lower()}"><h2>{priority} · {labels[priority]} '
                f'<span class="count">{len(items)}</span></h2>{"".join(cards)}</section>'
            )
        candidates = ""
        if self.candidate_programs:
            items = "".join(
                f'<li><strong>{esc(c.get("title", c.get("id", "")))}</strong> — {esc(c.get("rationale", c.get("focus", "")))}</li>'
                for c in self.candidate_programs
            )
            candidates = f'<section class="doc-section"><h2>Candidate programs</h2><ul class="plain">{items}</ul></section>'
        content = (
            '<header class="page-head"><h1>Future experiment queue</h1>'
            '<p class="lede">Proposals waiting to run, ordered by priority (P0 = do next). '
            'Each card shows its estimated effort and stage: <em>protocol ready</em> = fully specified, '
            '<em>program seed</em> = would open a new program, <em>idea</em> = not yet specified.</p></header>'
            f'<div id="queue-board" class="queue-board">{"".join(columns)}</div>{candidates}'
        )
        self.write_page(
            "queue/index.html",
            title=f"Queue · {SITE_NAME}",
            description="The future experiment queue: prioritized, protocol-ready proposals.",
            prefix=prefix,
            active="queue",
            content=content,
        )

    # -------------------------------------------------------------- notebook

    NOTEBOOK_DOCS = [
        ("synthesis", "Cross-program synthesis", "synthesis.md", "The living summary: what the corpus says when read as one body of evidence."),
        ("roadmap", "Research roadmap", "research_roadmap.md", "Where the portfolio goes next and why."),
        ("patterns", "Patterns", "patterns.md", "Recurring mechanisms and failure modes across experiments."),
        ("open-questions", "Open questions", "open_questions.md", "What we do not know yet, stated as testable questions."),
        ("scorecards", "Program scorecards", "program_scorecards.md", "Per-program health: evidence, gaps, and next bets."),
    ]

    def notebook_pages(self) -> None:
        cards = []
        for slug, title, filename, blurb in self.NOTEBOOK_DOCS:
            if not (KNOWLEDGE / filename).exists():
                continue
            cards.append(f'<a class="program-card" href="{slug}/"><h3>{esc(title)}</h3><p>{esc(blurb)}</p></a>')
        content = (
            '<header class="page-head"><h1>Notebook</h1>'
            '<p class="lede">The knowledge layer, rendered in full: synthesis, roadmap, patterns, open questions, and scorecards. '
            f'Generated indexes live in the <a href="{GITHUB}/tree/main/knowledge">knowledge/ folder on GitHub</a>.</p></header>'
            f'<div id="notebook" class="program-grid">{"".join(cards)}</div>'
        )
        self.write_page(
            "notebook/index.html",
            title=f"Notebook · {SITE_NAME}",
            description="Synthesis, roadmap, patterns, open questions, and program scorecards.",
            prefix="../",
            active="notebook",
            content=content,
        )
        for slug, title, filename, blurb in self.NOTEBOOK_DOCS:
            doc = KNOWLEDGE / filename
            if not doc.exists():
                continue
            prefix = "../../"
            resolver = KnowledgeResolver(doc, prefix, self.roster, self.program_ids)
            rendered = render_markdown(strip_leading_h1(read_text(doc)), resolver=resolver, heading_shift=1)
            doc_html = linkify_experiments(
                linkify_claims(rendered.html, prefix, self.claim_anchor_ids), prefix, self.roster
            )
            toc_items = "".join(
                f'<li class="lv{level}"><a href="#{esc(slug_)}">{esc(text)}</a></li>' for level, slug_, text in rendered.toc if level <= 3
            )
            toc = f'<aside class="detail-side"><nav class="side-toc" aria-label="On this page"><p>On this page</p><ol>{toc_items}</ol></nav></aside>' if toc_items else ""
            mobile_toc = f'<details class="mobile-toc"><summary>On this page</summary><ol>{toc_items}</ol></details>' if toc_items else ""
            header = (
                f'<nav class="crumbs"><a href="{prefix}">Home</a> / <a href="{prefix}notebook/">Notebook</a> / <span>{esc(title)}</span></nav>'
                f"<h1>{esc(title)}</h1>"
                f'<p class="doc-src muted">Rendered from <a href="{GITHUB}/blob/main/knowledge/{esc(filename)}">knowledge/{esc(filename)}</a></p>'
            )
            plain_panel = ""
            if filename == "synthesis.md":
                lb = self.learned_block(prefix)
                if lb:
                    plain_panel = (
                        '<section class="learned-panel"><p class="kicker">In plain language</p>'
                        f'{lb}<p class="learned-panel-note muted">The precise, technical synthesis follows below.</p></section>'
                    )
            self.write_page(
                f"notebook/{slug}/index.html",
                title=f"{title} · {SITE_NAME}",
                description=blurb,
                prefix=prefix,
                active="notebook",
                content=f'<div class="detail-layout"><div class="detail-main doc-page">{header}{mobile_toc}{plain_panel}{doc_html}</div>{toc}</div>',
            )

    def learned_block(self, prefix: str) -> str:
        """Plain-language 'what we've learned' (knowledge/synthesis_plain.json): a
        lead arc + tone-coded takeaways. Shared by the home and the notebook."""
        path = KNOWLEDGE / "synthesis_plain.json"
        data = read_json(path) if path.exists() else {}
        data = data if isinstance(data, dict) else {}
        tone_dot = {"positive": "var(--good)", "negative": "var(--critical)", "mixed": "var(--slot-3)", "open": "var(--baseline)"}
        items = "".join(
            f'<li class="learned-item"><span class="learned-dot" style="background:{tone_dot.get(str(t.get("tone")), "var(--baseline)")}" aria-hidden="true"></span>'
            f'<div><h3>{esc(t.get("headline", ""))}</h3><p>{self.rich_text(str(t.get("body", "")), prefix)}</p></div></li>'
            for t in data.get("takeaways", []) if isinstance(t, dict) and t.get("headline")
        )
        arc = str(data.get("arc", "")).strip()
        return (
            (f'<p class="learned-arc">{esc(arc)}</p>' if arc else "")
            + (f'<ol class="learned-list">{items}</ol>' if items else "")
        )

    # ------------------------------------------------------------------ home

    def home_page(self) -> None:
        prefix = ""
        # "Latest findings" must lead with actual findings, not the in-progress frontier:
        # show the newest FINISHED experiments (in-progress work is one click away via the tab).
        finished_feed = [exp for exp in self.experiments if exp["status"] == "finished"]
        feed = (finished_feed or self.experiments)[:12]
        feed_cards = "".join(
            feed_card(
                exp, prefix, self.slots, self.program_titles,
                big=True, rich=self.rich_text, chart=self.headline_chart(exp["id"]),
            )
            for exp in feed
        )

        cited_counts = Counter()
        for exp_id, claims in self.claims_by_exp.items():
            if exp_id in self.by_id:
                cited_counts[exp_id] = len(claims)
        bearing = [self.by_id[eid] for eid, _ in cited_counts.most_common(8)]
        bearing_cards = "".join(
            f'<li><a href="experiments/{esc(exp["id"])}/">{esc(exp["title"])}</a>'
            f'<span class="muted"> · cited by {cited_counts[exp["id"]]} claim{"s" if cited_counts[exp["id"]] != 1 else ""}</span>'
            f'<p class="muted clamp">{self.rich_text((exp.get("card_summary") or exp["finding"])[:200], prefix)}</p></li>'
            for exp in bearing
        )

        synthesis = read_text(KNOWLEDGE / "synthesis.md") if (KNOWLEDGE / "synthesis.md").exists() else ""
        exec_read = ""
        for head, _, body in _sections(synthesis):
            if head.strip().lower() == "executive read":
                resolver = KnowledgeResolver(KNOWLEDGE / "synthesis.md", prefix, self.roster, self.program_ids)
                exec_read = render_markdown(body, resolver=resolver, slug_prefix="exec-", heading_shift=2).html
                exec_read = linkify_experiments(
                    linkify_claims(exec_read, prefix, self.claim_anchor_ids), prefix, self.roster
                )
                break

        # A plain-language, fact-checked "what we've learned" (knowledge/synthesis_plain.json)
        # replaces the jargon-dense technical Executive Read as the home's answer.
        learned_html = self.learned_block(prefix)

        claim_counts = Counter(str(claim.get("status")) for claim in self.claims)
        confirmed = claim_counts.get("Confirmed", 0)
        figures_total = sum(len(exp["figures"]) for exp in self.experiments)
        run_dates = sorted(exp["when"] for exp in self.experiments if exp["when"])
        span_sub = format_range(run_dates[0], run_dates[-1]) if run_dates else ""
        tiles = (
            stat_tile("Experiments", str(len(self.experiments)), span_sub, "experiments/")
            + stat_tile("Programs", str(len(self.programs)), "lines of inquiry", "programs/")
            + stat_tile("Claims", str(len(self.claims)), f"{confirmed} confirmed", "claims/")
            + stat_tile("Figures", str(figures_total), "rendered on-site", "experiments/?sort=figs")
            + stat_tile("Queued", str(len(self.queue)), "proposals waiting", "queue/")
        )

        # A proportional status bar replaces the old wall of NN opaque C-id pills:
        # it reads at a glance and links to the ledger, where the ids actually live.
        claim_by_key = Counter(CLAIM_STATUS_META.get(str(c.get("status")), ("open",))[0] for c in self.claims)
        seg_order = [("check", "confirmed"), ("half", "promising"), ("open", "open"), ("cross", "ruled out"), ("pause", "retired")]
        bar_segs = "".join(
            f'<span class="cg-seg cg-{key}" style="flex:{claim_by_key[key]}" title="{claim_by_key[key]} {label}"></span>'
            for key, label in seg_order if claim_by_key.get(key)
        )
        counts_txt = " · ".join(
            (f"<b>{claim_by_key[key]} {label}</b>" if key == "check" else f"{claim_by_key[key]} {label}")
            for key, label in seg_order if claim_by_key.get(key)
        )
        claim_strip = (
            '<a class="claims-glance" href="claims/" aria-label="Claim ledger status breakdown">'
            f'<span class="cg-bar">{bar_segs}</span>'
            f'<span class="cg-legend">{len(self.claims)} durable claims, each tied to its evidence — '
            f'{counts_txt} <span class="cg-more">Open the ledger →</span></span></a>'
        )
        strip_legend = ""

        glossary_terms = [
            ("deployable", "a setting you could actually ship: greedy decoding, one sample, no oracle reranking"),
            ("greedy@1", "single deterministic answer (temperature 0) — the strictest deployable metric"),
            ("pass@k", "chance that at least one of k samples passes the hidden tests"),
            ("coverage", "fraction of tasks where at least one sample in the pool is correct — the ceiling that self-training can bank"),
            ("oracle", "the score if you could always pick the pool's best sample — an upper bound, not deployable"),
            ("visible / hidden tests", "the checks the model may see and run, vs the held-out checks that decide correctness"),
            ("false-pass", "a candidate that passes the visible tests but fails the hidden ones"),
            ("thinking / no-think", "generation with the model's native reasoning channel enabled vs disabled"),
            ("banking", "fine-tuning the fixed model on its own verified successes, so sampled wins become its greedy default"),
            ("import (lines Y / Z)", "the 2026-06-28 bulk import of the predecessor repo's two parallel working tracks"),
        ]
        glossary = (
            '<details id="glossary" class="glossary-box"><summary>Glossary — the corpus vocabulary, one line each</summary><dl>'
            + "".join(f"<dt>{esc(term)}</dt><dd>{esc(definition)}</dd>" for term, definition in glossary_terms)
            + "</dl></details>"
        )

        n_finished = sum(1 for exp in self.experiments if exp["status"] == "finished")
        n_inprog = len(self.experiments) - n_finished
        content = (
            '<section class="hero"><h1>What has this corpus learned?</h1>'
            '<p class="lede">One fixed Qwen3.5-4B — no scaling, no bigger teacher — pushed to see how much capability you can '
            'draw out of it and install back in. This is the reading surface: '
            f'<a href="experiments/">{n_finished} finished experiments</a> and '
            f'<a href="experiments/?status=in-progress">{n_inprog} in progress</a>, every claim tied to its evidence, '
            'every result rendered from its own data — newest first.</p>'
            f'<div class="stat-row">{tiles}</div>'
            f'{claim_strip}{strip_legend}{glossary}</section>'
            f'<section id="latest-feed" class="band"><div class="section-head"><h2>Latest findings</h2>'
            f'<a class="more" href="experiments/">All experiments →</a></div><div class="feed-grid">{feed_cards}</div></section>'
            + (
                '<section class="band split-band"><div class="split-a"><div class="section-head">'
                + (f'<h2>What we&rsquo;ve learned</h2>' if learned_html else '<h2>Executive read</h2>')
                + f'<a class="more" href="notebook/synthesis/">Full technical synthesis →</a></div>'
                + (learned_html or ('<p class="muted">The numbered takeaways from the living synthesis, updated as evidence lands.</p>'
                                    f'<div class="exec-read">{exec_read}</div>'))
                + '</div>'
                f'<div class="split-b"><div class="section-head"><h2>Load-bearing results</h2>'
                f'<a class="more" href="claims/">Claims →</a></div>'
                '<p class="muted">The experiments most often cited as evidence by the claim ledger.</p>'
                f'<ul class="plain exp-list">{bearing_cards}</ul></div></section>'
                if (learned_html or exec_read)
                else ""
            )
        )
        self.write_page(
            "index.html",
            title=f"{SITE_NAME} · Research Log",
            description="Latest findings, claims, and rendered reports from the small-model experimentation corpus.",
            prefix=prefix,
            active="home",
            content=content,
        )

    # -------------------------------------------------------------- payloads

    def data_payloads(self) -> None:
        experiments = [
            {
                "id": exp["id"],
                "title": exp["title"],
                "track": exp["track"],
                "programs": exp["programs"],
                "tags": exp["tags"],
                "first": exp["first"],
                "last": exp["last"],
                "ran_start": exp["ran_start"],
                "ran_end": exp["ran_end"],
                "when": exp["when"],
                "date_confidence": exp["date_confidence"],
                "commits": exp["commits"],
                "recent": exp["recent"],
                "finding": exp["finding"][:500],
                "finding_source": exp["finding_source"],
                "figures": len(exp["figures"]),
                "charts": len(self.viz.get(exp["id"], [])),
                "data_files": len(exp["data_files"]),
                "url": f"experiments/{exp['id']}/",
                "path": exp["path"],
                "anchor_ready": exp["anchor_ready"],
                "run_surface": exp["run_surface"],
            }
            for exp in self.experiments
        ]
        payload = {
            "generated_at": self.generated,
            "repo": {"name": "small-model-experimentation", "github": GITHUB},
            "summary": {
                "experiments": len(self.experiments),
                "programs": len(self.programs),
                "claims": len(self.claims),
                "future_proposals": len(self.queue),
                "figures": sum(len(exp["figures"]) for exp in self.experiments),
            },
            "experiments": experiments,
            "programs": [
                {
                    "id": str(p["id"]),
                    "title": self.program_titles[str(p["id"])],
                    "focus": str(p.get("focus", "")),
                    "url": f"programs/{p['id']}/",
                }
                for p in self.programs
            ],
            "claims": [
                {
                    "id": str(c.get("id")),
                    "title": str(c.get("title")),
                    "status": str(c.get("status")),
                    "summary": str(c.get("summary")),
                    "programs": [str(pid) for pid in c.get("programs", [])],
                    "url": f"claims/#{slugify(str(c.get('id')))}",
                }
                for c in self.claims
            ],
            "queue": [
                {
                    "id": str(item.get("id")),
                    "title": str(item.get("title")),
                    "priority": str(item.get("priority")),
                    "status": str(item.get("status")),
                    "question": str(item.get("question", "")),
                    "programs": [str(pid) for pid in item.get("programs", [])],
                }
                for item in self.queue
            ],
        }
        data_dir = self.out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "site-data.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        search: list[dict[str, str]] = []
        for exp in self.experiments:
            search.append(
                {
                    "k": "experiment",
                    "t": exp["title"],
                    "u": f"experiments/{exp['id']}/",
                    "d": exp["when"],
                    "x": f"{exp['id']} {' '.join(exp['tags'])} {' '.join(exp['programs'])} {exp['finding'][:500]}",
                    "b": search_vocab(exp["finding"], exp["readme_text"], exp["report_text"]),
                }
            )
        for claim in self.claims:
            _cp = self.claim_plain.get(str(claim.get("id")))
            _cp = _cp if isinstance(_cp, dict) else {}
            _ct = _cp.get("title") or claim.get("title")
            _cs = _cp.get("summary") or claim.get("summary")
            _ci = _cp.get("implication") or claim.get("implication", "")
            search.append(
                {
                    "k": "claim",
                    "t": f"{claim.get('id')} · {_ct}",
                    "u": f"claims/#{slugify(str(claim.get('id')))}",
                    "d": "",
                    # index the plain rewrite plus the raw summary so search hits
                    # whether the reader types plain words or the original jargon.
                    "x": f"{claim.get('status')} {_cs} {_ci} {claim.get('summary')}",
                }
            )
        for program in self.programs:
            search.append(
                {
                    "k": "program",
                    "t": self.program_titles[str(program["id"])],
                    "u": f"programs/{program['id']}/",
                    "d": "",
                    "x": f"{program['id']} {program.get('focus', '')}",
                }
            )
        for item in self.queue:
            search.append(
                {
                    "k": "queued",
                    "t": str(item.get("title")),
                    "u": f"queue/#{slugify(str(item.get('id') or item.get('title', '')))}",
                    "d": "",
                    "x": f"{item.get('priority')} {item.get('question', '')} {item.get('hypothesis', '')}",
                }
            )
        for slug, title, filename, blurb in self.NOTEBOOK_DOCS:
            if (KNOWLEDGE / filename).exists():
                search.append({"k": "notebook", "t": title, "u": f"notebook/{slug}/", "d": "", "x": blurb})
        (data_dir / "search-index.json").write_text(json.dumps(search, ensure_ascii=False) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        if self.out.exists():
            shutil.rmtree(self.out)
        (self.out / "assets").mkdir(parents=True, exist_ok=True)
        for asset in sorted((TEMPLATE / "assets").iterdir()):
            if asset.is_file():
                shutil.copyfile(asset, self.out / "assets" / asset.name)
        (self.out / ".nojekyll").write_text("", encoding="utf-8")

        for index, exp in enumerate(self.experiments):
            prev_exp = self.experiments[index - 1] if index > 0 else None
            next_exp = self.experiments[index + 1] if index + 1 < len(self.experiments) else None
            self.experiment_page(exp, prev_exp, next_exp)
        self.explorer_page()
        self.program_pages()
        self.claims_page()
        self.queue_page()
        self.notebook_pages()
        self.home_page()
        self.data_payloads()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="site", type=Path, help="output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    builder = SiteBuilder(out_dir)
    builder.build()
    pages = len(list(out_dir.rglob("index.html")))
    size = sum(path.stat().st_size for path in out_dir.rglob("*") if path.is_file())
    print(f"built site: {out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir} ({pages} pages, {size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
