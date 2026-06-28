from __future__ import annotations

import difflib
import os
import re
import subprocess
import tempfile
from pathlib import Path


DIFF_START_RE = re.compile(r"(?m)^(diff --git |--- a/)")


def unified_diff_for_file(path: str, before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    text = "".join(line if line.endswith("\n") else line + "\n" for line in diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def unified_diff_for_files(before: dict[str, str], after: dict[str, str]) -> str:
    parts: list[str] = []
    for path in sorted(set(before) | set(after)):
        parts.append(unified_diff_for_file(path, before.get(path, ""), after.get(path, "")))
    return "".join(part for part in parts if part.strip())


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


def extract_unified_diff(text: str) -> str:
    text = text.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.S)
        if blocks:
            text = max(blocks, key=len).strip()
    match = DIFF_START_RE.search(text)
    if match:
        text = text[match.start() :]
    lines = text.splitlines()
    kept: list[str] = []
    in_diff = False
    diff_prefixes = (
        "diff --git ",
        "index ",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "rename from ",
        "rename to ",
        "--- ",
        "+++ ",
        "@@ ",
        "+",
        "-",
        " ",
    )
    for line in lines:
        if line.startswith(diff_prefixes):
            in_diff = True
            kept.append(line)
        elif in_diff and line.startswith("\\ No newline"):
            kept.append(line)
        elif in_diff and line.strip() == "":
            kept.append(line)
        elif in_diff:
            break
    out = "\n".join(kept).strip()
    return out + "\n" if out else ""


def apply_patch_to_files(
    files: dict[str, str],
    patch_text: str,
    *,
    timeout: int = 20,
) -> tuple[bool, dict[str, str], str]:
    patch_text = extract_unified_diff(patch_text)
    if not patch_text:
        return False, files.copy(), "empty or unparsable unified diff"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_files(root, files)
        patch_path = root / "candidate.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        check = subprocess.run(
            ["git", "apply", "--check", "--recount", "-p1", str(patch_path)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if check.returncode != 0:
            return False, files.copy(), check.stdout
        apply = subprocess.run(
            ["git", "apply", "--whitespace=fix", "--recount", "-p1", str(patch_path)],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        if apply.returncode != 0:
            return False, files.copy(), apply.stdout
        updated: dict[str, str] = {}
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.name != "candidate.patch":
                rel = file_path.relative_to(root).as_posix()
                updated[rel] = file_path.read_text(encoding="utf-8")
        for rel in files:
            updated.setdefault(rel, "")
        return True, updated, apply.stdout


def patch_stats(patch_text: str) -> dict[str, int]:
    files = set()
    added = 0
    removed = 0
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"files_touched": len(files), "added_lines": added, "removed_lines": removed}
