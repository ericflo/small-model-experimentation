"""Canonical answer and action extraction for Menagerie model outputs."""

from __future__ import annotations

import re


_ANSWER_RE = re.compile(r"^answer\s*:\s*(.*)$", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"```[A-Za-z0-9_.+-]*")


def strip_think(text: str) -> str:
    """Remove completed or dangling thinking text from model output."""

    cleaned = _THINK_BLOCK_RE.sub("", text)
    closes = list(_THINK_CLOSE_RE.finditer(cleaned))
    if closes:
        return cleaned[closes[-1].end() :]
    return cleaned


def _strip_markdown_edge(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[*_`]+", "", line)
    line = re.sub(r"[*_`]+$", "", line)
    return line.strip()


def extract_answer(text: str) -> str | None:
    """Extract the last final-answer line, if present."""

    last: str | None = None
    for line in strip_think(text).splitlines():
        match = _ANSWER_RE.match(_strip_markdown_edge(line))
        if match:
            value = match.group(1).strip()
            value = re.sub(r"[*`]+$", "", value).rstrip()
            last = value
    return last


def normalize_action(text: str) -> str:
    """Normalize a single terse episode action from free-form model output."""

    candidates = []
    for line in strip_think(text).splitlines():
        stripped = line.strip()
        if _FENCE_RE.fullmatch(stripped):
            continue
        if stripped:
            candidates.append(stripped)
    if not candidates:
        return ""

    action = candidates[-1].strip()
    if len(action) >= 2 and action[0] == action[-1] and action[0] in {"'", '"', "`"}:
        action = action[1:-1].strip()
    for marker in ("- ", "* ", "> "):
        if action.startswith(marker):
            action = action[len(marker) :]
            break
    action = re.sub(r"\s+", " ", action.strip())
    return action


def canonical_action(text: str, mode: str) -> str:
    """Canonicalize model output into the action string sent to a family env."""

    if mode == "atom":
        value = extract_answer(text)
        if value is not None:
            return "ANSWER: " + value
        return normalize_action(text)
    if mode == "episode":
        return normalize_action(text)
    raise ValueError(f"unknown mode {mode!r}")
