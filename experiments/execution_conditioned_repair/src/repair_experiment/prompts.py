from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = (
    "You are a coding agent repairing a repository. Output only a unified diff.\n"
    "Do not explain. Do not include markdown fences."
)


def format_files(files: dict[str, str]) -> str:
    chunks: list[str] = []
    for path in sorted(files):
        chunks.append(f'<FILE path="{path}">\n{files[path].rstrip()}\n</FILE>')
    return "\n\n".join(chunks)


def build_user_prompt(
    record: dict[str, Any],
    mode: str = "trace",
    trace_override: str | None = None,
) -> str:
    if mode == "final_patch":
        files = record["buggy_files"]
        current_diff = ""
        test_output = ""
    else:
        files = record["current_files"]
        current_diff = record.get("wrong_patch", "")
        test_output = record.get("test_output_after_wrong_patch", "")

    if mode == "no_trace":
        test_output = ""
    elif mode == "wrong_patch_only":
        files = {}
        test_output = ""
    elif mode == "trace_only":
        current_diff = ""
    elif mode == "gold_file_removed":
        files = {
            path: "# file content intentionally withheld for ablation\n"
            for path in files
        }

    if trace_override is not None:
        test_output = trace_override

    return (
        "<ISSUE>\n"
        f"{record['issue'].strip()}\n"
        "</ISSUE>\n\n"
        "<REPO_CONTEXT>\n"
        f"{format_files(files)}\n"
        "</REPO_CONTEXT>\n\n"
        "<CURRENT_DIFF>\n"
        f"{current_diff.strip()}\n"
        "</CURRENT_DIFF>\n\n"
        "<TEST_OUTPUT_AFTER_CURRENT_DIFF>\n"
        f"{test_output.strip()}\n"
        "</TEST_OUTPUT_AFTER_CURRENT_DIFF>\n\n"
        "Task:\n"
        "Produce the minimal corrective unified diff to make the repository pass the tests."
    )


def messages_for_record(
    record: dict[str, Any],
    mode: str = "trace",
    trace_override: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record, mode, trace_override)},
    ]


def target_for_mode(record: dict[str, Any], mode: str) -> str:
    if mode == "final_patch":
        return record["base_buggy_diff"]
    return record["target_next_diff"]
