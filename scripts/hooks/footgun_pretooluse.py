#!/usr/bin/env python3
"""PreToolUse hook: block a Write/Edit whose NEW content trips scripts/check_footguns.py.

`make check` catches footguns at commit time, which is already too late twice over: the bad default has
been written, and in the worst case a GPU run has burned hours producing a number that measures the
harness rather than the model (a 512-token cap once made a reasoning arm read 0.040 against a true
~0.235). This hook fires at WRITE time instead.

THE SUBTLETY THAT MAKES OR BREAKS IT: on PreToolUse the new content is not on disk yet. Running the
checker against `file_path` would inspect the OLD file -- so the hook would pass while writing the very
violation it exists to stop. The proposed content arrives in the payload instead:
  Write -> tool_input.content
  Edit  -> tool_input.new_string   (the changed region only, which is exactly the ratchet we want)
It is written to a temp file carrying the target's extension, since the checker only scans .py/.sh.

Exit 0 allows the write; exit 2 blocks it and returns stderr to the model. Any internal error exits 0 --
a broken hook must not wedge the session, and `make check` still backstops at commit time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "check_footguns.py"


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    ti = payload.get("tool_input") or {}
    path = ti.get("file_path") or ""
    suffix = Path(path).suffix
    if suffix not in (".py", ".sh"):
        return 0

    # Write sends full content; Edit sends the replacement region. MultiEdit sends a list of edits.
    content = ti.get("content") or ti.get("new_string") or ""
    if not content and isinstance(ti.get("edits"), list):
        content = "\n".join(str(e.get("new_string", "")) for e in ti["edits"])
    if not content.strip():
        return 0

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="footgun_")
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        proc = subprocess.run([sys.executable, str(CHECKER), tmp],
                              text=True, stdout=subprocess.PIPE,  # footgun-ok: bounded checker output, not a floodable child
                              stderr=subprocess.STDOUT, timeout=30)
        if proc.returncode == 0:
            return 0
        # Re-point the temp path at the real file so the message is actionable.
        report = (proc.stdout or "").replace(tmp, path or "<new file>")
        sys.stderr.write(
            f"BLOCKED: this content trips the repo's footgun check (scripts/check_footguns.py).\n"
            f"These are rules the repo has ALREADY paid for -- see the cited incidents.\n\n{report}\n"
            f"Fix the content, or if the exception is genuinely justified add "
            f"'# footgun-ok: <reason>' on the offending line so the reason is recorded.\n")
        return 2
    except Exception:
        return 0                      # never wedge the session on a hook bug
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
