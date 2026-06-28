#!/usr/bin/env python3
"""Validate the generated static site bundle."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "site"
    if not site.is_absolute():
        site = ROOT / site
    errors: list[str] = []

    required = [
        site / "index.html",
        site / "assets" / "app.js",
        site / "assets" / "styles.css",
        site / "data" / "site-data.json",
        site / ".nojekyll",
    ]
    for path in required:
        if not path.exists():
            fail(errors, f"missing site file: {path.relative_to(ROOT)}")

    html_path = site / "index.html"
    data_path = site / "data" / "site-data.json"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8", errors="replace")
        if "__SITE_DATA__" in html:
            fail(errors, "site index still contains data placeholder")
        for marker in [
            'id="programNetwork"',
            'id="experimentTable"',
            'id="queueBoard"',
            'id="claimGraph"',
            'id="artifactKindChart"',
        ]:
            if marker not in html:
                fail(errors, f"site index missing marker: {marker}")
        match = re.search(r'<script id="site-data" type="application/json">(.*?)</script>', html, flags=re.S)
        if not match:
            fail(errors, "site index missing embedded site data")
        else:
            try:
                embedded = json.loads(match.group(1))
                if int(embedded.get("summary", {}).get("experiments", 0)) <= 0:
                    fail(errors, "embedded site data has no experiments")
            except json.JSONDecodeError as exc:
                fail(errors, f"embedded site data is invalid JSON: {exc}")

    if data_path.exists():
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            if len(data.get("experiments", [])) != int(data.get("summary", {}).get("experiments", -1)):
                fail(errors, "site data experiment count mismatch")
            if not data.get("queue"):
                fail(errors, "site data missing future queue")
            if not data.get("programs"):
                fail(errors, "site data missing programs")
        except json.JSONDecodeError as exc:
            fail(errors, f"site data JSON is invalid: {exc}")

    node = shutil.which("node")
    if node and (site / "assets" / "app.js").exists():
        result = subprocess.run(
            [node, "--check", str(site / "assets" / "app.js")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            fail(errors, "site JavaScript syntax check failed:\n" + result.stderr.strip())

    if errors:
        print("site check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("site check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
