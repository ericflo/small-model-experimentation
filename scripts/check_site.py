#!/usr/bin/env python3
"""Validate the generated research site: required files, page coverage, link integrity."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

URL_RE = re.compile(r'(?:href|src)="([^"]*)"')
SKIP_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)
ID_RE = re.compile(r'\bid="([^"]*)"')


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


class LinkChecker:
    def __init__(self, site: Path, errors: list[str]):
        self.site = site.resolve()
        self.errors = errors
        self.checked = 0
        self._ids: dict[Path, set[str]] = {}

    def ids_of(self, page: Path) -> set[str]:
        if page not in self._ids:
            text = page.read_text(encoding="utf-8", errors="replace")
            self._ids[page] = set(ID_RE.findall(text))
        return self._ids[page]

    def check_url(self, page: Path, url: str) -> None:
        if not url or SKIP_URL_RE.match(url):
            return
        target, fragment = (url.split("#", 1) + [""])[:2]
        target = target.split("?", 1)[0]
        if target:
            resolved = (page.parent / target).resolve()
            if resolved.is_dir():
                resolved = resolved / "index.html"
        else:
            resolved = page  # same-page anchor
        self.checked += 1
        if not resolved.exists():
            fail(self.errors, f"{page.relative_to(self.site)}: broken link {url}")
            return
        if not resolved.is_relative_to(self.site):
            fail(self.errors, f"{page.relative_to(self.site)}: link escapes the site root: {url}")
            return
        if fragment and resolved.suffix == ".html" and fragment not in self.ids_of(resolved):
            fail(self.errors, f"{page.relative_to(self.site)}: anchor #{fragment} missing in {url or 'this page'}")


def check_links(site: Path, errors: list[str]) -> int:
    checker = LinkChecker(site, errors)
    for page in sorted(site.rglob("*.html")):
        text = page.read_text(encoding="utf-8", errors="replace")
        if re.search(r"%%[A-Z_]+%%", text):
            fail(errors, f"{page.relative_to(site)}: unreplaced template placeholder")
        for url in URL_RE.findall(text):
            checker.check_url(page, url)
    return checker.checked


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
        site / "data" / "search-index.json",
        site / ".nojekyll",
    ]
    for path in required:
        if not path.exists():
            fail(errors, f"missing site file: {path.relative_to(site)}")

    markers = [
        (site / "index.html", 'id="latest-feed"'),
        (site / "experiments" / "index.html", 'id="explorer"'),
        (site / "claims" / "index.html", 'id="claims-board"'),
        (site / "queue" / "index.html", 'id="queue-board"'),
        (site / "notebook" / "index.html", 'id="notebook"'),
    ]
    for page, marker in markers:
        if page.exists():
            if marker not in page.read_text(encoding="utf-8", errors="replace"):
                fail(errors, f"{page.relative_to(site)}: missing marker {marker}")
        else:
            fail(errors, f"missing site page: {page.relative_to(site)}")

    catalog_path = ROOT / "knowledge" / "experiment_catalog.csv"
    catalog_ids: list[str] = []
    if catalog_path.exists():
        with catalog_path.open(newline="", encoding="utf-8") as handle:
            catalog_ids = [row["id"] for row in csv.DictReader(handle)]
    for exp_id in catalog_ids:
        if not (site / "experiments" / exp_id / "index.html").exists():
            fail(errors, f"missing experiment page: experiments/{exp_id}/")

    data_path = site / "data" / "site-data.json"
    if data_path.exists():
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            data = {}
            fail(errors, f"site data JSON is invalid: {exc}")
        if data:
            count = int(data.get("summary", {}).get("experiments", 0))
            if count <= 0:
                fail(errors, "site data has no experiments")
            if len(data.get("experiments", [])) != count:
                fail(errors, "site data experiment count mismatch")
            if catalog_ids and count != len(catalog_ids):
                fail(errors, f"site data has {count} experiments, catalog has {len(catalog_ids)}")
            if not data.get("programs"):
                fail(errors, "site data missing programs")
            if not data.get("queue"):
                fail(errors, "site data missing future queue")

    search_path = site / "data" / "search-index.json"
    if search_path.exists():
        try:
            search = json.loads(search_path.read_text(encoding="utf-8"))
            if not isinstance(search, list) or not search:
                fail(errors, "search index is empty")
            else:
                checker = LinkChecker(site, errors)
                for entry in search:
                    url = str(entry.get("u", ""))
                    checker.check_url(site / "index.html", url)
        except json.JSONDecodeError as exc:
            fail(errors, f"search index JSON is invalid: {exc}")

    checked = check_links(site, errors)

    node = shutil.which("node")
    if node and (site / "assets" / "app.js").exists():
        result = subprocess.run([node, "--check", str(site / "assets" / "app.js")], capture_output=True, text=True)
        if result.returncode != 0:
            fail(errors, f"site JavaScript syntax check failed:\n{result.stderr.strip()}")

    if errors:
        print("site check failed:")
        for error in errors[:60]:
            print(f"- {error}")
        if len(errors) > 60:
            print(f"- … and {len(errors) - 60} more")
        return 1
    pages = len(list(site.rglob("*.html")))
    print(f"site check passed ({pages} pages, {checked} internal links verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
