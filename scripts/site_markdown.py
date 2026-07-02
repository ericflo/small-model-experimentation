#!/usr/bin/env python3
"""Dependency-free GitHub-flavored-markdown subset renderer for the research site.

Covers the constructs the corpus actually uses (ATX headings, pipe tables, fenced
code, nested lists, blockquotes, emphasis, links, images). Raw HTML in sources is
always escaped: reports quote model output and must never inject markup.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field


@dataclass
class RenderResult:
    html: str
    toc: list[tuple[int, str, str]] = field(default_factory=list)
    first_image: str = ""
    plain_text: str = ""


class LinkResolver:
    """Base resolver: passes URLs through untouched."""

    def link(self, url: str) -> str:
        return url

    def image(self, url: str) -> str:
        return url

    def code_span_target(self, text: str) -> str:
        """Return a URL when a backticked token should become a link (else '')."""
        return ""

    def image_size(self, resolved: str) -> tuple[int, int] | None:
        """Return (width, height) for a resolved image URL when known.

        Setting intrinsic dimensions keeps lazy-loaded images from shifting
        layout after anchor jumps.
        """
        return None


_FENCE_RE = re.compile(r"^(\s{0,3})(```+|~~~+)\s*([\w+#.-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^\s{0,3}((-\s*){3,}|(\*\s*){3,}|(_\s*){3,})$")
_TABLE_DELIM_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d{1,3}[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")

_CODE_SPAN_RE = re.compile(r"(`+)(.+?)\1", re.S)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;|\s+\"[^)]*\")?\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;|\s+\"[^)]*\")?\)")
_AUTOLINK_RE = re.compile(r"(?<![\"'=\w])(https?://[^\s<>()\[\]]+[^\s<>()\[\].,;:!?'\"])")
_BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.S)
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])|(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])", re.S)
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~", re.S)

_INLINE_STRIP_RE = re.compile(r"[*_`]|~~|\[([^\]]*)\]\([^)]*\)|!\[[^\]]*\]\([^)]*\)")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _code_span_plain(match: re.Match[str]) -> str:
    # bare punctuation loses its meaning without the backticks ("`.` tokens"
    # must not flatten to ". tokens"), so quote it instead
    inner = match.group(1)
    if inner and not re.search(r"[A-Za-z0-9]", inner):
        return f"“{inner}”"
    return inner


def strip_inline(text: str) -> str:
    """Markdown inline syntax -> plain text (for excerpts and search)."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = re.sub(r"`([^`]*)`", _code_span_plain, text)
    text = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _attr(value: str) -> str:
    return html.escape(value, quote=True)


_ENTITY_SAFE_AMP_RE = re.compile(r"&(?![a-zA-Z][a-zA-Z0-9]{1,31};|#\d{1,7};|#[xX][0-9a-fA-F]{1,6};)")


def _escape_text(text: str) -> str:
    """Escape markup but keep entity references (sources embed e.g. &#x27;)."""
    return _ENTITY_SAFE_AMP_RE.sub("&amp;", text).replace("<", "&lt;").replace(">", "&gt;")


class _Renderer:
    def __init__(self, resolver: LinkResolver, slug_prefix: str = "", heading_shift: int = 0):
        self.resolver = resolver
        self.slug_prefix = slug_prefix
        self.heading_shift = heading_shift
        self.toc: list[tuple[int, str, str]] = []
        self.first_image = ""
        self.plain_parts: list[str] = []
        self._slug_counts: dict[str, int] = {}
        self._stash: list[str] = []

    # ---------------------------------------------------------------- inline

    def _stash_html(self, fragment: str) -> str:
        self._stash.append(fragment)
        return f"\x00{len(self._stash) - 1}\x00"

    def _restore(self, text: str) -> str:
        # stashed fragments may nest placeholders (code span inside a link label),
        # so expand repeatedly until stable
        for _ in range(10):
            expanded = re.sub(r"\x00(\d+)\x00", lambda m: self._stash[int(m.group(1))], text)
            if expanded == text:
                break
            text = expanded
        return text

    def inline(self, text: str) -> str:
        out = _escape_text(text)

        def code_span(match: re.Match[str]) -> str:
            content = match.group(2).strip()
            target = self.resolver.code_span_target(html.unescape(content))
            body = f"<code>{content}</code>"
            if target:
                body = f'<a href="{_attr(target)}">{body}</a>'
            return self._stash_html(body)

        out = _CODE_SPAN_RE.sub(code_span, out)

        def image(match: re.Match[str]) -> str:
            alt, src = match.group(1), html.unescape(match.group(2))
            resolved = self.resolver.image(src)
            if not resolved:
                return self._stash_html(f"<code>{html.escape(src)}</code>")
            if not self.first_image:
                self.first_image = resolved
            size = self.resolver.image_size(resolved)
            dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
            return self._stash_html(
                f'<img src="{_attr(resolved)}" alt="{_attr(html.unescape(alt))}"{dims} loading="lazy">'
            )

        out = _IMAGE_RE.sub(image, out)

        def link(match: re.Match[str]) -> str:
            label, href = match.group(1), html.unescape(match.group(2))
            resolved = self.resolver.link(href)
            body = self._emphasis(label)
            return self._stash_html(f'<a href="{_attr(resolved)}">{body}</a>')

        out = _LINK_RE.sub(link, out)
        out = _AUTOLINK_RE.sub(
            lambda m: self._stash_html(
                f'<a href="{_attr(html.unescape(m.group(1)))}">{m.group(1)}</a>'
            ),
            out,
        )
        out = self._emphasis(out)
        return self._restore(out)

    def _emphasis(self, text: str) -> str:
        text = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", text)
        text = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", text)
        return _STRIKE_RE.sub(lambda m: f"<del>{m.group(1)}</del>", text)

    # ---------------------------------------------------------------- blocks

    def render(self, text: str) -> str:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return self._blocks(lines)

    def _blocks(self, lines: list[str]) -> str:
        out: list[str] = []
        para: list[str] = []
        i = 0

        def flush() -> None:
            if para:
                joined = " ".join(part.strip() for part in para)
                self.plain_parts.append(strip_inline(joined))
                out.append(f"<p>{self.inline(joined)}</p>")
                para.clear()

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                flush()
                i += 1
                continue

            fence = _FENCE_RE.match(line)
            if fence:
                flush()
                marker, lang = fence.group(2)[:3], fence.group(3)
                body: list[str] = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(marker):
                    body.append(lines[i])
                    i += 1
                i += 1  # closing fence (or EOF)
                cls = f' class="language-{_attr(lang)}"' if lang else ""
                code = html.escape("\n".join(body), quote=False)
                out.append(f"<pre><code{cls}>{code}</code></pre>")
                continue

            heading = _HEADING_RE.match(line)
            if heading:
                flush()
                level = min(6, len(heading.group(1)) + self.heading_shift)
                raw = heading.group(2)
                plain = strip_inline(raw)
                slug = slugify(f"{self.slug_prefix}{plain}")
                count = self._slug_counts.get(slug, 0)
                self._slug_counts[slug] = count + 1
                if count:
                    slug = f"{slug}-{count}"
                self.toc.append((level, slug, plain))
                out.append(f'<h{level} id="{_attr(slug)}">{self.inline(raw)}</h{level}>')
                i += 1
                continue

            if _HR_RE.match(line):
                flush()
                out.append("<hr>")
                i += 1
                continue

            if _QUOTE_RE.match(line):
                flush()
                quoted: list[str] = []
                while i < len(lines):
                    m = _QUOTE_RE.match(lines[i])
                    if not m:
                        break
                    quoted.append(m.group(1))
                    i += 1
                out.append(f"<blockquote>{self._blocks(quoted)}</blockquote>")
                continue

            if "|" in stripped and i + 1 < len(lines) and _TABLE_DELIM_RE.match(lines[i + 1]) and "|" in lines[i + 1]:
                flush()
                table, i = self._table(lines, i)
                out.append(table)
                continue

            list_start = _LIST_ITEM_RE.match(line)
            if list_start:
                marker = list_start.group(2)
                # per GFM, an ordered list only interrupts a paragraph when it
                # starts at 1 — "…about\n250. More text" is wrapped prose
                if para and marker[:-1].isdigit() and marker[:-1] != "1":
                    para.append(line)
                    i += 1
                    continue
                flush()
                block: list[str] = []
                while i < len(lines):
                    current = lines[i]
                    if not current.strip():
                        # blank inside a list only continues it when an indented
                        # or new list line follows
                        nxt = lines[i + 1] if i + 1 < len(lines) else ""
                        if nxt.startswith(("  ", "\t")) or _LIST_ITEM_RE.match(nxt):
                            block.append(current)
                            i += 1
                            continue
                        break
                    if not _LIST_ITEM_RE.match(current) and not current.startswith(("  ", "\t")):
                        break
                    block.append(current)
                    i += 1
                out.append(self._list(block))
                continue

            para.append(line)
            i += 1

        flush()
        return "".join(out)

    def _table(self, lines: list[str], start: int) -> tuple[str, int]:
        def cells(row: str) -> list[str]:
            row = row.strip()
            if row.startswith("|"):
                row = row[1:]
            if row.endswith("|"):
                row = row[:-1]
            parts, current, escaped = [], [], False
            for ch in row:
                if escaped:
                    current.append(ch)
                    escaped = False
                elif ch == "\\":
                    current.append(ch)
                    escaped = True
                elif ch == "|":
                    parts.append("".join(current).strip())
                    current = []
                else:
                    current.append(ch)
            parts.append("".join(current).strip())
            return parts

        header = cells(lines[start])
        aligns = []
        for spec in cells(lines[start + 1]):
            left, right = spec.startswith(":"), spec.endswith(":")
            aligns.append("center" if left and right else "right" if right else "left" if left else "")
        i = start + 2
        rows: list[list[str]] = []
        while i < len(lines) and "|" in lines[i] and lines[i].strip():
            rows.append(cells(lines[i]))
            i += 1

        def render_row(row: list[str], tag: str) -> str:
            tds = []
            for idx, cell in enumerate(row[: len(header)] + [""] * max(0, len(header) - len(row))):
                align = aligns[idx] if idx < len(aligns) else ""
                style = f' style="text-align:{align}"' if align else ""
                tds.append(f"<{tag}{style}>{self.inline(cell)}</{tag}>")
            return f"<tr>{''.join(tds)}</tr>"

        for row in rows:
            self.plain_parts.append(strip_inline(" ".join(row)))
        thead = render_row(header, "th")
        tbody = "".join(render_row(row, "td") for row in rows)
        table = (
            '<div class="table-wrap"><table class="md-table" data-sortable>'
            f"<thead>{thead}</thead><tbody>{tbody}</tbody></table></div>"
        )
        return table, i

    def _list(self, block: list[str]) -> str:
        items: list[tuple[str, list[str]]] = []  # (marker, content lines)
        indent = None
        for line in block:
            match = _LIST_ITEM_RE.match(line)
            if match:
                this_indent = len(match.group(1).expandtabs(4))
                if indent is None:
                    indent = this_indent
                if this_indent <= indent:
                    items.append((match.group(2), [match.group(3)]))
                    continue
            if items:
                items[-1][1].append(line)
        if not items:
            return ""
        ordered = items[0][0][:-1].isdigit() if items[0][0] else False
        tag = "ol" if ordered else "ul"
        rendered = []
        for _, content in items:
            first = content[0]
            rest = content[1:]
            # trim one nesting level of indentation from continuation lines
            trimmed = []
            for line in rest:
                if line.startswith("\t"):
                    trimmed.append(line[1:])
                elif line.startswith("    ") and not _LIST_ITEM_RE.match(line[2:]):
                    trimmed.append(line[4:])
                elif line.startswith("  "):
                    trimmed.append(line[2:])
                else:
                    trimmed.append(line)
            has_block = any(
                _LIST_ITEM_RE.match(line) or _FENCE_RE.match(line) or _TABLE_DELIM_RE.match(line)
                for line in trimmed
            )
            self.plain_parts.append(strip_inline(first))
            if has_block or any(not line.strip() for line in trimmed):
                inner = self._blocks([first] + trimmed)
                rendered.append(f"<li>{inner}</li>")
            else:
                joined = " ".join([first] + [line.strip() for line in trimmed if line.strip()])
                rendered.append(f"<li>{self.inline(joined)}</li>")
        return f"<{tag}>{''.join(rendered)}</{tag}>"


def render_markdown(
    text: str,
    resolver: LinkResolver | None = None,
    slug_prefix: str = "",
    heading_shift: int = 0,
) -> RenderResult:
    renderer = _Renderer(resolver or LinkResolver(), slug_prefix, heading_shift)
    body = renderer.render(text)
    return RenderResult(
        html=body,
        toc=renderer.toc,
        first_image=renderer.first_image,
        plain_text=" ".join(part for part in renderer.plain_parts if part),
    )
