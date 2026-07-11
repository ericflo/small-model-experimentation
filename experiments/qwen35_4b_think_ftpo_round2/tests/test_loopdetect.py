"""Selftests for the mining detector and rejected-token localization."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import loopdetect  # noqa: E402


def test_no_loop_on_plain_text():
    text = "The quick brown fox jumps over the lazy dog. " * 2
    assert loopdetect.find_inner_repetition(text) is None


def _varied_filler(n_chars: int) -> str:
    """Non-repetitive filler long enough to host fingerprint probes."""
    words = ["alpha", "brine", "codex", "delta", "ember", "fjord", "gyre",
             "helix", "ionic", "joule", "krill", "lumen", "moss", "nadir"]
    out = []
    i = 0
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(words[i % len(words)] + str(i))
        i += 1
    return " ".join(out)[:n_chars]


def test_detects_simple_period_loop():
    span = "Wait, let me reconsider the constraint. "
    text = _varied_filler(200) + " " + span * 6 + "Answer: 4"
    hit = loopdetect.find_inner_repetition(text)
    assert hit is not None
    assert hit.repeats >= 4
    assert hit.period == len(span)
    # The detected pattern is a rotation of the repeated span; check the
    # doubled pattern so the probe word can straddle the rotation boundary.
    pattern = text[hit.start: hit.start + hit.period]
    assert "reconsider" in pattern + pattern


def test_short_repetition_below_thresholds_ignored():
    # 3 repeats of a 12-char span: repeats < 4 and total 36 < 60.
    text = _varied_filler(150) + "abcdefghijkl" * 3 + _varied_filler(150)
    assert loopdetect.find_inner_repetition(text) is None


def test_single_char_runaway_detected():
    # Filler long enough that a 128-interval probe lands inside the run.
    text = _varied_filler(250) + "!" * 90
    hit = loopdetect.find_inner_repetition(text)
    assert hit is not None
    assert hit.period * hit.repeats >= 60


def test_intervening_char_breaks_chain():
    span = "same segment repeated here! "
    text = span * 3 + "X" + span * 3  # 3+3 but not contiguous 4
    assert loopdetect.find_inner_repetition(text, min_repeats=4) is None


def test_locate_rejected_token_simple():
    # tokens: setup then a loop of ["Wait", ",", " let", " me", " retry", ". "]
    setup = ["Solve", " the", " task", ".", " Think", " hard", ". "]
    loop_unit = ["Wait", ",", " let", " me", " retry", ".", " "]
    tokens = setup + loop_unit * 6
    text = "".join(tokens)
    hit = loopdetect.find_inner_repetition(text)
    assert hit is not None
    idx = loopdetect.locate_rejected_token(tokens, hit)
    assert idx is not None
    # The rejected token must be inside the repeated region and readable.
    assert idx >= len(setup)
    assert tokens[idx].strip()
    # It should be the start of a repetition unit (a "Wait" occurrence).
    assert tokens[idx].strip().lower().startswith("wait")


def test_locate_rejected_skips_boundary_tokens():
    setup = ["Reason", " about", " it", ":"]
    loop_unit = ["\n", "-", " check", " the", " rule", " again", ";"]
    tokens = setup + loop_unit * 8
    text = "".join(tokens)
    hit = loopdetect.find_inner_repetition(text)
    assert hit is not None
    idx = loopdetect.locate_rejected_token(tokens, hit)
    assert idx is not None
    assert tokens[idx].strip()
    assert any(c.isalnum() for c in tokens[idx])


def test_char_token_mapping_roundtrip():
    tokens = ["ab", "c", "", "def", " gh"]
    offsets = loopdetect.token_char_offsets(tokens)
    text = "".join(tokens)
    for char_pos in range(len(text)):
        tok_idx = loopdetect.char_to_token_index(offsets, len(text), char_pos)
        assert tok_idx is not None
        start = offsets[tok_idx]
        assert start <= char_pos
        end = start + len(tokens[tok_idx])
        assert char_pos < end or len(tokens[tok_idx]) == 0
    assert loopdetect.char_to_token_index(offsets, len(text), len(text)) is None


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
