"""Think-block repetition detection and rejected-token localization.

Fingerprint-and-verify periodic-repetition scanner over decoded text plus
token-space refinement of the loop-initiation position. Reimplements the
published mining detector (Liquid AI's Apache-2.0 pipeline; see
docs/final_token_preference_optimization.md) with identical thresholds so
whitebox loop rates are comparable to the published numbers.

This detector LOCALIZES loop starts for FTPO row mining. It is deliberately
distinct from the corpus's strict content-blind 8k-tail gate detector
(verified-macro family), which classifies termination but cannot localize.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentence-restart connectives: when a detected period boundary sits
# mid-sentence but the repeated material restarts a sentence with one of
# these words, the rejected token moves onto the restart word.
RESTART_WORDS = {
    "actually", "after", "also", "alternatively", "because", "but", "finally",
    "first", "given", "hmm", "however", "in", "let", "looking", "maybe",
    "now", "okay", "perhaps", "second", "since", "so", "the", "then",
    "therefore", "this", "thus", "wait",
}


@dataclass(frozen=True)
class RepeatHit:
    start: int          # char offset of the earliest occurrence of the pattern
    end: int            # char offset one past the last full repeat
    period: int         # pattern length in chars
    repeats: int        # number of contiguous occurrences (incl. the first)
    snippet: str        # pattern text (truncated for logging)

    @property
    def repeat_start(self) -> int:
        """Char offset where the second occurrence (first repetition) begins."""
        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> RepeatHit | None:
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return None
    pattern = text[start_pos: start_pos + period]

    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos: pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos

    pos = start_pos - period
    while pos >= 0 and text[pos: pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period

    if reps >= min_repeats and reps * period >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return RepeatHit(start_pos, end_pos, period, reps, snippet)
    return None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> RepeatHit | None:
    """Scan for the first verified contiguous periodic repetition in text."""
    if not text or len(text) < min_total_repeated:
        return None
    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos: sample_pos + sample_len]

        other = text.find(fingerprint, sample_pos + sample_len)
        if other != -1 and min_period <= other - sample_pos <= max_period:
            hit = _verify_repetition_at(
                text, sample_pos, other - sample_pos,
                min_repeats=min_repeats, min_total_repeated=min_total_repeated)
            if hit is not None:
                return hit

        other = text.rfind(fingerprint, 0, sample_pos)
        if other != -1 and min_period <= sample_pos - other <= max_period:
            hit = _verify_repetition_at(
                text, other, sample_pos - other,
                min_repeats=min_repeats, min_total_repeated=min_total_repeated)
            if hit is not None:
                return hit
    return None


# ---------------------------------------------------------------------------
# char -> token mapping and rejected-token localization
# ---------------------------------------------------------------------------

def token_char_offsets(decoded_tokens: list[str]) -> list[int]:
    """Cumulative char start offset of each token in the concatenated text."""
    offsets = []
    pos = 0
    for tok in decoded_tokens:
        offsets.append(pos)
        pos += len(tok)
    return offsets


def char_to_token_index(offsets: list[int], total_len: int, char_pos: int) -> int | None:
    """Index of the token containing char_pos (linear scan; None if OOR)."""
    if char_pos < 0 or char_pos >= total_len:
        return None
    lo, hi = 0, len(offsets) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if offsets[mid] <= char_pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _is_boundary_only(surface: str) -> bool:
    s = surface.strip()
    if not s:
        return False  # pure whitespace handled separately
    return not any(ch.isalnum() or ch == "_" for ch in s)


def _skip_boundary_tokens(decoded: list[str], idx: int,
                          max_tokens: int = 4, max_chars: int = 12,
                          skip_whitespace: bool = False) -> int:
    """Advance past punctuation-only tokens (budgeted) and, optionally,
    whitespace-only tokens (unbudgeted) so the target lands on readable text.

    Micro-divergence from the reference implementation (which skips boundary
    tokens and whitespace in two separate passes and can land on punctuation
    after a newline): with skip_whitespace=True the two skips interleave, so
    a "\\n- item" loop start resolves to "item", not "-".
    """
    skipped_tokens = 0
    skipped_chars = 0
    while idx < len(decoded):
        surface = decoded[idx]
        if skip_whitespace and surface and not surface.strip():
            idx += 1
            continue
        if surface.strip() and _is_boundary_only(surface):
            skipped_tokens += 1
            skipped_chars += len(surface)
            if skipped_tokens > max_tokens or skipped_chars > max_chars:
                break
            idx += 1
            continue
        break
    return idx


def _normalized_word(surface: str) -> str:
    s = surface.strip().lower()
    start, end = 0, len(s)
    while start < end and not (s[start].isalnum() or s[start] == "_"):
        start += 1
    while end > start and not (s[end - 1].isalnum() or s[end - 1] == "_"):
        end -= 1
    return s[start:end]


def locate_rejected_token(
    decoded_tokens: list[str],
    hit: RepeatHit,
) -> int | None:
    """Token index of the first readable token of the first repetition.

    decoded_tokens are the decoded surfaces of the generated (think) tokens,
    aligned with the text the detector scanned (their concatenation).
    """
    total_len = sum(len(t) for t in decoded_tokens)
    offsets = token_char_offsets(decoded_tokens)

    seed_idx = char_to_token_index(offsets, total_len, hit.start)
    repeat_idx = char_to_token_index(offsets, total_len, hit.repeat_start)
    if seed_idx is None or repeat_idx is None or repeat_idx <= seed_idx:
        return None

    keys = [t.lstrip() for t in decoded_tokens]
    period_tokens = repeat_idx - seed_idx

    # Slide left while the token one token-period earlier matches — find the
    # earliest token-aligned start of the loop.
    while seed_idx - 1 >= 0 and seed_idx - 1 + period_tokens < len(keys) \
            and keys[seed_idx - 1] == keys[seed_idx - 1 + period_tokens]:
        seed_idx -= 1

    target = seed_idx + period_tokens
    if target >= len(decoded_tokens):
        return None
    target = _skip_boundary_tokens(decoded_tokens, target, skip_whitespace=True)
    if target >= len(decoded_tokens):
        return None

    # Sentence-restart heuristic: if the boundary is mid-sentence but the
    # repeated material restarts a sentence with a known connective, reject
    # the restart word instead.
    word = _normalized_word(decoded_tokens[target])
    if word:
        punct_start = target + 1
        punct_end = punct_start
        chars = 0
        while punct_end < len(decoded_tokens) and punct_end - punct_start < 3:
            surface = decoded_tokens[punct_end]
            if surface.strip() and _is_boundary_only(surface):
                chars += len(surface)
                if chars > 8:
                    break
                punct_end += 1
                continue
            break
        punct_text = "".join(decoded_tokens[punct_start:punct_end])
        seed_prev = seed_idx - 1
        repeat_prev = seed_idx + period_tokens - 1
        if (punct_end > punct_start
                and any(c in punct_text for c in ".!?")
                and seed_prev >= 0 and repeat_prev < len(decoded_tokens)
                and decoded_tokens[seed_prev] != decoded_tokens[repeat_prev]):
            nxt = _skip_boundary_tokens(decoded_tokens, punct_end,
                                        skip_whitespace=True)
            if nxt < len(decoded_tokens) \
                    and _normalized_word(decoded_tokens[nxt]) in RESTART_WORDS:
                target = _skip_boundary_tokens(decoded_tokens, punct_end,
                                               skip_whitespace=True)

    # Advance past whitespace-only tokens.
    while target < len(decoded_tokens) and not decoded_tokens[target].strip():
        target += 1
    if target >= len(decoded_tokens):
        return None
    return target
