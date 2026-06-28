from __future__ import annotations

import ast
import hashlib
from typing import Any

from .mbpp_env import infer_arity_from_cases, sanitize_reference_function, tokenize_text


def _arg_names(arity: int) -> list[str]:
    return [f"a{i}" for i in range(arity)]


def _make_code(entry_point: str, arity: int, body: str, imports: str = "") -> str:
    args = ", ".join(_arg_names(arity))
    import_block = imports.rstrip()
    if import_block:
        import_block += "\n\n"
    body_lines = "\n".join("    " + line if line else line for line in body.strip("\n").splitlines())
    return f"{import_block}def {entry_point}({args}):\n{body_lines}\n"


def _candidate(arm: str, template_id: str, entry_point: str, arity: int, body: str, imports: str = "") -> dict[str, Any]:
    code = _make_code(entry_point, arity, body, imports=imports)
    return {
        "arm": arm,
        "template_id": template_id,
        "code": code,
        "code_hash": hashlib.sha1(code.encode("utf-8")).hexdigest()[:16],
        "graph_size_proxy": len([line for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]),
    }


def _dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for cand in candidates:
        key = cand["code_hash"]
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def manual_core_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    entry = record["entry_point"]
    arity = infer_arity_from_cases(record["public_cases"], entry)
    cands: list[dict[str, Any]] = []

    # Generic safe fallbacks and common scalar/list/string operations.
    if arity == 1:
        cands.extend(
            [
                _candidate("manual_core", "identity", entry, arity, "return a0"),
                _candidate("manual_core", "len", entry, arity, "return len(a0)"),
                _candidate("manual_core", "sum", entry, arity, "return sum(a0)"),
                _candidate("manual_core", "max", entry, arity, "return max(a0)"),
                _candidate("manual_core", "min", entry, arity, "return min(a0)"),
                _candidate("manual_core", "sorted", entry, arity, "return sorted(a0)"),
                _candidate("manual_core", "reversed_list", entry, arity, "return list(reversed(a0))"),
                _candidate("manual_core", "unique_preserve_order", entry, arity, "return list(dict.fromkeys(a0))"),
                _candidate("manual_core", "word_at_start", entry, arity, "return 'Found a match!' if re.match(r'^\\w+', a0) else 'Not matched!'", imports="import re"),
                _candidate("manual_core", "lowercase_underscore_full", entry, arity, "return 'Found a match!' if re.match(r'^[a-z]+_[a-z]+$', a0) else 'Not matched!'", imports="import re"),
                _candidate("manual_core", "lowercase_underscore_search", entry, arity, "return 'Found a match!' if re.search(r'[a-z]+_[a-z]+', a0) else 'Not matched!'", imports="import re"),
            ]
        )

    if arity == 2:
        cands.extend(
            [
                _candidate("manual_core", "tuple_list_all_k", entry, arity, "return all(all(x == a1 for x in tup) for tup in a0)"),
                _candidate("manual_core", "tuple_list_any_k", entry, arity, "return any(any(x == a1 for x in tup) for tup in a0)"),
                _candidate("manual_core", "count_occurrences", entry, arity, "return a0.count(a1)"),
                _candidate("manual_core", "contains", entry, arity, "return a1 in a0"),
                _candidate("manual_core", "take_first_n", entry, arity, "return a0[:a1]"),
                _candidate("manual_core", "take_last_n", entry, arity, "return a0[-a1:]"),
                _candidate("manual_core", "topk_counter_most_common", entry, arity, "flat = [x for row in a0 for x in row]\nreturn [x for x, _ in collections.Counter(flat).most_common(a1)]", imports="import collections"),
            ]
        )

    if arity == 3:
        cands.extend(
            [
                _candidate("manual_core", "between_inclusive", entry, arity, "return a1 <= a0 <= a2"),
                _candidate("manual_core", "slice_range", entry, arity, "return a0[a1:a2]"),
                _candidate("manual_core", "replace_value", entry, arity, "return [a2 if x == a1 else x for x in a0]"),
            ]
        )

    return _dedupe(cands)


def manual_expanded_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    entry = record["entry_point"]
    arity = infer_arity_from_cases(record["public_cases"], entry)
    text = record["task_text"].lower()
    cands: list[dict[str, Any]] = []

    if arity == 1:
        cands.extend(
            [
                _candidate(
                    "manual_expanded",
                    "rearrange_no_adjacent_heap",
                    entry,
                    arity,
                    """
heap = [(-count, ch) for ch, count in collections.Counter(a0).items()]
heapq.heapify(heap)
out = []
while len(heap) >= 2:
    c1, ch1 = heapq.heappop(heap)
    c2, ch2 = heapq.heappop(heap)
    out.extend([ch1, ch2])
    c1 += 1
    c2 += 1
    if c1:
        heapq.heappush(heap, (c1, ch1))
    if c2:
        heapq.heappush(heap, (c2, ch2))
if heap:
    out.append(heap[0][1])
ans = ''.join(out)
return ans if all(ans[i] != ans[i + 1] for i in range(len(ans) - 1)) else False
""",
                    imports="import collections\nimport heapq",
                ),
                _candidate(
                    "manual_expanded",
                    "odd_bit_mask_positions_0_2_4",
                    entry,
                    arity,
                    """
n = int(a0)
mask = 0
i = 0
while (1 << i) <= max(n, 1):
    if i % 2 == 0:
        mask |= (1 << i)
    i += 1
return n | mask
""",
                ),
                _candidate(
                    "manual_expanded",
                    "alternating_digit_sums_divisible_by_11",
                    entry,
                    arity,
                    """
digits = [int(ch) for ch in str(abs(int(a0)))]
odd_pos = sum(digits[0::2])
even_pos = sum(digits[1::2])
return abs(odd_pos - even_pos) % 11 == 0
""",
                ),
                _candidate(
                    "manual_expanded",
                    "sum_even_digit_values_gt_odd",
                    entry,
                    arity,
                    """
digits = [int(ch) for ch in str(abs(int(a0)))]
even_digit_sum = sum(d for d in digits if d % 2 == 0)
odd_digit_sum = sum(d for d in digits if d % 2 == 1)
return even_digit_sum > odd_digit_sum
""",
                ),
            ]
        )

    if arity == 2:
        cands.extend(
            [
                _candidate(
                    "manual_expanded",
                    "topk_frequency_minheap_insertion_order",
                    entry,
                    arity,
                    """
freq = collections.Counter(x for row in a0 for x in row)
heap = []
for key, val in freq.items():
    item = (val, key)
    if len(heap) < a1:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)
return [key for _, key in heap]
""",
                    imports="import collections\nimport heapq",
                ),
                _candidate(
                    "manual_expanded",
                    "topk_frequency_minheap_pop_order",
                    entry,
                    arity,
                    """
freq = collections.Counter(x for row in a0 for x in row)
heap = []
for key, val in freq.items():
    if len(heap) < a1:
        heapq.heappush(heap, (val, key))
    elif val > heap[0][0]:
        heapq.heapreplace(heap, (val, key))
out = []
while heap:
    out.append(heapq.heappop(heap)[1])
return out
""",
                    imports="import collections\nimport heapq",
                ),
                _candidate(
                    "manual_expanded",
                    "topk_frequency_count_then_value",
                    entry,
                    arity,
                    "freq = collections.Counter(x for row in a0 for x in row)\nreturn [x for x, _ in sorted(freq.items(), key=lambda kv: (-kv[1], -kv[0]))[:a1]]",
                    imports="import collections",
                ),
                _candidate(
                    "manual_expanded",
                    "longest_subseq_adjacent_diff_le_1",
                    entry,
                    arity,
                    """
arr = list(a0)
n = min(int(a1), len(arr))
if n <= 0:
    return 0
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if abs(arr[i] - arr[j]) <= 1 and dp[i] < dp[j] + 1:
            dp[i] = dp[j] + 1
return max(dp)
""",
                ),
                _candidate(
                    "manual_expanded",
                    "longest_subseq_adjacent_diff_eq_1",
                    entry,
                    arity,
                    """
arr = list(a0)
n = min(int(a1), len(arr))
if n <= 0:
    return 0
dp = [1] * n
for i in range(1, n):
    for j in range(i):
        if abs(arr[i] - arr[j]) == 1 and dp[i] < dp[j] + 1:
            dp[i] = dp[j] + 1
return max(dp)
""",
                ),
                _candidate(
                    "manual_expanded",
                    "tuple_list_all_k_set",
                    entry,
                    arity,
                    "return all(set(tup) == {a1} for tup in a0)",
                ),
            ]
        )

    # Prompt-keyed variants are still generic kernels; the keyword gate only keeps
    # the search cheap and prevents irrelevant algorithms from dominating logs.
    if "underscore" in text and arity == 1:
        cands.append(
            _candidate(
                "manual_expanded",
                "lowercase_underscore_one_or_more_groups",
                entry,
                arity,
                "return 'Found a match!' if re.fullmatch(r'[a-z]+(?:_[a-z]+)+', a0) else 'Not matched!'",
                imports="import re",
            )
        )
    if "word at the beginning" in text and arity == 1:
        cands.append(
            _candidate(
                "manual_expanded",
                "word_at_start_alpha",
                entry,
                arity,
                "return 'Found a match!' if re.match(r'^[A-Za-z]+', a0) else 'Not matched!'",
                imports="import re",
            )
        )

    return _dedupe(cands)


def retrieved_transplant_candidates(
    record: dict[str, Any],
    train_library: list[dict[str, Any]],
    top_k: int = 80,
) -> list[dict[str, Any]]:
    entry = record["entry_point"]
    arity = infer_arity_from_cases(record["public_cases"], entry)
    query = tokenize_text(record["task_text"])
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in train_library:
        toks = tokenize_text(item.get("task_text", ""))
        if not toks:
            continue
        overlap = len(query & toks)
        union = len(query | toks)
        score = overlap / union if union else 0.0
        if overlap:
            scored.append((score, item))
    scored.sort(key=lambda row: (row[0], row[1].get("task_id", -1)), reverse=True)

    cands: list[dict[str, Any]] = []
    for rank, (_, item) in enumerate(scored[:top_k]):
        code = sanitize_reference_function(item.get("reference_code", ""), entry, arity)
        if not code:
            continue
        cands.append(
            {
                "arm": "retrieved_transplant",
                "template_id": f"retrieved_rank_{rank:03d}_task_{item.get('task_id')}",
                "source_task_id": item.get("task_id"),
                "code": code,
                "code_hash": hashlib.sha1(code.encode("utf-8")).hexdigest()[:16],
                "graph_size_proxy": len([line for line in code.splitlines() if line.strip() and not line.strip().startswith("#")]),
            }
        )
    return _dedupe(cands)


def generate_candidates(
    record: dict[str, Any],
    train_library: list[dict[str, Any]] | None,
    arms: list[str],
    retrieval_top_k: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if "manual_core" in arms:
        candidates.extend(manual_core_candidates(record))
    if "manual_expanded" in arms:
        candidates.extend(manual_expanded_candidates(record))
    if "retrieved_transplant" in arms and train_library is not None:
        candidates.extend(retrieved_transplant_candidates(record, train_library, top_k=retrieval_top_k))
    return _dedupe(candidates)
