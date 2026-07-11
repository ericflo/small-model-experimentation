"""Outcome-conditioned pivot mining over sampled think-token trajectories.

Given n verifier-scored rollouts of the same prompt, build a prefix tree over
their think-token-ID sequences and find divergence nodes where sibling
branches have a large verified success-rate gap. Each eligible node yields one
FTPO row: context = prompt + shared think prefix; rejected = the lower-success
branch's next token; chosen = the higher-success branches' next tokens.

The signal is Monte-Carlo process supervision distilled to single-token
preferences: every chosen/rejected token was actually sampled by the model
(provenance: the model's own distribution), and the label is the downstream
verifier outcome — no gold reasoning, no teacher.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _Node:
    count: int = 0
    successes: int = 0
    children: dict[int, "_Node"] = field(default_factory=dict)


@dataclass(frozen=True)
class PivotRow:
    depth: int                 # think-prefix length in tokens (== context suffix)
    prefix: tuple[int, ...]    # shared think token ids up to the node
    rejected_id: int
    chosen_ids: tuple[int, ...]
    gap: float                 # success-rate gap between best and worst branch
    n_rejected: int            # rollouts through the rejected branch
    n_chosen: int              # rollouts through the chosen branches (total)


def _build_tree(sequences: list[list[int]], successes: list[bool]) -> _Node:
    root = _Node()
    for seq, won in zip(sequences, successes):
        node = root
        node.count += 1
        node.successes += int(won)
        for token_id in seq:
            node = node.children.setdefault(token_id, _Node())
            node.count += 1
            node.successes += int(won)
    return root


def mine_pivots(
    sequences: list[list[int]],
    successes: list[bool],
    *,
    min_depth: int = 16,
    min_branch_rollouts: int = 2,
    min_gap: float = 0.5,
    max_nodes: int = 2,
) -> list[PivotRow]:
    """Return up to max_nodes pivot rows, largest gap first, ties deeper."""
    if len(sequences) != len(successes) or not sequences:
        return []
    root = _build_tree(sequences, successes)

    candidates: list[PivotRow] = []
    stack: list[tuple[_Node, list[int]]] = [(root, [])]
    while stack:
        node, prefix = stack.pop()
        depth = len(prefix)
        eligible = {
            tid: child for tid, child in node.children.items()
            if child.count >= min_branch_rollouts
        }
        if depth >= min_depth and len(eligible) >= 2:
            rates = {tid: c.successes / c.count for tid, c in eligible.items()}
            lo_tid = min(rates, key=lambda t: (rates[t], -eligible[t].count))
            hi_rate = max(rates.values())
            gap = hi_rate - rates[lo_tid]
            if gap >= min_gap:
                chosen = tuple(sorted(
                    tid for tid, rate in rates.items()
                    if tid != lo_tid and rate >= rates[lo_tid] + min_gap
                ))
                if chosen:
                    candidates.append(PivotRow(
                        depth=depth,
                        prefix=tuple(prefix),
                        rejected_id=lo_tid,
                        chosen_ids=chosen,
                        gap=gap,
                        n_rejected=eligible[lo_tid].count,
                        n_chosen=sum(eligible[t].count for t in chosen),
                    ))
        for tid, child in node.children.items():
            # No divergence signal can exist below a single-rollout path.
            if child.count >= min_branch_rollouts:
                stack.append((child, prefix + [tid]))

    candidates.sort(key=lambda r: (-r.gap, -r.depth))
    return candidates[:max_nodes]
