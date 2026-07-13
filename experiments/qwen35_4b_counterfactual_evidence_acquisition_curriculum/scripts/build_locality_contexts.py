#!/usr/bin/env python3
"""Materialize a fresh frozen non-coding block for the policy curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "A glass greenhouse stays cooler at noon when",
    "Before mailing a handmade card, its sender often",
    "On a windy footbridge, cyclists feel steadier after",
    "A brass door hinge stops squeaking once someone",
    "When seedlings crowd one tray, a careful gardener",
    "A canvas tent sheds overnight rain better if campers",
    "Before recording an acoustic guitar, the engineer usually",
    "Roasted coffee beans keep their aroma longest when they",
    "During a campsite blackout, a headlamp lets hikers",
    "A linen shirt keeps its shape while drying when",
    "When a whiteboard marker fades, the presenter",
    "Before packing ceramic bowls for a move, people often",
)

PREFIXES = (
    "Finish this everyday sentence in one brief clause:\n",
    "Supply a natural short completion and no explanation:\n",
    "Write only a sensible ending for this ordinary statement:\n",
    "Continue the sentence with a compact commonplace phrase:\n",
)

PRIOR_CONTEXTS = (
    EXP.parent / "qwen35_4b_validation_policy_counterexample_curriculum" / "data" / "locality_contexts.json",
    EXP.parent / "qwen35_4b_transaction_invariant_recovery_curriculum" / "data" / "locality_contexts.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_contexts.json")
    parser.add_argument("--seed", type=int, default=87600)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"policy-locality-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
                "messages": [{"role": "user", "content": content}],
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "contexts": contexts,
        "count": len(contexts),
        "forbidden_prior_context_files": [str(path) for path in PRIOR_CONTEXTS],
    }
    current_hashes = {row["content_sha256"] for row in contexts}
    overlap = set()
    for path in PRIOR_CONTEXTS:
        prior = json.loads(path.read_text())
        overlap.update(
            current_hashes & {row["content_sha256"] for row in prior["contexts"]}
        )
    if overlap:
        raise SystemExit(f"locality contexts overlap prior blocks: {sorted(overlap)}")
    payload["prior_content_overlap_count"] = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "count": len(contexts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
