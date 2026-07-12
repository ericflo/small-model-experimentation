#!/usr/bin/env python3
"""Skin-shuffled (+ repair) oracle traces: every item gets a fresh pseudo-
vocabulary applied consistently to prompt, trace, and answer, so no invented
surface form repeats across training rows. CPU-only.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
from gym import base  # noqa: E402
from gym.families import load as load_family  # noqa: E402

TRACE_FAMILIES = ("loomfix", "glyphgate", "stallwright", "packhouse",
                  "patchwheel", "kilnrite", "burrowmaze")
REPAIR_FAMILIES = ("loomfix", "glyphgate", "kilnrite", "stallwright",
                   "packhouse", "patchwheel", "burrowmaze")

def wrong_candidate(family, item):
    """A plausible wrong value: perturb the oracle answer deterministically."""
    value = base.extract_answer(family.oracle_atom(item))
    if value is None:
        return None
    rng = base.rng_for("wrongcand", item["id"])
    as_int = base.canon_int(value)
    if as_int is not None and str(as_int) == value.strip():
        return str(as_int + rng.choice([1, -1, 2]))
    parts = value.split("-")
    if len(parts) >= 3:  # tape/glyph string: swap two positions
        i, j = 0, len(parts) - 1
        parts[i], parts[j] = parts[j], parts[i]
        return "-".join(parts)
    return None

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-seed", type=int, default=18001)
    ap.add_argument("--per-level", type=int, default=25)
    ap.add_argument("--repair-per-level", type=int, default=12)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_skinshuffle.jsonl")
    args = ap.parse_args()

    rows, repair_count = [], 0
    for name in TRACE_FAMILIES:
        family = load_family(name)
        skinnable = getattr(family, "SKINNABLE", ())
        for level in family.LEVELS:
            items = family.gen_atoms(args.gen_seed, level, args.per_level)
            for index, item in enumerate(items):
                trace = family.oracle_trace(item)
                value = base.extract_answer(family.oracle_atom(item))
                if value is None:
                    continue
                mapping = base.skin_mapping(skinnable, base.rng_for("skin", item["id"]))
                prompt = base.apply_skin(item["prompt"], mapping)
                s_trace = base.apply_skin(trace, mapping)
                s_value = base.apply_skin(value, mapping)
                rows.append({"family": name, "level": level, "kind": "skin_trace",
                             "messages": [{"role": "user", "content": prompt}],
                             "think": s_trace, "answer": f"ANSWER: {s_value}",
                             "n_think_tokens": max(1, len(s_trace) // 4)})
                if (index < args.repair_per_level and name in REPAIR_FAMILIES
                        and hasattr(family, "oracle_trace_repair")):
                    wrong = wrong_candidate(family, item)
                    if wrong:
                        rtrace = family.oracle_trace_repair(item, wrong)
                        if rtrace:
                            rows.append({"family": name, "level": level,
                                         "kind": "skin_repair",
                                         "messages": [{"role": "user", "content": prompt}],
                                         "think": base.apply_skin(rtrace, mapping),
                                         "answer": f"ANSWER: {s_value}",
                                         "n_think_tokens": max(1, len(rtrace) // 4)})
                            repair_count += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("skin-shuffled rows:", len(rows), f"(repairs {repair_count})",
          dict(Counter(r["family"] for r in rows)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
