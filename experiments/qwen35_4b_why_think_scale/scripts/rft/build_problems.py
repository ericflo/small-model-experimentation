"""Build a runner-input JSONL of synthetic problems + a sidecar of (entry, asserts, meta)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import rft_lib as R

POOL = "/home/ericflo/Development/small-model-experimentation/large_artifacts/qwen35_4b_why_think_scale/corpora/why_think_40000.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=10000)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(POOL)][a.offset : a.offset + a.count]
    meta = {}
    with a.out.open("w") as f:
        for row in rows:
            entry, asserts = R.parse_problem(row)
            tid = row["task_id"]
            if not entry or not asserts:
                continue
            f.write(json.dumps({"id": tid, "messages": row["messages"]}, ensure_ascii=False) + "\n")
            meta[tid] = {
                "entry": entry, "asserts": asserts,
                "family": row.get("family", "cognitive_core"), "cat": row.get("cat", ""),
                "family_fn": row.get("family_fn", ""), "n_tests": len(asserts),
            }
    a.meta.write_text(json.dumps(meta))
    print(f"[build] wrote {len(meta)} problems (offset {a.offset}) -> {a.out}")


if __name__ == "__main__":
    main()
