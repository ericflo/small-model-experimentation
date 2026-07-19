"""Filter native samples by execution -> rejection-sampled SFT corpus (native think + clean code)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "/home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_coding_fitness_harness/src")
sys.path.insert(0, str(Path(__file__).parent))
import rft_lib as R
import code_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, required=True)      # runner output JSONL
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--problems", type=Path, required=True)  # runner input (for messages)
    ap.add_argument("--out", type=Path, required=True)      # rft corpus
    ap.add_argument("--max-per-problem", type=int, default=1)
    a = ap.parse_args()
    meta = json.loads(a.meta.read_text())
    messages_by_id = {}
    for line in a.problems.read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            messages_by_id[p["id"]] = p["messages"]
    gen = {}
    for line in a.gen.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            gen[row["id"]] = row

    n_problems = len(meta)
    n_solved = 0
    n_samples_total = 0
    n_samples_pass = 0
    kept = []
    for tid, m in meta.items():
        row = gen.get(tid)
        if row is None:
            continue
        entry = m["entry"]
        winners = []
        for out in row["outputs"]:
            n_samples_total += 1
            text = out["text"]
            think, after = R.split_think_answer(text)
            cand, _reason = code_env.extract_candidate_code(after, entry)
            if cand is None:
                cand, _reason = code_env.extract_candidate_code(text, entry)
            if R.code_passes(cand, m["asserts"]):
                n_samples_pass += 1
                # skip degenerate/empty thinking; keep clean code (no comments to strip - native code)
                if think and len(think) > 20:
                    winners.append((think, cand, out.get("n_thinking_tokens", 0)))
        if winners:
            n_solved += 1
            # keep the shortest-thinking winners (concise, high quality), up to max
            winners.sort(key=lambda w: w[2])
            for think, cand, _nt in winners[: a.max_per_problem]:
                kept.append({
                    "messages": messages_by_id[tid],
                    "think": think,
                    "answer": f"```python\n{cand.strip()}\n```",
                    "kind": "rft",
                    "family": m["family"], "cat": m["cat"], "family_fn": m["family_fn"],
                    "n_tests": m["n_tests"], "n_why_comments": 0,
                    "row_weight": 1.0, "task_id": tid,
                })

    # messages aren't in the runner output; reload from problems file if needed
    with a.out.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[filter] problems={n_problems} solved={n_solved} ({n_solved/max(1,n_problems):.1%}) "
          f"| samples={n_samples_total} pass={n_samples_pass} ({n_samples_pass/max(1,n_samples_total):.1%}) "
          f"| kept_rows={len(kept)} -> {a.out}")


if __name__ == "__main__":
    main()
