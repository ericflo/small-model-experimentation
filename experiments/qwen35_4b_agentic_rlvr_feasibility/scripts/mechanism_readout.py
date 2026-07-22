"""Answer the token-cap kill rule from the instrumented proxy log (Arm 0 mechanism readout).

QUESTION (pre-registered kill rule 1): in a TIMEOUT episode, is the wall consumed by
  (a) ONE dominant near-wall call  -- a single generation decoding ~4096+ tokens, i.e. the per-call
      cap is the binding constraint and lowering maxTokens 8192->2560 mechanically restores the loop;
  (b) MANY sub-cap calls           -- the loop already turns over and the cap is NOT binding.
KILL Arm 1 if >=80% of timeout episodes are shape (b).

The proxy log is a flat stream of per-call records from N concurrent workers, so episode membership
must be RECONSTRUCTED. pi resends the whole conversation each call, so within an episode the message
list strictly grows and each call's list begins with the previous call's list. Chains are rebuilt by
prefix matching on (role, content-head) signatures; concurrent episodes interleave in time but their
histories never prefix-match each other (different tool outputs / assistant text after turn 1; and
same-task episodes at temperature 1.0 diverge immediately).

Decoded size uses n_content_chars + n_tool_arg_chars (the server runs without --reasoning-parser, so
<think> arrives as content). Chars->tokens uses the standard ~3.5 chars/token for this tokenizer's
English+code mix; the dominant-call threshold (4096 tok) sits far from typical productive turns
(~210 tok median), so classification is insensitive to the exact ratio.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_agentic_rlvr_feasibility"

CHARS_PER_TOK = 3.5
DOMINANT_TOK = 4096          # a call decoding more than this is "dominant near-wall"


def _sig(msg):
    c = msg.get("content")
    if isinstance(c, list):
        c = "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return (msg.get("role"), (c or "")[:400])


def build_chains(records):
    """Group per-call records into episode chains by strict prefix matching."""
    chains = []          # each: {"sigs": [...], "calls": [rec, ...]}
    for rec in sorted(records, key=lambda r: r.get("t", 0)):
        sigs = [_sig(m) for m in rec.get("messages", [])]
        best = None
        for ch in chains:
            prev = ch["sigs"]
            if len(sigs) > len(prev) and sigs[: len(prev)] == prev:
                if best is None or len(prev) > len(best["sigs"]):
                    best = ch
        if best is not None:
            best["sigs"] = sigs
            best["calls"].append(rec)
        else:
            chains.append({"sigs": sigs, "calls": [rec]})
    return chains


def decoded_chars(rec):
    return (rec.get("n_content_chars") or 0) + (rec.get("n_tool_arg_chars") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-log", default=str(OUTD / "logs" / "pi_capladder_proxy.jsonl"))
    ap.add_argument("--episodes", default=str(OUTD / "reports" / "pi_arm0_reanchor.json"),
                    help="pi_episode result whose timeout count anchors the chain classification")
    ap.add_argument("--wall-secs", type=float, default=600.0)
    a = ap.parse_args()

    records = [json.loads(l) for l in open(a.proxy_log) if l.strip()]
    # drop probe traffic: real pi calls carry pi's system prompt as message 0
    records = [r for r in records if r.get("messages") and r["messages"][0].get("role") == "system"]
    chains = build_chains(records)
    ep = json.load(open(a.episodes)) if Path(a.episodes).exists() else {}
    n_timeouts = sum(1 for e in ep.get("episodes", []) if e.get("exit") == 124)

    print(f"proxy calls: {len(records)} -> reconstructed chains: {len(chains)} "
          f"(episode file: {len(ep.get('episodes', []))} episodes, {n_timeouts} timeouts)")

    rows = []
    for ch in chains:
        calls = ch["calls"]
        dur = calls[-1]["t"] - calls[0]["t"]
        sizes = [decoded_chars(c) for c in calls]
        max_tok = max(sizes) / CHARS_PER_TOK
        rows.append({"n_calls": len(calls), "dur": dur, "max_call_tok": max_tok,
                     "total_tok": sum(sizes) / CHARS_PER_TOK})

    # timeout-shaped chains: the N longest-duration chains, N = the episode file's timeout count
    # (chain duration undercounts the episode wall -- it excludes the final in-flight call that the
    # external kill truncates, which never completes and so never logs -- so rank, don't threshold)
    rows.sort(key=lambda r: -r["dur"])
    to_chains = rows[:n_timeouts] if n_timeouts else [r for r in rows if r["dur"] > 0.8 * a.wall_secs]

    print(f"\n{'chain':>5s} {'calls':>5s} {'dur_s':>7s} {'max_call_tok':>12s} {'total_tok':>9s}  shape")
    dominant = 0
    for i, r in enumerate(rows):
        is_to = r in to_chains
        shape = "DOMINANT-CALL" if r["max_call_tok"] >= DOMINANT_TOK else "many-sub-cap"
        if is_to and r["max_call_tok"] >= DOMINANT_TOK:
            dominant += 1
        print(f"{i:5d} {r['n_calls']:5d} {r['dur']:7.0f} {r['max_call_tok']:12.0f} "
              f"{r['total_tok']:9.0f}  {shape}{'  <-- timeout-shaped' if is_to else ''}")

    if to_chains:
        frac_sub = 1 - dominant / len(to_chains)
        print(f"\ntimeout-shaped chains: {len(to_chains)} | dominant-call: {dominant} "
              f"| many-sub-cap: {len(to_chains) - dominant} ({100 * frac_sub:.0f}%)")
        print("KILL RULE 1 (kill Arm 1 if >=80% many-sub-cap):",
              "KILL -- cap not binding" if frac_sub >= 0.8 else "PROCEED -- cap is binding")
    else:
        print("\nno timeout-shaped chains found; nothing to classify")


if __name__ == "__main__":
    main()
