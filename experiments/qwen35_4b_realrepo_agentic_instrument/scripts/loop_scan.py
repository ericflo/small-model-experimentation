#!/usr/bin/env python3
"""Are our TERMINATION failures doom loops? A free text scan over pi trajectories we already logged.

Two independent surfaces in this session put TERMINATION, not capability, at the bottleneck:
  * real-repo agentic coding through pi: 20/62 held-out episodes hit the 600s wall; the warm-start's
    median episode ran 574s vs base's 381s and timed out 8 more times;
  * held-out digit induction: 29.5% of CoT episodes never emitted an answer even at a 3072-token cap,
    while accuracy AMONG committed episodes was 0.972.

If those failures are periodic repetition (doom loops), there is an off-the-shelf intervention aimed
exactly at them: loop-targeted FTPO (docs/final_token_preference_optimization.md), which Liquid AI report
taking Qwen3.5-4B -- our exact model -- from 22.9% to 1% looping with eval scores rising, at a 1-2
GPU-hour LoRA. This repo already documents that work, already has the FTPO machinery, and the doc
already names this experiment as the outstanding one. What it does NOT have is a measurement of the loop
rate at the budgets where our current work actually lives; C52's "loops are ~0.1%" was measured at
think@1024-2048, and the same doc notes loops "dominate many 32k+ contacts".

Detector is Liquid's: a span repeating at least 4 times over at least 60 characters. Applied to the
assistant text inside pi's --mode json event stream, so it costs nothing but a file read.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_realrepo_agentic_instrument"

MIN_REPEATS = 4
MIN_CHARS = 60


def find_loop(text: str, min_repeats=MIN_REPEATS, min_chars=MIN_CHARS):
    """Return (span, n_repeats) for the longest periodic repetition, or None.

    Scans candidate periods from the END of the text, which is where a doom loop lives: the tail is the
    span that ran to the budget. Periods are tried coarse-to-fine so the reported span is the actual
    repeating unit rather than a multiple of it.
    """
    t = text.rstrip()
    if len(t) < min_chars * min_repeats:
        return None
    best = None
    for period in range(min_chars, min(len(t) // min_repeats, 2000) + 1):
        tail = t[-period * min_repeats:]
        unit = tail[:period]
        if tail == unit * min_repeats:
            # extend: how many times does it actually repeat?
            n = min_repeats
            while len(t) >= period * (n + 1) and t[-period * (n + 1):] == unit * (n + 1):
                n += 1
            if best is None or period * n > best[1] * len(best[0]):
                best = (unit, n)
    return best


def assistant_messages(path: Path):
    """Return the text of each assistant message, as complete as the stream got.

    THE BUG THIS REPLACES, and it mattered: the first version filtered on event types
    message_start/message_end/message and therefore read only COMPLETED messages. A doom-looping
    message never completes -- it runs to the token wall and emits no message_end (C62 measured exactly
    this: "92 stream deltas, 1 message_end, content frozen"). So the scan structurally excluded the
    pathology it was built to find, and duly reported 0%.

    pi streams `message_update` events whose `assistantMessageEvent.partial` is a cumulative snapshot of
    the in-flight message. Keeping the LONGEST snapshot per responseId recovers the full text of messages
    that never finished. Messages are returned SEPATELY rather than concatenated, because a loop lives
    inside one message and joining them could both mask a loop and manufacture a false one at a seam.
    """
    best = {}
    order = []

    def note(rid, text):
        if not rid or not isinstance(text, str):
            return
        if rid not in best:
            order.append(rid)
            best[rid] = text          # first sight must ALWAYS assign; an empty first
        elif len(text) > len(best[rid]):   # snapshot otherwise left the id in `order`
            best[rid] = text               # with no entry in `best` -> KeyError

    def text_of(msg):
        if not isinstance(msg, dict):
            return None
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = [x.get("text", "") for x in c if isinstance(x, dict) and isinstance(x.get("text"), str)]
            reason = [x.get("reasoning", "") for x in c if isinstance(x, dict) and isinstance(x.get("reasoning"), str)]
            return "".join(reason) + "".join(parts)
        return None

    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                ame = ev.get("assistantMessageEvent")
                if isinstance(ame, dict):
                    part = ame.get("partial")
                    t = text_of(part)
                    if t is not None:
                        note((part or {}).get("responseId") or "cur", t)
                msg = ev.get("message")
                t = text_of(msg)
                if t is not None:
                    note((msg or {}).get("responseId") or "cur", t)
    except OSError:
        return []
    return [best[r] for r in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="base,warmstart")
    ap.add_argument("--out", default=str(HERE.parent / "reports" / "loop_scan.json"))
    a = ap.parse_args()

    results = {"detector": {"min_repeats": MIN_REPEATS, "min_chars": MIN_CHARS}, "arms": {}}
    for label in a.labels.split(","):
        label = label.strip()
        rep = OUTD / "reports" / f"pi_{label}_test.json"
        traj_dir = OUTD / "trajectories" / label
        if not rep.exists():
            print(f"  [{label}] no episode record; skipping", flush=True)
            continue
        eps = json.loads(rep.read_text()).get("episodes", [])
        by_traj = {}
        for e in eps:
            if e.get("traj"):
                by_traj[e["traj"]] = e
        looped_timeout = looped_ok = n_timeout = n_ok = 0
        total_chars = 0
        initiators = Counter()
        examples = []
        for rel, e in by_traj.items():
            p = OUTD / rel
            if not p.exists():
                continue
            msgs = assistant_messages(p)
            hit = None
            for mtxt in msgs:                       # a loop lives INSIDE one message
                hit = find_loop(mtxt)
                if hit:
                    break
            total_chars += sum(len(m) for m in msgs)
            timed_out = e.get("pi_exit") == 124
            if timed_out:
                n_timeout += 1
                looped_timeout += bool(hit)
            else:
                n_ok += 1
                looped_ok += bool(hit)
            if hit:
                unit = hit[0].strip()
                initiators[unit[:40]] += 1
                if len(examples) < 5:
                    examples.append({"task": e.get("task_id"), "timed_out": timed_out,
                                     "repeats": hit[1], "span_chars": len(hit[0]),
                                     "span": unit[:160]})
        arm = {"n_trajectories": len(by_traj), "chars_scanned": total_chars,
               "n_timeout": n_timeout, "n_completed": n_ok,
               "loop_rate_timeout": round(looped_timeout / max(1, n_timeout), 3),
               "loop_rate_completed": round(looped_ok / max(1, n_ok), 3),
               "loop_rate_overall": round((looped_timeout + looped_ok) / max(1, len(by_traj)), 3),
               "top_repeating_spans": initiators.most_common(8), "examples": examples}
        results["arms"][label] = arm
        print(f"  [{label}] {total_chars/1e6:.1f}M chars scanned", flush=True)
        print(f"  [{label}] {len(by_traj)} trajectories | timeouts {n_timeout} "
              f"({arm['loop_rate_timeout']:.0%} looping) | completed {n_ok} "
              f"({arm['loop_rate_completed']:.0%} looping) | overall {arm['loop_rate_overall']:.0%}",
              flush=True)
        for ex in examples[:2]:
            print(f"      x{ex['repeats']} ({ex['span_chars']}ch) {ex['span'][:110]!r}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
