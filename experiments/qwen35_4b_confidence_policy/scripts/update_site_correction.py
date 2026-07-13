#!/usr/bin/env python3
"""Rewrite the confidence_policy brief + viz for the corrected (powered-up) result."""
import json
from pathlib import Path

EID = "qwen35_4b_confidence_policy"

brief = {
    "verdict_tag": "Verifier-free SELECT + ABSTAIN work; selectively escalating hard tasks to more think budget is a null result at n=400",
    "concept_primer": (
        "Picture the model answering a coding problem several times. A cheap read of its own "
        "confidence (one logit: 'is this right?') picks the best try and flags the ones it is "
        "unsure about. A tempting extra step is to let the model think LONGER on the flagged-hard "
        "problems instead of drawing more tries. This experiment tests all three moves at matched "
        "compute — and powers up the shakiest one until the answer is trustworthy."
    ),
    "plain_question": (
        "Can the model's own confidence be turned into one deployable policy — pick the best "
        "answer, abstain when unsure, and spend extra compute wisely — with no test-execution or "
        "grader?"
    ),
    "plain_answer": (
        "Partly. Picking the answer with the highest single-token P(True) confidence is the best "
        "grader-free selector at every compute level, and abstaining below a confidence threshold "
        "trades coverage for accuracy cleanly — both robust. But the appealing third move — "
        "spending compute on more THINKING for the flagged-hard tasks rather than more samples — "
        "showed no benefit once powered up to 400 tasks: an early small-sample win did not "
        "replicate (every 95% interval now includes zero). Thinking longer helps a little across "
        "the board, just not selectively on the hard tail — because these coding tasks nearly "
        "saturate the model's thinking budget already."
    ),
    "why_it_matters": (
        "It gives a shippable, verifier-free recipe (select + abstain from one logit) AND a "
        "methodological lesson: a pre-registered, under-powered subset win was flagged as shaky "
        "and then REVERSED on the larger sample — power up before you claim. No grader, no bigger "
        "model, just the fixed 4B's own logits."
    ),
    "verdict_tone": "neutral",
    "key_numbers": [
        {"label": "P(True)-select accuracy @k=9", "value": "0.762",
         "sub": "best verifier-free selector; majority-vote 0.742, mean-logprob 0.725, greedy 0.701"},
        {"label": "Abstention: accuracy at 68% coverage", "value": "0.866",
         "sub": "up from 0.757 at full coverage; solvability AUROC 0.72"},
        {"label": "Escalate vs breadth (n=400)", "value": "CIs span 0",
         "sub": "esc-minus-breadth +0.004..+0.022 across abstain fractions — a null; the n=24-60 win did not replicate"},
        {"label": "MBPP tasks powered up to", "value": "400",
         "sub": "regenerated on vLLM (~10x faster than the initial HF pass) with bootstrap CIs"},
    ],
    "charts": [
        {"index": 0,
         "chart_plain_title": "P(True) confidence-select is the best verifier-free selector",
         "chart_read": "Bars are MBPP accuracy at nine candidates for each selection rule. Single-token P(True)-select leads the verifier-free rules; the execution line and oracle need a grader."},
        {"index": 1,
         "chart_plain_title": "Escalating hard tasks to more think budget: a null result at n=400",
         "chart_read": "Each group is the hardest X% of tasks by confidence. Escalating them to a bigger think budget (left) barely differs from spending the same compute on more samples (right) — every 95% interval on the difference includes zero."},
    ],
}

sel = {"kind": "bar",
       "headline": True,
       "title": "P(True) confidence-select is the best verifier-free selector (MBPP, k=9)",
       "categories": ["greedy", "majority-vote", "mean-logprob", "P(True)-select", "execution-line", "oracle"],
       "series": [{"label": "MBPP accuracy at 9 candidates",
                   "values": [0.7008, 0.7418, 0.7254, 0.7623, 0.8402, 0.8484]}],
       "x_label": "selection rule", "y_label": "accuracy", "y_format": "decimal",
       "note": "greedy/majority/mean-logprob/P(True) are verifier-free; execution-line and oracle require a grader.",
       "source": "experiments/qwen35_4b_confidence_policy/runs/frontier.json"}

esc = {"kind": "bar",
       "title": "Escalate vs breadth on the abstained tail: null at n=400 (matched compute)",
       "categories": ["hardest 20%", "hardest 30%", "hardest 40%", "hardest 50%"],
       "series": [
           {"label": "escalate to budget 2048", "values": [0.289, 0.306, 0.357, 0.406]},
           {"label": "more samples @ 256 (matched compute)", "values": [0.268, 0.302, 0.34, 0.4]},
       ],
       "x_label": "abstained (low-confidence) task subset",
       "y_label": "MBPP accuracy on the subset", "y_format": "decimal",
       "note": "Powered up to n=80-200 per subset; every esc-minus-breadth 95% bootstrap CI spans 0.",
       "source": "experiments/qwen35_4b_confidence_policy/runs/escalation.json"}


def main():
    bp = Path("knowledge/experiment_brief.json")
    b = json.loads(bp.read_text())
    b["experiments"][EID] = brief
    bp.write_text(json.dumps(b, indent=1, ensure_ascii=False) + "\n")
    vp = Path("knowledge/experiment_viz.json")
    v = json.loads(vp.read_text())
    v["experiments"][EID] = {"charts": [sel, esc]}
    vp.write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n")
    print("brief + viz corrected for", EID)


if __name__ == "__main__":
    main()
