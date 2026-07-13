#!/usr/bin/env python3
"""Add the confidence_policy experiment's brief + native chart specs. Idempotent."""
import json
from pathlib import Path

EID = "qwen35_4b_confidence_policy"
BRIEF = Path("knowledge/experiment_brief.json")
VIZ = Path("knowledge/experiment_viz.json")

brief = {
    "verdict_tag": "A verifier-free select+abstain+escalate policy: for hard tasks, more think budget beats more samples",
    "concept_primer": (
        "Picture the model answering a coding problem several times. A cheap read of its own "
        "confidence (one logit: 'is this right?') picks the best try and flags the ones it is "
        "unsure about. For those flagged-hard problems you can either draw more tries, or let the "
        "model think longer on the same problem. This experiment measures which is the better use "
        "of the same compute."
    ),
    "plain_question": (
        "Can the model's own confidence be turned into one deployable policy — pick the best answer, "
        "abstain when unsure, and spend extra compute wisely — without any test-execution or grader?"
    ),
    "plain_answer": (
        "Yes. Picking the answer with the highest single-token P(True) confidence is the best "
        "grader-free selector at every compute level, abstaining below a confidence threshold trades "
        "coverage for accuracy cleanly, and for the flagged-hard tasks giving the model a bigger "
        "thinking budget beats drawing more samples at matched compute — at every abstain fraction."
    ),
    "why_it_matters": (
        "It fuses two separate research threads into one shippable recipe: the model's confidence "
        "readout says WHICH tasks are hard, and more serial thinking (not more sampling) is the right "
        "way to spend compute on them. No verifier, no bigger model — just the fixed 4B and its own logits."
    ),
    "verdict_tone": "positive",
    "key_numbers": [
        {"label": "P(True)-select accuracy @k=9", "value": "0.762",
         "sub": "best verifier-free selector; majority-vote 0.742, mean-logprob 0.725, greedy 0.701"},
        {"label": "Abstention: accuracy at 68% coverage", "value": "0.866",
         "sub": "up from 0.757 at full coverage — a clean risk-coverage trade"},
        {"label": "Hardest-20% tasks: escalate vs breadth", "value": "0.458 vs 0.304",
         "sub": "re-solving at a higher think budget beats matched-compute extra samples (base 0.250)"},
        {"label": "MBPP tasks x candidates", "value": "244 / 120",
         "sub": "cached pool for selection; fresh two-budget pool for escalation"},
    ],
    "charts": [
        {"index": 0,
         "chart_plain_title": "For hard tasks, more think budget beats more samples",
         "chart_read": "Each group is the hardest X% of tasks by confidence. Escalating them to a bigger think budget (middle bar) beats spending the same compute on more samples (right bar) and the base policy (left bar), at every fraction."},
        {"index": 1,
         "chart_plain_title": "P(True) confidence-select is the best verifier-free selector",
         "chart_read": "Bars are MBPP accuracy at nine candidates for each selection rule. Single-token P(True)-select leads the verifier-free rules; the execution line and oracle need a grader."},
    ],
}

viz = {"charts": [
    {"headline": True, "kind": "bar",
     "title": "Escalating hard tasks to more think budget beats more samples (matched compute)",
     "categories": ["hardest 20%", "hardest 30%", "hardest 40%", "hardest 50%"],
     "series": [
         {"label": "base (2 samples @ budget 256)", "values": [0.25, 0.361, 0.354, 0.4]},
         {"label": "escalate to budget 2048", "values": [0.458, 0.5, 0.479, 0.467]},
         {"label": "more samples @ 256 (matched compute)", "values": [0.304, 0.417, 0.416, 0.423]},
     ],
     "x_label": "abstained (low-confidence) task subset",
     "y_label": "MBPP accuracy on the subset", "y_format": "decimal",
     "note": "Verifier-free confidence-select within each arm; token-matched compute. n=24-60 per subset.",
     "source": "experiments/qwen35_4b_confidence_policy/runs/escalation.json"},
    {"kind": "bar",
     "title": "P(True) confidence-select is the best verifier-free selector (MBPP, k=9)",
     "categories": ["greedy", "majority-vote", "mean-logprob", "P(True)-select", "execution-line", "oracle"],
     "series": [{"label": "MBPP accuracy at 9 candidates",
                 "values": [0.7008, 0.7418, 0.7254, 0.7623, 0.8402, 0.8484]}],
     "x_label": "selection rule", "y_label": "accuracy", "y_format": "decimal",
     "note": "greedy/majority/mean-logprob/P(True) are verifier-free; execution-line and oracle require a grader.",
     "source": "experiments/qwen35_4b_confidence_policy/runs/frontier.json"},
]}


def main():
    b = json.loads(BRIEF.read_text())
    b["experiments"][EID] = brief
    BRIEF.write_text(json.dumps(b, indent=1, ensure_ascii=False) + "\n")
    v = json.loads(VIZ.read_text())
    v["experiments"][EID] = viz
    VIZ.write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n")
    print("added brief + viz for", EID)


if __name__ == "__main__":
    main()
