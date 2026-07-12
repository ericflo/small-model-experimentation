#!/usr/bin/env python3
"""Derive the compact terminal metrics from the committed G0 result rows."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "analysis" / "metrics.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selected(rows: list[dict], **matches) -> list[dict]:
    return [row for row in rows if all(row[key] == value for key, value in matches.items())]


def target_rate(rows: list[dict]) -> float:
    assert rows
    return fmean(float(row["target_selected"]) for row in rows)


def mean(rows: list[dict], key: str) -> float:
    assert rows
    return fmean(float(row[key]) for row in rows)


def main() -> None:
    result = read_json(RUNS / "positive_control.json")
    rows = read_jsonl(RUNS / "positive_control_rows.jsonl")
    alpha = float(result["selection"]["alpha"])
    layers = sorted(int(layer) for layer in result["confirmation_by_layer"])

    def confirmation(kind: str, condition: str, layer: int) -> list[dict]:
        return selected(
            rows,
            split_half="confirmation",
            prompt_kind=kind,
            condition=condition,
            layers=[layer],
            alpha=alpha,
        )

    layer24 = {
        (kind, condition): confirmation(kind, condition, 24)
        for kind in ("direct", "consequence")
        for condition in ("j", "random", "logit")
    }
    alpha_margins = {}
    for candidate_alpha in (0.5, 1.0, 2.0, 4.0):
        alpha_margins[str(candidate_alpha)] = {
            kind: mean(
                selected(
                    rows,
                    split_half="selection",
                    prompt_kind=kind,
                    condition="j",
                    layers=[24],
                    alpha=candidate_alpha,
                ),
                "target_minus_source_logit",
            )
            for kind in ("direct", "consequence")
        }

    metrics = {
        "schema_version": 1,
        "decision": result["decision"],
        "confirmation_n": result["counts"]["confirmation_items"],
        "clean_accuracy": {
            kind: result["baseline_confirmation"][kind]["source_accuracy"]
            for kind in ("direct", "consequence")
        },
        "direct_target_rate_by_layer": {
            str(layer): result["confirmation_by_layer"][str(layer)]["direct"]["j"]["target_rate"]
            for layer in layers
        },
        "consequence_target_rate_by_layer": {
            str(layer): result["confirmation_by_layer"][str(layer)]["consequence"]["j"]["target_rate"]
            for layer in layers
        },
        "layer24_controls": {
            "direct_random_target_rate": target_rate(layer24[("direct", "random")]),
            "direct_logit_lens_target_rate": target_rate(layer24[("direct", "logit")]),
            "consequence_random_target_rate": target_rate(layer24[("consequence", "random")]),
            "direct_j_mean_delta_norm": mean(layer24[("direct", "j")], "delta_norm"),
            "direct_random_mean_delta_norm": mean(layer24[("direct", "random")], "delta_norm"),
            "consequence_j_mean_delta_norm": mean(layer24[("consequence", "j")], "delta_norm"),
            "consequence_random_mean_delta_norm": mean(layer24[("consequence", "random")], "delta_norm"),
        },
        "selection_layer24_margin_by_alpha": alpha_margins,
        "g1_run": False,
        "g2_run": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} from {len(rows):,} rows")


if __name__ == "__main__":
    main()
