#!/usr/bin/env python3
"""Apply the frozen repair-verifier 2AFC probe readout (no promotion).

Lifecycle 21, eval-only FEASIBILITY GATE for a possible on-policy training
charter. The menders-shaped eliminative-repair skill is closed to SFT at
every tested dose (three pedagogies, 80-800 rows; the dose-scale candidate
scored exactly the untrained controls' guess floor). On-policy training
(learn from own attempts + live feedback) is the one remaining mechanism
class, and it only has signal if the model can VERIFY repairs it cannot
propose (C29: read-only 2AFC verifier 0.81 while generation collapsed;
C47: think-judges rescue substrate-scoped no-think judges). This probe
measures that verification signal directly.

ONE evaluated model — the hygiene_explore composite (tree 9eb653d7…),
tree-recomputed at the boundary. No training, no benchmark seed, no
promotion, no aggregate seed. TWO arms, same model, one decode config
each (a within-model contrast): ``think`` (natural thinking, 1,024-token
cap; C44/C47 say give the model CoT) and ``nothink`` (thinking suppressed
via the runner's ``--thinking off`` channel, exactly how predecessors'
no-think evals suppress it). 200 frozen 2AFC items per arm = 400
judgments.

THE ITEM IS FULLY SYMMETRIC (no repair history, no attempt narration):
machine spec with its legality clauses + the original written sequence +
BOTH trials' setups, wanted outcomes, and the OBSERVED outcomes of the
original (broken) sequence on both trials — pure failure evidence — then
two candidate single-step changes labeled A and B in identical
grammatical form with no provenance markers. One is the unique legal fix
that makes BOTH trials come out as wanted; the other is a legal fix
consistent with trial-one evidence only. Deciding requires SIMULATING
each candidate against both trials — exactly the execution-based
self-check an on-policy loop would use as its reward signal. The measured
signal is EXECUTION-BASED FIX VERIFICATION.

PREREGISTERED READINGS, all from the per-arm graded rows:

(a) ``2afc_accuracy`` per arm with the exact (Clopper-Pearson) two-sided
    95% binomial confidence interval, read against the 0.5 chance floor;
(b) per-formalism accuracy (25 items per formalism) per arm, descriptive;
(c) position-bias check: accuracy on A-correct versus B-correct items
    (100 each by construction), descriptive;
(d) cap-contact diagnostic (preregistered): the think arm's cap-contact
    rate at the 1,024-token cap. If it exceeds ``CAP_SCOPE_THRESHOLD``
    (0.20), any SIGNAL_ABSENT reading is SCOPED as possibly
    budget-limited (recorded in the consequence; it never creates a
    third verdict state);
(e) the frozen ORDERED, TOTAL consequence partition on the think arm
    (no third state; the nothink arm is descriptive — a C47
    substrate-scoping check):
    - think accuracy >= 0.65 AND the 95% CI excludes 0.5 ->
      ``SIGNAL_PRESENT``;
    - think accuracy < 0.65 OR the CI includes 0.5 -> ``SIGNAL_ABSENT``.

The receipt records everything and the process exits 0 on ANY complete
event: there is no seed to open and no promotion to grant or refuse.

ANSWER NORMALIZATION (frozen grading rule, byte-identical to the
retention-calibration and dose-scale cells' ``normalize_answer`` and
applied identically to both arms by the evaluator): both the parsed and
the expected answer pass through ``normalize_answer`` before comparison —
collapse runs of whitespace to a single space, strip, then remove any
spaces immediately adjacent to '>' or ';'. The expected answer is always
the single letter ``A`` or ``B``; grading is exact match after
normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


CONSTRUCTION_SEED = 77160
PROBE_ROWS = 200
PER_FORMALISM = 25
FORMALISMS = (
    "troughline",
    "trinketcord",
    "crankwheel",
    "sigilslate",
    "barrowyoke",
    "balesled",
    "millround",
    "skeinreel",
)
PROBE_KIND = "u_verify2afc"
POSITIONS = ("A", "B")
ROWS_PER_POSITION = 100
# Frozen run order: the think arm first, then the nothink arm.
ARMS = ("think", "nothink")
GATING_ARM = "think"
ARM_THINKING = {"think": "natural", "nothink": "off"}
MAX_TOKENS = 1024
CHANCE_FLOOR = 0.5
SIGNAL_MIN_ACCURACY = 0.65
CONFIDENCE = 0.95
OUTCOME = "PROBE_READ_COMPLETE"
VERDICTS = ("SIGNAL_PRESENT", "SIGNAL_ABSENT")
# The frozen consequence statements: an ordered, total, two-state
# partition on the think arm. No third state exists. The measured signal
# is EXECUTION-BASED FIX VERIFICATION (simulate each candidate against
# both trials), the self-check an on-policy loop would use as reward.
CONSEQUENCES = {
    "SIGNAL_PRESENT": (
        "SIGNAL_PRESENT: the on-policy episode charter is fundable — "
        "execution-based fix-verification signal exists for the skill "
        "generation cannot produce (C29-class dissociation): the model "
        "can simulate candidate repairs against two-trial evidence above "
        "the frozen bar"
    ),
    "SIGNAL_ABSENT": (
        "SIGNAL_ABSENT: the eliminative-repair skill lacks even "
        "execution-based fix-verification signal at this instrument; the "
        "on-policy class closes for menders and the program map is "
        "complete at demonstrated-not-confirmed"
    ),
}
# Preregistered cap-contact scoping for the consequence: simulating two
# candidates over two trials each may press the 1,024-token think cap. A
# SIGNAL_ABSENT reading with think-arm cap contacts on more than 20% of
# items is scoped as possibly budget-limited. The scope annotates the
# frozen consequence; it never creates a third verdict state.
CAP_SCOPE_THRESHOLD = 0.20
CAP_SCOPE_NOTE = (
    "SCOPED: think-arm cap contacts exceed 20% of items, so this "
    "SIGNAL_ABSENT reading is possibly budget-limited at the 1,024-token "
    "cap — it closes the instrument reading, not the question; a "
    "bigger-budget re-probe is the permitted follow-up"
)
CAP_NO_SCOPE_NOTE = (
    "cap contacts within budget (<= 20% of items) or verdict "
    "SIGNAL_PRESENT; no budget scoping applies"
)
ANSWER_NORMALIZATION = {
    "function": "check_local.normalize_answer",
    "definition": (
        "re.sub(r'\\s+', ' ', s).strip(), then re.sub(r'\\s*>\\s*', '>', s), "
        "then re.sub(r'\\s*;\\s*', ';', s)"
    ),
    "applied_to": (
        "both the parsed and the expected answer, both arms, all 200 items"
    ),
    "rationale": (
        "byte-identical to the retention-calibration and dose-scale cells' "
        "frozen rule (unchanged since the seed-88021 gate); the expected "
        "answer here is always a single letter, so normalization only "
        "forgives whitespace"
    ),
    "prospective": True,
}


def normalize_answer(value: str) -> str:
    """Frozen grading normalization; see ANSWER_NORMALIZATION."""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*>\s*", ">", value)
    value = re.sub(r"\s*;\s*", ";", value)
    return value


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Exact P(X <= k) for X ~ Binomial(n, p); math.comb keeps it exact-ish."""
    if not 0 <= p <= 1 or n < 0:
        raise ValueError(f"binomial parameters out of range: n={n}, p={p}")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(total, 1.0)


def _bisect(function, target: float, *, increasing: bool) -> float:
    """Solve function(p) = target for p in [0, 1] by deterministic bisection."""
    low, high = 0.0, 1.0
    for _ in range(200):
        mid = (low + high) / 2
        value = function(mid)
        if (value < target) == increasing:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def clopper_pearson_interval(
    k: int, n: int, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Exact two-sided binomial CI (Clopper-Pearson) via tail bisection."""
    if not 0 <= k <= n or n < 1 or not 0 < confidence < 1:
        raise ValueError(f"interval parameters out of range: k={k}, n={n}")
    alpha = 1.0 - confidence
    if k == 0:
        low = 0.0
    else:
        # P(X >= k | p) = 1 - cdf(k-1, n, p) is increasing in p.
        low = _bisect(
            lambda p: 1.0 - binomial_cdf(k - 1, n, p), alpha / 2, increasing=True
        )
    if k == n:
        high = 1.0
    else:
        # P(X <= k | p) = cdf(k, n, p) is decreasing in p.
        high = _bisect(
            lambda p: binomial_cdf(k, n, p), alpha / 2, increasing=False
        )
    if low > high:
        raise ValueError(f"degenerate interval for k={k}, n={n}")
    return low, high


def ci_excludes_chance(low: float, high: float) -> bool:
    return not (low <= CHANCE_FLOOR <= high)


def signal_verdict(accuracy: float, ci_low: float, ci_high: float) -> str:
    """The frozen ordered TOTAL two-state partition (no third state)."""
    if not (
        math.isfinite(accuracy)
        and math.isfinite(ci_low)
        and math.isfinite(ci_high)
    ):
        raise ValueError("verdict inputs must be finite")
    if accuracy >= SIGNAL_MIN_ACCURACY and ci_excludes_chance(ci_low, ci_high):
        return "SIGNAL_PRESENT"
    return "SIGNAL_ABSENT"


def selected_rows(payload: dict, arm: str) -> list[dict]:
    rows = [row for row in payload.get("rows", []) if row.get("arm") == arm]
    if arm not in ARMS or len(rows) != PROBE_ROWS:
        raise ValueError(f"local receipt does not contain {PROBE_ROWS} rows for {arm}")
    return rows


def validate_receipt_layout(payload: dict) -> None:
    rows = payload.get("rows")
    expected_total = PROBE_ROWS * len(ARMS)
    if not isinstance(rows, list) or len(rows) != expected_total:
        raise ValueError(
            f"local receipt must contain exactly {expected_total} graded rows"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("local receipt rows must be JSON objects")
    if {row.get("arm") for row in rows} != set(ARMS):
        raise ValueError("local receipt arm set changed")
    prefix = f"probe{CONSTRUCTION_SEED}_"
    reference: dict[str, tuple[str, str]] | None = None
    for arm in ARMS:
        selected = selected_rows(payload, arm)
        task_ids = [row.get("task_id") for row in selected]
        if (
            any(
                not isinstance(task_id, str) or not task_id.startswith(prefix)
                for task_id in task_ids
            )
            or len(set(task_ids)) != PROBE_ROWS
            or any(row.get("kind") != PROBE_KIND for row in selected)
            or any(row.get("surface") not in FORMALISMS for row in selected)
            or any(row.get("expected") not in POSITIONS for row in selected)
            or any(type(row.get("correct")) is not bool for row in selected)
            or any(type(row.get("cap_contact")) is not bool for row in selected)
            or any(
                row.get("parsed") is None and row.get("correct")
                for row in selected
            )
        ):
            raise ValueError(f"local receipt row schema changed for {arm}")
        surface_counts = {
            formalism: sum(row["surface"] == formalism for row in selected)
            for formalism in FORMALISMS
        }
        if any(count != PER_FORMALISM for count in surface_counts.values()):
            raise ValueError(f"local receipt formalism balance changed for {arm}")
        expected_counts = {
            position: sum(row["expected"] == position for row in selected)
            for position in POSITIONS
        }
        if any(count != ROWS_PER_POSITION for count in expected_counts.values()):
            raise ValueError(f"local receipt position balance changed for {arm}")
        mapping = {
            row["task_id"]: (row["expected"], row["surface"]) for row in selected
        }
        if reference is None:
            reference = mapping
        elif mapping != reference:
            raise ValueError("local arms disagree on tasks, keys, or surfaces")


def arm_summary(rows: list[dict]) -> dict:
    correct = sum(bool(row["correct"]) for row in rows)
    n = len(rows)
    low, high = clopper_pearson_interval(correct, n)
    per_formalism = {}
    for formalism in FORMALISMS:
        formalism_rows = [row for row in rows if row["surface"] == formalism]
        per_formalism[formalism] = {
            "n": len(formalism_rows),
            "correct": sum(bool(row["correct"]) for row in formalism_rows),
            "accuracy": sum(bool(row["correct"]) for row in formalism_rows)
            / len(formalism_rows),
        }
    position_bias = {}
    for position in POSITIONS:
        position_rows = [row for row in rows if row["expected"] == position]
        position_bias[f"{position.lower()}_correct_items"] = {
            "n": len(position_rows),
            "correct": sum(bool(row["correct"]) for row in position_rows),
            "accuracy": sum(bool(row["correct"]) for row in position_rows)
            / len(position_rows),
        }
    position_bias["accuracy_gap_a_minus_b"] = (
        position_bias["a_correct_items"]["accuracy"]
        - position_bias["b_correct_items"]["accuracy"]
    )
    cap_contacts = sum(bool(row.get("cap_contact")) for row in rows)
    return {
        "rows": n,
        "correct": correct,
        "2afc_accuracy": correct / n,
        "ci95_exact": [low, high],
        "ci95_excludes_chance": ci_excludes_chance(low, high),
        "chance_floor": CHANCE_FLOOR,
        "parsed": sum(row.get("parsed") is not None for row in rows),
        "cap_contacts": cap_contacts,
        "cap_contact_rate": cap_contacts / n,
        "mean_sampled_tokens": sum(row["n_sampled_tokens"] for row in rows) / n,
        "per_formalism": per_formalism,
        "position_bias": position_bias,
    }


def evaluate_probe(payload: dict) -> dict:
    if (
        payload.get("seed") != CONSTRUCTION_SEED
        or payload.get("rows_per_arm") != PROBE_ROWS
    ):
        raise ValueError("local receipt seed or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt arm order changed")
    validate_receipt_layout(payload)
    arm_readings = {arm: arm_summary(selected_rows(payload, arm)) for arm in ARMS}
    think = arm_readings[GATING_ARM]
    verdict = signal_verdict(
        think["2afc_accuracy"], think["ci95_exact"][0], think["ci95_exact"][1]
    )
    if verdict not in VERDICTS:
        raise ValueError(f"verdict left the frozen space: {verdict}")
    scope_applies = (
        verdict == "SIGNAL_ABSENT"
        and think["cap_contact_rate"] > CAP_SCOPE_THRESHOLD
    )
    return {
        "schema_version": 1,
        "seed": CONSTRUCTION_SEED,
        "feasibility_probe": True,
        "evaluated_model": "hygiene_explore composite (tree 9eb653d7...)",
        "arms": list(ARMS),
        "gating_arm": GATING_ARM,
        "rows_per_arm": PROBE_ROWS,
        "answer_normalization": ANSWER_NORMALIZATION,
        "readings": {
            "per_arm": arm_readings,
            "consequence": {
                "verdict": verdict,
                "statement": CONSEQUENCES[verdict],
                "think_accuracy": think["2afc_accuracy"],
                "think_correct": think["correct"],
                "think_ci95": think["ci95_exact"],
                "cap_contact_diagnostic": {
                    "think_cap_contacts": think["cap_contacts"],
                    "think_cap_contact_rate": think["cap_contact_rate"],
                    "scope_threshold": CAP_SCOPE_THRESHOLD,
                    "budget_limited_scope_applies": scope_applies,
                    "note": CAP_SCOPE_NOTE if scope_applies else CAP_NO_SCOPE_NOTE,
                },
                "ordered_total_no_third_state": True,
            },
            "nothink_descriptive": {
                "gating": False,
                "role": (
                    "C47 substrate-scoping check: whether the no-think judge "
                    "seat holds on this substrate is reported, never gated"
                ),
                "nothink_accuracy": arm_readings["nothink"]["2afc_accuracy"],
                "think_minus_nothink": (
                    think["2afc_accuracy"]
                    - arm_readings["nothink"]["2afc_accuracy"]
                ),
            },
        },
        "thresholds": {
            "signal_min_accuracy": SIGNAL_MIN_ACCURACY,
            "chance_floor": CHANCE_FLOOR,
            "confidence": CONFIDENCE,
            "cap_scope_threshold": CAP_SCOPE_THRESHOLD,
            "rule": (
                "SIGNAL_PRESENT iff think-arm 2afc_accuracy >= 0.65 AND the "
                "exact two-sided 95% binomial CI excludes 0.5; otherwise "
                "SIGNAL_ABSENT. Ordered, total, no third state; the nothink "
                "arm is descriptive. A SIGNAL_ABSENT reading with think-arm "
                "cap contacts on more than 20% of items is annotated as "
                "possibly budget-limited at the 1,024-token cap"
            ),
        },
        "consequence_statements": dict(CONSEQUENCES),
        "no_promotion_in_feasibility_probe": True,
        "outcome": OUTCOME,
        "eligible": [],
        "promoted": None,
    }


def finalize_probe(
    result: dict, receipt_path: Path, raw: bytes, design_receipt: Path | None = None
) -> dict:
    """Shared writer fields: keeps the eval-internal and recovery receipts
    schema-identical; the smoke harness hard-requires these fields."""
    exp = Path(__file__).resolve().parents[1]
    if design_receipt is None:
        design_receipt = exp / "data" / "local_design_receipt.json"
    finalized = dict(result)
    finalized.update({
        "experiment_id": exp.name,
        "local_receipt": str(receipt_path.resolve()),
        "local_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "design_receipt_sha256": hashlib.sha256(
            design_receipt.read_bytes()
        ).hexdigest(),
        "backend": "vllm_merged_composite",
        "aggregate_seed": None,
        "aggregate_seed_open": False,
        "benchmark_data_read": False,
    })
    return finalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        raw = args.receipt.read_bytes()
        result = evaluate_probe(json.loads(raw.decode("utf-8")))
        result = finalize_probe(result, args.receipt, raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite the probe readout receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # A complete probe event always exits 0: there is no seed to open and no
    # promotion to grant; the outputs live in the receipt.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
