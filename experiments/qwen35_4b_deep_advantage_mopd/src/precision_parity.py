"""Pure helpers for the interpretation-only NF4/bf16 parity diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


DIAGNOSTIC_SEED = 42
DIAGNOSTIC_ROUNDS = (0, 1, 2, 3)
PROBE_TARGET_COUNTS = {"deep": 6, "soup": 2}
REPLAY_ABSOLUTE_TOLERANCE = 1e-5
REPLAY_RELATIVE_TOLERANCE = 1e-3
UPDATE_NORM_DEGENERATE_EPSILON = 1e-12


def select_registered_probe_units(
    samples: Sequence[dict[str, Any]],
    unit_ledger: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct the trainer's pre-existing lexicographic 6/2 probe.

    The trainer first fixes the complete consume-once unit set and then sorts
    that set by sample ID inside target buckets.  This helper deliberately
    takes the consumed unit ledger, rather than selecting from every cache
    row, so route-control rows and unconsumed rows cannot enter the diagnostic.
    """

    sample_by_id: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id in sample_by_id:
            raise ValueError(f"duplicate target-cache sample ID: {sample_id}")
        sample_by_id[sample_id] = sample

    ledger_ids = [str(row["sample_id"]) for row in unit_ledger]
    if len(ledger_ids) != len(set(ledger_ids)):
        raise ValueError("training unit ledger is not consume-once unique")
    missing = sorted(set(ledger_ids) - set(sample_by_id))
    if missing:
        raise ValueError(f"training units absent from target cache: {missing[:3]}")

    buckets: dict[str, list[dict[str, Any]]] = {"deep": [], "soup": []}
    for row in sorted(unit_ledger, key=lambda value: str(value["sample_id"])):
        target = str(row["target"])
        if target not in buckets:
            raise ValueError(f"unexpected primary probe target: {target}")
        sample = sample_by_id[str(row["sample_id"])]
        role = str(sample["meta"]["role"])
        expected_role = "capability" if target == "deep" else "anchor"
        if role != expected_role or str(row.get("role")) != expected_role:
            raise ValueError(
                f"probe role/target mismatch for {sample['id']}: "
                f"target={target} sample_role={role} ledger_role={row.get('role')}"
            )
        if (
            int(sample["meta"].get("prompt_tokens_truncated", 0)) != 0
            or int(row.get("prompt_tokens_truncated", 0)) != 0
        ):
            raise ValueError(f"registered probe requires a full prefix: {sample['id']}")
        if target not in sample["targets"]:
            raise ValueError(f"target {target} absent for probe sample {sample['id']}")
        buckets[target].append({"sample": sample, "target": target})

    for target, required in PROBE_TARGET_COUNTS.items():
        if len(buckets[target]) < required:
            raise ValueError(
                f"insufficient {target} units for registered probe: "
                f"{len(buckets[target])} < {required}"
            )
    selected = (
        buckets["deep"][: PROBE_TARGET_COUNTS["deep"]]
        + buckets["soup"][: PROBE_TARGET_COUNTS["soup"]]
    )
    counts = {
        target: sum(unit["target"] == target for unit in selected)
        for target in PROBE_TARGET_COUNTS
    }
    if counts != PROBE_TARGET_COUNTS:
        raise AssertionError(f"registered probe mixture changed: {counts}")
    return selected


def probe_identity_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "sample_id": str(unit["sample"]["id"]),
            "target": str(unit["target"]),
        }
        for unit in units
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def canonical_tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor values in a device- and source-dtype-independent format."""

    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = json.dumps(
        {"dtype": "float32", "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_logit_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _center(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean()


def endpoint_logit_metrics(
    nf4_logits: torch.Tensor,
    bf16_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    top_k: int,
) -> dict[str, float | bool]:
    """Compare one NF4 and bf16 full-vocabulary next-token readout."""

    nf4 = _require_logit_vector("nf4 logits", nf4_logits)
    bf16 = _require_logit_vector("bf16 logits", bf16_logits)
    if nf4.shape != bf16.shape:
        raise ValueError(f"endpoint vocabulary mismatch: {nf4.shape} != {bf16.shape}")
    if not 1 <= int(top_k) <= int(nf4.numel()):
        raise ValueError(f"top_k outside vocabulary: {top_k}")
    teacher = teacher_indices.detach().to(device="cpu", dtype=torch.long)
    if teacher.ndim != 1 or teacher.numel() != int(top_k):
        raise ValueError(
            f"teacher support must contain exactly {top_k} indices, got {teacher.shape}"
        )
    if bool((teacher < 0).any()) or bool((teacher >= nf4.numel()).any()):
        raise ValueError("teacher support index outside vocabulary")

    centered_error = (_center(nf4) - _center(bf16)).abs()
    nf4_log_probs = torch.log_softmax(nf4, dim=-1)
    bf16_log_probs = torch.log_softmax(bf16, dim=-1)
    nf4_probs = nf4_log_probs.exp()
    bf16_probs = bf16_log_probs.exp()
    mixture_log_probs = torch.logaddexp(nf4_log_probs, bf16_log_probs) - math.log(2.0)
    js = 0.5 * (
        (nf4_probs * (nf4_log_probs - mixture_log_probs)).sum()
        + (bf16_probs * (bf16_log_probs - mixture_log_probs)).sum()
    )
    nf4_top = torch.topk(nf4, k=int(top_k)).indices
    bf16_top = torch.topk(bf16, k=int(top_k)).indices
    top_overlap = (
        (nf4_top.unsqueeze(-1) == bf16_top.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean()
    )
    teacher_error = (
        nf4_log_probs.index_select(0, teacher)
        - bf16_log_probs.index_select(0, teacher)
    ).abs()
    return {
        "median_abs_centered_logit_error": float(torch.median(centered_error)),
        "rms_centered_logit_error": float(torch.sqrt(torch.mean(centered_error.square()))),
        "p95_abs_centered_logit_error": float(torch.quantile(centered_error, 0.95)),
        "maximum_abs_centered_logit_error": float(centered_error.max()),
        "total_variation": float(0.5 * (nf4_probs - bf16_probs).abs().sum()),
        "jensen_shannon_divergence_nats": float(js),
        "top1_agreement": bool(int(nf4.argmax()) == int(bf16.argmax())),
        "topk_overlap_fraction": float(top_overlap),
        "teacher_support_mean_abs_logprob_error": float(teacher_error.mean()),
        "teacher_support_maximum_abs_logprob_error": float(teacher_error.max()),
    }


def update_logit_metrics(
    nf4_before: torch.Tensor,
    nf4_after: torch.Tensor,
    bf16_before: torch.Tensor,
    bf16_after: torch.Tensor,
) -> dict[str, float | bool | None]:
    """Compare the softmax-relevant pre/post movement in the two views."""

    values = [
        _require_logit_vector("nf4 before logits", nf4_before),
        _require_logit_vector("nf4 after logits", nf4_after),
        _require_logit_vector("bf16 before logits", bf16_before),
        _require_logit_vector("bf16 after logits", bf16_after),
    ]
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("update parity requires one shared vocabulary shape")
    nf4_delta = _center(values[1] - values[0])
    bf16_delta = _center(values[3] - values[2])
    difference = (nf4_delta - bf16_delta).abs()
    nf4_norm = torch.linalg.vector_norm(nf4_delta)
    bf16_norm = torch.linalg.vector_norm(bf16_delta)
    nf4_degenerate = float(nf4_norm) <= UPDATE_NORM_DEGENERATE_EPSILON
    bf16_degenerate = float(bf16_norm) <= UPDATE_NORM_DEGENERATE_EPSILON
    cosine_defined = not nf4_degenerate and not bf16_degenerate
    ratio_defined = not nf4_degenerate
    cosine = (
        float(torch.dot(nf4_delta, bf16_delta) / (nf4_norm * bf16_norm))
        if cosine_defined
        else None
    )
    ratio = float(bf16_norm / nf4_norm) if ratio_defined else None
    return {
        "nf4_update_l2_norm": float(nf4_norm),
        "bf16_update_l2_norm": float(bf16_norm),
        "nf4_update_degenerate": nf4_degenerate,
        "bf16_update_degenerate": bf16_degenerate,
        "update_cosine_defined": cosine_defined,
        "update_norm_ratio_defined": ratio_defined,
        "bf16_to_nf4_update_norm_ratio": ratio,
        "update_cosine_similarity": cosine,
        "median_abs_centered_update_error": float(torch.median(difference)),
        "rms_centered_update_error": float(torch.sqrt(torch.mean(difference.square()))),
        "p95_abs_centered_update_error": float(torch.quantile(difference, 0.95)),
        "maximum_abs_centered_update_error": float(difference.max()),
    }


def replay_comparison(
    observed: Sequence[float],
    registered: Sequence[float],
    *,
    absolute_tolerance: float = REPLAY_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = REPLAY_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    if len(observed) != len(registered) or not observed:
        raise ValueError("replay and registered vectors must have the same non-zero length")
    rows = []
    for actual, expected in zip(observed, registered):
        actual = float(actual)
        expected = float(expected)
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError("replay comparison contains a non-finite value")
        absolute = abs(actual - expected)
        allowed = float(absolute_tolerance) + float(relative_tolerance) * abs(expected)
        rows.append(
            {
                "observed": actual,
                "registered": expected,
                "absolute_error": absolute,
                "allowed_error": allowed,
                "within_engineering_tolerance": absolute <= allowed,
            }
        )
    return {
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "maximum_absolute_error": max(row["absolute_error"] for row in rows),
        "mean_absolute_error": statistics.mean(row["absolute_error"] for row in rows),
        "passed": all(row["within_engineering_tolerance"] for row in rows),
        "rows": rows,
    }


def position_objective_metrics(
    nf4_before: torch.Tensor,
    nf4_after: torch.Tensor,
    bf16_before: torch.Tensor,
    bf16_after: torch.Tensor,
) -> dict[str, float]:
    """Measure objective parity without allowing position-level cancellation."""

    values = [
        value.detach().to(device="cpu", dtype=torch.float32)
        for value in (nf4_before, nf4_after, bf16_before, bf16_after)
    ]
    if any(value.ndim != 1 for value in values):
        raise ValueError("position-objective inputs must be one-dimensional")
    if len({tuple(value.shape) for value in values}) != 1 or not values[0].numel():
        raise ValueError("position-objective inputs require one shared non-empty shape")
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("position-objective inputs contain non-finite values")
    pre_error = (values[2] - values[0]).abs()
    post_error = (values[3] - values[1]).abs()
    nf4_gain = values[0] - values[1]
    bf16_gain = values[2] - values[3]
    gain_error = (bf16_gain - nf4_gain).abs()
    nf4_zero = nf4_gain == 0.0
    bf16_zero = bf16_gain == 0.0
    return {
        "mean_abs_bf16_nf4_objective_error_before": float(pre_error.mean()),
        "maximum_abs_bf16_nf4_objective_error_before": float(pre_error.max()),
        "mean_abs_bf16_nf4_objective_error_after": float(post_error.mean()),
        "maximum_abs_bf16_nf4_objective_error_after": float(post_error.max()),
        "mean_abs_bf16_nf4_gain_error": float(gain_error.mean()),
        "maximum_abs_bf16_nf4_gain_error": float(gain_error.max()),
        "gain_sign_agreement_fraction": float(
            (torch.sign(nf4_gain) == torch.sign(bf16_gain)).float().mean()
        ),
        "nf4_zero_gain_fraction": float(nf4_zero.float().mean()),
        "bf16_zero_gain_fraction": float(bf16_zero.float().mean()),
    }


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson vectors must have the same length of at least two")
    x = torch.tensor(list(left), dtype=torch.float64)
    y = torch.tensor(list(right), dtype=torch.float64)
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("Pearson vectors contain non-finite values")
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(x, y) / denominator)


def summarize_objective_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty objective cohort")
    views = ("nf4_before", "nf4_after", "bf16_before", "bf16_after")
    values = {
        view: [float(row["objective"][view]) for row in rows]
        for view in views
    }
    if not all(math.isfinite(value) for series in values.values() for value in series):
        raise ValueError("objective cohort contains non-finite values")
    nf4_gains = [
        before - after
        for before, after in zip(values["nf4_before"], values["nf4_after"])
    ]
    bf16_gains = [
        before - after
        for before, after in zip(values["bf16_before"], values["bf16_after"])
    ]
    gain_errors = [bf16 - nf4 for bf16, nf4 in zip(bf16_gains, nf4_gains)]
    gain_sign_agreements = [
        (nf4 > 0.0) - (nf4 < 0.0) == (bf16 > 0.0) - (bf16 < 0.0)
        for nf4, bf16 in zip(nf4_gains, bf16_gains)
    ]
    pre_gaps = [
        bf16 - nf4
        for bf16, nf4 in zip(values["bf16_before"], values["nf4_before"])
    ]
    post_gaps = [
        bf16 - nf4
        for bf16, nf4 in zip(values["bf16_after"], values["nf4_after"])
    ]
    return {
        "unit_count": len(rows),
        "equal_unit_macro_mean": {
            view: statistics.mean(series) for view, series in values.items()
        },
        "mean_bf16_minus_nf4_objective_before": statistics.mean(pre_gaps),
        "mean_bf16_minus_nf4_objective_after": statistics.mean(post_gaps),
        "mean_abs_bf16_nf4_objective_gap_before": statistics.mean(map(abs, pre_gaps)),
        "mean_abs_bf16_nf4_objective_gap_after": statistics.mean(map(abs, post_gaps)),
        "mean_nf4_objective_gain": statistics.mean(nf4_gains),
        "mean_bf16_objective_gain": statistics.mean(bf16_gains),
        "mean_bf16_minus_nf4_gain": statistics.mean(gain_errors),
        "mean_abs_bf16_nf4_gain_error": statistics.mean(map(abs, gain_errors)),
        "maximum_abs_bf16_nf4_gain_error": max(map(abs, gain_errors)),
        "gain_sign_agreement_count": sum(gain_sign_agreements),
        "gain_sign_agreement_fraction": statistics.mean(gain_sign_agreements),
        "nf4_zero_gain_count": sum(value == 0.0 for value in nf4_gains),
        "nf4_zero_gain_fraction": statistics.mean(value == 0.0 for value in nf4_gains),
        "bf16_zero_gain_count": sum(value == 0.0 for value in bf16_gains),
        "bf16_zero_gain_fraction": statistics.mean(value == 0.0 for value in bf16_gains),
        "gain_pearson_correlation": pearson_correlation(nf4_gains, bf16_gains),
    }


def mean_numeric_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    boolean_keys: Iterable[str] = (),
    nullable_keys: Iterable[str] = (),
) -> dict[str, float | None]:
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate an empty metric collection")
    bools = set(boolean_keys)
    nullable = set(nullable_keys)
    if bools & nullable:
        raise ValueError("metric keys cannot be both boolean and nullable")
    keys = set(values[0])
    if any(set(row) != keys for row in values):
        raise ValueError("metric rows have inconsistent keys")
    result: dict[str, float | None] = {}
    for key in sorted(keys):
        series = [row[key] for row in values]
        if key in bools:
            result[f"{key}_fraction"] = statistics.mean(bool(value) for value in series)
            continue
        if key in nullable:
            defined = [float(value) for value in series if value is not None]
            if not all(math.isfinite(value) for value in defined):
                raise ValueError(f"metric {key} contains a non-finite value")
            result[f"{key}_defined_fraction"] = len(defined) / len(series)
            result[f"mean_{key}"] = statistics.mean(defined) if defined else None
            result[f"maximum_{key}"] = max(defined) if defined else None
            continue
        numeric = [float(value) for value in series]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"metric {key} contains non-finite values")
        result[f"mean_{key}"] = statistics.mean(numeric)
        result[f"maximum_{key}"] = max(numeric)
    return result
