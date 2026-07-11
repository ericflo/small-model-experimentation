"""Canonical MOPD equation (5) with an auditable teacher-top-k correction."""

from __future__ import annotations

import torch


def bias_corrected_topk_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Evaluate MOPD's corrected reverse KL on the teacher's top-k set.

    For every selected token ``v``, equation (5) is

    ``p_s(v) * log(p_s(v) / p_t(v)) - p_s(v) + p_t(v)``.

    The teacher distribution is treated as frozen. Both normalizers are over
    the complete vocabulary; no renormalized top-k or lumped tail distribution
    is introduced.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {student_logits.shape} != {teacher_logits.shape}"
        )
    if student_logits.ndim < 1:
        raise ValueError("logits must have a vocabulary dimension")
    vocabulary = int(student_logits.shape[-1])
    if top_k < 1 or top_k > vocabulary:
        raise ValueError(f"top_k must be in [1, {vocabulary}], got {top_k}")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be one of: none, mean, sum")

    teacher_logits = teacher_logits.detach()
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    indices = torch.topk(teacher_log_probs, k=top_k, dim=-1).indices
    student_selected_log = student_log_probs.gather(-1, indices)
    teacher_selected_log = teacher_log_probs.gather(-1, indices)
    student_selected = student_selected_log.exp()
    teacher_selected = teacher_selected_log.exp()
    per_position = (
        student_selected * (student_selected_log - teacher_selected_log)
        - student_selected
        + teacher_selected
    ).sum(dim=-1)
    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    return per_position.mean()


def full_reverse_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Full-vocabulary reference used only by the synthetic correctness gate."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher shape mismatch")
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.detach().float(), dim=-1)
    per_position = (
        student_log_probs.exp() * (student_log_probs - teacher_log_probs)
    ).sum(dim=-1)
    if reduction == "none":
        return per_position
    if reduction == "sum":
        return per_position.sum()
    if reduction == "mean":
        return per_position.mean()
    raise ValueError("reduction must be one of: none, mean, sum")
