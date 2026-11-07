"""Forward KL to the teacher prior, plus masked entropy."""

from __future__ import annotations

import torch


def prior_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over legal actions; zero at matching distributions."""
    teacher = torch.log_softmax(teacher_logits.masked_fill(~mask, -1e9), dim=-1)
    student = torch.log_softmax(student_logits.masked_fill(~mask, -1e9), dim=-1)
    return (teacher.exp() * (teacher - student)).sum(-1).mean()


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    return -(logp.exp() * logp).sum(-1).mean()
