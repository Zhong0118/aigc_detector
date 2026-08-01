"""Composable Phase3 family, auxiliary, hierarchy, and contrastive losses.

PyTorch is imported lazily so configuration and ``--dry-run`` validation remain
available on a CPU-only control machine that has not installed the server wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase3LossOutput:
    total: Any
    family: Any
    model: Any
    consistency: Any
    contrastive: Any


def hierarchical_consistency_loss(
    family_logits: Any,
    model_logits: Any,
    model_to_family_ids: Any,
    *,
    epsilon: float = 1e-8,
) -> Any:
    """KL from aggregated auxiliary-model probabilities to family probabilities."""

    torch, functional = _require_torch()
    if family_logits.ndim != 2 or model_logits.ndim != 2:
        raise ValueError("family_logits and model_logits must be rank-two tensors")
    if family_logits.shape[0] != model_logits.shape[0]:
        raise ValueError("family and model logits must have the same batch size")
    mapping = torch.as_tensor(
        model_to_family_ids, dtype=torch.long, device=model_logits.device
    )
    if mapping.ndim != 1 or mapping.numel() != model_logits.shape[1]:
        raise ValueError("model_to_family_ids must have one entry per model logit")
    family_count = family_logits.shape[1]
    if mapping.numel() == 0 or mapping.min().item() < 0 or mapping.max().item() >= family_count:
        raise ValueError("model_to_family_ids contains an invalid family label")

    model_probabilities = functional.softmax(model_logits, dim=-1)
    aggregated = model_probabilities.new_zeros(
        (model_probabilities.shape[0], family_count)
    )
    aggregated.scatter_add_(
        1,
        mapping.unsqueeze(0).expand(model_probabilities.shape[0], -1),
        model_probabilities,
    )
    aggregated = aggregated.clamp_min(epsilon)
    aggregated = aggregated / aggregated.sum(dim=-1, keepdim=True)
    family_probabilities = functional.softmax(family_logits, dim=-1)
    aggregated_log_probabilities = aggregated.log()
    loss = functional.kl_div(
        aggregated_log_probabilities,
        family_probabilities,
        reduction="batchmean",
    ).clamp_min(0.0)
    _require_finite(loss, "hierarchical consistency")
    return loss


def supervised_contrastive_loss(
    projections: Any,
    family_labels: Any,
    group_ids: Any,
    temperature: float,
) -> Any:
    """Supervised contrastive loss with cross-semantic-group positives.

    A pair is positive only when it shares a family and comes from a different
    semantic prompt group.  Self pairs are always excluded from both numerator and
    denominator.
    """

    torch, functional = _require_torch()
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if projections.ndim != 2:
        raise ValueError("contrastive projections must be rank two")
    family_labels = torch.as_tensor(
        family_labels, dtype=torch.long, device=projections.device
    )
    group_ids = torch.as_tensor(
        group_ids, dtype=torch.long, device=projections.device
    )
    batch_size = projections.shape[0]
    if family_labels.shape != (batch_size,) or group_ids.shape != (batch_size,):
        raise ValueError("contrastive labels must have one value per projection")

    normalized = functional.normalize(projections, p=2, dim=-1)
    similarities = normalized @ normalized.transpose(0, 1)
    logits = similarities / temperature
    identity = torch.eye(batch_size, dtype=torch.bool, device=projections.device)
    denominator_mask = ~identity
    positive_mask = (
        family_labels[:, None].eq(family_labels[None, :])
        & group_ids[:, None].ne(group_ids[None, :])
        & denominator_mask
    )
    positive_counts = positive_mask.sum(dim=1)
    if not bool(torch.all(positive_counts > 0).item()):
        invalid = torch.nonzero(positive_counts == 0).flatten().tolist()
        raise ValueError(
            "supervised contrastive loss requires a valid anchor positive for "
            f"every row; invalid anchors={invalid[:8]!r}"
        )

    masked_logits = logits.masked_fill(~denominator_mask, float("-inf"))
    log_denominator = torch.logsumexp(masked_logits, dim=1, keepdim=True)
    log_probabilities = logits - log_denominator
    loss = -(
        log_probabilities.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_counts
    ).mean()
    _require_finite(loss, "supervised contrastive")
    return loss


def compute_phase3_loss(
    *,
    family_logits: Any,
    family_labels: Any,
    label_smoothing_factor: float,
    auxiliary_weight: float = 0.0,
    model_logits: Any | None = None,
    model_labels: Any | None = None,
    consistency_weight: float = 0.0,
    model_to_family_ids: Any | None = None,
    contrastive_weight: float = 0.0,
    projections: Any | None = None,
    contrastive_family_labels: Any | None = None,
    contrastive_group_ids: Any | None = None,
    contrastive_temperature: float = 0.07,
) -> Phase3LossOutput:
    """Compute exactly the components enabled by an immutable Phase3 recipe."""

    torch, functional = _require_torch()
    for name, weight in (
        ("auxiliary_weight", auxiliary_weight),
        ("consistency_weight", consistency_weight),
        ("contrastive_weight", contrastive_weight),
    ):
        if weight < 0:
            raise ValueError(f"{name} must be nonnegative")
    if not 0 <= label_smoothing_factor < 1:
        raise ValueError("label_smoothing_factor must be in [0, 1)")

    family = functional.cross_entropy(
        family_logits,
        family_labels,
        label_smoothing=label_smoothing_factor,
    )
    _require_finite(family, "family classification")
    zero = family_logits.new_zeros(())

    model = zero
    if auxiliary_weight > 0:
        if model_logits is None:
            raise ValueError("enabled auxiliary loss requires model_logits")
        if model_labels is None:
            raise ValueError("enabled auxiliary loss requires model_labels")
        valid = model_labels.ne(-100)
        if bool(valid.any().item()):
            model = functional.cross_entropy(
                model_logits[valid],
                model_labels[valid],
                label_smoothing=label_smoothing_factor,
            )
            _require_finite(model, "auxiliary model classification")

    consistency = zero
    if consistency_weight > 0:
        if model_logits is None or model_to_family_ids is None:
            raise ValueError(
                "enabled consistency loss requires model_logits and "
                "model_to_family_ids"
            )
        consistency = hierarchical_consistency_loss(
            family_logits, model_logits, model_to_family_ids
        )

    contrastive = zero
    if contrastive_weight > 0:
        if (
            projections is None
            or contrastive_family_labels is None
            or contrastive_group_ids is None
        ):
            raise ValueError(
                "enabled contrastive loss requires projections, family labels, "
                "and group IDs"
            )
        contrastive = supervised_contrastive_loss(
            projections,
            contrastive_family_labels,
            contrastive_group_ids,
            contrastive_temperature,
        )

    total = (
        family
        + auxiliary_weight * model
        + consistency_weight * consistency
        + contrastive_weight * contrastive
    )
    _require_finite(total, "total Phase3")
    return Phase3LossOutput(
        total=total,
        family=family,
        model=model,
        consistency=consistency,
        contrastive=contrastive,
    )


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised on server dependency check
        raise RuntimeError(
            "Phase3 tensor losses require PyTorch; install the server CUDA wheel"
        ) from exc
    return torch, functional


def _require_finite(value: Any, name: str) -> None:
    torch, _ = _require_torch()
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite {name} loss")
