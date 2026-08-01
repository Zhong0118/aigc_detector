from __future__ import annotations

import importlib

import pytest


def test_loss_module_exports_phase3_contract() -> None:
    module = importlib.import_module("research.text_attribution.attribution_losses")
    assert hasattr(module, "Phase3LossOutput")
    assert callable(module.compute_phase3_loss)
    assert callable(module.hierarchical_consistency_loss)
    assert callable(module.supervised_contrastive_loss)


def _torch_and_losses():
    torch = pytest.importorskip("torch")
    from research.text_attribution import attribution_losses

    return torch, attribution_losses


def test_consistency_is_zero_when_probabilities_match() -> None:
    torch, losses = _torch_and_losses()
    family_logits = torch.log(torch.tensor([[0.7, 0.3]], dtype=torch.float32))
    model_logits = torch.log(
        torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    )
    result = losses.hierarchical_consistency_loss(
        family_logits,
        model_logits,
        torch.tensor([0, 0, 1, 1]),
    )
    assert result.item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_is_positive_and_has_finite_gradients() -> None:
    torch, losses = _torch_and_losses()
    family_logits = torch.tensor([[3.0, -1.0]], requires_grad=True)
    model_logits = torch.tensor([[0.0, 0.0, 3.0, 2.0]], requires_grad=True)
    result = losses.hierarchical_consistency_loss(
        family_logits,
        model_logits,
        torch.tensor([0, 0, 1, 1]),
    )
    assert result.item() > 0
    result.backward()
    assert torch.isfinite(family_logits.grad).all()
    assert torch.isfinite(model_logits.grad).all()


def test_consistency_uses_family_to_aggregated_kl_direction() -> None:
    torch, losses = _torch_and_losses()
    family_logits = torch.log(torch.tensor([[0.8, 0.2]], dtype=torch.float64))
    model_logits = torch.log(
        torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    )
    actual = losses.hierarchical_consistency_loss(
        family_logits,
        model_logits,
        torch.tensor([0, 0, 1, 1]),
    )
    family = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    aggregated = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    expected = (family * (family.log() - aggregated.log())).sum()
    reverse = (aggregated * (aggregated.log() - family.log())).sum()

    assert actual.item() == pytest.approx(expected.item())
    assert actual.item() != pytest.approx(reverse.item())


def test_supcon_rejects_batch_without_valid_positive() -> None:
    torch, losses = _torch_and_losses()
    with pytest.raises(ValueError, match="valid anchor"):
        losses.supervised_contrastive_loss(
            projections=torch.eye(3),
            family_labels=torch.tensor([0, 1, 2]),
            group_ids=torch.tensor([0, 1, 2]),
            temperature=0.07,
        )


def test_supcon_uses_cross_group_same_family_positives() -> None:
    torch, losses = _torch_and_losses()
    projections = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        requires_grad=True,
    )
    result = losses.supervised_contrastive_loss(
        projections,
        family_labels=torch.tensor([0, 0, 1, 1]),
        group_ids=torch.tensor([0, 1, 0, 1]),
        temperature=0.1,
    )
    assert torch.isfinite(result)
    assert result.item() >= 0
    result.backward()
    assert torch.isfinite(projections.grad).all()


def test_compute_loss_ignores_unseen_model_labels() -> None:
    torch, losses = _torch_and_losses()
    output = losses.compute_phase3_loss(
        family_logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True),
        family_labels=torch.tensor([0, 1]),
        label_smoothing_factor=0.05,
        auxiliary_weight=0.3,
        model_logits=torch.tensor(
            [[3.0, 0.0], [0.0, 3.0]], requires_grad=True
        ),
        model_labels=torch.tensor([0, -100]),
    )
    expected_model = torch.nn.functional.cross_entropy(
        torch.tensor([[3.0, 0.0]]), torch.tensor([0]), label_smoothing=0.05
    )
    torch.testing.assert_close(output.model, expected_model)
    torch.testing.assert_close(output.total, output.family + 0.3 * output.model)


def test_disabled_components_are_exact_scalar_zero() -> None:
    torch, losses = _torch_and_losses()
    output = losses.compute_phase3_loss(
        family_logits=torch.tensor([[1.0, 0.0]], requires_grad=True),
        family_labels=torch.tensor([0]),
        label_smoothing_factor=0.0,
        auxiliary_weight=0.0,
        consistency_weight=0.0,
        contrastive_weight=0.0,
    )
    assert output.model.shape == output.consistency.shape == output.contrastive.shape == ()
    assert output.model.item() == 0.0
    assert output.consistency.item() == 0.0
    assert output.contrastive.item() == 0.0
    torch.testing.assert_close(output.total, output.family)


def test_enabled_component_requires_its_inputs() -> None:
    torch, losses = _torch_and_losses()
    with pytest.raises(ValueError, match="model_logits"):
        losses.compute_phase3_loss(
            family_logits=torch.tensor([[1.0, 0.0]]),
            family_labels=torch.tensor([0]),
            label_smoothing_factor=0.0,
            auxiliary_weight=0.1,
        )
