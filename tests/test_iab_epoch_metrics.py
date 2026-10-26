import torch
from torch.utils.data import DataLoader, TensorDataset


def test_epoch_evaluation_returns_metrics_loss_and_class_counts():
    from scripts.run_image_experiment import evaluate_epoch

    inputs = torch.tensor([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    result = evaluate_epoch(
        torch.nn.Identity(), loader, "cpu", "detection", ["real", "ai"],
        torch.nn.CrossEntropyLoss(),
    )

    assert result["n"] == 4
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["auroc"] == 1.0
    assert result["fpr_0_5"] == 0.0
    assert result["tpr_0_5"] == 1.0
    assert result["class_counts"] == {"real": 2, "ai": 2}
    assert result["loss"] > 0
