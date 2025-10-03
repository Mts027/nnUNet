import torch
import pytest

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerInstanceLosses import LossMixer


class ConstantLoss(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = torch.tensor(value, dtype=torch.float32)

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.value.to(device=y_pred.device, dtype=y_pred.dtype)


@pytest.fixture
def dummy_inputs():
    y_pred = torch.ones((1, 1))
    y = torch.zeros_like(y_pred)
    return y_pred, y


def test_lossmixer_combines_weighted_losses(dummy_inputs):
    module = LossMixer([
        ("a", ConstantLoss(1.0), 0.5),
        ("b", ConstantLoss(3.0), 0.25),
    ])
    y_pred, y = dummy_inputs
    result = module(y_pred, y)
    assert torch.isclose(result, torch.tensor(1.25))


def test_lossmixer_component_losses_buffer(dummy_inputs):
    module = LossMixer([
        ("a", ConstantLoss(2.0), 1.0),
        ("b", ConstantLoss(1.0), 0.0),
    ])
    y_pred, y = dummy_inputs
    result = module(y_pred, y)
    assert torch.isclose(result, torch.tensor(2.0))
    assert set(module.component_losses.keys()) == {"a"}
    assert torch.isclose(module.component_losses["a"], torch.tensor(2.0))


def test_lossmixer_requires_components(dummy_inputs):
    with pytest.raises(ValueError):
        LossMixer([])


def test_lossmixer_unique_names(dummy_inputs):
    with pytest.raises(ValueError):
        LossMixer([
            ("dup", ConstantLoss(1.0), 1.0),
            ("dup", ConstantLoss(2.0), 1.0),
        ])
