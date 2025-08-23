#!/usr/bin/env python3
import pytest

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.utils import count_model_params


class TinyNet(nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4, 3, bias=True)  # params: 4*3 + 3 = 15
        self.lin2 = nn.Linear(3, 2, bias=False)  # params: 3*2 = 6

    def forward(self, x):  # pragma: no cover
        return self.lin2(self.lin1(x))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_count_model_params_total_and_trainable():
    model = TinyNet()

    # Freeze second layer
    for p in model.lin2.parameters():
        p.requires_grad = False

    total = count_model_params(model, only_trainable=False)
    trainable = count_model_params(model, only_trainable=True)

    assert total == 21
    assert trainable == 15


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_count_model_params_type_error():
    with pytest.raises(TypeError):
        count_model_params(object())  # type: ignore