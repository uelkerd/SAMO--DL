#!/usr/bin/env python3
"""Utility functions for model inspection and metrics."""

from typing import Optional

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - allow import in environments without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


def count_model_params(model: "nn.Module", only_trainable: bool = True) -> int:
    """Count parameters in a PyTorch model.

    Args:
        model: A torch.nn.Module instance.
        only_trainable: If True, count only parameters with requires_grad=True.

    Returns:
        Total number of parameters as an integer.
    """
    if nn is None:
        raise RuntimeError("PyTorch is required for count_model_params but is not available")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())