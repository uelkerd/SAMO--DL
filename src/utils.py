#!/usr/bin/env python3
"""Utility functions for the SAMO-DL project."""

import torch
from typing import Union


def count_model_params(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model to count parameters for
        only_trainable: If True, only count trainable parameters
        
    Returns:
        int: Number of parameters (total or trainable only)
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
