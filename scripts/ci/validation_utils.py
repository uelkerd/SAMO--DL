#!/usr/bin/env python3
"""
Reusable validation helpers for CI scripts.

These helpers reduce boilerplate and provide clearer failure semantics
using specific exception types.
"""

from typing import Any, Iterable


def validate_metric_ranges(metrics: dict[str, Any], fields: Iterable[str]) -> None:
    """Ensure each metric in fields is within [0, 1].

    Raises:
        ValueError: if any metric is missing or out of range.
    """
    for field in fields:
        if field not in metrics:
            raise ValueError(f"Missing metric: {field}")
        value = metrics[field]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:  # non-numeric
            raise ValueError(f"Metric '{field}' is not numeric: {value}") from exc
        if not 0.0 <= numeric <= 1.0:
            pretty = field.replace("_", " ").capitalize()
            raise ValueError(f"{pretty} should be between 0 and 1")


def validate_required_keys(obj: dict[str, Any], keys: Iterable[str], label: str = "object") -> None:
    """Validate that all keys exist in obj.

    Raises:
        KeyError: if a required key is missing.
    """
    for key in keys:
        if key not in obj:
            pretty = key.replace("_", " ")
            raise KeyError(f"{label} should have {pretty}")


def validate_hasattrs(instance: Any, attrs: Iterable[str], label: str = "object") -> None:
    """Validate that instance has all attributes in attrs.

    Raises:
        AttributeError: if a required attribute is missing.
    """
    for attr in attrs:
        if not hasattr(instance, attr):
            raise AttributeError(f"{label} should have '{attr}' attribute")


def ensure(condition: bool, message: str) -> None:
    """Fail the test with AssertionError if condition is False.

    Keeps test bodies declarative by avoiding explicit if blocks.
    """
    if not condition:
        raise AssertionError(message)
