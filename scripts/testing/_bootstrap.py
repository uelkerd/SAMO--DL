#!/usr/bin/env python3
"""Testing bootstrap utilities.

- Robust project root discovery using marker files
- Consistent sys.path insertion (idempotent)
- Simple logging configuration helper
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    "README.md",
    ".git",
)


def get_project_root(start: Optional[Path] = None, markers: Iterable[str] = _MARKERS) -> Path:
    """Discover the project root by walking up directories until a marker is found.

    Falls back to two levels up from this file if no markers are found.
    """
    current = (start or Path(__file__).resolve()).parent
    for _ in range(10):  # reasonable safety bound
        if any((current / m).exists() for m in markers):
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback: two levels above this file (scripts/testing/*)
    return Path(__file__).resolve().parents[2]


def ensure_project_root_on_sys_path(start: Optional[Path] = None) -> Path:
    """Ensure the discovered project root is on sys.path, returning the root path."""
    root = get_project_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_path(p: Path) -> None:
    """Idempotently add an arbitrary path to sys.path."""
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def configure_basic_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure simple logging for scripts and return a module-level logger."""
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)