from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start_path: Path, markers: tuple[str, ...] = ("pyproject.toml", "setup.py", ".git", "src")) -> Path:
    """Find repository root by walking parents looking for a marker.

    Raises FileNotFoundError if no marker is found.
    """
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(
        f"Could not determine repository root from {start_path}. Markers: {markers}"
    )


def add_repo_src_to_path(start_path: Path | None = None) -> Path:
    """Compute repo root and insert <root>/src at sys.path[0].

    Returns the repo root path.
    """
    base = start_path or Path(__file__)
    root = find_repo_root(base)
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return root