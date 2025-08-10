"""String utility helpers for common patterns."""

from typing import Optional


def is_truthy(value: Optional[str]) -> bool:
    """Return True if value looks like a truthy flag.

    Accepts common values case-insensitively and ignores surrounding whitespace.
    Recognized truthy values: "1", "true", "yes".
    """
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}

