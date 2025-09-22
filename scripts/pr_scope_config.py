"""Configuration module for PR scope checking.

This module contains all static configuration values used by the PR scope checker
to avoid duplication and make maintenance easier.
"""

from typing import Dict, List, Set

# File type detection patterns
FILE_TYPE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "code": {
        "extensions": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h"],
    },
    "docs": {
        "extensions": [".md", ".rst", ".txt", ".adoc"],
    },
    "tests": {
        "directories": ["tests/", "test/"],
        "suffixes": ["_test.py", "_test.js", "_test.ts", "_test.java", "_test.cpp", "_test.c", "_test.h"],
        "prefixes": ["test_"],
        "patterns": [r"(^|/)(test_.*|.*_test\.(py|js|ts|java|cpp|c|h))$"],
    },
    "config": {
        "directories": ["config/"],
        "extensions": [".yml", ".yaml", ".json", ".toml", ".cfg"],
        "exact_files": ["config.py", "config.js", "config.ts", "config.java", "config.cpp", "config.c", "config.h"],
    },
    "docker": {
        "keywords": ["docker", "Dockerfile"],
    },
    "api": {
        "keywords": ["api", "endpoint"],
    },
    "ml": {
        "keywords": ["model", "ml", "ai"],
    },
}

# Acceptable file type combinations for single PRs
ACCEPTABLE_COMBINATIONS: List[Set[str]] = [
    {"code", "tests"},  # Feature + tests
    {"code", "tests", "config"},  # Feature + tests + config
    {"code", "tests", "docs"},  # Feature + tests + docs
    {"code", "tests", "config", "docs"},  # Feature + tests + config + docs
    {"code", "config"},  # Code changes with config
    {"code", "docs"},  # Code changes with docs
    {"config", "docs"},  # Config changes with docs
    {"tests", "config"},  # Tests with config
]

# Size limits
MAX_FILES_CHANGED = 50
MAX_LINES_CHANGED = 1500
MAX_FILE_TYPES_FOR_MIXED_CONCERNS = 4
MAX_FILE_TYPES_FOR_WARNING = 2

# Branch naming pattern
BRANCH_NAME_PATTERN = r"^(feat|fix|chore|refactor|docs|test)/[a-z]+(?:-[a-z]+)*$"

# Commit message patterns
SINGLE_PURPOSE_KEYWORDS = [
    "feat:",
    "fix:",
    "chore:",
    "refactor:",
    "docs:",
    "test:",
]

MIXING_INDICATORS = [" and ", " also ", " plus ", " & ", " in addition "]

# Display limits
MAX_FILES_TO_DISPLAY = 10
