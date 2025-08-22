#!/usr/bin/env python3
"""Repository inventory tool for cleanup planning.

- Summarizes top-level sizes and lists files (skipping ignored dirs)
- Scans for references to candidate paths
- Configurable via configs/repo_inventory.json and CLI flags
"""
import os
import json
import argparse
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from shutil import which
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / ".logs"
REPORT = LOGS / "repo_inventory.json"
CONFIG_PATH = ROOT / "configs" / "repo_inventory.json"

IGNORES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "artifacts",
    ".logs",
    "htmlcov",
}

GIT_BIN = which("git") or "git"
RG_BIN = which("rg") or "rg"


def is_git_tracked(path: Path) -> bool:
    """Return True if the given path is tracked by git (O(1) set lookup)."""
    tracked = _load_tracked_files()
    try:
        p = str(path.relative_to(ROOT))
    except ValueError:
        return False
    return p in tracked


@lru_cache(maxsize=1)
def _load_tracked_files() -> frozenset[str]:
    """Load all git-tracked files once and cache them in memory."""
    try:
        result = subprocess.run(
            [GIT_BIN, "ls-files"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return frozenset(result.stdout.splitlines())
    except (subprocess.CalledProcessError, OSError):
        return frozenset()


def list_all_files() -> List[Path]:
    """Walk the repository and return a list of files, skipping ignored dirs.

    Follows neither symlinks nor ignored directories to keep traversal fast and
    avoid accidental recursion into large caches.
    """
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        # Prune ignored directories early
        dirnames[:] = [d for d in dirnames if d not in IGNORES]
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                rel_parts = p.relative_to(ROOT).parts
            except ValueError:
                continue
            if any(part in IGNORES for part in rel_parts):
                continue
            # Include regular files and symlinks-to-files; skip vanished/dirs
            try:
                if p.is_file() or (p.is_symlink() and p.exists() and p.resolve().is_file()):
                    files.append(p)
            except OSError:
                continue
    return files


def gather_metadata(p: Path) -> Dict[str, Any]:
    """Collect lightweight metadata for a file for reporting purposes."""
    try:
        stat = p.stat()
        size = stat.st_size
    except OSError:
        size = 0
    try:
        rel = str(p.relative_to(ROOT))
    except ValueError:
        rel = str(p)
    return {
        "path": rel,
        "size_bytes": size,
        "is_tracked": is_git_tracked(p) if p.exists() else False,
    }


def gather_top_level(files_meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize first-level entries under repo root with sizes aggregated from files_meta."""
    # Initialize entries and types from a shallow scan
    top: Dict[str, Dict[str, Any]] = {}
    for entry in ROOT.iterdir():
        if entry.name in IGNORES:
            continue
        try:
            etype = "dir" if entry.is_dir() else "file"
        except OSError:
            etype = "unknown"
        top[entry.name] = {"type": etype, "size_bytes": 0}

    # Aggregate sizes by first path segment from files_meta
    for fm in files_meta:
        parts = Path(fm["path"]).parts
        if not parts:
            continue
        first = parts[0]
        if first in IGNORES:
            continue
        bucket = top.setdefault(first, {"type": "dir", "size_bytes": 0})
        bucket["size_bytes"] += int(fm.get("size_bytes", 0))

    return top


def find_references(paths: List[str]) -> Dict[str, List[str]]:
    """Search for textual references to provided paths using ripgrep.

    Returns a mapping from candidate path string to a list of example matches
    (capped) to help update docs/CI after moves.
    """
    refs: Dict[str, List[str]] = {p: [] for p in paths}
    # Use ripgrep to find textual references for moved/deprecated candidates
    for p in paths:
        try:
            rg = subprocess.run(
                [
                    RG_BIN,
                    "-n",
                    "-S",
                    "--hidden",
                    "--glob",
                    "!{.git,node_modules,artifacts,.logs,htmlcov}",
                    p,
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            if rg.returncode in (0, 1):  # 0 found, 1 not found
                lines = [ln for ln in rg.stdout.splitlines() if ln.strip()]
                refs[p] = lines[:200]
        except OSError:
            pass
    return refs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration for candidates and defaults; return {} on error."""
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return {}
        except OSError:
            return {}
    return {}


def parse_args(default_candidates: List[str], default_cap: int) -> argparse.Namespace:
    """Parse CLI arguments, defaulting to values loaded from config."""
    parser = argparse.ArgumentParser(description="Repo inventory and reference scout")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=default_candidates,
        help="List of candidate paths to search references for (overrides config)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=default_cap,
        help="Cap on number of file entries in report (for readability)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(REPORT),
        help="Path to write JSON report",
    )
    return parser.parse_args()


def main() -> int:
    """Generate a JSON inventory report for the repository structure.

    - Loads defaults from configs/repo_inventory.json
    - Allows CLI overrides for candidates, cap, and report path
    - Produces a summary suitable for cleanup planning
    """
    LOGS.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CONFIG_PATH)
    default_candidates = list(cfg.get("candidates", []))
    default_cap = int(cfg.get("file_inventory_cap", 2000))

    args = parse_args(default_candidates, default_cap)

    all_files = list_all_files()
    files_meta = [gather_metadata(p) for p in all_files]
    files_meta.sort(key=lambda x: (-x["size_bytes"], x["path"]))

    top_level = gather_top_level(files_meta)

    candidates = args.candidates
    refs = find_references(candidates)

    report = {
        "root": str(ROOT),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_level_summary": top_level,
        "file_inventory_cap": args.cap,
        "file_inventory": files_meta[: args.cap],
        "candidates": candidates,
        "references": refs,
        "config_path": str(CONFIG_PATH),
    }

    try:
        commit = subprocess.check_output(
            [GIT_BIN, "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        report["git_commit"] = commit
    except Exception:
        report["git_commit"] = None

    # Allow overriding report path
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote inventory report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
