#!/usr/bin/env python3
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
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

TRACKED_CACHE: Dict[str, bool] = {}


def is_git_tracked(path: Path) -> bool:
    p = str(path.relative_to(ROOT))
    if p in TRACKED_CACHE:
        return TRACKED_CACHE[p]
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", p],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        TRACKED_CACHE[p] = True
    except subprocess.CalledProcessError:
        TRACKED_CACHE[p] = False
    return TRACKED_CACHE[p]


def list_all_files() -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        # Prune ignored directories early
        dirnames[:] = [d for d in dirnames if d not in IGNORES]
        for fname in filenames:
            p = Path(dirpath) / fname
            try:
                rel_parts = p.relative_to(ROOT).parts
            except Exception:
                continue
            if any(part in IGNORES for part in rel_parts):
                continue
            try:
                # Include regular files and symlinks; skip vanished entries
                if p.is_file() or p.is_symlink():
                    files.append(p)
            except FileNotFoundError:
                continue
    return files


def gather_metadata(p: Path) -> Dict[str, Any]:
    try:
        stat = p.stat()
        size = stat.st_size
    except FileNotFoundError:
        size = 0
    try:
        rel = str(p.relative_to(ROOT))
    except Exception:
        rel = str(p)
    return {
        "path": rel,
        "size_bytes": size,
        "is_tracked": is_git_tracked(p) if p.exists() else False,
    }


def gather_top_level() -> Dict[str, Any]:
    top = {}
    for entry in ROOT.iterdir():
        if entry.name in IGNORES:
            continue
        try:
            if entry.is_dir():
                size = 0
                for dp, dn, fn in os.walk(entry):
                    dn[:] = [d for d in dn if d not in IGNORES]
                    for f in fn:
                        fp = Path(dp) / f
                        try:
                            size += fp.stat().st_size
                        except Exception:
                            pass
                top[entry.name] = {"type": "dir", "size_bytes": size}
            else:
                top[entry.name] = {
                    "type": "file",
                    "size_bytes": entry.stat().st_size,
                }
        except Exception:
            # Best effort
            top[entry.name] = {"type": "unknown", "size_bytes": 0}
    return top


def find_references(paths: List[str]) -> Dict[str, List[str]]:
    refs: Dict[str, List[str]] = {p: [] for p in paths}
    # Use ripgrep to find textual references for moved/deprecated candidates
    for p in paths:
        try:
            rg = subprocess.run(
                [
                    "rg",
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
            )
            if rg.returncode in (0, 1):  # 0 found, 1 not found
                lines = [ln for ln in rg.stdout.splitlines() if ln.strip()]
                refs[p] = lines[:200]
        except Exception:
            pass
    return refs


def load_config(config_path: Path) -> Dict[str, Any]:
    if config_path.exists():
        try:
            return json.loads(config_path.read_text())
        except Exception:
            return {}
    return {}


def parse_args(default_candidates: List[str], default_cap: int) -> argparse.Namespace:
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
    LOGS.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CONFIG_PATH)
    default_candidates = list(cfg.get("candidates", []))
    default_cap = int(cfg.get("file_inventory_cap", 2000))

    args = parse_args(default_candidates, default_cap)

    all_files = list_all_files()
    files_meta = [gather_metadata(p) for p in all_files]
    files_meta.sort(key=lambda x: (-x["size_bytes"], x["path"]))

    top_level = gather_top_level()

    candidates = args.candidates
    refs = find_references(candidates)

    report = {
        "root": str(ROOT),
        "top_level_summary": top_level,
        "file_inventory_cap": args.cap,
        "file_inventory": files_meta[: args.cap],
        "candidates": candidates,
        "references": refs,
        "config_path": str(CONFIG_PATH),
    }

    # Allow overriding report path
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote inventory report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())