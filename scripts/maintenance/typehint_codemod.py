#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

PY39_TO_PY38_PATTERNS = [
    (re.compile(r"\bdict\[(.*?)\]"), r"Dict[\1]"),
    (re.compile(r"\blist\[(.*?)\]"), r"List[\1]"),
    (re.compile(r"\btuple\[(.*?)\]"), r"Tuple[\1]"),
    (re.compile(r"\bset\[(.*?)\]"), r"Set[\1]"),
    # A | B -> Union[A, B] (simple cases only)
    (re.compile(r"(?<!['\"])(?P<a>\b[A-Za-z_][A-Za-z0-9_\[\],\. ]*?)\s*\|\s*(?P<b>[A-Za-z_][A-Za-z0-9_\[\],\. ]*\b)"), r"Union[\g<a>, \g<b>]"),
    # X | None -> Optional[X]
    (re.compile(r"Union\[(.*?)\s*,\s*NoneType\]"), r"Optional[\1]"),
    (re.compile(r"Union\[(NoneType)\s*,\s*(.*?)\]"), r"Optional[\2]"),
]

NEEDED_IMPORTS = {"Dict", "List", "Tuple", "Set", "Optional", "Union"}


def ensure_typing_imports(lines):
    for i, line in enumerate(lines):
        if line.startswith("from typing import "):
            existing = set(x.strip() for x in line.split("import", 1)[1].split(","))
            merged = sorted(existing | NEEDED_IMPORTS)
            lines[i] = f"from typing import {', '.join(merged)}\n"
            return lines
    # no existing import; add one after last import block
    insert_at = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
        elif line.strip() and not line.startswith(("#", '"', "'")):
            break
    lines.insert(insert_at, f"from typing import {', '.join(sorted(NEEDED_IMPORTS))}\n")
    return lines


def transform_file(path: Path, dry_run: bool = False) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    for pattern, repl in PY39_TO_PY38_PATTERNS:
        updated = pattern.sub(repl, updated)
    if updated == original:
        return False
    lines = updated.splitlines(keepends=True)
    if any(tok in updated for tok in ("Dict[", "List[", "Tuple[", "Set[", "Optional[", "Union[")):
        lines = ensure_typing_imports(lines)
        updated = "".join(lines)
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Codemod: convert 3.9+/3.10 type hints to Python 3.8-compatible typing aliases")
    parser.add_argument("target", type=str, help="Target directory (e.g., src)")
    parser.add_argument("--dry-run", action="store_true", help="Only report files that would change")
    args = parser.parse_args()

    target = Path(args.target)
    changed = []
    for root, _dirs, files in os.walk(target):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = Path(root) / fname
            if transform_file(fpath, dry_run=args.dry_run):
                changed.append(str(fpath))

    if args.dry_run:
        print("Files to change:")
        for c in changed:
            print(c)
    else:
        print(f"Updated {len(changed)} files.")


if __name__ == "__main__":
    main()