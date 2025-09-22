#!/usr/bin/env python3
"""Auto-scan legacy code and update LEGACY_TRACKING.md"""

import subprocess
import re
import os
from collections import defaultdict

def scan_violations():
    """Run quality tools on entire codebase, collect violations per file"""
    violations = defaultdict(list)

    # Run ruff on all Python files
    try:
        result = subprocess.run(
            ['ruff', 'check', '.', '--output-format', 'json'],
            capture_output=True, text=True, cwd='.'
        )
        if result.stdout:
            import json
            data = json.loads(result.stdout)
            for item in data:
                file_path = item.get('filename', '')
                code = item.get('code', '')
                violations[file_path].append(f"ruff:{code}")
    except Exception as e:
        print(f"Warning: Could not run ruff scan: {e}")

    # Run bandit scan
    try:
        result = subprocess.run(
            ['bandit', '-r', '.', '-f', 'json'],
            capture_output=True, text=True
        )
        if result.stdout:
            import json
            data = json.loads(result.stdout)
            for item in data.get('results', []):
                file_path = item.get('filename', '')
                test_id = item.get('test_id', '')
                violations[file_path].append(f"bandit:{test_id}")
    except Exception as e:
        print(f"Warning: Could not run bandit scan: {e}")

    return dict(violations)

def update_tracking_file(violations_data):
    """Update LEGACY_TRACKING.md with current violations"""
    if not os.path.exists('LEGACY_TRACKING.md'):
        print("LEGACY_TRACKING.md not found, skipping update")
        return

    with open('LEGACY_TRACKING.md', 'r') as f:
        content = f.read()

    # Generate quarantined section
    quarantined_section = "## 🔴 QUARANTINED FILES (Auto-populated)\n"
    quarantined_section += "<!-- Auto-updated by scripts/update-legacy-tracking.py -->\n"

    for file_path, violation_list in sorted(violations_data.items()):
        if file_path.startswith('src/quality_enforced/'):
            continue  # Skip fortress files
        count = len(violation_list)
        quarantined_section += f"- [ ] {file_path} ({count} violations) #legacy-quarantined\n"

    # Replace the quarantined section
    pattern = r'## 🔴 QUARANTINED FILES \(Auto-populated\).*?(?=## |\Z)'
    new_content = re.sub(pattern, quarantined_section, content, flags=re.DOTALL)

    with open('LEGACY_TRACKING.md', 'w') as f:
        f.write(new_content)

    print(f"✅ Updated LEGACY_TRACKING.md with {len(violations_data)} quarantined files")

def main():
    """Main function to scan violations and update tracking file"""
    violations = scan_violations()
    update_tracking_file(violations)

if __name__ == "__main__":
    main()