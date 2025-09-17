#!/usr/bin/env python3
"""Script to fix common syntax errors in Python files"""

import os
import re

def fix_syntax_errors(file_path):
    """Fix common syntax errors in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix 1: Move shebang to the top
        if '#!/usr/bin/env python3' in content:
            # Remove shebang from anywhere in the file
            content = re.sub(r'#!/usr/bin/env python3\n?', '', content)
            # Add shebang at the top
            content = '#!/usr/bin/env python3\n' + content

        # Fix 2: Fix indentation issues - remove leading spaces from lines that should be at module level
        lines = content.split('\n')
        fixed_lines = []
        in_function = False
        indent_level = 0

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append('')
                continue

            # Check if this is a function definition
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
                indent_level = 0
                fixed_lines.append(line)
                continue

            # Check if this is a comment or import at module level
            if (stripped.startswith('import ') or
                stripped.startswith('from ') or
                stripped.startswith('#') or
                stripped.startswith('"""') or
                stripped.startswith("'''")):

                # If we're in a function but this looks like module-level code, fix indentation
                if in_function and not line.startswith('    '):
                    # This should be at module level
                    in_function = False
                    indent_level = 0
                    fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
                continue

            # Check if this is a return statement or other function content
            if (stripped.startswith('return ') or
                stripped.startswith('if ') or
                stripped.startswith('for ') or
                stripped.startswith('while ') or
                stripped.startswith('try:') or
                stripped.startswith('except ') or
                stripped.startswith('finally:') or
                stripped.startswith('else:') or
                stripped.startswith('elif ')):

                if not in_function and not line.startswith('    '):
                    # This should be in a function, add indentation
                    fixed_lines.append('    ' + line)
                else:
                    fixed_lines.append(line)
                continue

            # Default: keep the line as is
            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # Fix 3: Remove duplicate imports
        lines = content.split('\n')
        seen_imports = set()
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    fixed_lines.append(line)
                # Skip duplicate imports
            else:
                fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        print(f"No changes needed: {file_path}")
        return False

    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix syntax errors in all Python files."""
    # Get list of files with syntax errors
    files_to_fix = [
        "scripts/training/pre_training_validation.py",
        "scripts/training/minimal_working_training.py",
        "scripts/training/focal_loss_training.py",
        "scripts/training/fixed_training_with_optimized_config.py",
        "scripts/training/final_bulletproof_training_cell.py",
        "scripts/training/bulletproof_training_cell_fixed.py",
        "scripts/training/bulletproof_training_cell.py",
        "scripts/testing/test_domain_adaptation.py",
        "scripts/testing/standalone_focal_test.py",
        "scripts/testing/simple_test.py",
        "scripts/testing/simple_temperature_test_local.py",
        "scripts/testing/quick_focal_test.py",
        "scripts/testing/quick_f1_test.py",
        "scripts/testing/local_validation_debug.py",
        "scripts/maintenance/vertex_ai_setup_fixed.py",
        "scripts/legacy/vertex_ai_setup.py",
        "scripts/legacy/validate_and_train.py",
        "scripts/legacy/threshold_optimization.py",
        "scripts/legacy/temperature_scaling.py",
        "scripts/legacy/start_monitoring_dashboard.py",
        "scripts/legacy/simple_vertex_ai_validation.py",
        "scripts/legacy/simple_validation.py",
        "scripts/legacy/simple_finalize_model.py",
        "scripts/legacy/model_optimization.py",
        "scripts/legacy/model_monitoring.py",
        "scripts/legacy/minimal_validation.py",
        "scripts/legacy/fine_tune_emotion_model.py"
    ]

    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_syntax_errors(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")

    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()
