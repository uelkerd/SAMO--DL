#!/usr/bin/env python3
"""
SAMO-DL Auto-Fix Code Quality

This script automatically fixes common code quality issues to prevent
recurring DeepSource warnings.

Auto-fixes:
- FLK-W291: Trailing whitespace
- FLK-W292: Missing newlines at end of file
- FLK-W293: Blank line whitespace
- FLK-E501: Line length violations (basic)
- PTC-W0027: f-strings without expressions
- Basic import organization
"""
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeQualityAutoFixer:
    """Automatically fixes common code quality issues."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.fixes_applied = 0
        self.files_modified = 0
        self.fixes_by_type = {}

    def fix_file(self, file_path: Path) -> Dict[str, Any]:
        """Fix quality issues in a single Python file."""
        # Validate file path for security
        if not self._is_safe_file_path(file_path):
            return {
                'file': str(file_path),
                'error': 'Unsafe file path detected',
                'modified': False
            }

        logger.info("Fixing: %s", file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes = []

            # Apply various fixes
            content, file_fixes = self._fix_trailing_whitespace(content)
            fixes.extend(file_fixes)

            content, file_fixes = self._fix_missing_newlines(content)
            fixes.extend(file_fixes)

            content, file_fixes = self._fix_blank_line_whitespace(content)
            fixes.extend(file_fixes)

            content, file_fixes = self._fix_f_strings_without_expressions(content)
            fixes.extend(file_fixes)

            content, file_fixes = self._fix_basic_line_length(content)
            fixes.extend(file_fixes)

            content, file_fixes = self._fix_import_organization(content)
            fixes.extend(file_fixes)

            # Apply fixes if not dry run
            if content != original_content and not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.files_modified += 1

            # Update statistics
            self.fixes_applied += len(fixes)
            for fix_type in [f['type'] for f in fixes]:
                self.fixes_by_type[fix_type] = self.fixes_by_type.get(fix_type, 0) + 1

            return {
                'file': str(file_path),
                'fixes': fixes,
                'modified': content != original_content
            }

        except Exception as e:
            logger.error("Error fixing %s: %s", file_path, e)
            return {
                'file': str(file_path),
                'error': str(e),
                'modified': False
            }

    @staticmethod
    def _fix_trailing_whitespace(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix trailing whitespace issues."""
        fixes = []
        lines = content.splitlines()
        modified = False

        for i, line in enumerate(lines):
            if line.rstrip() != line:
                lines[i] = line.rstrip()
                modified = True
                fixes.append({
                    'type': 'FLK-W291',
                    'line': i + 1,
                    'description': 'Removed trailing whitespace'
                })

        if modified:
            content = '\n'.join(lines) + ('\n' if content.endswith('\n') else '')

        return content, fixes

    @staticmethod
    def _fix_missing_newlines(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix missing newlines at end of file."""
        fixes = []

        if content and not content.endswith('\n'):
            content += '\n'
            fixes.append({
                'type': 'FLK-W292',
                'line': len(content.splitlines()),
                'description': 'Added missing newline at end of file'
            })

        return content, fixes

    @staticmethod
    def _fix_blank_line_whitespace(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix blank lines containing whitespace."""
        fixes = []
        lines = content.splitlines()
        modified = False

        for i, line in enumerate(lines):
            if line.strip() == '' and line != '':
                lines[i] = ''
                modified = True
                fixes.append({
                    'type': 'FLK-W293',
                    'line': i + 1,
                    'description': 'Removed whitespace from blank line'
                })

        if modified:
            content = '\n'.join(lines) + ('\n' if content.endswith('\n') else '')

        return content, fixes

    @staticmethod
    def _fix_f_strings_without_expressions(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix f-strings without expressions."""
        fixes = []

        # Pattern to find f-strings without expressions
        pattern = r'f["\']([^"\']*?)["\']'

        def replace_f_string(match):
            """Replace f-string without expressions with regular string."""
            string_content = match.group(1)
            if not re.search(r'\{[^}]*\}', string_content):
                fixes.append({
                    'type': 'PTC-W0027',
                    'line': content[:match.start()].count('\n') + 1,
                    'description': (
                        f'Converted f-string to regular string: {match.group(0)}'
                    )
                })
                return ""{string_content}"'
            return match.group(0)

        content = re.sub(pattern, replace_f_string, content)

        return content, fixes

    @staticmethod
    def _fix_basic_line_length(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix basic line length violations (simple cases)."""
        fixes = []
        lines = content.splitlines()
        modified = False

        for i, line in enumerate(lines):
            if len(line) > 88:
                # Try to break long lines at common break points
                if 'import ' in line and line.count(',') > 2:
                    # Break long import lines
                    parts = line.split(',')
                    if len(parts) > 3:
                        # Split into multiple lines
                        import_start = line[:line.find('import') + 6]
                        indent = len(line) - len(line.lstrip())

                        new_lines = [import_start + parts[0] + ',']
                        for part in parts[1:-1]:
                            new_lines.append(' ' * (indent + 4) + part + ',')
                        new_lines.append(' ' * (indent + 4) + parts[-1])

                        lines[i:i+1] = new_lines
                        modified = True
                        fixes.append({
                            'type': 'FLK-E501',
                            'line': i + 1,
                            'description': 'Broke long import line into multiple lines'
                        })

                elif 'def ' in line and line.count('(') > 0 and line.count(')') == 0:
                    # Break long function definitions
                    if line.count(',') > 2:
                        # Split parameters
                        func_start = line[:line.find('(') + 1]
                        params_part = line[line.find('(') + 1:]
                        indent = len(line) - len(line.lstrip())

                        # Find the last parameter
                        last_comma = params_part.rfind(',')
                        if last_comma > 0:
                            first_params = params_part[:last_comma + 1]
                            last_param = params_part[last_comma + 1:]

                            new_lines = [
                                func_start + first_params,
                                ' ' * (indent + 4) + last_param
                            ]

                            lines[i:i+1] = new_lines
                            modified = True
                            fixes.append({
                                'type': 'FLK-E501',
                                'line': i + 1,
                                'description': (
                                    'Broke long function definition into multiple lines'
                                )
                            })

        if modified:
            content = '\n'.join(lines) + ('\n' if content.endswith('\n') else '')

        return content, fixes

    @staticmethod
    def _fix_import_organization(content: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Fix basic import organization issues."""
        fixes = []
        lines = content.splitlines()
        modified = False

        # Find import sections
        import_start = -1
        import_end = -1

        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                if import_start == -1:
                    import_start = i
                import_end = i
            elif import_start != -1 and line.strip() == '':
                import_end = i - 1
                break

        if import_start != -1 and import_end != -1:
            # Sort imports within the section
            import_lines = lines[import_start:import_end + 1]
            sorted_imports = sorted(import_lines, key=lambda x: (
                # Standard library first
                0 if (
                    not x.strip().startswith('from ') and
                    not any(pkg in x for pkg in [
                        'django', 'flask', 'numpy', 'pandas',
                        'torch', 'transformers'
                    ])
                ) else 1,
                # Then by import type
                0 if x.strip().startswith('import ') else 1,
                # Then alphabetically
                x.strip().lower()
            ))

            if sorted_imports != import_lines:
                lines[import_start:import_end + 1] = sorted_imports
                modified = True
                fixes.append({
                    'type': 'Import Organization',
                    'line': import_start + 1,
                    'description': 'Reorganized imports alphabetically'
                })

        if modified:
            content = '\n'.join(lines) + ('\n' if content.endswith('\n') else '')

        return content, fixes

    @staticmethod
    def _is_safe_file_path(file_path: Path) -> bool:
        """Validate that file path is safe for processing."""
        try:
            # Resolve to absolute path to prevent path traversal
            resolved_path = file_path.resolve()

            # Check if path contains suspicious patterns
            path_str = str(resolved_path)
            suspicious_patterns = [
                '..',  # Path traversal
                '~',   # Home directory
                '/etc', '/var', '/usr', '/bin', '/sbin',  # System directories
                'C:\\', 'D:\\',  # Windows system drives
            ]

            for pattern in suspicious_patterns:
                if pattern in path_str:
                    return False

            # Ensure it's a Python file
            if not path_str.endswith('.py'):
                return False

            return True

        except Exception:
            return False

    def fix_directory(self, directory: Path) -> Dict[str, Any]:
        """Fix quality issues in all Python files in a directory."""
        logger.info("Fixing directory: %s", directory)

        python_files = list(directory.rglob("*.py"))
        logger.info("Found %d Python files", len(python_files))

        results = []

        for file_path in python_files:
            # Skip certain directories
            if any(part in str(file_path) for part in [
                '__pycache__', '.git', '.venv', '.env', 'build', 'dist',
                '.eggs', '.tox', '.coverage', 'htmlcov', '.cache',
                '.logs', 'results', 'samples', 'notebooks', 'website'
            ]):
                continue

            result = self.fix_file(file_path)
            results.append(result)

        return {
            'files_processed': len(results),
            'files_modified': self.files_modified,
            'total_fixes': self.fixes_applied,
            'fixes_by_type': self.fixes_by_type,
            'results': results
        }

    @staticmethod
    def generate_report(results: Dict[str, Any]) -> str:
        """Generate a comprehensive fix report."""
        report = """
🔧 CODE QUALITY AUTO-FIX REPORT
{'='*50}

📊 SUMMARY:
- Files processed: {results['files_processed']}
- Files modified: {results['files_modified']}
- Total fixes applied: {results['total_fixes']}

🛠️ FIXES BY TYPE:
"""

        for fix_type, count in sorted(results['fixes_by_type'].items()):
            report += f"- {fix_type}: {count} fixes\n"

        report += """

📋 DETAILED RESULTS:
{'-'*50}
"""

        for result in results['results']:
            if result.get('modified', False):
                report += "✅ {result["file']}: {len(result['fixes'])} fixes applied\n"
                for fix in result['fixes']:
                    report += "   - {fix["type']}: {fix['description']}\n"
            elif 'error' in result:
                report += "❌ {result["file']}: Error - {result['error']}\n"
            else:
                report += "⏭️ {result["file']}: No fixes needed\n"

        return report

    def run_fixes(self, directory: Path) -> bool:
        """Run all auto-fixes and return success status."""
        logger.info("Starting code quality auto-fixes...")

        if self.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")

        results = self.fix_directory(directory)

        # Generate and display report
        report = self.generate_report(results)
        print(report)

        # Return success (True if no errors, False if any errors occurred)
        errors = [r for r in results['results'] if 'error' in r]
        return len(errors) == 0


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Automatically fix common code quality issues'
    )
    parser.add_argument(
        'directory',
        help='Directory to process'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be fixed without making changes'
    )

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)

    fixer = CodeQualityAutoFixer(dry_run=args.dry_run)
    success = fixer.run_fixes(directory)

    if success:
        print("\n✅ All auto-fixes completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some errors occurred during auto-fixes!")
        sys.exit(1)

if __name__ == "__main__":
    main()
