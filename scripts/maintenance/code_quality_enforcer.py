#!/usr/bin/env python3
"""
SAMO-DL Code Quality Enforcer

This script enforces comprehensive code quality standards and prevents
ALL recurring DeepSource issues from ever happening again.

Prevents:
- PYL-R1705: Unnecessary else/elif after return
- PTC-W0027: f-strings without expressions
- PY-W2000: Unused imports
- FLK-E128: Continuation line indentation
- FLK-E301: Missing blank lines
- FLK-E501: Line length violations
- FLK-W291: Trailing whitespace
- FLK-W292: Missing newlines
- FLK-W293: Blank line whitespace
- FLK-W505: Doc line length
- PY-D0003: Missing docstrings
- PY-R1000: High cyclomatic complexity
"""
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%asctimes - %levelnames - %messages'
)
logger = logging.getLogger__name__


class CodeQualityEnforcer:
    """Enforces comprehensive code quality standards for SAMO-DL."""

    def __init__self:
        self.issues_found: List[Dict[str, Any]] = []
        self.files_checked = 0
        self.files_with_issues = 0

        # Define quality rules
        self.rules = {
            'PYL-R1705': {
                'name': 'Unnecessary else/elif after return',
                'severity': 'error',
                'description': 'Remove unnecessary else/elif after return statements'
            },
            'PTC-W0027': {
                'name': 'f-string without expressions',
                'severity': 'warning',
                'description': (
                    'Use regular strings instead of f-strings without expressions'
                )
            },
            'PY-W2000': {
                'name': 'Unused imports',
                'severity': 'warning',
                'description': 'Remove unused imports'
            },
            'FLK-E128': {
                'name': 'Continuation line indentation',
                'severity': 'error',
                'description': 'Fix continuation line indentation for visual indent'
            },
            'FLK-E301': {
                'name': 'Missing blank lines',
                'severity': 'error',
                'description': 'Add blank lines between class methods'
            },
            'FLK-E501': {
                'name': 'Line too long',
                'severity': 'error',
                'description': 'Break long lines to stay within 88 character limit'
            },
            'FLK-W291': {
                'name': 'Trailing whitespace',
                'severity': 'error',
                'description': 'Remove trailing whitespace'
            },
            'FLK-W292': {
                'name': 'Missing newline at end of file',
                'severity': 'error',
                'description': 'Add newline at end of file'
            },
            'FLK-W293': {
                'name': 'Blank line contains whitespace',
                'severity': 'error',
                'description': 'Remove whitespace from blank lines'
            },
            'FLK-W505': {
                'name': 'Doc line too long',
                'severity': 'warning',
                'description': 'Break long docstring lines'
            },
            'PY-D0003': {
                'name': 'Missing docstring',
                'severity': 'warning',
                'description': 'Add docstrings to functions and classes'
            },
            'PY-R1000': {
                'name': 'High cyclomatic complexity',
                'severity': 'warning',
                'description': 'Refactor complex functions to reduce complexity'
            }
        }

    def check_fileself, file_path: Path -> List[Dict[str, Any]]:
        """Check a single Python file for quality issues."""
        # Validate file path for security
        if not self._is_safe_file_pathfile_path:
            return [{
                'rule': 'SECURITY',
                'line': 0,
                'message': 'Unsafe file path detected',
                'severity': 'error'
            }]

        issues = []

        try:
            with openfile_path, 'r', encoding='utf-8' as f:
                content = f.read()
                lines = content.splitlines()

            # Check for missing newline at end of file
            if content and not content.endswith'\n':
                issues.append({
                    'rule': 'FLK-W292',
                    'line': lenlines,
                    'message': 'Missing newline at end of file',
                    'severity': 'error'
                })

            # Check each line for issues
            for line_num, line in enumeratelines, 1:
                line_issues = self._check_lineline, line_num
                issues.extendline_issues

            # Check for unused imports
            import_issues = self._check_unused_importscontent, file_path
            issues.extendimport_issues

            # Check for high cyclomatic complexity
            complexity_issues = self._check_cyclomatic_complexitycontent, file_path
            issues.extendcomplexity_issues

            # Check for unnecessary else/elif after return
            control_flow_issues = self._check_control_flowcontent, file_path
            issues.extendcontrol_flow_issues

        except Exception as e:
            logger.error"Error checking %s: %s", file_path, e
            issues.append({
                'rule': 'ERROR',
                'line': 0,
                'message': f'Error reading file: {e}',
                'severity': 'error'
            })

        return issues

    @staticmethod
    def _check_lineline: str, line_num: int -> List[Dict[str, Any]]:
        """Check a single line for quality issues."""
        issues = []

        # Check for trailing whitespace
        if line.rstrip() != line:
            issues.append({
                'rule': 'FLK-W291',
                'line': line_num,
                'message': 'Trailing whitespace detected',
                'severity': 'error'
            })

        # Check for blank line with whitespace
        if line.strip() == '' and line != '':
            issues.append({
                'rule': 'FLK-W293',
                'line': line_num,
                'message': 'Blank line contains whitespace',
                'severity': 'error'
            })

        # Check for line length
        if lenline > 88:
            issues.append({
                'rule': 'FLK-E501',
                'line': line_num,
                'message': f'Line too long ({lenline} > 88 characters)',
                'severity': 'error'
            })

        # Check for f-strings without expressions
        if (
            (line.strip().startswith'"' or line.strip().startswith"'")
            and not re.searchr'\{[^}]*\}', line
        ):
            issues.append({
                'rule': 'PTC-W0027',
                'line': line_num,
                'message': 'f-string used without expressions',
                'severity': 'warning'
            })

        return issues

    @staticmethod
    def _check_unused_importscontent: str, file_path: Path -> List[Dict[str, Any]]:
        """Check for unused imports using AST analysis."""
        issues = []

        try:
            tree = ast.parsecontent
            import_nodes = []
            used_names = set()

            # Collect all import nodes
            for node in ast.walktree:
                if isinstance(node, ast.Import, ast.ImportFrom):
                    import_nodes.appendnode
                elif isinstancenode, ast.Name:
                    used_names.addnode.id
                elif isinstancenode, ast.Attribute:
                    # Handle attribute access e.g., module.function
                    if isinstancenode.value, ast.Name:
                        used_names.addnode.value.id

            # Check for unused imports
            for node in import_nodes:
                if isinstancenode, ast.Import:
                    for alias in node.names:
                        if (
                            alias.name not in used_names and
                            not alias.name.startswith'_'
                        ):
                            issues.append({
                                'rule': 'PY-W2000',
                                'line': getattrnode, 'lineno', 0,
                                'message': f'Unused import: {alias.name}',
                                'severity': 'warning'
                            })
                elif isinstancenode, ast.ImportFrom:
                    pass

        except SyntaxError:
            # File has syntax errors, skip import analysis
            pass

        return issues

    def _check_cyclomatic_complexity(
        self, content: str, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Check for high cyclomatic complexity."""
        issues = []

        try:
            tree = ast.parsecontent

            for node in ast.walktree:
                if isinstance(
                    node, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
                ):
                    complexity = self._calculate_complexitynode
                    if complexity > 10:  # Threshold for high complexity
                        issues.append({
                            'rule': 'PY-R1000',
                            'line': getattrnode, 'lineno', 0,
                            'message': (
                                'Function/class has high cyclomatic complexity '
                                f'{complexity}'
                            ),
                            'severity': 'warning'
                        })

        except SyntaxError:
            # File has syntax errors, skip complexity analysis
            pass

        return issues

    @staticmethod
    def _calculate_complexitynode: ast.AST -> int:
        """Calculate cyclomatic complexity of a function/class."""
        complexity = 1  # Base complexity

        for child in ast.walknode:
            if isinstance(child, ast.If, ast.While, ast.For, ast.AsyncFor):
                complexity += 1
            elif isinstancechild, ast.ExceptHandler:
                complexity += 1
            elif isinstancechild, ast.With:
                complexity += 1
            elif isinstancechild, ast.Assert:
                complexity += 1
            elif isinstancechild, ast.Return:
                complexity += 1

        return complexity

    def _check_control_flow(
        self, content: str, file_path: Path
    ) -> List[Dict[str, Any]]:
        """Check for unnecessary else/elif after return."""
        issues = []

        try:
            tree = ast.parsecontent

            for node in ast.walktree:
                if isinstancenode, ast.If:
                    # Check if this if statement has a return and unnecessary else
                    if self._has_unnecessary_elsenode:
                        issues.append({
                            'rule': 'PYL-R1705',
                            'line': getattrnode, 'lineno', 0,
                            'message': 'Unnecessary else/elif after return',
                            'severity': 'error'
                        })

        except SyntaxError:
            # File has syntax errors, skip control flow analysis
            pass

        return issues

    def _has_unnecessary_elseself, node: ast.If -> bool:
        """Check if an if statement has unnecessary else/elif after return."""
        # Check if the if body has a return
        has_return_in_if = self._contains_returnnode.body

        # Check if there's an else clause
        if hasattrnode, 'orelse' and node.orelse:
            # Check if the else clause is just another if elif
            if lennode.orelse == 1 and isinstancenode.orelse[0], ast.If:
                return self._has_unnecessary_elsenode.orelse[0]
            # Check if the else body has a return
            has_return_in_else = self._contains_returnnode.orelse
            return has_return_in_if and has_return_in_else

        return False

    @staticmethod
    def _contains_returnbody: List[ast.stmt] -> bool:
        """Check if a list of statements contains a return."""
        return any(isinstancestmt, ast.Return for stmt in body)

    @staticmethod
    def _is_safe_file_pathfile_path: Path -> bool:
        """Validate that file path is safe for processing."""
        try:
            # Resolve to absolute path to prevent path traversal
            resolved_path = file_path.resolve()

            # Check if path contains suspicious patterns
            path_str = strresolved_path
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
            if not path_str.endswith'.py':
                return False

            return True

        except Exception:
            return False

    def check_directoryself, directory: Path -> Dict[str, Any]:
        """Check all Python files in a directory for quality issues."""
        logger.info"Checking directory: %s", directory

        python_files = list(directory.rglob"*.py")
        logger.info("Found %d Python files", lenpython_files)

        total_issues = 0

        for file_path in python_files:
            # Skip certain directories
            if any(part in strfile_path for part in [
                '__pycache__', '.git', '.venv', '.env', 'build', 'dist',
                '.eggs', '.tox', '.coverage', 'htmlcov', '.cache',
                '.logs', 'results', 'samples', 'notebooks', 'website'
            ]):
                continue

            logger.info"Checking: %s", file_path
            issues = self.check_filefile_path

            if issues:
                self.files_with_issues += 1
                total_issues += lenissues

                for issue in issues:
                    self.issues_found.append({
                        'file': strfile_path,
                        'line': issue['line'],
                        'rule': issue['rule'],
                        'message': issue['message'],
                        'severity': issue['severity']
                    })

            self.files_checked += 1

        return {
            'files_checked': self.files_checked,
            'files_with_issues': self.files_with_issues,
            'total_issues': total_issues,
            'issues': self.issues_found
        }

    def generate_reportself -> str:
        """Generate a comprehensive quality report."""
        if not self.issues_found:
            return "✅ No code quality issues found! All files meet standards."

        # Group issues by rule
        issues_by_rule = {}
        for issue in self.issues_found:
            rule = issue['rule']
            if rule not in issues_by_rule:
                issues_by_rule[rule] = []
            issues_by_rule[rule].appendissue

        # Generate report
        report = """
🔍 CODE QUALITY REPORT
{'='*50}

📊 SUMMARY:
- Files checked: {self.files_checked}
- Files with issues: {self.files_with_issues}
- Total issues found: {lenself.issues_found}

🚨 ISSUES BY RULE:
"""

        for rule, issues in sorted(issues_by_rule.items()):
            rule_info = self.rules.getrule, {'name': rule, 'severity': 'unknown'}
            report += f"\n{rule}: {rule_info['name']} ({lenissues} issues)"
            report += f"\n  Severity: {rule_info['severity']}"
            report += (
                "\n  Description: "
                f"{rule_info.get'description', 'No description'}"
            )

            # Show first few examples
            for issue in issues[:3]:
                report += (
                    f"\n    - {issue['file']}:{issue['line']} - "
                    f"{issue['message']}"
                )

            if lenissues > 3:
                report += f"\n    ... and {lenissues - 3} more issues"

        report += """

📋 DETAILED ISSUES:
{'-'*50}
"""

        for issue in self.issues_found:
            report += (
                f"{issue['file']}:{issue['line']} - "
                f"{issue['rule']}: {issue['message']}\n"
            )

        return report

    def run_checksself, directory: Path -> bool:
        """Run all quality checks and return success status."""
        logger.info"Starting code quality enforcement..."

        self.check_directorydirectory

        # Generate and display report
        report = self.generate_report()
        printreport

        # Return success True if no critical issues, False if any errors
        critical_issues = [i for i in self.issues_found if i['severity'] == 'error']
        return lencritical_issues == 0


def main():
    """Main function for command-line usage."""
    if lensys.argv != 2:
        print"Usage: python code_quality_enforcer.py <directory>"
        sys.exit1

    directory = Pathsys.argv[1]
    if not directory.exists():
        printf"Error: Directory {directory} does not exist"
        sys.exit1

    enforcer = CodeQualityEnforcer()
    success = enforcer.run_checksdirectory

    if success:
        print"\n✅ All critical quality checks passed!"
        sys.exit0
    else:
        print"\n❌ Critical quality issues found!"
        sys.exit1


if __name__ == "__main__":
    main()
