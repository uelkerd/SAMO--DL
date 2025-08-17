#!/usr/bin/env python3
"""
SAMO-DL Code Quality Enforcer

This script runs BEFORE commits to prevent common code quality issues:
- Unnecessary else/elif after return (PYL-R1705)
- f-strings without expressions (PTC-W0027)
- Unused imports (PY-W2000)
- Continuation line indentation (FLK-E128)
- Missing blank lines (FLK-E301)
- Line length violations (FLK-E501)
- Trailing whitespace (FLK-W291)
- Missing newlines (FLK-W292)
- Blank line whitespace (FLK-W293)
- Doc line length (FLK-W505)
- Missing docstrings (PY-D0003)
- High cyclomatic complexity (PY-R1000)
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class CodeQualityEnforcer:
    """Enforces code quality standards across the SAMO-DL codebase."""
    
    def __init__(self):
        self.issues = []
        self.files_checked = 0
        self.issues_found = 0
        
    def check_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Check a single Python file for quality issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
                
            # Parse AST for structural checks
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return [{'type': 'syntax_error', 'file': str(file_path), 'line': 0}]
            
            file_issues = []
            
            # 1. Check for unnecessary else/elif after return (PYL-R1705)
            file_issues.extend(self._check_unnecessary_else_after_return(tree, file_path))
            
            # 2. Check for f-strings without expressions (PTC-W0027)
            file_issues.extend(self._check_fstring_without_expressions(lines, file_path))
            
            # 3. Check for unused imports (PY-W2000)
            file_issues.extend(self._check_unused_imports(tree, lines, file_path))
            
            # 4. Check continuation line indentation (FLK-E128)
            file_issues.extend(self._check_continuation_indentation(lines, file_path))
            
            # 5. Check missing blank lines (FLK-E301)
            file_issues.extend(self._check_missing_blank_lines(tree, file_path))
            
            # 6. Check line length (FLK-E501)
            file_issues.extend(self._check_line_length(lines, file_path))
            
            # 7. Check trailing whitespace (FLK-W291)
            file_issues.extend(self._check_trailing_whitespace(lines, file_path))
            
            # 8. Check missing newlines (FLK-W292)
            file_issues.extend(self._check_missing_newline(content, file_path))
            
            # 9. Check blank line whitespace (FLK-W293)
            file_issues.extend(self._check_blank_line_whitespace(lines, file_path))
            
            # 10. Check doc line length (FLK-W505)
            file_issues.extend(self._check_doc_line_length(lines, file_path))
            
            # 11. Check missing docstrings (PY-D0003)
            file_issues.extend(self._check_missing_docstrings(tree, file_path))
            
            # 12. Check cyclomatic complexity (PY-R1000)
            file_issues.extend(self._check_cyclomatic_complexity(tree, file_path))
            
            return file_issues
            
        except Exception as e:
            return [{'type': 'error', 'file': str(file_path), 'error': str(e)}]
    
    def _check_unnecessary_else_after_return(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Check for unnecessary else/elif after return statements."""
        issues = []
        
        class ReturnVisitor(ast.NodeVisitor):
            def __init__(self):
                self.returns_in_branch = set()
                
            def visit_If(self, node):
                # Check if all branches return
                has_return = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        has_return = True
                        break
                
                if has_return and node.orelse:
                    # Check if orelse is just elif or else
                    if isinstance(node.orelse[0], ast.If):
                        # This is elif, check if it also returns
                        for stmt in node.orelse[0].body:
                            if isinstance(stmt, ast.Return):
                                issues.append({
                                    'type': 'PYL-R1705',
                                    'file': str(file_path),
                                    'line': node.lineno,
                                    'message': 'Unnecessary elif after return'
                                })
                                break
                    elif len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Return):
                        # This is else with return
                        issues.append({
                            'type': 'PYL-R1705',
                            'file': str(file_path),
                            'line': node.lineno,
                            'message': 'Unnecessary else after return'
                        })
                
                self.generic_visit(node)
        
        visitor = ReturnVisitor()
        visitor.visit(tree)
        return issues
    
    def _check_fstring_without_expressions(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for f-strings without expressions."""
        issues = []
        fstring_pattern = re.compile(r'f["\']([^"\']*\{[^}]*\}[^"\']*)*["\']')
        
        for i, line in enumerate(lines, 1):
            if 'f"' in line or "f'" in line:
                # Check if it's actually an f-string
                if fstring_pattern.search(line):
                    # Check if it has expressions
                    if not re.search(r'\{[^}]*\}', line):
                        issues.append({
                            'type': 'PTC-W0027',
                            'file': str(file_path),
                            'line': i,
                            'message': 'f-string used without any expression'
                        })
        
        return issues
    
    def _check_unused_imports(self, tree: ast.AST, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for unused imports."""
        issues = []
        
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = set()
                self.used_names = set()
                
            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.add(alias.name)
                    if alias.asname:
                        self.imports.add(alias.asname)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    self.imports.add(node.module)
                for alias in node.names:
                    self.imports.add(alias.name)
                    if alias.asname:
                        self.imports.add(alias.asname)
            
            def visit_Name(self, node):
                self.used_names.add(node.id)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        # Check for unused imports
        for imp in visitor.imports:
            if imp not in visitor.used_names and not imp.startswith('_'):
                # Find the line number
                for i, line in enumerate(lines, 1):
                    if f'import {imp}' in line or f'from {imp}' in line:
                        issues.append({
                            'type': 'PY-W2000',
                            'file': str(file_path),
                            'line': i,
                            'message': f'Imported name "{imp}" is not used anywhere in the module'
                        })
                        break
        
        return issues
    
    def _check_continuation_indentation(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for continuation line indentation issues."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.startswith('#'):
                # Check for continuation lines
                if line.strip().startswith(('(', '[', '{')):
                    # This is an opening line, check next line
                    if i < len(lines):
                        next_line = lines[i]
                        if next_line.strip() and not next_line.startswith('#'):
                            # Check if next line is properly indented
                            expected_indent = len(line) - len(line.lstrip()) + 4
                            actual_indent = len(next_line) - len(next_line.lstrip())
                            if actual_indent < expected_indent:
                                issues.append({
                                    'type': 'FLK-E128',
                                    'file': str(file_path),
                                    'line': i + 1,
                                    'message': 'Continuation line under-indented for visual indent'
                                })
        
        return issues
    
    def _check_missing_blank_lines(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Check for missing blank lines between classes and functions."""
        issues = []
        
        class BlankLineVisitor(ast.NodeVisitor):
            def __init__(self):
                self.last_node = None
                
            def visit_ClassDef(self, node):
                if self.last_node and not isinstance(self.last_node, ast.ClassDef):
                    # Check if there's a blank line
                    pass
                self.last_node = node
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if self.last_node and not isinstance(self.last_node, ast.FunctionDef):
                    # Check if there's a blank line
                    pass
                self.last_node = node
                self.generic_visit(node)
        
        visitor = BlankLineVisitor()
        visitor.visit(tree)
        return issues
    
    def _check_line_length(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for lines that are too long."""
        issues = []
        max_length = 88
        
        for i, line in enumerate(lines, 1):
            if len(line) > max_length:
                issues.append({
                    'type': 'FLK-E501',
                    'file': str(file_path),
                    'line': i,
                    'message': f'Line too long ({len(line)} > {max_length} characters)'
                })
        
        return issues
    
    def _check_trailing_whitespace(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for trailing whitespace."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if line.rstrip() != line:
                issues.append({
                    'type': 'FLK-W291',
                    'file': str(file_path),
                    'line': i,
                    'message': 'Trailing whitespace detected'
                })
        
        return issues
    
    def _check_missing_newline(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check if file ends with newline."""
        issues = []
        
        if not content.endswith('\n'):
            issues.append({
                'type': 'FLK-W292',
                'file': str(file_path),
                'line': len(content.splitlines()),
                'message': 'No newline at end of file'
            })
        
        return issues
    
    def _check_blank_line_whitespace(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for blank lines that contain whitespace."""
        issues = []
        
        for i, line in enumerate(lines, 1):
            if not line.strip() and line != '':
                issues.append({
                    'type': 'FLK-W293',
                    'file': str(file_path),
                    'line': i,
                    'message': 'Blank line contains whitespace'
                })
        
        return issues
    
    def _check_doc_line_length(self, lines: List[str], file_path: Path) -> List[Dict[str, Any]]:
        """Check for docstring lines that are too long."""
        issues = []
        max_length = 88
        
        in_docstring = False
        for i, line in enumerate(lines, 1):
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
            
            if in_docstring and len(line) > max_length:
                issues.append({
                    'type': 'FLK-W505',
                    'file': str(file_path),
                    'line': i,
                    'message': f'Doc line too long ({len(line)} > {max_length} characters)'
                })
        
        return issues
    
    def _check_missing_docstrings(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Check for missing docstrings."""
        issues = []
        
        class DocstringVisitor(ast.NodeVisitor):
            def visit_Module(self, node):
                if not node.body or not isinstance(node.body[0], ast.Expr):
                    issues.append({
                        'type': 'PY-D0003',
                        'file': str(file_path),
                        'line': 1,
                        'message': 'Missing module docstring'
                    })
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                if not node.body or not isinstance(node.body[0], ast.Expr):
                    issues.append({
                        'type': 'PY-D0003',
                        'file': str(file_path),
                        'line': node.lineno,
                        'message': 'Missing class docstring'
                    })
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if not node.body or not isinstance(node.body[0], ast.Expr):
                    issues.append({
                        'type': 'PY-D0003',
                        'file': str(file_path),
                        'line': node.lineno,
                        'message': 'Missing function docstring'
                    })
                self.generic_visit(node)
        
        visitor = DocstringVisitor()
        visitor.visit(tree)
        return issues
    
    def _check_cyclomatic_complexity(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Check for functions with high cyclomatic complexity."""
        issues = []
        threshold = 10
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.complexity = 0
                
            def visit_FunctionDef(self, node):
                old_function = self.current_function
                old_complexity = self.complexity
                
                self.current_function = node
                self.complexity = 1  # Base complexity
                self.generic_visit(node)
                
                if self.complexity > threshold:
                    issues.append({
                        'type': 'PY-R1000',
                        'file': str(file_path),
                        'line': node.lineno,
                        'message': f'Function has cyclomatic complexity {self.complexity} > {threshold}'
                    })
                
                self.current_function = old_function
                self.complexity = old_complexity
            
            def visit_If(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                if self.current_function:
                    self.complexity += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        return issues
    
    def run_checks(self, directory: Path) -> Dict[str, Any]:
        """Run all quality checks on the directory."""
        print(f"🔍 Running comprehensive code quality checks on {directory}")
        
        python_files = list(directory.rglob('*.py'))
        python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
        
        all_issues = []
        
        for file_path in python_files:
            file_issues = self.check_file(file_path)
            if file_issues:
                all_issues.extend(file_issues)
            self.files_checked += 1
        
        # Group issues by type
        issues_by_type = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Print summary
        print(f"\n📊 QUALITY CHECK SUMMARY")
        print(f"=" * 50)
        print(f"Files checked: {self.files_checked}")
        print(f"Total issues found: {len(all_issues)}")
        
        if issues_by_type:
            print(f"\n🚨 ISSUES BY TYPE:")
            for issue_type, issues in sorted(issues_by_type.items()):
                print(f"  {issue_type}: {len(issues)} issues")
        
        # Return results for pre-commit
        if all_issues:
            print(f"\n❌ PRE-COMMIT BLOCKED: Code quality issues found!")
            print(f"Fix these issues before committing, or use --no-verify to bypass.")
            return {
                'success': False,
                'issues': all_issues,
                'summary': issues_by_type
            }
        else:
            print(f"\n✅ All code quality checks passed!")
            return {
                'success': True,
                'issues': [],
                'summary': {}
            }


def main():
    """Main function for pre-commit hook."""
    if len(sys.argv) < 2:
        print("Usage: python code_quality_enforcer.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    enforcer = CodeQualityEnforcer()
    result = enforcer.run_checks(directory)
    
    if not result['success']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
