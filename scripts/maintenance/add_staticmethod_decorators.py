#!/usr/bin/env python3
"""
Script to add @staticmethod decorators to methods that don't use self.
"""

import ast
import re
from pathlib import Path
from typing import List, Set


class StaticMethodVisitor(ast.NodeVisitor):
    """AST visitor to find methods that should be static."""

    def __init__(self):
        self.static_methods = []
        self.current_class = None

    def visit_ClassDef(self, node):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definitions within classes."""
        if self.current_class and not node.name.startswith("_"):
            # Check if method uses self
            if self._should_be_static(node):
                self.static_methods.append(
                    {"class": self.current_class, "method": node.name, "line": node.lineno}
                )

    def _should_be_static(self, node) -> bool:
        """Check if a method should be static."""
        # Skip if it has decorators already
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in [
                "staticmethod",
                "classmethod",
                "property",
            ]:
                return False
            if isinstance(decorator, ast.Attribute) and decorator.attr in [
                "staticmethod",
                "classmethod",
                "property",
            ]:
                return False

        # Skip special methods
        if node.name.startswith("_"):
            return False

        # Skip if no parameters or first parameter is not 'self'
        if not node.args.args or node.args.args[0].arg != "self":
            return False

        # Check if method uses self
        uses_self = False
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "self"
            ):
                uses_self = True
                break
            if (
                isinstance(child, ast.Name)
                and child.id == "self"
                and isinstance(child.ctx, ast.Load)
            ):
                uses_self = True
                break

        return not uses_self


def find_static_methods(file_path: Path) -> List[dict]:
    """Find methods that should be static in a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        visitor = StaticMethodVisitor()
        visitor.visit(tree)

        return visitor.static_methods
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def add_staticmethod_decorators(file_path: Path, static_methods: List[dict]):
    """Add @staticmethod decorators to the specified methods."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Sort by line number in reverse order to avoid line number shifts
        static_methods.sort(key=lambda x: x["line"], reverse=True)

        modified = False
        for method_info in static_methods:
            line_idx = method_info["line"] - 1  # Convert to 0-based index

            if line_idx < len(lines):
                # Find the indentation of the method
                method_line = lines[line_idx]
                indent = len(method_line) - len(method_line.lstrip())

                # Add @staticmethod decorator
                decorator_line = " " * indent + "@staticmethod\n"
                lines.insert(line_idx, decorator_line)
                modified = True
                print(
                    f"Added @staticmethod to {method_info['class']}.{method_info['method']} in {file_path}"
                )

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

    except Exception as e:
        print(f"Error modifying {file_path}: {e}")


def process_files(root_dir: Path, file_patterns: List[str] = None):
    """Process Python files to add @staticmethod decorators."""
    if file_patterns is None:
        file_patterns = ["**/*.py"]

    for pattern in file_patterns:
        for file_path in root_dir.glob(pattern):
            if file_path.name == "__init__.py":
                continue

            # Skip certain directories
            if any(part in str(file_path) for part in [".git", "__pycache__", ".pytest_cache"]):
                continue

            static_methods = find_static_methods(file_path)
            if static_methods:
                add_staticmethod_decorators(file_path, static_methods)


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    print("Adding @staticmethod decorators to methods that don't use self...")

    # Focus on test files first (where most issues were found)
    test_patterns = ["tests/**/*.py"]
    process_files(root_dir, test_patterns)

    # Then process other files
    other_patterns = ["src/**/*.py", "deployment/**/*.py"]
    process_files(root_dir, other_patterns)

    print("Done!")
