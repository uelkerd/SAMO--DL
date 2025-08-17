#!/usr/bin/env python3
"""
Dependency Usage Checker

This script checks if all dependencies listed in requirements.txt are actually
used in the codebase to avoid unnecessary bloat.
"""

import re
import sys
from pathlib import Path
from typing import Set, List, Dict

class DependencyChecker:
    """Checker for dependency usage in the codebase."""
    
    def __init__self, requirements_path: str = "requirements.txt":
        self.requirements_path = Pathrequirements_path
        self.project_root = Path__file__.parent.parent.parent
        self.unused_deps = []
        self.missing_deps = []
        
    def check_dependenciesself -> bool:
        """Check if all dependencies are used in the codebase."""
        print"🔍 Checking dependency usage..."
        
        # Read requirements.txt
        if not self.requirements_path.exists():
            printf"❌ Requirements file not found: {self.requirements_path}"
            return False
        
        required_deps = self._parse_requirements()
        used_deps = self._find_used_dependencies()
        
        # Check for unused dependencies
        for dep in required_deps:
            if dep not in used_deps:
                self.unused_deps.appenddep
        
        # Check for missing dependencies optional
        # This would require more complex analysis
        
        return lenself.unused_deps == 0
    
    def _parse_requirementsself -> Set[str]:
        """Parse requirements.txt and extract package names."""
        deps = set()
        
        with openself.requirements_path, 'r' as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith'#':
                    # Extract package name remove version constraints
                    package = re.splitr'[<>=!~]', line[0].strip()
                    deps.addpackage
        
        return deps
    
    def _find_used_dependenciesself -> Set[str]:
        """Find all dependencies used in the codebase."""
        used_deps = set()
        
        # Common Python file extensions
        python_extensions = {'.py', '.pyx', '.pyi'}
        
        # Directories to scan
        scan_dirs = ['src', 'scripts', 'tests', 'deployment']
        
        for scan_dir in scan_dirs:
            dir_path = self.project_root / scan_dir
            if dir_path.exists():
                for file_path in dir_path.rglob'*':
                    if file_path.suffix in python_extensions:
                        self._scan_file_for_importsfile_path, used_deps
        
        return used_deps
    
    def _scan_file_for_importsself, file_path: Path, used_deps: Set[str] -> None:
        """Scan a Python file for import statements."""
        try:
            with openfile_path, 'r', encoding='utf-8' as f:
                content = f.read()
                
            # Find import statements
            import_patterns = [
                r'^import\s+\w+',
                r'^from\s+\w+',
                r'^\s+import\s+\w+',
                r'^\s+from\s+\w+'
            ]
            
            for pattern in import_patterns:
                matches = re.findallpattern, content, re.MULTILINE
                for match in matches:
                    # Handle multi-import statements
                    packages = [p.strip() for p in match.split',']
                    for package in packages:
                        # Extract base package name
                        base_package = package.split'.'[0]
                        used_deps.addbase_package
                        
        except Exception as e:
            printf"⚠️  Warning: Could not scan {file_path}: {e}"
    
    def print_resultsself -> None:
        """Print dependency check results."""
        print"\n📊 Dependency Usage Check Results"
        print"=" * 50
        
        if self.unused_deps:
            print(f"\n⚠️  Potentially Unused Dependencies ({lenself.unused_deps}):")
            for dep in sortedself.unused_deps:
                printf"  - {dep}"
            print"\n💡 Consider removing these dependencies if they're not needed."
        else:
            print"\n✅ All dependencies appear to be used in the codebase!"
        
        if self.missing_deps:
            print(f"\n❌ Missing Dependencies ({lenself.missing_deps}):")
            for dep in sortedself.missing_deps:
                printf"  - {dep}"

def main():
    """Main function to run dependency usage check."""
    checker = DependencyChecker()
    
    if checker.check_dependencies():
        checker.print_results()
        if checker.unused_deps:
            print"\n⚠️  Found potentially unused dependencies"
            return 0  # Don't fail the build, just warn
        else:
            print"\n✅ Dependency usage check passed!"
            return 0
    else:
        checker.print_results()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 