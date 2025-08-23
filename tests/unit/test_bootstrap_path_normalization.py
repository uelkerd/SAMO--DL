#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import importlib
import types
import pytest


def _simulate_script_sys_path_adjustment(tmp_path):
    # Simulate project structure: tmp/project_root/src
    project_root = tmp_path / "project_root"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)

    # Create a dummy module inside src
    dummy_mod = src_dir / "dummy_module.py"
    dummy_mod.write_text("value = 42\n", encoding="utf-8")

    # Simulate script two levels down: scripts/testing/debug_rate_limiter.py style
    scripts_dir = project_root / "scripts" / "testing"
    scripts_dir.mkdir(parents=True)

    script = scripts_dir / "fake_script.py"
    script.write_text(
        (
            "import os, sys\n"
            "CURRENT_DIR = os.path.dirname(__file__)\n"
            "PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..'))\n"
            "if PROJECT_ROOT not in sys.path:\n"
            "    sys.path.insert(0, PROJECT_ROOT)\n"
            "import importlib\n"
            "m = importlib.import_module('src.dummy_module')\n"
            "print(m.value)\n"
        ),
        encoding="utf-8",
    )

    return project_root, script


@pytest.mark.parametrize("levels", [2])
def test_bootstrap_path_normalization(tmp_path, capsys, levels):
    project_root, script = _simulate_script_sys_path_adjustment(tmp_path)

    # Execute the script in a clean interpreter context
    env = os.environ.copy()
    cmd = (
        f"{sys.executable} -c \"import runpy; runpy.run_path(r'{str(script)}', run_name='__main__')\""
    )
    import subprocess

    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(project_root))
    assert proc.returncode == 0, proc.stderr
    assert "42" in proc.stdout