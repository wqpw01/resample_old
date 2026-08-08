from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_main_cli_exposes_case_config_dry_run_and_steps():
    project_root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--case-config" in result.stdout
    assert "--auto-case-config" in result.stdout
    assert "--rejected-audit-config" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--steps" in result.stdout
    assert "--backend" in result.stdout
    assert "--gpu-device" in result.stdout
    assert "--gpu-batch-size" in result.stdout


def test_interrupted_metadata_recovery_cli_exposes_required_evidence_arguments():
    project_root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/recover_interrupted_run_metadata.py", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--case-config" in result.stdout
    assert "--expected-completed-count" in result.stdout
    assert "--reason" in result.stdout
    assert "--exit-code" in result.stdout
