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


def test_manual_preprocessing_cli_exposes_inputs_and_explicit_overwrite():
    project_root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/preprocess_manual_segmentation_case.py", "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    for option in (
        "--ct",
        "--segmentation",
        "--artery-model",
        "--vein-model",
        "--output",
        "--registration-module",
        "--output-root",
        "--case-id",
        "--overwrite",
    ):
        assert option in result.stdout


def test_manual_preprocessing_cli_refuses_nonempty_output_without_overwrite(tmp_path):
    from scripts.preprocess_manual_segmentation_case import main

    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    exit_code = main(
        [
            "--ct",
            str(tmp_path / "ct.nrrd"),
            "--segmentation",
            str(tmp_path / "seg.nrrd"),
            "--artery-model",
            str(tmp_path / "artery.ply"),
            "--vein-model",
            str(tmp_path / "vein.ply"),
            "--output",
            str(output),
            "--registration-module",
            str(tmp_path / "2021.py"),
            "--output-root",
            str(tmp_path / "gallery"),
            "--case-id",
            "case_2",
        ]
    )

    assert exit_code == 1
    assert sentinel.read_text(encoding="utf-8") == "keep"
