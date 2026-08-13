from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from ct_vascular_resampling.output_audit import audit_output


CORE_HASH = "d" * 64
BUILD_COMMIT = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _valid_output(tmp_path: Path) -> Path:
    case = tmp_path / "case_2"
    gallery = case / "gallery"
    image = Image.new("RGB", (3, 3), "white")
    vessel = image.copy()
    vessel.putpixel((1, 1), (255, 0, 0))
    for directory, rendered in {
        "ct": image,
        "boundary_only": image,
        "ct_overlay": image,
        "organ_vessel_boundary": image,
        "eus_vessel_boundary": vessel,
        "ct_eus_vessel_overlay": vessel,
    }.items():
        destination = gallery / directory / "sample.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination)

    record = {
        "slice_id": "stomach-000000-rp000-pp000-yp000",
        "status": "gallery",
        "organ": "stomach",
        "coordinate_system": "RAS",
        "core_design_sha256": CORE_HASH,
        "build_git_commit": BUILD_COMMIT,
        "source_region": "stomach",
        "yaw_policy": "standard",
        "angles_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "ct_png": "ct/sample.png",
        "boundary_only_png": "boundary_only/sample.png",
        "ct_overlay_png": "ct_overlay/sample.png",
        "organ_vessel_boundary_png": "organ_vessel_boundary/sample.png",
        "organ_metadata_schema_version": "eus-organ-metadata/v1",
        "eus_vessel_boundary_png": "eus_vessel_boundary/sample.png",
        "ct_eus_vessel_overlay_png": "ct_eus_vessel_overlay/sample.png",
        "eus_vessel_metadata_schema_version": "eus-vessel-metadata/v1",
        "organ_labels": ["aorta"],
        "eus_candidate_organ_labels": ["aorta"],
        "eus_vessel_labels": ["aorta"],
        "eus_vessel_features": [
            {"label": "aorta", "x_mm": 1.0, "y_mm": 1.0, "area_mm2": 1.0}
        ],
        "features": [],
    }
    _write_jsonl(case / "manifest.jsonl", [record])
    _write_jsonl(gallery / "gallery.jsonl", [record])
    _write_json(
        case / "run_metadata.json",
        {
            "run_state": "complete",
            "total_squares": 1,
            "completed_pose_count": 1,
            "status_counts": {"gallery": 1},
            "core_design_filename": "基于目标器官的采样方法-20260813.docx",
            "core_design_sha256": CORE_HASH,
            "build_git_commit": BUILD_COMMIT,
            "pose_angles_degrees": {
                "roll": [0.0],
                "pitch": [0.0],
                "yaw": {"standard": [0.0], "duodenum_bulb": [0.0], "pancreas_special": [0.0]},
            },
            "quality_filtering": {"black_ratio_limit": 0.6},
            "manual_segmentation": {
                "eus_vessel_colors": {
                    "aorta": [255, 0, 0],
                    "inferior_vena_cava": [0, 0, 255],
                    "portal_vein": [170, 85, 255],
                }
            },
        },
    )
    _write_json(
        case / "library_summary.json",
        {
            "indexed_feature_count": 1,
            "organ_label_counts": {"aorta": 1},
            "eus_candidate_organ_label_counts": {"aorta": 1},
            "eus_vessel_label_counts": {"aorta": 1},
            "eus_vessel_feature_counts": {"aorta": 1},
            "eus_vessel_colors": {
                "aorta": [255, 0, 0],
                "inferior_vena_cava": [0, 0, 255],
                "portal_vein": [170, 85, 255],
            },
        },
    )
    return case


def test_audit_output_accepts_consistent_structure_and_pixels(tmp_path):
    report = audit_output(_valid_output(tmp_path), check_pixels=True)

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["manifest_lines"] == 1
    assert report["gallery_lines"] == 1
    assert report["pixel_audit"]["frames_with_color"] == {"aorta": 1}


def test_audit_output_rejects_feature_without_boundary_color(tmp_path):
    case = _valid_output(tmp_path)
    Image.new("RGB", (3, 3), "white").save(
        case / "gallery" / "eus_vessel_boundary" / "sample.png"
    )

    report = audit_output(case, check_pixels=True)

    assert report["passed"] is False
    assert report["pixel_audit"]["feature_without_color"] == {"aorta": 1}
    assert any("feature_without_color" in error for error in report["errors"])


def test_audit_cli_writes_report_and_returns_nonzero_for_failure(tmp_path):
    case = _valid_output(tmp_path)
    Image.new("RGB", (3, 3), "white").save(
        case / "gallery" / "eus_vessel_boundary" / "sample.png"
    )
    report_path = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_resampling_output.py",
            str(case),
            "--report",
            str(report_path),
            "--check-pixels",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False
    assert not (tmp_path / ".audit.json.tmp").exists()
