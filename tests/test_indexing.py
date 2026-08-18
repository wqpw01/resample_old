import json

import pytest

from ct_vascular_resampling.config import ManualSegmentationConfig
from ct_vascular_resampling.eus_organs import EUS_ORGAN_METADATA_SCHEMA_VERSION
from ct_vascular_resampling.indexing import build_library_summary
from ct_vascular_resampling.manual_segmentation import EUS_VESSEL_METADATA_SCHEMA_VERSION


def test_internal_index_builds_summary_without_external_registration_module(tmp_path):
    gallery = tmp_path / "case" / "gallery"
    gallery.mkdir(parents=True)
    for relative in ("organ.png", "eus.png", "overlay.png"):
        (gallery / relative).touch()
    record = {
        "status": "gallery",
        "features": [
            {"label": "artery", "x_mm": 1.0, "y_mm": 2.0, "area_mm2": 3.0},
            {"label": "vein", "x_mm": 4.0, "y_mm": 5.0, "area_mm2": 6.0},
        ],
        "organ_metadata_schema_version": EUS_ORGAN_METADATA_SCHEMA_VERSION,
        "organ_vessel_boundary_png": "organ.png",
        "organ_labels": ["aorta", "liver"],
        "eus_candidate_organ_labels": ["aorta", "liver"],
        "eus_vessel_metadata_schema_version": EUS_VESSEL_METADATA_SCHEMA_VERSION,
        "eus_vessel_labels": ["aorta"],
        "eus_vessel_features": [
            {"label": "aorta", "x_mm": 1.0, "y_mm": 2.0, "area_mm2": 3.0}
        ],
        "eus_vessel_boundary_png": "eus.png",
        "ct_eus_vessel_overlay_png": "overlay.png",
    }
    (gallery / "gallery.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manual = ManualSegmentationConfig(
        path=tmp_path / "segmentation.nrrd",
        organ_label_values={"aorta": (8,), "liver": (6,)},
        eus_vessel_label_values={
            "aorta": (8,),
            "inferior_vena_cava": (9,),
            "portal_vein": (26,),
        },
        eus_vessel_colors={
            "aorta": (255, 0, 0),
            "inferior_vena_cava": (0, 0, 255),
            "portal_vein": (170, 85, 255),
        },
    )

    summary = build_library_summary(
        case_id="case",
        case_directory=tmp_path / "case",
        manual_segmentation=manual,
    )

    assert summary["indexed_feature_count"] == 1
    assert summary["feature_label_counts"] == {"artery": 1, "vein": 1}
    assert summary["organ_label_counts"] == {"aorta": 1, "liver": 1}
    assert summary["eus_vessel_feature_counts"] == {"aorta": 1}


def test_internal_index_rejects_invalid_original_vessel_feature(tmp_path):
    gallery = tmp_path / "case" / "gallery"
    gallery.mkdir(parents=True)
    for relative in ("organ.png", "eus.png", "overlay.png"):
        (gallery / relative).touch()
    record = {
        "status": "gallery",
        "features": [{"label": "artery", "x_mm": 1.0, "y_mm": 2.0, "area_mm2": -1.0}],
        "organ_metadata_schema_version": EUS_ORGAN_METADATA_SCHEMA_VERSION,
        "organ_vessel_boundary_png": "organ.png",
        "organ_labels": [],
        "eus_candidate_organ_labels": [],
        "eus_vessel_metadata_schema_version": EUS_VESSEL_METADATA_SCHEMA_VERSION,
        "eus_vessel_labels": [],
        "eus_vessel_features": [],
        "eus_vessel_boundary_png": "eus.png",
        "ct_eus_vessel_overlay_png": "overlay.png",
    }
    (gallery / "gallery.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manual = ManualSegmentationConfig(
        path=tmp_path / "segmentation.nrrd",
        organ_label_values={},
        eus_vessel_label_values={"aorta": (8,), "inferior_vena_cava": (9,), "portal_vein": (26,)},
        eus_vessel_colors={"aorta": (255, 0, 0), "inferior_vena_cava": (0, 0, 255), "portal_vein": (170, 85, 255)},
    )

    with pytest.raises(ValueError, match="features|area_mm2|正数"):
        build_library_summary(
            case_id="case",
            case_directory=tmp_path / "case",
            manual_segmentation=manual,
        )
