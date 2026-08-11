from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ct_vascular_resampling.config import ManualSegmentationConfig
from ct_vascular_resampling.geometry import SectionContour
from ct_vascular_resampling.manual_segmentation import (
    analyze_manual_label_plane,
    apply_manual_label_analysis,
)
from ct_vascular_resampling.rendering import VesselLayer, render_sample_images


CONFIG = ManualSegmentationConfig(
    path=Path("unused.seg.nrrd"),
    organ_label_values={
        "spleen": (1,),
        "kidney_right": (2,),
        "kidney_left": (3,),
        "gallbladder": (4,),
        "esophagus": (5,),
        "liver": (6,),
        "stomach": (7,),
        "aorta": (8,),
        "inferior_vena_cava": (9,),
        "pancreas": (11,),
        "adrenal_gland_right": (12,),
        "adrenal_gland_left": (13,),
        "duodenum": (14,),
        "portal_vein": (23, 26, 33, 34, 35, 36, 37),
    },
    eus_vessel_label_values={
        "aorta": (8,),
        "inferior_vena_cava": (9,),
        "portal_vein": (26, 33, 34, 35, 36, 37),
    },
    eus_vessel_colors={
        "aorta": (255, 0, 0),
        "inferior_vena_cava": (0, 0, 255),
        "portal_vein": (170, 85, 255),
    },
)


def _colors(image: np.ndarray) -> set[tuple[int, int, int]]:
    return set(map(tuple, image.reshape(-1, 3)))


def test_one_pixel_and_full_frame_organs_are_labels_without_artificial_frame():
    labels = np.full((9, 9), 6, dtype=np.uint8)
    labels[4, 4] = 11

    result = analyze_manual_label_plane(labels, 100.0, 100.0, CONFIG)

    assert result.organ_labels == ["liver", "pancreas"]
    assert np.all(result.organ_boundary_rgb[0] == 255)
    assert np.all(result.organ_boundary_rgb[-1] == 255)
    assert np.all(result.organ_boundary_rgb[:, 0] == 255)
    assert np.all(result.organ_boundary_rgb[:, -1] == 255)
    assert np.any(result.organ_boundary_rgb[3:6, 3:6] != 255)


def test_absent_tangent_label_does_not_appear_without_sampled_pixels():
    result = analyze_manual_label_plane(np.zeros((9, 9), dtype=np.uint8), 100.0, 100.0, CONFIG)

    assert result.organ_labels == []
    assert result.eus_vessel_labels == []
    assert result.eus_vessel_features == []


def test_main_portal_vein_23_is_organ_only_but_smv_26_has_dual_identity():
    main_portal = np.zeros((9, 9), dtype=np.uint8)
    main_portal[3:6, 3:6] = 23
    smv = np.zeros((9, 9), dtype=np.uint8)
    smv[3:6, 3:6] = 26

    main_result = analyze_manual_label_plane(main_portal, 80.0, 80.0, CONFIG)
    smv_result = analyze_manual_label_plane(smv, 80.0, 80.0, CONFIG)

    assert main_result.organ_labels == ["portal_vein"]
    assert main_result.eus_vessel_labels == []
    assert main_result.eus_vessel_features == []
    assert smv_result.organ_labels == ["portal_vein"]
    assert smv_result.eus_vessel_labels == ["portal_vein"]
    assert [feature["label"] for feature in smv_result.eus_vessel_features] == ["portal_vein"]


def test_incomplete_eus_vessel_is_drawn_and_labelled_but_not_featured():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[4:7, 4:7] = 8
    labels[0:3, 8:11] = 9

    result = analyze_manual_label_plane(labels, 110.0, 110.0, CONFIG)

    assert result.eus_vessel_labels == ["aorta", "inferior_vena_cava"]
    assert [feature["label"] for feature in result.eus_vessel_features] == ["aorta"]
    assert (255, 0, 0) in _colors(result.eus_vessel_boundary_rgb)
    assert (0, 0, 255) in _colors(result.eus_vessel_boundary_rgb)


def test_portal_source_labels_merge_before_components_and_boundary_extraction():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[4:7, 4:6] = 26
    labels[4:7, 6:8] = 33

    result = analyze_manual_label_plane(labels, 110.0, 110.0, CONFIG)

    assert len(result.eus_vessel_features) == 1
    assert result.eus_vessel_features[0]["label"] == "portal_vein"
    assert result.eus_vessel_features[0]["area_mm2"] == pytest.approx(1200.0)
    assert tuple(result.eus_vessel_boundary_rgb[5, 5]) == (255, 255, 255)
    assert tuple(result.eus_vessel_boundary_rgb[5, 6]) == (255, 255, 255)
    assert (170, 85, 255) in _colors(result.eus_vessel_boundary_rgb)


def test_eus_features_use_eight_connectivity_and_square_local_millimetres():
    labels = np.zeros((11, 11), dtype=np.uint8)
    labels[4, 4] = 8
    labels[5, 5] = 8

    result = analyze_manual_label_plane(labels, 100.0, 200.0, CONFIG)

    assert result.eus_vessel_features == [
        {
            "label": "aorta",
            "x_mm": 45.0,
            "y_mm": 90.0,
            "area_mm2": 400.0,
        }
    ]


def test_manual_rendering_adds_unfiltered_eus_images_without_changing_original_vessels():
    contour = SectionContour(
        points_mm=np.asarray([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]]),
        complete=True,
        centroid_mm=np.asarray([5.0, 5.0]),
        area_mm2=36.0,
    )
    baseline = render_sample_images(
        np.full((12, 12), 127, dtype=np.uint8),
        11.0,
        11.0,
        [VesselLayer("artery_tree", "artery", (255, 82, 0), [contour])],
    )
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[4:7, 4:7] = 8
    labels[0:3, 8:11] = 9
    analysis = analyze_manual_label_plane(labels, 11.0, 11.0, CONFIG)

    manual = apply_manual_label_analysis(baseline, analysis)

    assert manual.features == baseline.features
    assert manual.boundary_only.tobytes() == baseline.boundary_only.tobytes()
    assert manual.ct_overlay.tobytes() == baseline.ct_overlay.tobytes()
    assert manual.organ_labels == ["aorta", "inferior_vena_cava"]
    assert [feature["label"] for feature in manual.eus_vessel_features] == ["aorta"]
    assert manual.eus_vessel_labels == ["aorta", "inferior_vena_cava"]
    assert manual.eus_vessel_boundary is not None
    assert manual.ct_eus_vessel_overlay is not None
    boundary_colors = set(manual.eus_vessel_boundary.getdata())
    overlay_colors = set(manual.ct_eus_vessel_overlay.getdata())
    assert (255, 0, 0) in boundary_colors
    assert (0, 0, 255) in boundary_colors
    assert (255, 0, 0) in overlay_colors
    assert (0, 0, 255) in overlay_colors
    assert manual.eus_vessel_boundary.getpixel((5, 5)) == (255, 255, 255)
    assert manual.eus_vessel_boundary.getpixel((9, 1)) == (255, 255, 255)
    assert manual.ct_eus_vessel_overlay.getpixel((5, 5)) == (127, 127, 127)
    assert manual.ct_eus_vessel_overlay.getpixel((9, 1)) == (127, 127, 127)


@pytest.mark.parametrize(
    ("labels", "width_mm", "length_mm", "message"),
    [
        (np.zeros((2, 2, 2), dtype=np.uint8), 100.0, 100.0, "二维"),
        (np.zeros((1, 2), dtype=np.uint8), 100.0, 100.0, "至少 2"),
        (np.zeros((2, 2), dtype=np.uint8), 0.0, 100.0, "大于零"),
        (np.zeros((2, 2), dtype=np.uint8), 100.0, -1.0, "大于零"),
    ],
)
def test_manual_label_plane_rejects_invalid_geometry(labels, width_mm, length_mm, message):
    with pytest.raises(ValueError, match=message):
        analyze_manual_label_plane(labels, width_mm, length_mm, CONFIG)
