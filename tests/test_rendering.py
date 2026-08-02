from __future__ import annotations

import numpy as np

import ct_vascular_resampling.rendering as rendering
from ct_vascular_resampling.geometry import SectionContour
from ct_vascular_resampling.rendering import VesselLayer, render_sample_images


def test_renderer_creates_ct_boundary_overlay_and_feature_triplets():
    contour = SectionContour(
        points_mm=np.asarray([[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]]),
        complete=True,
        centroid_mm=np.asarray([3.5, 3.5]),
        area_mm2=9.0,
    )

    rendered = render_sample_images(
        ct_pixels=np.full((20, 20), 127, dtype=np.uint8),
        width_mm=10.0,
        length_mm=10.0,
        layers=[VesselLayer("portal_tree", "portal", (255, 0, 255), [contour])],
    )

    assert rendered.ct.mode == "L"
    assert rendered.boundary_only.mode == "RGB"
    assert rendered.ct_overlay.mode == "RGB"
    assert rendered.features == [{"label": "portal", "x_mm": 3.5, "y_mm": 3.5, "area_mm2": 9.0}]
    assert (255, 0, 255) in set(rendered.boundary_only.getdata())


def test_renderer_keeps_vessel_outputs_isolated_and_draws_organs_beneath_vessels():
    vessel_color = (255, 0, 255)
    organ_color = (31, 119, 180)
    shared_contour = SectionContour(
        points_mm=np.asarray([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]]),
        complete=True,
        centroid_mm=np.asarray([5.0, 5.0]),
        area_mm2=36.0,
    )
    clipped_contour = SectionContour(
        points_mm=np.asarray([[0.0, 3.0], [4.0, 3.0], [4.0, 6.0], [0.0, 6.0]]),
        complete=False,
        centroid_mm=np.asarray([2.0, 4.5]),
        area_mm2=12.0,
    )

    rendered = render_sample_images(
        ct_pixels=np.full((20, 20), 127, dtype=np.uint8),
        width_mm=10.0,
        length_mm=10.0,
        layers=[VesselLayer("portal_tree", "portal", vessel_color, [shared_contour])],
        organ_layers=[
            rendering.OrganLayer("stomach", "stomach", (127, 127, 127), [clipped_contour]),
            rendering.OrganLayer("liver-a", "liver", organ_color, [shared_contour]),
            rendering.OrganLayer("liver-b", "liver", organ_color, [clipped_contour]),
        ],
    )

    vessel_pixels = set(rendered.boundary_only.getdata())
    overlay_pixels = set(rendered.ct_overlay.getdata())
    combined_pixels = set(rendered.organ_vessel_boundary.getdata())
    assert organ_color not in vessel_pixels
    assert organ_color not in overlay_pixels
    assert organ_color in combined_pixels
    assert vessel_color in combined_pixels
    assert rendered.organ_vessel_boundary.getpixel((4, 4)) == vessel_color
    assert rendered.organ_labels == ["liver", "stomach"]
    assert rendered.features == [{"label": "portal", "x_mm": 5.0, "y_mm": 5.0, "area_mm2": 36.0}]
