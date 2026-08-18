from __future__ import annotations

import numpy as np

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
        layers=[VesselLayer("artery_tree", "artery", (255, 82, 0), [contour])],
    )

    assert rendered.ct.mode == "L"
    assert rendered.boundary_only.mode == "RGB"
    assert rendered.ct_overlay.mode == "RGB"
    assert rendered.features == [{"label": "artery", "x_mm": 3.5, "y_mm": 3.5, "area_mm2": 9.0}]
    assert (255, 82, 0) in set(rendered.boundary_only.getdata())
