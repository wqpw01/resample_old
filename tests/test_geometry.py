from __future__ import annotations

import numpy as np
import trimesh

from ct_vascular_resampling.geometry import frame_from_vertices, intersect_mesh_with_square


def test_mesh_section_is_clipped_to_square_and_reported_in_local_millimetres():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    frame = frame_from_vertices(
        np.asarray(
            [
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ]
        )
    )

    contours = intersect_mesh_with_square(mesh, frame)

    assert len(contours) == 1
    assert contours[0].complete is True
    assert np.allclose(contours[0].centroid_mm, [2.0, 2.0])
    assert np.isclose(contours[0].area_mm2, 1.0)
