from __future__ import annotations

import numpy as np

from ct_vascular_resampling.sampling import sample_points_with_normals


def test_fps_uses_one_deterministic_index_set_for_points_and_normals():
    points = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [2.0, 0.0, 10.0], [3.0, 0.0, 10.0]])

    first_points, first_normals = sample_points_with_normals(points, normals, count=3, seed=0)
    second_points, second_normals = sample_points_with_normals(points, normals, count=3, seed=0)

    assert np.array_equal(first_points, second_points)
    assert np.array_equal(first_normals, second_normals)
    for point, normal in zip(first_points, first_normals, strict=True):
        original_index = int(point[0] / 10.0)
        assert np.array_equal(normal, normals[original_index])
