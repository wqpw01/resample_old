from __future__ import annotations

import numpy as np

from ct_vascular_resampling.sampling import sample_points_with_normals
from ct_vascular_resampling.squares import generate_sampling_squares


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


def test_square_generator_restores_twenty_seven_rotation_variants_at_100_mm():
    squares = generate_sampling_squares(
        point=np.asarray([0.0, 0.0, 0.0]),
        normal=np.asarray([0.0, 0.0, 1.0]),
        side_length_mm=100.0,
        use_reverse_normal=False,
        reference_axis="x",
    )

    assert squares.shape == (27, 4, 3)
    assert np.allclose(squares[13], [[0.0, -50.0, 0.0], [0.0, 50.0, 0.0], [0.0, 50.0, 100.0], [0.0, -50.0, 100.0]])


def test_edge_angle_rotates_the_entire_square_about_its_near_edge():
    squares = generate_sampling_squares(
        point=np.asarray([0.0, 0.0, 0.0]),
        normal=np.asarray([0.0, 0.0, 1.0]),
        side_length_mm=100.0,
        use_reverse_normal=False,
        reference_axis="x",
    )

    # Variants 10, 13 and 16 share normal_angle=0 and plane_angle=0;
    # only edge_angle is -5, 0 and +5 degrees respectively.
    negative, neutral, positive = squares[10], squares[13], squares[16]

    assert np.allclose(negative[:2], neutral[:2])
    assert np.allclose(positive[:2], neutral[:2])
    assert not np.allclose(negative[2:], neutral[2:])
    assert not np.allclose(positive[2:], neutral[2:])
    assert not np.allclose(negative[2:], positive[2:])
