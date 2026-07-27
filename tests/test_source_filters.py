from __future__ import annotations

import numpy as np

from ct_vascular_resampling.sampling import (
    build_esophagus_samples,
    filter_duodenum_remainder_points,
    filter_duodenum_upper_points,
    filter_liver_points,
    filter_pancreas_points,
    filter_stomach_points,
)


def test_stomach_filter_keeps_target_points_in_forward_normal_range():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 200.0, 0.0]])
    normals = np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_voxels = np.asarray([[5.0, 0.0, 0.0], [-5.0, 100.0, 0.0], [10.5, 200.0, 0.0]])

    filtered_points, filtered_normals = filter_stomach_points(points, normals, target_voxels)

    assert np.array_equal(filtered_points, [[0.0, 0.0, 0.0]])
    assert np.allclose(filtered_normals, [[1.0, 0.0, 0.0]])


def test_liver_filter_applies_source_coordinate_and_normal_constraints():
    liver = np.asarray([[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [30.0, 60.0, 0.0], [70.0, 60.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    esophagus = np.asarray([[1.0, 40.0, 5.0], [1.0, 20.0, 10.0]])
    gallbladder = np.asarray([[65.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    filtered_points, filtered_normals = filter_liver_points(liver, normals, esophagus, gallbladder)

    assert np.array_equal(filtered_points, [[30.0, 60.0, 0.0]])
    assert np.array_equal(filtered_normals, [[0.0, 0.0, -1.0]])


def test_pancreas_and_duodenum_filters_match_source_extrema_rules():
    pancreas = np.asarray([[5.0, 0.0, 0.0], [11.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    pancreas_normals = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    duodenum_reference = np.asarray([[10.0, 0.0, 5.0], [1.0, 0.0, 1.0]])

    pancreas_points, _ = filter_pancreas_points(pancreas, pancreas_normals, duodenum_reference)
    upper_points, _ = filter_duodenum_upper_points(
        np.asarray([[0.0, 0.0, 6.0], [0.0, 0.0, 4.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, 5.0]]),
    )
    remainder_points, _ = filter_duodenum_remainder_points(
        np.asarray([[11.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.asarray([[10.0, 0.0, 0.0]]),
    )

    assert np.array_equal(pancreas_points, [[5.0, 0.0, 0.0]])
    assert np.array_equal(upper_points, [[0.0, 0.0, 6.0]])
    assert np.array_equal(remainder_points, [[11.0, 0.0, 0.0]])


def test_esophagus_samples_preserve_requested_count_and_include_downward_copy():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 8.0]])
    normals = np.asarray([[0.0, 0.0, 1.0]] * 3)

    sampled_points, sampled_normals = build_esophagus_samples(points, normals, count=3, seed=0)

    assert len(sampled_points) == 3
    assert len(sampled_normals) == 3
    assert np.any(sampled_points[:, 2] < 0.0)
