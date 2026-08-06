from __future__ import annotations

import numpy as np
import pytest
import trimesh

from ct_vascular_resampling.centerline import (
    centerline_tangents,
    extract_duodenum_centerline,
    order_skeleton_indices,
)
from ct_vascular_resampling.sampling import (
    build_esophagus_samples,
    sample_points_with_minimum_spacing,
)


def test_constrained_fps_stops_at_ten_mm_and_preserves_point_normal_pairs():
    points = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    normals = np.asarray([[index, 0.0, 1.0] for index in range(5)], dtype=np.float64)

    first = sample_points_with_minimum_spacing(points, normals, count=5, seed=0, minimum_spacing_mm=10.0)
    second = sample_points_with_minimum_spacing(points, normals, count=5, seed=0, minimum_spacing_mm=10.0)

    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.normals, second.normals)
    assert first.stats.requested_count == 5
    assert first.stats.candidate_count == 5
    assert first.stats.actual_count == 3
    assert first.stats.actual_minimum_distance_mm == pytest.approx(10.0)
    assert np.array_equal(first.indices, [4, 0, 2])
    for point, normal in zip(first.points, first.normals, strict=True):
        assert normal[0] == points.tolist().index(point.tolist())


def test_esophagus_copies_the_entire_valid_span_before_spacing_limited_fps():
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
    normals = np.asarray([[1.0, 0.0, 0.0]] * 3)

    result = build_esophagus_samples(
        points,
        normals,
        count=10,
        seed=0,
        minimum_spacing_mm=10.0,
    )

    assert result.stats.requested_count == 10
    assert result.stats.candidate_count == 5
    assert result.stats.actual_count == 5
    assert np.min(result.points[:, 2]) == -20.0
    assert np.max(result.points[:, 2]) == 20.0
    assert result.stats.actual_minimum_distance_mm == pytest.approx(10.0)


def test_esophagus_extension_uses_explicit_full_valid_segment_span():
    points = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 8.0]])
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = build_esophagus_samples(
        points,
        normals,
        count=4,
        seed=0,
        minimum_spacing_mm=1.0,
        translation_span_mm=20.0,
    )

    assert set(result.points[:, 2]) == {-18.0, -12.0, 2.0, 8.0}


def test_skeleton_order_prunes_only_a_short_terminal_spur():
    skeleton = np.zeros((9, 9, 9), dtype=bool)
    center = np.asarray([4, 4, 4])
    skeleton[tuple(center)] = True
    for direction, length in (([-1, -1, -1], 3), ([1, 1, -1], 2), ([1, -1, 1], 1)):
        for step in range(1, length + 1):
            skeleton[tuple(center + np.asarray(direction) * step)] = True

    ordered = order_skeleton_indices(skeleton, np.asarray([[1.0, 1.0, 1.0]]), voxel_pitch_mm=1.0, max_terminal_spur_mm=5.0)

    assert len(ordered) == 6
    assert not np.any(np.all(ordered == [5, 3, 5], axis=1))


def test_skeleton_order_rejects_a_branch_when_all_terminal_arms_are_long():
    skeleton = np.zeros((15, 15, 15), dtype=bool)
    center = np.asarray([7, 7, 7])
    skeleton[tuple(center)] = True
    for direction in ([-1, -1, -1], [1, 1, -1], [1, -1, 1]):
        for step in range(1, 5):
            skeleton[tuple(center + np.asarray(direction) * step)] = True

    with pytest.raises(ValueError, match="分叉"):
        order_skeleton_indices(skeleton, np.asarray([[3.0, 3.0, 3.0]]), voxel_pitch_mm=1.0, max_terminal_spur_mm=5.0)


def test_ten_mm_chord_tangents_are_oriented_from_proximal_to_distal():
    points = np.column_stack([np.zeros(21), np.zeros(21), np.arange(21, dtype=np.float64)])

    tangents = centerline_tangents(points, window_mm=10.0)

    assert np.allclose(tangents, np.asarray([[0.0, 0.0, 1.0]] * 21))


def test_duodenum_centerline_extraction_returns_single_proximal_to_distal_path():
    duodenum = trimesh.creation.cylinder(radius=3.0, height=30.0, sections=32)
    stomach_reference = np.asarray([[0.0, 0.0, -20.0]])

    centerline = extract_duodenum_centerline(
        duodenum,
        stomach_reference,
        voxel_pitch_mm=1.0,
        tangent_window_mm=10.0,
    )

    assert len(centerline.points) > 10
    assert centerline.points[0, 2] < centerline.points[-1, 2]
    assert np.all(centerline.tangents[:, 2] > 0.9)
