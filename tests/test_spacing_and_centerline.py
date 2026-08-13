from __future__ import annotations

import numpy as np
import pytest
import trimesh

from ct_vascular_resampling.centerline import (
    centerline_tangents,
    extract_duodenum_centerline,
    order_skeleton_indices,
    select_skeleton_indices_by_endpoint_hints,
)
from ct_vascular_resampling.sampling import sample_points_with_minimum_spacing


def branched_tree_skeleton() -> np.ndarray:
    skeleton = np.zeros((21, 21, 21), dtype=bool)
    center = np.asarray([10, 10, 10])
    skeleton[tuple(center)] = True
    for direction, length in (
        (np.asarray([-1, -1, -1]), 7),
        (np.asarray([1, 1, -1]), 7),
        (np.asarray([1, -1, 1]), 6),
    ):
        for step in range(1, length + 1):
            skeleton[tuple(center + direction * step)] = True
    return skeleton


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


def test_manual_endpoint_hints_select_unique_path_without_pruning_long_side_branches():
    skeleton = branched_tree_skeleton()
    skeleton_indices = np.argwhere(skeleton)
    world_points = skeleton_indices.astype(np.float64)

    selected, audit = select_skeleton_indices_by_endpoint_hints(
        skeleton,
        world_points,
        proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
        distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
        match_tolerance_mm=1.0,
    )

    assert np.array_equal(selected[0], [3, 3, 3])
    assert np.array_equal(selected[-1], [17, 17, 3])
    assert not np.any(np.all(selected == [16, 4, 16], axis=1))
    assert len(selected) == 15
    assert audit.mode == "manual_endpoint_hints"
    assert audit.automatic_terminal_spur_pruning_applied is False
    assert audit.excluded_endpoints_ras_mm == ((16.0, 4.0, 16.0),)


def test_manual_endpoint_hints_reject_match_outside_tolerance():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)

    with pytest.raises(ValueError, match="近端.*误差"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 5.0]),
            distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_same_resolved_endpoint():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)

    with pytest.raises(ValueError, match="同一个端点"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            distal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_disconnected_skeleton():
    skeleton = np.zeros((12, 12, 12), dtype=bool)
    skeleton[1:5, 1, 1] = True
    skeleton[7:11, 10, 10] = True
    points = np.argwhere(skeleton).astype(np.float64)

    with pytest.raises(ValueError, match="单连通"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([1.0, 1.0, 1.0]),
            distal_hint_ras_mm=np.asarray([10.0, 10.0, 10.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_cyclic_skeleton():
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 2] = True
    skeleton[2, 2, 3] = True
    skeleton[2, 3, 2] = True
    points = np.argwhere(skeleton).astype(np.float64)

    with pytest.raises(ValueError, match="无环树"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([2.0, 2.0, 2.0]),
            distal_hint_ras_mm=np.asarray([2.0, 2.0, 3.0]),
            match_tolerance_mm=2.0,
        )


def test_manual_endpoint_hints_reject_equal_distance_endpoint_match():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)

    with pytest.raises(ValueError, match="无法唯一匹配"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([10.0, 10.0, 3.0]),
            distal_hint_ras_mm=np.asarray([16.0, 4.0, 16.0]),
            match_tolerance_mm=20.0,
        )


def test_manual_endpoint_hints_reject_nonfinite_world_coordinates():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)
    points[0, 0] = np.nan

    with pytest.raises(ValueError, match="有限"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
            match_tolerance_mm=1.0,
        )


def test_duodenum_centerline_extraction_uses_manual_endpoint_path(monkeypatch):
    skeleton = branched_tree_skeleton()

    class FakeVoxelGrid:
        matrix = skeleton

        def fill(self):
            return self

        def indices_to_points(self, indices):
            return np.asarray(indices, dtype=np.float64)

        def points_to_indices(self, points):
            return np.asarray(points, dtype=np.int64)

    monkeypatch.setattr(trimesh.Trimesh, "voxelized", lambda _self, pitch: FakeVoxelGrid())
    duodenum = trimesh.creation.box()

    centerline = extract_duodenum_centerline(
        duodenum,
        np.asarray([[3.0, 3.0, 3.0]]),
        voxel_pitch_mm=1.0,
        tangent_window_mm=10.0,
        max_terminal_spur_mm=5.0,
        endpoint_hints_ras_mm=((3.0, 3.0, 3.0), (17.0, 17.0, 3.0)),
        endpoint_match_tolerance_mm=1.0,
    )

    assert np.array_equal(centerline.points[0], [3.0, 3.0, 3.0])
    assert np.array_equal(centerline.points[-1], [17.0, 17.0, 3.0])
    assert len(centerline.points) == 15
    assert centerline.selection_audit is not None
    assert centerline.selection_audit.mode == "manual_endpoint_hints"


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
