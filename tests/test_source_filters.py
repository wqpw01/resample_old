from __future__ import annotations

import numpy as np
import trimesh

from ct_vascular_resampling.sampling import (
    filter_duodenum_bulb_points,
    filter_duodenum_remainder_points,
    filter_liver_region_one_points,
    filter_liver_region_two_points,
    filter_pancreas_points,
    filter_points_by_target_rays,
)


def test_target_ray_keeps_only_forward_intersection_within_100_mm():
    target = trimesh.creation.box(extents=(2.0, 2.0, 2.0), transform=trimesh.transformations.translation_matrix([20.0, 0.0, 0.0]))
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 10.0, 0.0]])
    normals = np.asarray([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    result = filter_points_by_target_rays(points, normals, {"pancreas": target}, ray_length_mm=100.0)

    assert np.array_equal(result.points, [[0.0, 0.0, 0.0]])
    assert np.array_equal(result.normals, [[1.0, 0.0, 0.0]])
    assert result.target_ids == ("pancreas",)
    assert np.allclose(result.distances_mm, [19.0])


def test_target_ray_excludes_intersection_beyond_100_mm():
    target = trimesh.creation.box(extents=(2.0, 2.0, 2.0), transform=trimesh.transformations.translation_matrix([102.0, 0.0, 0.0]))

    result = filter_points_by_target_rays(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0]]),
        {"liver": target},
        ray_length_mm=100.0,
    )

    assert len(result.points) == 0


def test_pancreas_filter_uses_extreme_plateau_centroid_and_strict_angle_boundaries():
    cos_105 = np.cos(np.deg2rad(105.0))
    sin_105 = np.sin(np.deg2rad(105.0))
    pancreas = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    normals = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [sin_105, 0.0, cos_105],
            [0.0, 0.0, 1.0],
        ]
    )
    duodenum = np.asarray([[2.0, 0.0, 9.0], [8.0, 2.0, 9.0], [20.0, 0.0, 5.0]])

    filtered_points, _ = filter_pancreas_points(pancreas, normals, duodenum)

    assert np.array_equal(filtered_points, [[4.0, 0.0, 0.0]])


def test_liver_region_one_uses_ras_twenty_mm_closed_ranges_and_inferior_normal():
    liver = np.asarray([[-100.0, 80.0, 0.0], [-80.0, 60.0, 0.0], [-10.0, 10.0, 0.0], [-50.0, 20.0, 0.0]])
    normals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    esophagus = np.asarray([[0.0, 10.0, 0.0]])
    vena_cava = np.asarray([[-10.0, 0.0, 0.0]])

    points, _ = filter_liver_region_one_points(liver, normals, esophagus, vena_cava)

    assert np.array_equal(points, [[-80.0, 60.0, 0.0], [-10.0, 10.0, 0.0]])


def test_liver_region_two_uses_closed_x_range_and_strict_y_and_normal_angles():
    liver = np.asarray([[-40.0, 19.0, 0.0], [30.0, 19.0, 0.0], [-5.0, 20.0, 0.0], [-5.0, 19.0, 0.0]])
    normals = np.asarray([[0.0, -1.0, -1.0], [0.0, -1.0, -1.0], [0.0, -1.0, -1.0], [0.0, 0.0, -1.0]])
    spleen = np.asarray([[-40.0, 0.0, 0.0]])
    vena_cava = np.asarray([[30.0, 0.0, 0.0]])
    pancreas = np.asarray([[0.0, 20.0, 0.0]])

    points, _ = filter_liver_region_two_points(liver, normals, spleen, vena_cava, pancreas)

    assert np.array_equal(points, [[-40.0, 19.0, 0.0], [30.0, 19.0, 0.0]])


def test_duodenum_bulb_and_remainder_are_strict_and_mutually_exclusive():
    duodenum = np.asarray([[5.0, 0.0, 11.0], [5.0, 0.0, 10.0], [11.0, 0.0, 9.0], [10.0, 0.0, 9.0]])
    normals = np.asarray([[0.0, 0.0, 1.0]] * 4)
    right_adrenal = np.asarray([[0.0, 0.0, 10.0]])
    aorta = np.asarray([[10.0, 0.0, 0.0]])

    bulb_points, _ = filter_duodenum_bulb_points(duodenum, normals, right_adrenal)
    remainder_points, _ = filter_duodenum_remainder_points(duodenum, normals, aorta, right_adrenal)

    assert np.array_equal(bulb_points, [[5.0, 0.0, 11.0]])
    assert np.array_equal(remainder_points, [[11.0, 0.0, 9.0]])
