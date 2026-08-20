from __future__ import annotations

import numpy as np

from ct_vascular_resampling.centerline import CenterlinePath
from ct_vascular_resampling.squares import (
    LIVER_REGION_TWO_YAW,
    PANCREAS_SPECIAL_YAW,
    STANDARD_YAW,
    DUODENUM_BULB_YAW,
    duodenum_local_frame,
    generate_pose_variants,
    ordinary_local_frame,
)


def test_ordinary_zero_frame_uses_forward_normal_and_anchor_to_point_direction():
    frame = ordinary_local_frame(
        point=np.asarray([0.0, 0.0, 0.0]),
        forward=np.asarray([2.0, 0.0, 0.0]),
        esophagus_anchor=np.asarray([0.0, -10.0, 0.0]),
    )

    assert np.allclose(frame.x_axis, [1.0, 0.0, 0.0])
    assert np.allclose(frame.y_axis, [0.0, 1.0, 0.0])
    assert np.allclose(frame.z_axis, [0.0, 0.0, 1.0])
    assert np.allclose(np.cross(frame.x_axis, frame.y_axis), frame.z_axis)


def test_duodenum_zero_frame_is_exactly_in_tangent_radial_plane():
    points = np.column_stack([np.zeros(11), np.zeros(11), np.arange(11, dtype=np.float64)])
    tangents = np.asarray([[0.0, 0.0, 1.0]] * 11)
    centerline = CenterlinePath(points, tangents, np.arange(11, dtype=np.float64))

    frame = duodenum_local_frame(
        point=np.asarray([2.0, 0.0, 5.0]),
        outward_normal=np.asarray([1.0, 0.0, 0.0]),
        centerline=centerline,
    )

    assert np.allclose(frame.x_axis, [1.0, 0.0, 0.0])
    assert np.allclose(frame.y_axis, [0.0, 0.0, -1.0])
    assert np.allclose(frame.z_axis, [0.0, 1.0, 0.0])
    assert np.allclose(np.cross(frame.x_axis, frame.y_axis), frame.z_axis)


def test_pose_variants_use_confirmed_roll_pitch_and_region_yaw_ranges():
    frame = ordinary_local_frame(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
    )

    standard = generate_pose_variants(np.zeros(3), frame, 100.0, STANDARD_YAW)
    bulb = generate_pose_variants(np.zeros(3), frame, 100.0, DUODENUM_BULB_YAW)
    special = generate_pose_variants(np.zeros(3), frame, 100.0, PANCREAS_SPECIAL_YAW)
    liver_two = generate_pose_variants(np.zeros(3), frame, 100.0, LIVER_REGION_TWO_YAW)

    assert {value.roll_degrees for value in standard} == set(np.arange(-45.0, 46.0, 5.0))
    assert {value.pitch_degrees for value in standard} == set(np.arange(-30.0, 31.0, 5.0))
    assert {value.yaw_degrees for value in standard} == set(np.arange(-30.0, 31.0, 5.0))
    assert {value.yaw_degrees for value in bulb} == set(np.arange(-120.0, 31.0, 5.0))
    assert {value.yaw_degrees for value in special} == set(np.arange(-120.0, 31.0, 5.0))
    assert {value.yaw_degrees for value in liver_two} == set(np.arange(-60.0, 61.0, 5.0))
    assert len(standard) == 3211
    assert len(bulb) == 7657
    assert len(special) == 7657
    assert len(liver_two) == 6175


def test_positive_yaw_is_counterclockwise_when_viewed_from_local_positive_z():
    frame = ordinary_local_frame(
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
    )

    positive = next(
        value
        for value in generate_pose_variants(np.zeros(3), frame, 100.0, STANDARD_YAW)
        if value.roll_degrees == value.pitch_degrees == 0.0 and value.yaw_degrees == 5.0
    )

    assert positive.local_frame.x_axis[1] > 0.0
    assert np.allclose(
        (positive.vertices[0] + positive.vertices[1]) / 2.0,
        np.zeros(3),
        rtol=0.0,
        atol=1e-12,
    )


def test_zero_pose_keeps_probe_at_bottom_center_and_uses_100_mm_forward_depth():
    frame = ordinary_local_frame(np.zeros(3), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, -1.0, 0.0]))
    variants = generate_pose_variants(np.zeros(3), frame, 100.0, STANDARD_YAW)
    neutral = next(
        value
        for value in variants
        if value.roll_degrees == value.pitch_degrees == value.yaw_degrees == 0.0
    )

    bottom_center = (neutral.vertices[0] + neutral.vertices[1]) / 2.0
    top_center = (neutral.vertices[2] + neutral.vertices[3]) / 2.0
    assert np.allclose(bottom_center, [0.0, 0.0, 0.0])
    assert np.allclose(top_center, [100.0, 0.0, 0.0])
    assert np.allclose(neutral.local_frame.x_axis, [1.0, 0.0, 0.0])


def test_nonzero_pose_uses_intrinsic_local_z_y_x_matrix_order():
    frame = ordinary_local_frame(np.zeros(3), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, -1.0, 0.0]))
    variant = next(
        value
        for value in generate_pose_variants(np.zeros(3), frame, 100.0, STANDARD_YAW)
        if value.roll_degrees == value.pitch_degrees == value.yaw_degrees == 5.0
    )
    angle = np.deg2rad(5.0)
    cosine, sine = np.cos(angle), np.sin(angle)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])
    ry = np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    rz = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])

    actual = np.column_stack(
        [variant.local_frame.x_axis, variant.local_frame.y_axis, variant.local_frame.z_axis]
    )

    assert np.allclose(actual, rz @ ry @ rx, rtol=0.0, atol=1e-12)
