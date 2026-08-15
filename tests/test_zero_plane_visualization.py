from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from ct_vascular_resampling.zero_plane_visualization import select_zero_planes


def _record(index: int = 0, *, organ: str = "stomach") -> dict:
    return {
        "slice_id": f"{organ}-{index:06d}-rp000-pp000-yp000",
        "organ": organ,
        "probe_point_world": [0.0, 0.0, 0.0],
        "angles_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "square_vertices_world": [
            [0.0, -50.0, 0.0],
            [0.0, 50.0, 0.0],
            [100.0, 50.0, 0.0],
            [100.0, -50.0, 0.0],
        ],
        "local_axes_world": {
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        },
        "input_normal_world": [1.0, 0.0, 0.0],
        "coordinate_system": "RAS",
    }


def test_select_zero_planes_filters_nonzero_angles_and_preserves_recorded_geometry():
    nonzero = _record(0)
    nonzero["angles_degrees"]["roll"] = 5.0
    selected = select_zero_planes([nonzero, _record(0)], {"stomach": 1})

    assert len(selected) == 1
    plane = selected[0]
    assert plane.slice_id == "stomach-000000-rp000-pp000-yp000"
    assert plane.organ == "stomach"
    assert plane.point_index == 0
    assert np.array_equal(plane.probe, np.zeros(3))
    assert np.array_equal(
        plane.vertices,
        np.asarray(
            [
                [0.0, -50.0, 0.0],
                [0.0, 50.0, 0.0],
                [100.0, 50.0, 0.0],
                [100.0, -50.0, 0.0],
            ]
        ),
    )


def test_select_zero_planes_sorts_by_expected_organ_order_and_point_index():
    selected = select_zero_planes(
        [_record(1), _record(0), _record(0, organ="liver")],
        {"stomach": 2, "liver": 1},
    )

    assert [(item.organ, item.point_index) for item in selected] == [
        ("stomach", 0),
        ("stomach", 1),
        ("liver", 0),
    ]


def test_select_zero_planes_rejects_duplicate_sample():
    with pytest.raises(ValueError, match="重复"):
        select_zero_planes([_record(), _record()], {"stomach": 1})


def test_select_zero_planes_rejects_missing_expected_sample():
    with pytest.raises(ValueError, match="计数"):
        select_zero_planes([_record()], {"stomach": 2})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda record: record.update(coordinate_system="LPS"),
            "RAS",
        ),
        (
            lambda record: record["square_vertices_world"].__setitem__(
                2, [99.0, 50.0, 0.0]
            ),
            "100 mm 正方形",
        ),
        (
            lambda record: record.__setitem__(
                "probe_point_world", [0.0, 1.0, 0.0]
            ),
            "底边中点",
        ),
        (
            lambda record: record["local_axes_world"].__setitem__(
                "z", [0.0, 0.0, -1.0]
            ),
            "右手",
        ),
        (
            lambda record: record["local_axes_world"].__setitem__(
                "y", [1.0, 1.0, 0.0]
            ),
            "正交",
        ),
    ],
)
def test_select_zero_planes_rejects_invalid_recorded_geometry(mutate, message):
    record = deepcopy(_record())
    mutate(record)

    with pytest.raises(ValueError, match=message):
        select_zero_planes([record], {"stomach": 1})
