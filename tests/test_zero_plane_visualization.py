from __future__ import annotations

import csv
from copy import deepcopy
import json

import numpy as np
import pytest

from ct_vascular_resampling.zero_plane_visualization import (
    select_zero_planes,
    write_structured_exports,
)


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


def test_write_structured_exports_preserves_geometry_and_provenance(tmp_path):
    records = select_zero_planes([_record()], {"stomach": 1})
    provenance = {
        "source_manifest_sha256": "a" * 64,
        "core_design_sha256": "b" * 64,
        "build_git_commit": "c" * 40,
    }

    write_structured_exports(records, tmp_path, provenance)

    point_text = (tmp_path / "sampling_points.ply").read_text(encoding="utf-8")
    edge_text = (tmp_path / "zero_planes_edges.ply").read_text(encoding="utf-8")
    face_text = (tmp_path / "zero_planes_faces.ply").read_text(encoding="utf-8")
    assert "element vertex 1" in point_text
    assert "property float nx" in point_text
    assert "property uchar red" in point_text
    assert "element vertex 4" in edge_text
    assert "element edge 4" in edge_text
    assert "element face 1" in face_text

    payload = json.loads(
        (tmp_path / "sampling_points_zero_planes.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "zero-plane-visualization/v1"
    assert payload["coordinate_system"] == "RAS"
    assert payload["unit"] == "mm"
    assert payload["record_count"] == 1
    assert payload["organ_counts"] == {"stomach": 1}
    assert payload["provenance"] == provenance
    assert payload["records"][0]["vertices_world"][2] == [100.0, 50.0, 0.0]

    with (tmp_path / "sampling_points_zero_planes.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["slice_id"] == "stomach-000000-rp000-pp000-yp000"
    assert float(rows[0]["v2_x_mm"]) == 100.0
