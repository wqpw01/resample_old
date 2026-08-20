from __future__ import annotations

import numpy as np
import trimesh

from ct_vascular_resampling.artifacts import write_square_samples_ply, write_surface_samples_ply
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


class _StreamingOnlyIterator:
    def __init__(self, values):
        self._values = iter(values)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._values)

    def __length_hint__(self):
        raise AssertionError("writer must stream instead of materializing the iterable")


def test_intermediate_ply_writers_preserve_source_field_layout(tmp_path):
    points_path = tmp_path / "FPS-Stomach.ply"
    squares_path = tmp_path / "Stomach-vertex.ply"
    write_surface_samples_ply(
        points_path,
        SurfaceSamples(np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 0.0, 1.0]])),
    )
    write_square_samples_ply(
        squares_path,
        [
            SquareSample(
                "stomach-000000-x-00",
                "stomach",
                np.zeros(3),
                np.asarray([0.0, 0.0, 1.0]),
                np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 100.0, 0.0]]),
            )
        ],
    )

    point_text = points_path.read_text(encoding="utf-8")
    square_text = squares_path.read_text(encoding="utf-8")
    assert "property double nx" in point_text
    assert "element vertex 4" in square_text


def test_surface_sample_ply_preserves_six_decimal_precision_at_patient_coordinates(tmp_path):
    path = tmp_path / "FPS-Liver.ply"
    points = np.asarray(
        [
            [4.627197123, 8.222656789, 742.912345],
            [14.627197123, 8.222656789, 742.912345],
        ]
    )
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    write_surface_samples_ply(path, SurfaceSamples(points, normals))

    loaded = trimesh.load(path, process=False)
    data = loaded.metadata["_ply_raw"]["vertex"]["data"]
    round_tripped = np.column_stack([data[axis] for axis in ("x", "y", "z")])
    assert np.max(np.abs(round_tripped - np.round(points, 6))) < 1e-12
    assert np.isclose(
        np.linalg.norm(round_tripped[1] - round_tripped[0]),
        10.0,
        rtol=0.0,
        atol=1e-12,
    )


def test_square_ply_writer_streams_a_single_pass_iterable(tmp_path):
    sample = SquareSample(
        "stomach-000000",
        "stomach",
        np.zeros(3),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )

    write_square_samples_ply(tmp_path / "streamed.ply", _StreamingOnlyIterator([sample]))

    assert "element vertex 4" in (tmp_path / "streamed.ply").read_text(encoding="utf-8")
