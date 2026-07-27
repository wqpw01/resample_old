from __future__ import annotations

import numpy as np

from ct_vascular_resampling.artifacts import write_square_samples_ply, write_surface_samples_ply
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


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
    assert "property float nx" in point_text
    assert "element vertex 4" in square_text
