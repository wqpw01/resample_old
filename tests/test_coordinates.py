from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ct_vascular_resampling.coordinates import to_ras_points, to_ras_vectors
from ct_vascular_resampling.ct_resampling import CTVolume
from ct_vascular_resampling.mesh_io import load_surface_mesh


def test_lps_points_and_vectors_are_reflected_into_ras():
    values = np.asarray([[10.0, 20.0, 30.0], [-4.0, 5.0, 6.0]])

    assert np.array_equal(to_ras_points(values, "LPS"), [[-10.0, -20.0, 30.0], [4.0, -5.0, 6.0]])
    assert np.array_equal(to_ras_vectors(values, "LPS"), [[-10.0, -20.0, 30.0], [4.0, -5.0, 6.0]])
    assert np.array_equal(to_ras_points(values, "RAS"), values)


def test_ct_volume_converts_lps_origin_and_direction_without_resampling_voxels():
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    volume = CTVolume.from_sitk(image, input_coordinate_system="LPS")
    point_lps = np.asarray(image.TransformIndexToPhysicalPoint((2, 1, 1)))
    point_ras = to_ras_points(point_lps, "LPS")

    assert np.array_equal(volume.data_zyx, sitk.GetArrayFromImage(image))
    assert np.array_equal(volume.origin_xyz, [-10.0, -20.0, 30.0])
    assert np.array_equal(volume.direction_xyz, np.diag([-1.0, -1.0, 1.0]) @ np.asarray(image.GetDirection()).reshape(3, 3))
    assert np.allclose(volume.world_to_continuous_indices(point_ras), [2.0, 1.0, 1.0])


def test_mesh_loader_converts_lps_vertices_to_ras(tmp_path):
    mesh_path = tmp_path / "surface.obj"
    mesh_path.write_text(
        """v 10 20 30
v 11 20 30
v 10 21 30
v 10 20 31
f 1 2 3
f 1 4 2
f 1 3 4
f 2 4 3
""",
        encoding="utf-8",
    )

    mesh = load_surface_mesh(mesh_path, input_coordinate_system="LPS")

    assert np.any(np.all(np.isclose(mesh.vertices, [-10.0, -20.0, 30.0]), axis=1))
    assert np.allclose(np.linalg.norm(mesh.vertex_normals, axis=1), 1.0)
