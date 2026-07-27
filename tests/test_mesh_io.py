from __future__ import annotations

import pytest

from ct_vascular_resampling.mesh_io import load_surface_mesh


def test_surface_mesh_loader_preserves_vertices_and_computes_normals(tmp_path):
    mesh_path = tmp_path / "surface.obj"
    mesh_path.write_text(
        """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 4 2
f 1 3 4
f 2 4 3
""",
        encoding="utf-8",
    )

    mesh = load_surface_mesh(mesh_path)

    assert mesh.vertices.shape == (4, 3)
    assert mesh.vertex_normals.shape == (4, 3)


def test_surface_mesh_loader_rejects_point_cloud_without_faces(tmp_path):
    cloud_path = tmp_path / "points.ply"
    cloud_path.write_text(
        """ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="三角面"):
        load_surface_mesh(cloud_path)
