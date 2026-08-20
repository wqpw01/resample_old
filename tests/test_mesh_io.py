from __future__ import annotations

import numpy as np
import pytest
import trimesh

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


def test_surface_mesh_loader_can_keep_only_the_largest_closed_shell(tmp_path):
    outer = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    inner = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = tmp_path / "nested.ply"
    trimesh.util.concatenate((outer, inner)).export(mesh_path)

    surface = load_surface_mesh(mesh_path, main_outer_surface_only=True)

    assert len(surface.mesh.faces) == len(outer.faces)
    assert surface.surface_audit is not None
    assert surface.surface_audit.input_component_count == 2
    assert surface.surface_audit.kept_face_count == len(outer.faces)
    assert surface.surface_audit.discarded_face_count == len(inner.faces)
    distances = np.linalg.norm(surface.vertices, axis=1)
    assert np.allclose(distances, 5.0)


def test_outer_surface_mode_welds_stl_triangle_vertices_before_shell_selection(tmp_path):
    mesh_path = tmp_path / "surface.stl"
    expected = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    expected.export(mesh_path)

    surface = load_surface_mesh(mesh_path, main_outer_surface_only=True)

    assert surface.mesh.is_watertight
    assert surface.mesh.is_winding_consistent
    assert len(surface.mesh.faces) == len(expected.faces)
    assert np.allclose(np.linalg.norm(surface.vertices, axis=1), 5.0, atol=1e-5)


def test_outer_surface_mode_repairs_inverted_shell_to_outward_normals(tmp_path):
    inverted = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    inverted.faces = inverted.faces[:, ::-1]
    mesh_path = tmp_path / "inverted.ply"
    inverted.export(mesh_path)

    surface = load_surface_mesh(mesh_path, main_outer_surface_only=True)

    assert surface.mesh.is_winding_consistent
    assert surface.mesh.volume > 0.0
    assert np.all(np.einsum("ij,ij->i", surface.vertices, surface.vertex_normals) > 0.0)


def test_outer_surface_mode_repairs_winding_before_selecting_geometric_outer_shell(tmp_path):
    outer = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    outer.faces[: len(outer.faces) // 2] = outer.faces[: len(outer.faces) // 2, ::-1]
    inner = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh_path = tmp_path / "mixed-winding.ply"
    trimesh.util.concatenate((outer, inner)).export(mesh_path)

    surface = load_surface_mesh(mesh_path, main_outer_surface_only=True)

    assert len(surface.mesh.faces) == len(outer.faces)
    assert surface.mesh.is_watertight
    assert surface.mesh.is_winding_consistent
    assert surface.mesh.volume > 0.0
    assert np.allclose(np.linalg.norm(surface.vertices, axis=1), 5.0)


def test_outer_surface_mode_rejects_ambiguous_open_outer_shell(tmp_path):
    outer = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
    outer.update_faces(np.arange(len(outer.faces) - 1))
    inner = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    mesh_path = tmp_path / "open-outer.ply"
    trimesh.util.concatenate((outer, inner)).export(mesh_path)

    with pytest.raises(ValueError, match="非闭合|外壳"):
        load_surface_mesh(mesh_path, main_outer_surface_only=True)


def test_outer_surface_mode_preserves_single_shell_vertex_order(tmp_path):
    mesh_path = tmp_path / "single.ply"
    trimesh.creation.icosphere(subdivisions=2, radius=5.0).export(mesh_path)

    ordinary = load_surface_mesh(mesh_path)
    outer_only = load_surface_mesh(mesh_path, main_outer_surface_only=True)

    assert np.array_equal(outer_only.vertices, ordinary.vertices)
    assert np.array_equal(outer_only.vertex_normals, ordinary.vertex_normals)
