"""三角网格读取与输入预检。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .coordinates import to_ras_points, to_ras_vectors


SUPPORTED_MESH_SUFFIXES = {".obj", ".stl", ".ply"}


@dataclass(frozen=True)
class SurfaceMeshAudit:
    input_component_count: int
    input_face_count: int
    kept_face_count: int
    discarded_face_count: int
    selected_enclosed_volume_mm3: float

    def to_record(self) -> dict[str, int | float | bool]:
        return {
            "enabled": True,
            "selection_rule": "largest_watertight_absolute_volume",
            "input_component_count": self.input_component_count,
            "input_face_count": self.input_face_count,
            "kept_face_count": self.kept_face_count,
            "discarded_face_count": self.discarded_face_count,
            "selected_enclosed_volume_mm3": self.selected_enclosed_volume_mm3,
        }


@dataclass(frozen=True)
class SurfaceMesh:
    mesh: trimesh.Trimesh
    vertices: np.ndarray
    vertex_normals: np.ndarray
    surface_audit: SurfaceMeshAudit | None = None


def load_surface_mesh(
    path: str | Path,
    *,
    input_coordinate_system: str = "RAS",
    main_outer_surface_only: bool = False,
) -> SurfaceMesh:
    """读取可进行截面相交的 OBJ/STL/PLY 三角网格。"""

    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(f"仅支持 OBJ、STL 或 PLY 网格: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"网格文件不存在: {source}")
    loaded = trimesh.load(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"网格文件没有几何对象: {source}")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"网格必须包含三角面，不能使用纯点云: {source}")
    surface_audit: SurfaceMeshAudit | None = None
    if main_outer_surface_only:
        topology = loaded.copy()
        topology.merge_vertices()
        topology.remove_unreferenced_vertices()
        components = topology.split(only_watertight=False, repair=False)
        if any(not component.is_watertight for component in components):
            raise ValueError(f"采样源网格包含非闭合连通分量，无法可靠判定主外壳: {source}")
        watertight: list[trimesh.Trimesh] = []
        for component in components:
            if not component.is_watertight:
                continue
            component.fix_normals(multibody=False)
            component = trimesh.Trimesh(
                vertices=np.asarray(component.vertices).copy(),
                faces=np.asarray(component.faces).copy(),
                process=False,
            )
            if component.is_winding_consistent and float(component.volume) > 0.0:
                watertight.append(component)
        if not watertight:
            raise ValueError(f"采样源网格没有可验证的闭合外壳: {source}")
        if (
            len(components) == 1
            and loaded.is_watertight
            and loaded.is_winding_consistent
            and float(loaded.volume) > 0.0
        ):
            retained = loaded
        else:
            retained = max(
                watertight,
                key=lambda component: (float(component.volume), len(component.faces)),
            )
        surface_audit = SurfaceMeshAudit(
            input_component_count=len(components),
            input_face_count=len(loaded.faces),
            kept_face_count=len(retained.faces),
            discarded_face_count=len(loaded.faces) - len(retained.faces),
            selected_enclosed_volume_mm3=float(retained.volume),
        )
        loaded = retained
    vertices = to_ras_points(np.asarray(loaded.vertices, dtype=np.float64), input_coordinate_system)
    if main_outer_surface_only:
        face_normals, valid_faces = trimesh.triangles.normals(
            np.asarray(loaded.vertices)[np.asarray(loaded.faces)]
        )
        if not np.all(valid_faces):
            raise ValueError(f"采样源主外壳包含退化三角面: {source}")
        source_normals = trimesh.geometry.weighted_vertex_normals(
            vertex_count=len(loaded.vertices),
            faces=loaded.faces,
            face_normals=face_normals,
            face_angles=loaded.face_angles,
        )
    else:
        source_normals = loaded.vertex_normals
    normals = to_ras_vectors(np.asarray(source_normals, dtype=np.float64), input_coordinate_system)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
        raise ValueError(f"网格顶点或法线格式无效: {source}")
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(magnitudes < 1e-8):
        raise ValueError(f"网格包含零法线顶点: {source}")
    loaded.vertices = vertices
    return SurfaceMesh(
        mesh=loaded,
        vertices=vertices,
        vertex_normals=normals / magnitudes,
        surface_audit=surface_audit,
    )
