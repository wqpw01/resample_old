"""三角网格读取与输入预检。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_MESH_SUFFIXES = {".obj", ".stl", ".ply"}


@dataclass(frozen=True)
class SurfaceMesh:
    mesh: trimesh.Trimesh
    vertices: np.ndarray
    vertex_normals: np.ndarray


def load_surface_mesh(path: str | Path) -> SurfaceMesh:
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
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    normals = np.asarray(loaded.vertex_normals, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
        raise ValueError(f"网格顶点或法线格式无效: {source}")
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(magnitudes < 1e-8):
        raise ValueError(f"网格包含零法线顶点: {source}")
    return SurfaceMesh(mesh=loaded, vertices=vertices, vertex_normals=normals / magnitudes)
