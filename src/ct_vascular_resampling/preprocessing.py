"""手工分割输入共享的医学影像几何与网格工具。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage.measure import marching_cubes

from .contract import BLACK_RATIO_LIMIT


def validate_geometry(ct: sitk.Image, segmentation: sitk.Image, atol_mm: float = 1e-6) -> None:
    """确认 CT 与标签图使用同一离散网格和患者物理空间。"""

    if ct.GetDimension() != 3 or segmentation.GetDimension() != 3:
        raise ValueError("CT 与分割必须都是三维图像")
    if ct.GetSize() != segmentation.GetSize():
        raise ValueError("CT 与分割的 Size 不一致")
    if not np.allclose(ct.GetSpacing(), segmentation.GetSpacing(), atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与分割的 Spacing 不一致")
    if not np.allclose(ct.GetOrigin(), segmentation.GetOrigin(), atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与分割的 Origin 不一致")
    if not np.allclose(ct.GetDirection(), segmentation.GetDirection(), atol=1e-8, rtol=0.0):
        raise ValueError("CT 与分割的 Direction 不一致")


def build_binary_masks(
    segmentation: sitk.Image,
    mappings: Mapping[str, tuple[int, ...]],
) -> dict[str, sitk.Image]:
    """从离散标签图提取保持 Size/Spacing/Origin/Direction 的非空二值掩膜。"""

    labels = sitk.GetArrayViewFromImage(segmentation)
    masks: dict[str, sitk.Image] = {}
    for name, values in mappings.items():
        binary = np.isin(labels, values).astype(np.uint8)
        if not np.any(binary):
            raise ValueError(f"{name} 没有分割体素")
        image = sitk.GetImageFromArray(binary)
        image.CopyInformation(segmentation)
        masks[name] = image
    return masks


def mask_to_mesh(mask: sitk.Image) -> trimesh.Trimesh:
    """将二值掩膜转换为保持 ITK 患者物理坐标的闭合三角网格。"""

    values = sitk.GetArrayViewFromImage(mask)
    if values.ndim != 3 or not np.any(values):
        raise ValueError("网格提取需要非空三维二值掩膜")
    padded = np.pad(np.asarray(values, dtype=np.uint8), 1, mode="constant")
    vertices_zyx, faces, _, _ = marching_cubes(padded.astype(np.float32), level=0.5)
    indices_xyz = vertices_zyx[:, [2, 1, 0]] - 1.0
    scaled = indices_xyz * np.asarray(mask.GetSpacing(), dtype=np.float64)
    direction = np.asarray(mask.GetDirection(), dtype=np.float64).reshape(3, 3)
    vertices_xyz = scaled @ direction.T + np.asarray(mask.GetOrigin(), dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)
    mesh.fix_normals(multibody=True)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("掩膜未能生成三角网格")
    return mesh


def _case_config(
    case_id: str,
    output_root: str = "resampling_output",
) -> dict[str, object]:
    """生成手工预处理器补全器官、血管和标签字段前的基础配置。"""

    return {
        "case_id": case_id,
        "ct_path": "ct.nrrd",
        "output_root": output_root,
        "organ_models": {},
        "vessel_models": [],
        "geometry": {"input_coordinate_system": "LPS", "canonical_coordinate_system": "RAS"},
        "sampling": {
            "point_counts": {
                "stomach": 1000,
                "liver": 500,
                "pancreas": 500,
                "duodenum_part1": 500,
                "duodenum_part2": 500,
                "esophagus": 200,
            },
            "ray_length_mm": 100.0,
            "ray_batch_size": 2048,
            "minimum_spacing_mm": 10.0,
            "centerline_voxel_pitch_mm": 1.0,
            "centerline_tangent_window_mm": 10.0,
            "centerline_max_terminal_spur_mm": 5.0,
        },
        "square": {"side_length_mm": 100.0},
        "ct": {
            "output_resolution": 300,
            "window_level": 40.0,
            "window_width": 400.0,
            "fill_hu_value": -1000.0,
        },
        "filtering": {
            "black_threshold": 50,
            "black_ratio_limit": BLACK_RATIO_LIMIT,
            "line_min_diagonal_fraction": 0.70,
            "black_side_min_ratio": 0.90,
            "valid_side_max_black_ratio": 0.10,
        },
        "runtime": {"seed": 0, "workers": 8, "backend": "auto", "gpu_device": 0, "gpu_batch_size": 8},
    }
