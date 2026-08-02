"""DICOM CT 与 3D Slicer 分割的下游重采样输入预处理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import SimpleITK as sitk
import trimesh
import yaml
from skimage.measure import marching_cubes


ORGAN_LABEL_VALUES = {
    "spleen": (1,),
    "kidney_right": (2,),
    "kidney_left": (3,),
    "gallbladder": (4,),
    "esophagus": (5,),
    "liver": (6,),
    "stomach": (7,),
    "aorta": (8,),
    "inferior_vena_cava": (9,),
    "pancreas": (11,),
    "adrenal_gland_right": (12,),
    "adrenal_gland_left": (13,),
    "duodenum": (14,),
}

# 依据病例 2 的 3D Slicer 红色/蓝色标注锁定，不将导管或实体器官并入血管树。
ARTERY_LABEL_VALUES = (8, 20, 22, 24, 25, 39, 40)
VEIN_LABEL_VALUES = (9, 23, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 41, 42)
PORTAL_AUXILIARY_LABEL_VALUES = (23, 26, 33, 34, 35, 36, 37)


def case2_model_label_values() -> dict[str, tuple[int, ...]]:
    """返回正式重采样所需器官、辅助门静脉和两类血管的标签映射。"""

    return {
        **ORGAN_LABEL_VALUES,
        "portal_vein_and_splenic_vein": PORTAL_AUXILIARY_LABEL_VALUES,
        "artery_tree": ARTERY_LABEL_VALUES,
        "vein_tree": VEIN_LABEL_VALUES,
    }


def validate_geometry(ct: sitk.Image, segmentation: sitk.Image, atol_mm: float = 1e-6) -> None:
    """确认 CT 与标签图使用同一离散网格和物理空间。"""

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
    """从一个离散标签图提取保持原空间的非空二值掩膜。"""

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
    """将二值掩膜转换为保持 ITK 物理空间的闭合三角网格。"""

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
    registration_module_path: Path,
    output_root: str = "resampling_output",
    deduplicate_degenerate_edge_angles: bool = False,
) -> dict:
    organ_models = {
        name: f"models/{name}.ply"
        for name in (*ORGAN_LABEL_VALUES, "portal_vein_and_splenic_vein")
    }
    return {
        "case_id": case_id,
        "ct_path": "ct/ct_venous.nrrd",
        "output_root": output_root,
        "registration_module_path": str(registration_module_path),
        "organ_models": organ_models,
        "vessel_models": [
            {"id": "artery_tree", "path": "models/artery_tree.ply", "label": "artery", "color": [255, 82, 0]},
            {"id": "vein_tree", "path": "models/vein_tree.ply", "label": "vein", "color": [0, 188, 212]},
        ],
        "sampling": {
            "point_counts": {
                "stomach": 1000,
                "liver": 500,
                "pancreas": 500,
                "duodenum_part1": 500,
                "duodenum_part2": 500,
                "esophagus": 200,
            },
            "stomach_search_distance_mm": 10.0,
            "stomach_voxel_pitch_mm": 1.0,
        },
        "square": {
            "side_length_mm": 100.0,
            "deduplicate_degenerate_edge_angles": deduplicate_degenerate_edge_angles,
        },
        "ct": {"output_resolution": 300, "window_level": 40.0, "window_width": 400.0, "fill_hu_value": -1000.0},
        "filtering": {
            "black_threshold": 50,
            "black_ratio_limit": 0.50,
            "line_min_diagonal_fraction": 0.70,
            "black_side_min_ratio": 0.90,
            "valid_side_max_black_ratio": 0.10,
        },
        "runtime": {"seed": 0, "workers": 8},
    }


def write_preprocessed_masks_case(
    ct: sitk.Image,
    masks: Mapping[str, sitk.Image],
    source_label_values: Mapping[str, tuple[int, ...]],
    output_directory: str | Path,
    registration_module_path: str | Path,
    case_id: str = "case_2",
    output_root: str = "resampling_output",
    provenance: Mapping[str, object] | None = None,
    manifest_metadata: Mapping[str, object] | None = None,
    deduplicate_degenerate_edge_angles: bool = False,
) -> dict[str, object]:
    """将已对齐的命名掩膜转换为下游所需网格和病例 YAML。"""

    required = set(ORGAN_LABEL_VALUES) | {"portal_vein_and_splenic_vein", "artery_tree", "vein_tree"}
    missing = sorted(required - set(masks))
    if missing:
        raise ValueError(f"缺少下游重采样必需掩膜: {', '.join(missing)}")
    unexpected = sorted(set(masks) - required)
    if unexpected:
        raise ValueError(f"包含不支持的掩膜: {', '.join(unexpected)}")
    if set(source_label_values) != required:
        raise ValueError("source_label_values 必须与掩膜名称完全一致")
    if ct.GetDimension() != 3:
        raise ValueError("CT 必须是三维图像")
    for name, mask in masks.items():
        validate_geometry(ct, mask)
        if not np.any(sitk.GetArrayViewFromImage(mask)):
            raise ValueError(f"{name} 没有分割体素")

    destination = Path(output_directory)
    ct_directory = destination / "ct"
    mask_directory = destination / "masks"
    model_directory = destination / "models"
    for directory in (ct_directory, mask_directory, model_directory):
        directory.mkdir(parents=True, exist_ok=True)

    ct_path = ct_directory / "ct_venous.nrrd"
    sitk.WriteImage(ct, str(ct_path), useCompression=True)
    records: dict[str, dict[str, object]] = {}
    for name in sorted(required):
        mask = masks[name]
        mask_path = mask_directory / f"{name}.nrrd"
        model_path = model_directory / f"{name}.ply"
        sitk.WriteImage(mask, str(mask_path), useCompression=True)
        mesh = mask_to_mesh(mask)
        mesh.export(str(model_path), file_type="ply")
        records[name] = {
            "source_label_values": list(source_label_values[name]),
            "voxel_count": int(np.count_nonzero(sitk.GetArrayViewFromImage(mask))),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "mask": str(mask_path.relative_to(destination)),
            "mesh": str(model_path.relative_to(destination)),
        }

    manifest = {
        "case_id": case_id,
        "ct": {
            "path": str(ct_path.relative_to(destination)),
            "size": list(ct.GetSize()),
            "spacing": list(ct.GetSpacing()),
            "origin": list(ct.GetOrigin()),
            "direction": list(ct.GetDirection()),
        },
        "models": records,
    }
    if provenance:
        manifest["provenance"] = dict(provenance)
    if manifest_metadata:
        reserved = set(manifest).intersection(manifest_metadata)
        if reserved:
            raise ValueError(f"manifest_metadata 不能覆盖保留字段: {', '.join(sorted(reserved))}")
        manifest.update(manifest_metadata)
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    case_config_path = destination / "case_preprocessed.yaml"
    case_config_path.write_text(
        yaml.safe_dump(
            _case_config(
                case_id,
                Path(registration_module_path),
                output_root=output_root,
                deduplicate_degenerate_edge_angles=deduplicate_degenerate_edge_angles,
            ),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "output_directory": str(destination),
        "ct_path": str(ct_path),
        "manifest_path": str(manifest_path),
        "case_config_path": str(case_config_path),
        "model_count": len(records),
    }


def write_preprocessed_case(
    ct: sitk.Image,
    segmentation: sitk.Image,
    output_directory: str | Path,
    registration_module_path: str | Path,
    case_id: str = "case_2",
) -> dict[str, object]:
    """写出可直接供病例重采样使用的 CT、掩膜、网格、清单与 YAML。"""

    validate_geometry(ct, segmentation)
    label_values = case2_model_label_values()
    masks = build_binary_masks(segmentation, label_values)
    for mask in masks.values():
        mask.CopyInformation(ct)
    origin_delta = np.abs(np.asarray(ct.GetOrigin()) - np.asarray(segmentation.GetOrigin()))
    return write_preprocessed_masks_case(
        ct=ct,
        masks=masks,
        source_label_values=label_values,
        output_directory=output_directory,
        registration_module_path=registration_module_path,
        case_id=case_id,
        manifest_metadata={"segmentation_origin_max_delta_mm": float(np.max(origin_delta))},
    )
