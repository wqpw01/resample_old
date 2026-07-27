"""将二维 10 cm 裁剪标签导出为 2021.py 可加载的检索记录。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import SimpleITK as sitk
from scipy import ndimage


VEIN_LABEL_IDS = frozenset({26, 27, 28, 29, 30, 31, 32})
ARTERY_LABEL_IDS = frozenset({33, 34, 35, 36, 37, 38, 39, 40})
_LABEL_GROUPS = (("vein", VEIN_LABEL_IDS), ("artery", ARTERY_LABEL_IDS))
_SCHEMA_VERSION = "cropped-retrieval-features/v1"


@dataclass(frozen=True)
class CroppedFeatureResult:
    folder: Path
    feature_path: Path
    gallery_path: Path
    label_white_path: Path
    record: dict[str, Any]


@dataclass(frozen=True)
class CroppedRootSummary:
    root: Path
    folder_count: int
    gallery_record_count: int


def _tar_members(tar_path: Path) -> tuple[dict[str, Any], bytes | None]:
    with tarfile.open(tar_path) as archive:
        json_members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".json")]
        nifti_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and (member.name.endswith(".nii") or member.name.endswith(".nii.gz"))
        ]
        if len(json_members) != 1 or len(nifti_members) > 1:
            raise ValueError(f"标签 TAR 必须包含一个 JSON 和至多一个 NIfTI: {tar_path}")
        json_file = archive.extractfile(json_members[0])
        if json_file is None:
            raise ValueError(f"无法读取标签 TAR 内容: {tar_path}")
        metadata = json.load(json_file)
        if not nifti_members:
            return metadata, None
        nifti_file = archive.extractfile(nifti_members[0])
        if nifti_file is None:
            raise ValueError(f"无法读取标签 TAR 内容: {tar_path}")
        return metadata, nifti_file.read()


def _read_label_image(payload: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as temporary:
        temporary.write(payload)
        temporary.flush()
        values = sitk.GetArrayFromImage(sitk.ReadImage(temporary.name))
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"裁剪标签必须是二维或单层三维 NIfTI，实际维度: {values.shape}")
    return np.asarray(values)


def _empty_label_image(metadata: dict[str, Any]) -> np.ndarray:
    try:
        width = int(metadata["FileInfo"]["Width"])
        height = int(metadata["FileInfo"]["Height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("无 NIfTI 的标签 JSON 必须提供 FileInfo.Width 和 FileInfo.Height") from error
    if width < 2 or height < 2:
        raise ValueError(f"无 NIfTI 标签的图像尺寸无效: {width} x {height}")
    return np.zeros((height, width), dtype=np.uint16)


def _label_table(metadata: dict[str, Any]) -> dict[int, dict[str, Any]]:
    try:
        entries = metadata["Models"]["ColorLabelTableModel"]
    except (KeyError, TypeError) as error:
        raise ValueError("标签 JSON 缺少 Models.ColorLabelTableModel") from error
    if not isinstance(entries, list):
        raise ValueError("ColorLabelTableModel 必须是列表")
    table: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("ID"), int):
            continue
        color = entry.get("Color", [0, 0, 0])
        if not isinstance(color, list) or len(color) < 3 or any(not isinstance(value, int) for value in color[:3]):
            color = [0, 0, 0]
        table[entry["ID"]] = {
            "description": str(entry.get("Desc", f"label_{entry['ID']}")),
            "color_rgb": [int(value) for value in color[:3]],
        }
    return table


def _ensure_label_white(path: Path, labels: np.ndarray, table: dict[int, dict[str, Any]]) -> None:
    height, width = labels.shape
    if path.is_file():
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"现有白底标签图尺寸与 NIfTI 不一致: {path}")
        return
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for label_id, entry in table.items():
        pixels[labels == label_id] = entry["color_rgb"]
    Image.fromarray(pixels, mode="RGB").save(path)


def _features(
    labels: np.ndarray,
    table: dict[int, dict[str, Any]],
    pixel_spacing_mm: tuple[float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    height, width = labels.shape
    x_spacing, y_spacing = pixel_spacing_mm
    structure = np.ones((3, 3), dtype=np.uint8)
    for feature_label, label_ids in _LABEL_GROUPS:
        for label_id in sorted(label_ids):
            components, count = ndimage.label(labels == label_id, structure=structure)
            for component_id in range(1, count + 1):
                points_yx = np.argwhere(components == component_id)
                if not len(points_yx):
                    continue
                y_values, x_values = points_yx[:, 0], points_yx[:, 1]
                base = {
                    "label": feature_label,
                    "label_id": label_id,
                    "label_desc": table.get(label_id, {}).get("description", f"label_{label_id}"),
                    "component_index": component_id,
                    "area_px": int(len(points_yx)),
                    "centroid_px": [float(np.mean(x_values)), float(np.mean(y_values))],
                }
                touches_edge = bool(
                    np.any(x_values == 0)
                    or np.any(x_values == width - 1)
                    or np.any(y_values == 0)
                    or np.any(y_values == height - 1)
                )
                if touches_edge:
                    skipped.append({**base, "reason": "touches_image_edge"})
                    continue
                features.append(
                    {
                        **base,
                        "x_mm": float(np.mean(x_values) * x_spacing),
                        "y_mm": float(np.mean(y_values) * y_spacing),
                        "area_mm2": float(len(points_yx) * x_spacing * y_spacing),
                    }
                )
    return features, skipped


def _gallery_record(
    stem: str,
    features: list[dict[str, Any]],
    width_mm: float,
    length_mm: float,
    pixel_spacing_mm: tuple[float, float],
) -> dict[str, Any]:
    adapter_features = [
        {key: feature[key] for key in ("label", "x_mm", "y_mm", "area_mm2")}
        for feature in features
    ]
    center = [width_mm / 2.0, length_mm / 2.0, 0.0]
    return {
        "frame_id": stem,
        "slice_id": f"{stem}_cropped",
        "status": "gallery" if adapter_features else "unindexed",
        "organ": "unknown",
        "source": "cropped_label_tar",
        "probe_point_world": center,
        "input_normal_world": [0.0, 0.0, 1.0],
        "input_direction_world": [0.0, 1.0, 0.0],
        "square_vertices_world": [[0.0, 0.0, 0.0], [width_mm, 0.0, 0.0], [width_mm, length_mm, 0.0], [0.0, length_mm, 0.0]],
        "origin_world": [0.0, 0.0, 0.0],
        "center_world": center,
        "u_axis_world": [1.0, 0.0, 0.0],
        "v_axis_world": [0.0, 1.0, 0.0],
        "normal_world": [0.0, 0.0, 1.0],
        "width_mm": width_mm,
        "length_mm": length_mm,
        "pixel_spacing_mm": list(pixel_spacing_mm),
        "ct_png": f"{stem}_cropped.jpg",
        "boundary_only_png": f"{stem}_cropped_label_white.png",
        "ct_overlay_png": f"{stem}_cropped_overlay.png",
        "features": adapter_features,
        "quality": {"accepted": True, "reason": None, "black_ratio": None, "line_length_px": None, "black_side_ratio": None, "valid_side_black_ratio": None},
        "resampling_backend": "label_tar_2d",
        "pose_coordinate_system": "synthetic_2d_10cm_crop",
        "patient_world_pose": False,
    }


def process_cropped_folder(folder: str | Path, width_mm: float = 100.0, length_mm: float = 100.0) -> CroppedFeatureResult:
    """提取一个裁剪帧的完整血管截面，并写入详细 JSON 与单帧 JSONL。"""

    directory = Path(folder)
    stem = directory.name
    tar_path = directory / f"{stem}_cropped_jpg_Label.tar"
    if not tar_path.is_file():
        raise FileNotFoundError(f"未找到裁剪标签 TAR: {tar_path}")
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("裁剪物理尺寸必须大于零")
    metadata, nifti_payload = _tar_members(tar_path)
    if nifti_payload is None:
        labels = _empty_label_image(metadata)
        label_source = "empty_label_json"
    else:
        labels = _read_label_image(nifti_payload)
        label_source = "nifti"
    table = _label_table(metadata)
    height, width = labels.shape
    pixel_spacing_mm = (width_mm / (width - 1), length_mm / (height - 1))
    white_path = directory / f"{stem}_cropped_label_white.png"
    _ensure_label_white(white_path, labels, table)
    features, skipped = _features(labels, table, pixel_spacing_mm)
    record = _gallery_record(stem, features, width_mm, length_mm, pixel_spacing_mm)
    feature_path = directory / f"{stem}_cropped_retrieval_features.json"
    gallery_path = directory / f"{stem}_cropped_gallery.jsonl"
    details = {
        "schema_version": _SCHEMA_VERSION,
        "frame_id": stem,
        "label_tar": tar_path.name,
        "label_source": label_source,
        "label_white_png": white_path.name,
        "image_size_px": [width, height],
        "crop_size_mm": [width_mm, length_mm],
        "pixel_spacing_mm": list(pixel_spacing_mm),
        "feature_coordinate_system": "top_left_origin_x_right_y_down_mm",
        "features": features,
        "skipped_components": skipped,
        "adapter_record": record,
    }
    feature_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gallery_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return CroppedFeatureResult(directory, feature_path, gallery_path, white_path, record)


def process_cropped_root(root: str | Path, width_mm: float = 100.0, length_mm: float = 100.0) -> CroppedRootSummary:
    """处理根目录下所有带匹配裁剪标签 TAR 的帧，结果仅写入各自帧目录。"""

    directory = Path(root)
    folders = [
        child
        for child in sorted(directory.iterdir())
        if child.is_dir() and (child / f"{child.name}_cropped_jpg_Label.tar").is_file()
    ]
    results = [process_cropped_folder(folder, width_mm, length_mm) for folder in folders]
    gallery_record_count = sum(bool(result.record["features"]) for result in results)
    return CroppedRootSummary(directory, len(results), gallery_record_count)
