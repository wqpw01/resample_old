"""从一个二维手工标签平面派生器官与独立 EUS 血管结果。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import (
    DEFAULT_ORGAN_COLORS,
    EUS_VESSEL_IDS,
    ORGAN_BOUNDARY_IDS,
    ManualSegmentationConfig,
)
from .rendering import RenderedSample


EUS_VESSEL_METADATA_SCHEMA_VERSION = "eus-vessel-metadata/v1"


@dataclass(frozen=True)
class ManualLabelPlaneAnalysis:
    organ_labels: list[str]
    organ_boundary_rgb: np.ndarray
    eus_vessel_labels: list[str]
    eus_vessel_features: list[dict[str, float | str]]
    eus_vessel_boundary_rgb: np.ndarray


def _visible_foreground_boundary(mask: np.ndarray) -> np.ndarray:
    """只标记图像内前景/背景变化，不把图像外假定为背景。"""

    boundary = np.zeros(mask.shape, dtype=bool)
    vertical_change = mask[1:] != mask[:-1]
    boundary[1:] |= mask[1:] & vertical_change
    boundary[:-1] |= mask[:-1] & vertical_change
    horizontal_change = mask[:, 1:] != mask[:, :-1]
    boundary[:, 1:] |= mask[:, 1:] & horizontal_change
    boundary[:, :-1] |= mask[:, :-1] & horizontal_change
    return boundary


def _touches_image_edge(points_yx: np.ndarray, height: int, width: int) -> bool:
    y_values = points_yx[:, 0]
    x_values = points_yx[:, 1]
    return bool(
        np.any(y_values == 0)
        or np.any(y_values == height - 1)
        or np.any(x_values == 0)
        or np.any(x_values == width - 1)
    )


def analyze_manual_label_plane(
    labels: np.ndarray,
    width_mm: float,
    length_mm: float,
    config: ManualSegmentationConfig,
) -> ManualLabelPlaneAnalysis:
    """按像素出现语义分析器官，并提取完整 EUS 血管连通域特征。"""

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("手工标签平面必须是二维数组")
    height, width = values.shape
    if height < 2 or width < 2:
        raise ValueError("手工标签平面每个方向至少 2 个像素")
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("方形物理尺寸必须大于零")

    organ_boundary = np.full((height, width, 3), 255, dtype=np.uint8)
    organ_labels: list[str] = []
    for identifier in ORGAN_BOUNDARY_IDS:
        mask = np.isin(values, config.organ_label_values[identifier])
        if not np.any(mask):
            continue
        organ_labels.append(identifier)
        organ_boundary[_visible_foreground_boundary(mask)] = DEFAULT_ORGAN_COLORS[identifier]

    eus_boundary = np.full((height, width, 3), 255, dtype=np.uint8)
    eus_labels: list[str] = []
    eus_features: list[dict[str, float | str]] = []
    x_spacing_mm = width_mm / (width - 1)
    y_spacing_mm = length_mm / (height - 1)
    structure = np.ones((3, 3), dtype=np.uint8)
    for identifier in EUS_VESSEL_IDS:
        mask = np.isin(values, config.eus_vessel_label_values[identifier])
        if not np.any(mask):
            continue
        eus_labels.append(identifier)
        eus_boundary[_visible_foreground_boundary(mask)] = config.eus_vessel_colors[identifier]
        components, component_count = ndimage.label(mask, structure=structure)
        for component_id in range(1, component_count + 1):
            points_yx = np.argwhere(components == component_id)
            if not len(points_yx) or _touches_image_edge(points_yx, height, width):
                continue
            y_values = points_yx[:, 0]
            x_values = points_yx[:, 1]
            eus_features.append(
                {
                    "label": identifier,
                    "x_mm": float(np.mean(x_values) * x_spacing_mm),
                    "y_mm": float(np.mean(y_values) * y_spacing_mm),
                    "area_mm2": float(len(points_yx) * x_spacing_mm * y_spacing_mm),
                }
            )

    return ManualLabelPlaneAnalysis(
        organ_labels=sorted(organ_labels),
        organ_boundary_rgb=organ_boundary,
        eus_vessel_labels=sorted(eus_labels),
        eus_vessel_features=eus_features,
        eus_vessel_boundary_rgb=eus_boundary,
    )


def apply_manual_label_analysis(
    rendered: RenderedSample,
    analysis: ManualLabelPlaneAnalysis,
) -> RenderedSample:
    """附加手工标签图像，同时保持原血管图像和特征不变。"""

    size = rendered.ct.size
    expected_shape = (size[1], size[0], 3)
    if analysis.organ_boundary_rgb.shape != expected_shape:
        raise ValueError("器官边界图尺寸与 CT 不一致")
    if analysis.eus_vessel_boundary_rgb.shape != expected_shape:
        raise ValueError("EUS 血管边界图尺寸与 CT 不一致")

    original_vessels = np.asarray(rendered.boundary_only.convert("RGB"), dtype=np.uint8)
    combined = np.asarray(analysis.organ_boundary_rgb, dtype=np.uint8).copy()
    original_vessel_mask = np.any(original_vessels != 255, axis=2)
    combined[original_vessel_mask] = original_vessels[original_vessel_mask]

    eus_boundary = np.asarray(analysis.eus_vessel_boundary_rgb, dtype=np.uint8)
    eus_mask = np.any(eus_boundary != 255, axis=2)
    ct_eus_overlay = np.asarray(rendered.ct.convert("RGB"), dtype=np.uint8).copy()
    ct_eus_overlay[eus_mask] = eus_boundary[eus_mask]

    return replace(
        rendered,
        organ_vessel_boundary=Image.fromarray(combined, mode="RGB"),
        organ_labels=list(analysis.organ_labels),
        eus_vessel_boundary=Image.fromarray(eus_boundary, mode="RGB"),
        ct_eus_vessel_overlay=Image.fromarray(ct_eus_overlay, mode="RGB"),
        eus_vessel_features=list(analysis.eus_vessel_features),
        eus_vessel_labels=list(analysis.eus_vessel_labels),
    )
