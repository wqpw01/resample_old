"""CT、血管边界与叠加图渲染。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .geometry import SectionContour


@dataclass(frozen=True)
class VesselLayer:
    identifier: str
    label: str
    color: tuple[int, int, int]
    contours: list[SectionContour]


@dataclass(frozen=True)
class RenderedSample:
    ct: Image.Image
    boundary_only: Image.Image
    ct_overlay: Image.Image
    organ_vessel_boundary: Image.Image
    features: list[dict[str, float | str]]
    organ_labels: list[str]
    eus_vessel_boundary: Image.Image | None = None
    ct_eus_vessel_overlay: Image.Image | None = None
    eus_vessel_features: list[dict[str, float | str]] | None = None
    eus_vessel_labels: list[str] | None = None


def _to_pixels(points_mm: np.ndarray, width_mm: float, length_mm: float, size: tuple[int, int]) -> list[tuple[int, int]]:
    width_px, height_px = size
    return [
        (
            round(float(point[0]) / width_mm * (width_px - 1)),
            round(float(point[1]) / length_mm * (height_px - 1)),
        )
        for point in points_mm
    ]


def render_sample_images(
    ct_pixels: np.ndarray,
    width_mm: float,
    length_mm: float,
    layers: Iterable[VesselLayer],
) -> RenderedSample:
    """在同一物理坐标范围内生成 CT、分层边界图和截面特征。"""

    pixels = np.asarray(ct_pixels, dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("ct_pixels 必须是二维灰度数组")
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("方形物理尺寸必须大于零")
    ct = Image.fromarray(pixels, mode="L")
    boundary_only = Image.new("RGB", ct.size, "white")
    organ_vessel_boundary = Image.new("RGB", ct.size, "white")
    overlay = ct.convert("RGB")
    boundary_draw = ImageDraw.Draw(boundary_only)
    combined_draw = ImageDraw.Draw(organ_vessel_boundary)
    overlay_draw = ImageDraw.Draw(overlay)
    line_width = max(1, round(min(ct.size) / 150.0))
    features: list[dict[str, float | str]] = []
    for layer in layers:
        for contour in layer.contours:
            points = _to_pixels(contour.points_mm, width_mm, length_mm, ct.size)
            if len(points) >= 2:
                closed = points + [points[0]]
                boundary_draw.line(closed, fill=layer.color, width=line_width)
                combined_draw.line(closed, fill=layer.color, width=line_width)
                overlay_draw.line(closed, fill=layer.color, width=line_width)
            if contour.complete:
                features.append(
                    {
                        "label": layer.label,
                        "x_mm": float(contour.centroid_mm[0]),
                        "y_mm": float(contour.centroid_mm[1]),
                        "area_mm2": float(contour.area_mm2),
                    }
                )
    return RenderedSample(
        ct=ct,
        boundary_only=boundary_only,
        ct_overlay=overlay,
        organ_vessel_boundary=organ_vessel_boundary,
        features=features,
        organ_labels=[],
    )
