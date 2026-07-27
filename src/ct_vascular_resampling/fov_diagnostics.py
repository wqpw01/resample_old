"""CT 体积边界与黑色拒绝切片的空间归因。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FilterConfig
from .ct_resampling import CTVolume, SquareFovDiagnosis, diagnose_square_fov
from .quality import QualityResult


@dataclass(frozen=True)
class RejectedFovAssessment:
    """拒绝 CT 切片中黑区与 CT FOV 的对应关系。"""

    fov: SquareFovDiagnosis
    black_out_of_bounds_overlap_ratio: float
    out_of_bounds_black_ratio: float
    boundary_line_oob_alignment_ratio: float | None
    cause: str

    def to_record(self) -> dict[str, object]:
        return {
            **self.fov.to_record(),
            "black_out_of_bounds_overlap_ratio": self.black_out_of_bounds_overlap_ratio,
            "out_of_bounds_black_ratio": self.out_of_bounds_black_ratio,
            "boundary_line_oob_alignment_ratio": self.boundary_line_oob_alignment_ratio,
            "cause": self.cause,
        }


def _line_oob_alignment(segment: tuple[int, int, int, int] | None, out_of_bounds_mask: np.ndarray) -> float | None:
    if segment is None:
        return None
    mask = np.asarray(out_of_bounds_mask, dtype=np.uint8)
    edges = cv2.Canny(mask * 255, 50, 150)
    boundary_band = cv2.dilate(edges, np.ones((5, 5), dtype=np.uint8)) > 0
    x1, y1, x2, y2 = segment
    point_count = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.rint(np.linspace(x1, x2, point_count)).astype(int)
    ys = np.rint(np.linspace(y1, y2, point_count)).astype(int)
    valid = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    if not np.any(valid):
        return 0.0
    return float(np.mean(boundary_band[ys[valid], xs[valid]]))


def assess_rejected_fov(
    volume: CTVolume,
    vertices_world: np.ndarray,
    pixels: np.ndarray,
    filter_settings: FilterConfig,
    quality: QualityResult,
    *,
    probe_point_world: np.ndarray | None = None,
) -> RejectedFovAssessment:
    """评估已拒绝切片的黑色区域是否由 CT FOV 常量填充产生。"""

    values = np.asarray(pixels)
    if values.ndim != 2:
        raise ValueError("CT PNG 必须是二维灰度数组")
    fov = diagnose_square_fov(volume, vertices_world, values.shape[0], probe_point_world)
    if values.shape != fov.out_of_bounds_mask.shape:
        raise ValueError("CT PNG 与方形 FOV 掩码尺寸不一致")
    black_mask = values < filter_settings.black_threshold
    overlap = black_mask & fov.out_of_bounds_mask
    black_count = int(np.count_nonzero(black_mask))
    out_of_bounds_count = int(np.count_nonzero(fov.out_of_bounds_mask))
    black_overlap_ratio = float(np.count_nonzero(overlap) / black_count) if black_count else 0.0
    out_of_bounds_black_ratio = float(np.count_nonzero(overlap) / out_of_bounds_count) if out_of_bounds_count else 0.0
    line_alignment = _line_oob_alignment(quality.line_segment_px, fov.out_of_bounds_mask)
    if not fov.contains_ct_fov_exceedance:
        cause = "in_volume_low_hu_or_padding"
    elif line_alignment is not None and line_alignment >= 0.8:
        cause = "fov_boundary_aligned"
    elif black_overlap_ratio >= 0.8:
        cause = "ct_fov_exceeded"
    else:
        cause = "mixed_fov_and_in_volume_black"
    return RejectedFovAssessment(fov, black_overlap_ratio, out_of_bounds_black_ratio, line_alignment, cause)
