"""CT PNG 的黑色区域与直线黑边质量筛选。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FilterConfig


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reason: str | None
    black_ratio: float
    line_length_px: float | None = None
    black_side_ratio: float | None = None
    valid_side_black_ratio: float | None = None
    line_segment_px: tuple[int, int, int, int] | None = None
    black_ratio_exceeded: bool = False


def _line_quality(mask: np.ndarray, config: FilterConfig) -> tuple[float, float, float, tuple[int, int, int, int]] | None:
    height, width = mask.shape
    min_length = config.line_min_diagonal_fraction * float(np.hypot(width, height))
    edges = cv2.Canny((mask.astype(np.uint8) * 255), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(20, int(min_length / 2.0)),
        minLineLength=int(np.ceil(min_length)),
        maxLineGap=3,
    )
    if lines is None:
        return None
    grid_y, grid_x = np.indices(mask.shape)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        signed_distance = (dx * (grid_y - y1) - dy * (grid_x - x1)) / length
        first_side = signed_distance > 1.5
        second_side = signed_distance < -1.5
        if np.mean(first_side) < 0.10 or np.mean(second_side) < 0.10:
            continue
        first_black = float(np.mean(mask[first_side]))
        second_black = float(np.mean(mask[second_side]))
        black_side = max(first_black, second_black)
        valid_side = min(first_black, second_black)
        if black_side >= config.black_side_min_ratio and valid_side <= config.valid_side_max_black_ratio:
            return length, black_side, valid_side, (int(x1), int(y1), int(x2), int(y2))
    return None


def evaluate_ct_quality(pixels: np.ndarray, config: FilterConfig) -> QualityResult:
    """评估单通道窗口化 CT PNG 是否可进入最终图库。"""

    values = np.asarray(pixels)
    if values.ndim != 2:
        raise ValueError("CT PNG 必须是二维灰度数组")
    if values.size == 0:
        raise ValueError("CT PNG 不能为空")
    black_mask = values < config.black_threshold
    black_ratio = float(np.mean(black_mask))
    line = _line_quality(black_mask, config)
    black_ratio_exceeded = black_ratio > config.black_ratio_limit
    if line is not None:
        length, black_side, valid_side, segment = line
        return QualityResult(
            False,
            "black_boundary_line",
            black_ratio,
            length,
            black_side,
            valid_side,
            segment,
            black_ratio_exceeded,
        )
    if black_ratio_exceeded:
        return QualityResult(False, "black_ratio", black_ratio, black_ratio_exceeded=True)
    return QualityResult(True, None, black_ratio, black_ratio_exceeded=False)
