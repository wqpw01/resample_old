from __future__ import annotations

import numpy as np

from ct_vascular_resampling.config import FilterConfig
from ct_vascular_resampling.quality import evaluate_ct_quality


def _pixels_with_black_ratio(ratio: float) -> np.ndarray:
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    rng = np.random.default_rng(0)
    pixels.ravel()[rng.choice(pixels.size, int(pixels.size * ratio), replace=False)] = 0
    return pixels


def test_quality_accepts_black_pixels_below_fifty_percent():
    result = evaluate_ct_quality(_pixels_with_black_ratio(0.40), FilterConfig())

    assert result.accepted is True
    assert result.reason is None
    assert result.black_ratio == 0.40


def test_quality_rejects_black_pixels_over_fifty_percent():
    pixels = _pixels_with_black_ratio(0.51)

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_ratio"
    assert result.black_ratio == 0.51


def test_quality_rejects_long_vertical_boundary_with_black_on_one_side():
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :20] = 0

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_boundary_line"
    assert result.line_length_px is not None
    assert result.line_length_px >= 0.7 * np.hypot(100, 100)


def test_quality_prioritizes_boundary_line_when_black_ratio_also_exceeds_limit():
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :60] = 0

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_boundary_line"
    assert result.black_ratio_exceeded is True
    assert result.line_length_px is not None
    assert result.line_segment_px is not None


def test_quality_accepts_ct_without_black_region_or_artifact_line():
    pixels = np.full((100, 100), 127, dtype=np.uint8)

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is True
    assert result.reason is None
