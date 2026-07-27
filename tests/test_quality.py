from __future__ import annotations

import numpy as np

from ct_vascular_resampling.config import FilterConfig
from ct_vascular_resampling.quality import evaluate_ct_quality


def test_quality_rejects_black_pixels_over_thirty_percent():
    pixels = np.full((10, 10), 127, dtype=np.uint8)
    pixels.ravel()[:31] = 0

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_ratio"
    assert result.black_ratio == 0.31


def test_quality_rejects_long_vertical_boundary_with_black_on_one_side():
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :20] = 0

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_boundary_line"
    assert result.line_length_px is not None
    assert result.line_length_px >= 0.7 * np.hypot(100, 100)


def test_quality_keeps_boundary_line_evidence_when_black_ratio_already_rejects():
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :60] = 0

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is False
    assert result.reason == "black_ratio"
    assert result.line_length_px is not None
    assert result.line_segment_px is not None


def test_quality_accepts_ct_without_black_region_or_artifact_line():
    pixels = np.full((100, 100), 127, dtype=np.uint8)

    result = evaluate_ct_quality(pixels, FilterConfig())

    assert result.accepted is True
    assert result.reason is None
