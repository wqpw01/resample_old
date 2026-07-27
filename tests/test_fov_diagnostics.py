from __future__ import annotations

import importlib

import numpy as np
import SimpleITK as sitk

from ct_vascular_resampling.config import FilterConfig
from ct_vascular_resampling.ct_resampling import CTVolume
from ct_vascular_resampling.quality import evaluate_ct_quality


def test_rejected_fov_assessment_identifies_an_out_of_bounds_black_boundary():
    module = importlib.import_module("ct_vascular_resampling.fov_diagnostics")
    volume = CTVolume.from_sitk(sitk.GetImageFromArray(np.full((1, 100, 100), 40, dtype=np.int16)))
    vertices = np.asarray(
        [
            [-100.0, 0.0, 0.0],
            [99.0, 0.0, 0.0],
            [99.0, 99.0, 0.0],
            [-100.0, 99.0, 0.0],
        ]
    )
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :50] = 0
    settings = FilterConfig()
    quality = evaluate_ct_quality(pixels, settings)

    assessment = module.assess_rejected_fov(
        volume,
        vertices,
        pixels,
        settings,
        quality,
        probe_point_world=np.asarray([0.0, 0.0, 0.0]),
    )

    assert assessment.fov.contains_ct_fov_exceedance is True
    assert assessment.fov.out_of_bounds_ratio == 0.5
    assert assessment.black_out_of_bounds_overlap_ratio == 1.0
    assert assessment.boundary_line_oob_alignment_ratio is not None
    assert assessment.cause == "fov_boundary_aligned"
