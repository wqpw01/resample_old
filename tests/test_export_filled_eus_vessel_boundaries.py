from pathlib import Path

import numpy as np
import pytest

from scripts.export_filled_eus_vessel_boundaries import (
    CLASS_SPECS,
    render_filled_rgb,
    validate_boundary_subset,
    validate_filename_sets,
)


def test_render_filled_rgb_fills_edge_touching_pixels() -> None:
    labels = np.zeros((4, 5), dtype=np.uint8)
    labels[:, :2] = 8

    rendered = render_filled_rgb(labels, CLASS_SPECS["aorta"])

    assert np.all(rendered[:, :2] == (255, 0, 0))
    assert np.all(rendered[:, 2:] == 255)


def test_render_filled_rgb_merges_only_eus_portal_labels() -> None:
    labels = np.asarray([[23, 26, 33, 34, 35, 36, 37]], dtype=np.uint8)

    rendered = render_filled_rgb(labels, CLASS_SPECS["portal_vein"])

    assert np.all(rendered[0, 0] == 255)
    assert np.all(rendered[0, 1:] == (170, 85, 255))


def test_validate_boundary_subset_rejects_pixels_outside_source_mask() -> None:
    source_mask = np.asarray([[False, True], [False, False]])
    boundary_mask = np.asarray([[True, True], [False, False]])

    with pytest.raises(ValueError, match="边界像素不属于重采样标签"):
        validate_boundary_subset(boundary_mask, source_mask, Path("sample.png"))


def test_validate_filename_sets_requires_exact_100_file_match() -> None:
    names = {f"sample-{index:03d}.png" for index in range(100)}
    validate_filename_sets({key: names for key in CLASS_SPECS}, names)

    actual = {key: names for key in CLASS_SPECS}
    actual["aorta"] = names - {"sample-000.png"}
    with pytest.raises(ValueError, match="文件名集合不一致"):
        validate_filename_sets(actual, names)
