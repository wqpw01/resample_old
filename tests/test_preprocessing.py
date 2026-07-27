from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import yaml


def _image(array: np.ndarray, origin: tuple[float, float, float]) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.25, 1.5, 2.0))
    image.SetOrigin(origin)
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


@pytest.fixture
def ct_image() -> sitk.Image:
    return _image(np.zeros((4, 4, 4), dtype=np.int16), (10.0, 20.0, 30.0))


@pytest.fixture
def segmentation_image() -> sitk.Image:
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[1, 1, 1] = 8
    labels[2, 2, 2] = 20
    return _image(labels, (10.0 + 5e-7, 20.0, 30.0))


def test_case2_vessel_taxonomy_matches_confirmed_slicer_colours():
    from ct_vascular_resampling.preprocessing import (
        ARTERY_LABEL_VALUES,
        PORTAL_AUXILIARY_LABEL_VALUES,
        VEIN_LABEL_VALUES,
    )

    assert ARTERY_LABEL_VALUES == (8, 20, 22, 24, 25, 39, 40)
    assert VEIN_LABEL_VALUES == (9, 23, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 41, 42)
    assert PORTAL_AUXILIARY_LABEL_VALUES == (23, 26, 33, 34, 35, 36, 37)


def test_validate_geometry_accepts_sub_micrometre_origin_difference(ct_image, segmentation_image):
    from ct_vascular_resampling.preprocessing import validate_geometry

    validate_geometry(ct_image, segmentation_image)


def test_build_binary_masks_unions_requested_label_values(segmentation_image):
    from ct_vascular_resampling.preprocessing import build_binary_masks

    masks = build_binary_masks(segmentation_image, {"artery_tree": (8, 20)})

    assert np.array_equal(
        sitk.GetArrayFromImage(masks["artery_tree"]),
        np.isin(sitk.GetArrayFromImage(segmentation_image), (8, 20)).astype(np.uint8),
    )
    assert masks["artery_tree"].GetOrigin() == segmentation_image.GetOrigin()
    assert masks["artery_tree"].GetSpacing() == segmentation_image.GetSpacing()
    assert masks["artery_tree"].GetDirection() == segmentation_image.GetDirection()


def test_mask_to_mesh_uses_origin_spacing_and_direction():
    from ct_vascular_resampling.preprocessing import mask_to_mesh

    mask = _image(np.zeros((4, 4, 4), dtype=np.uint8), (10.0, 20.0, 30.0))
    values = sitk.GetArrayFromImage(mask)
    values[1:3, 1:3, 1:3] = 1
    mask = _image(values, (10.0, 20.0, 30.0))

    mesh = mask_to_mesh(mask)
    expected = np.asarray(mask.TransformContinuousIndexToPhysicalPoint((0.5, 1.0, 1.0)))

    assert np.any(np.linalg.norm(mesh.vertices - expected, axis=1) < 1e-6)
    assert len(mesh.faces) > 0


def test_write_preprocessed_case_writes_ct_masks_meshes_manifest_and_case_yaml(tmp_path, ct_image):
    from ct_vascular_resampling.preprocessing import (
        ARTERY_LABEL_VALUES,
        ORGAN_LABEL_VALUES,
        PORTAL_AUXILIARY_LABEL_VALUES,
        VEIN_LABEL_VALUES,
        write_preprocessed_case,
    )

    label_values = sorted(
        set(value for values in ORGAN_LABEL_VALUES.values() for value in values)
        | set(ARTERY_LABEL_VALUES)
        | set(VEIN_LABEL_VALUES)
        | set(PORTAL_AUXILIARY_LABEL_VALUES)
    )
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    for value, coordinate in zip(label_values, np.ndindex((6, 6, 6))):
        labels[coordinate[0] + 1, coordinate[1] + 1, coordinate[2] + 1] = value
    case_ct = _image(np.zeros_like(labels, dtype=np.int16), ct_image.GetOrigin())
    segmentation = _image(labels, ct_image.GetOrigin())

    result = write_preprocessed_case(case_ct, segmentation, tmp_path, Path('/tmp/2021.py'))

    assert (tmp_path / 'ct' / 'ct_venous.nrrd').is_file()
    assert (tmp_path / 'masks' / 'artery_tree.nrrd').is_file()
    assert (tmp_path / 'models' / 'vein_tree.ply').is_file()
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    config = yaml.safe_load((tmp_path / 'case_preprocessed.yaml').read_text(encoding='utf-8'))
    assert result['model_count'] == 16
    assert manifest['models']['artery_tree']['source_label_values'] == list(ARTERY_LABEL_VALUES)
    assert manifest['segmentation_origin_max_delta_mm'] < 1e-6
    assert config['organ_models']['portal_vein_and_splenic_vein'] == 'models/portal_vein_and_splenic_vein.ply'
    assert [model['label'] for model in config['vessel_models']] == ['artery', 'vein']


def test_case2_runner_parser_requires_dicom_segmentation_and_output():
    from scripts.preprocess_slicer_case import build_parser

    args = build_parser().parse_args(['--dicom-dir', 'dicom', '--segmentation', 'seg.nrrd', '--output', 'out'])

    assert args.series_id is None
