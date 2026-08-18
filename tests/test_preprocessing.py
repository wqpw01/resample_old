from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh
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


MANUAL_ORGAN_LABEL_VALUES = {
    "spleen": (1,),
    "kidney_right": (2,),
    "kidney_left": (3,),
    "gallbladder": (4,),
    "esophagus": (5,),
    "liver": (6,),
    "stomach": (7,),
    "aorta": (8,),
    "inferior_vena_cava": (9,),
    "pancreas": (11,),
    "adrenal_gland_right": (12,),
    "adrenal_gland_left": (13,),
    "duodenum": (14,),
    "portal_vein": (23, 26, 33, 34, 35, 36, 37),
}


def _manual_case_inputs(tmp_path: Path) -> dict[str, Path]:
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    label_values = sorted({value for values in MANUAL_ORGAN_LABEL_VALUES.values() for value in values})
    for value, coordinate in zip(label_values, np.ndindex((6, 6, 6))):
        labels[coordinate[0] + 1, coordinate[1] + 1, coordinate[2] + 1] = value
    ct = _image(np.zeros_like(labels, dtype=np.int16), (10.0, 20.0, 30.0))
    segmentation = _image(labels, (10.0, 20.0, 30.0))
    ct_path = tmp_path / "ct.nrrd"
    segmentation_path = tmp_path / "EUS-main-organ.seg.nrrd"
    sitk.WriteImage(ct, str(ct_path))
    sitk.WriteImage(segmentation, str(segmentation_path))
    artery_path = tmp_path / "artery_tree.ply"
    vein_path = tmp_path / "vein_tree.ply"
    trimesh.creation.box().export(artery_path)
    trimesh.creation.icosphere().export(vein_path)
    return {
        "ct": ct_path,
        "segmentation": segmentation_path,
        "artery": artery_path,
        "vein": vein_path,
    }


def test_manual_preprocessing_writes_only_organs_and_references_external_vessels(tmp_path):
    from ct_vascular_resampling.config import load_case_config
    from ct_vascular_resampling.manual_preprocessing import write_manual_segmentation_case

    inputs = _manual_case_inputs(tmp_path)
    output = tmp_path / "prepared"
    source_segmentation = inputs["segmentation"].read_bytes()
    artery_before = inputs["artery"].read_bytes()
    vein_before = inputs["vein"].read_bytes()

    result = write_manual_segmentation_case(
        ct_path=inputs["ct"],
        segmentation_path=inputs["segmentation"],
        artery_model_path=inputs["artery"],
        vein_model_path=inputs["vein"],
        output_directory=output,
        output_root=tmp_path / "gallery-output",
        case_id="case_2_manual",
    )

    mask_paths = sorted((output / "masks").glob("*.nrrd"))
    model_paths = sorted((output / "models").glob("*.ply"))
    copied_segmentation = output / "segmentation" / "EUS-main-organ.seg.nrrd"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(result["case_config_path"]).read_text(encoding="utf-8"))
    loaded = load_case_config(result["case_config_path"])

    assert len(mask_paths) == len(model_paths) == 14
    assert {path.stem for path in mask_paths} == set(config["organ_models"])
    assert {path.stem for path in model_paths} == set(config["organ_models"])
    assert "artery_tree" not in {path.stem for path in model_paths}
    assert "vein_tree" not in {path.stem for path in model_paths}
    assert copied_segmentation.read_bytes() == source_segmentation
    assert manifest["segmentation"]["sha256"] == hashlib.sha256(source_segmentation).hexdigest()
    assert manifest["organ_models"]["portal_vein_and_splenic_vein"]["source_label_values"] == [
        23,
        26,
        33,
        34,
        35,
        36,
        37,
    ]
    assert manifest["external_vessel_models"]["artery_tree"]["path"] == str(
        inputs["artery"].resolve()
    )
    assert manifest["external_vessel_models"]["vein_tree"]["path"] == str(
        inputs["vein"].resolve()
    )
    assert manifest["external_vessel_models"]["artery_tree"]["sha256"] == hashlib.sha256(
        artery_before
    ).hexdigest()
    assert manifest["external_vessel_models"]["vein_tree"]["sha256"] == hashlib.sha256(
        vein_before
    ).hexdigest()
    assert inputs["artery"].read_bytes() == artery_before
    assert inputs["vein"].read_bytes() == vein_before
    assert config["filtering"]["black_ratio_limit"] == 0.60
    assert config["manual_segmentation"]["organ_label_values"] == {
        name: list(values) for name, values in MANUAL_ORGAN_LABEL_VALUES.items()
    }
    assert config["manual_segmentation"]["eus_vessel_label_values"]["portal_vein"] == [
        26,
        33,
        34,
        35,
        36,
        37,
    ]
    assert config["manual_segmentation"]["eus_vessel_colors"] == {
        "aorta": [255, 0, 0],
        "inferior_vena_cava": [0, 0, 255],
        "portal_vein": [170, 85, 255],
    }
    assert loaded.manual_segmentation is not None
    assert loaded.vessel_models[0].path == inputs["artery"].resolve()
    assert loaded.vessel_models[1].path == inputs["vein"].resolve()


def test_manual_preprocessing_accepts_a_selected_dicom_series(monkeypatch, tmp_path):
    from ct_vascular_resampling import manual_preprocessing

    inputs = _manual_case_inputs(tmp_path)
    dicom_directory = tmp_path / "dicom"
    dicom_directory.mkdir()
    (dicom_directory / "slice-001.dcm").write_bytes(b"test-dicom")
    ct_image = sitk.ReadImage(str(inputs["ct"]))

    def read_selected(source, *, dicom_series_uid):
        assert source == dicom_directory.resolve()
        assert dicom_series_uid == "1.2.840.test"
        return ct_image

    monkeypatch.setattr(manual_preprocessing, "read_ct_image", read_selected)

    result = manual_preprocessing.write_manual_segmentation_case(
        ct_path=dicom_directory,
        dicom_series_uid="1.2.840.test",
        segmentation_path=inputs["segmentation"],
        artery_model_path=inputs["artery"],
        vein_model_path=inputs["vein"],
        output_directory=tmp_path / "prepared",
        output_root=tmp_path / "gallery-output",
        case_id="dicom-manual",
    )

    generated = yaml.safe_load(Path(result["case_config_path"]).read_text(encoding="utf-8"))
    assert generated["ct_path"] == str(dicom_directory.resolve())
    assert generated["dicom_series_uid"] == "1.2.840.test"


@pytest.mark.parametrize(
    "failure",
    ["geometry", "missing_label", "missing_portal_segment", "missing_external_vessel"],
)
def test_manual_preprocessing_preflight_failures_write_nothing(tmp_path, failure):
    from ct_vascular_resampling.manual_preprocessing import write_manual_segmentation_case

    inputs = _manual_case_inputs(tmp_path)
    if failure == "geometry":
        segmentation = sitk.ReadImage(str(inputs["segmentation"]))
        segmentation.SetOrigin((10.1, 20.0, 30.0))
        sitk.WriteImage(segmentation, str(inputs["segmentation"]))
    elif failure == "missing_label":
        segmentation = sitk.ReadImage(str(inputs["segmentation"]))
        labels = sitk.GetArrayFromImage(segmentation)
        labels[labels == 14] = 0
        replacement = sitk.GetImageFromArray(labels)
        replacement.CopyInformation(segmentation)
        sitk.WriteImage(replacement, str(inputs["segmentation"]))
    elif failure == "missing_portal_segment":
        segmentation = sitk.ReadImage(str(inputs["segmentation"]))
        labels = sitk.GetArrayFromImage(segmentation)
        labels[labels == 37] = 0
        replacement = sitk.GetImageFromArray(labels)
        replacement.CopyInformation(segmentation)
        sitk.WriteImage(replacement, str(inputs["segmentation"]))
    else:
        inputs["vein"].unlink()
    output = tmp_path / "prepared"

    with pytest.raises((FileNotFoundError, ValueError)):
        write_manual_segmentation_case(
            ct_path=inputs["ct"],
            segmentation_path=inputs["segmentation"],
            artery_model_path=inputs["artery"],
            vein_model_path=inputs["vein"],
            output_directory=output,
            output_root=tmp_path / "gallery-output",
            case_id="case_2_manual",
        )

    assert not output.exists()
