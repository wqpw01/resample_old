from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml


def _image(values: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.25, 1.5, 2.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _mask() -> sitk.Image:
    values = np.zeros((8, 8, 8), dtype=np.uint8)
    values[2:6, 2:6, 2:6] = 1
    return _image(values)


def test_auto_preprocessing_merges_portal_into_vein_and_writes_case_config(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import (
        AUTO_ORGAN_IDS,
        write_auto_preprocessed_case,
    )

    ct = _image(np.zeros((8, 8, 8), dtype=np.int16))
    vascular_values = np.zeros((8, 8, 8), dtype=np.uint8)
    vascular_values[2, 2, 2] = 11
    vascular_values[3, 3, 3] = 22
    vascular_values[4, 4, 4] = 33
    result = write_auto_preprocessed_case(
        ct=ct,
        organ_masks={name: _mask() for name in AUTO_ORGAN_IDS},
        vascular_segmentation=_image(vascular_values),
        vessel_label_values={"artery": (11,), "vein": (22,), "portal": (33,)},
        output_directory=tmp_path,
        registration_module_path=Path("/tmp/2021.py"),
        case_id="auto_case",
        total_segmentator_metadata={"task": "total"},
    )

    vein = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "masks" / "vein_tree.nrrd")))
    assert vein[3, 3, 3] == 1
    assert vein[4, 4, 4] == 1
    assert vein[2, 2, 2] == 0
    config = yaml.safe_load((tmp_path / "case_preprocessed.yaml").read_text(encoding="utf-8"))
    assert result["model_count"] == 16
    assert [item["label"] for item in config["vessel_models"]] == ["artery", "vein"]


def test_totalsegmentator_command_requests_only_required_organs(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import AUTO_ORGAN_IDS, build_totalsegmentator_command

    command = build_totalsegmentator_command(
        executable="TotalSegmentator",
        ct_path=tmp_path / "ct.nii.gz",
        output_directory=tmp_path / "segmentations",
        device="gpu:0",
    )

    assert command[:7] == [
        "TotalSegmentator",
        "-i",
        str(tmp_path / "ct.nii.gz"),
        "-o",
        str(tmp_path / "segmentations"),
        "--task",
        "total",
    ]
    assert command[command.index("--roi_subset") + 1 : command.index("--device")] == list(AUTO_ORGAN_IDS)
    assert command[-2:] == ["--device", "gpu:0"]


def test_auto_case_config_resolves_paths_and_label_values(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import load_auto_case_config

    config_path = tmp_path / "auto_case.yaml"
    config_path.write_text(
        """
case_id: auto_case
ct_path: input/ct.nrrd
vascular_segmentation_path: input/vessels.nrrd
registration_module_path: registration/2021.py
output_root: output
vessel_label_values:
  artery: [11]
  vein: [22]
  portal: [33]
""".strip(),
        encoding="utf-8",
    )

    config = load_auto_case_config(config_path)

    assert config.case_id == "auto_case"
    assert config.ct_path == tmp_path / "input" / "ct.nrrd"
    assert config.vessel_label_values == {"artery": (11,), "vein": (22,), "portal": (33,)}
