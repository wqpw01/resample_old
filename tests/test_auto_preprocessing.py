from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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


def _write_auto_config(
    tmp_path: Path,
    vessel_label_values: dict[str, list[int]],
    totalsegmentator: dict[str, object] | None = None,
) -> Path:
    config_path = tmp_path / "auto_case.yaml"
    payload: dict[str, object] = {
        "case_id": "auto_case",
        "ct_path": "input/ct.nrrd",
        "vascular_segmentation_path": "input/vessels.nrrd",
        "registration_module_path": "registration/2021.py",
        "output_root": "output",
        "vessel_label_values": vessel_label_values,
    }
    if totalsegmentator is not None:
        payload["totalsegmentator"] = totalsegmentator
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _write_totalsegmentator_masks(directory: Path) -> None:
    from ct_vascular_resampling.auto_preprocessing import AUTO_ORGAN_IDS

    directory.mkdir(parents=True, exist_ok=True)
    for structure in AUTO_ORGAN_IDS:
        sitk.WriteImage(_mask(), str(directory / f"{structure}.nrrd"), useCompression=True)


def _auto_case_config(tmp_path: Path, cache_directory: Path | None):
    from ct_vascular_resampling.auto_preprocessing import AutoCaseConfig

    ct_path = tmp_path / "ct.nrrd"
    vascular_path = tmp_path / "mixed_labels.nrrd"
    registration_path = tmp_path / "registration" / "2021.py"
    sitk.WriteImage(_image(np.zeros((8, 8, 8), dtype=np.int16)), str(ct_path), useCompression=True)
    vascular_values = np.zeros((8, 8, 8), dtype=np.uint8)
    vascular_values[2, 2, 2] = 1
    vascular_values[3, 3, 3] = 2
    vascular_values[4, 4, 4] = 99
    sitk.WriteImage(_image(vascular_values), str(vascular_path), useCompression=True)
    registration_path.parent.mkdir(parents=True)
    registration_path.write_text("# test registration module\n", encoding="utf-8")
    return AutoCaseConfig(
        case_id="auto_case",
        ct_path=ct_path,
        vascular_segmentation_path=vascular_path,
        vessel_label_values={"artery": (1,), "vein": (2,)},
        registration_module_path=registration_path,
        output_root=tmp_path / "output",
        totalsegmentator_cache_directory=cache_directory,
    )


def test_auto_preprocessing_merges_legacy_portal_and_ignores_unconfigured_organ_label(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import (
        AUTO_ORGAN_IDS,
        write_auto_preprocessed_case,
    )

    ct = _image(np.zeros((8, 8, 8), dtype=np.int16))
    vascular_values = np.zeros((8, 8, 8), dtype=np.uint8)
    vascular_values[2, 2, 2] = 11
    vascular_values[3, 3, 3] = 22
    vascular_values[4, 4, 4] = 33
    unconfigured_liver_label = 6
    vascular_values[5, 5, 5] = unconfigured_liver_label
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
    assert vein[5, 5, 5] == 0
    artery = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "masks" / "artery_tree.nrrd")))
    assert artery[2, 2, 2] == 1
    assert artery[5, 5, 5] == 0
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

    config_path = _write_auto_config(tmp_path, {"artery": [11], "vein": [22, 33]})

    config = load_auto_case_config(config_path)

    assert config.case_id == "auto_case"
    assert config.ct_path == tmp_path / "input" / "ct.nrrd"
    assert config.vessel_label_values == {"artery": (11,), "vein": (22, 33)}


def test_auto_case_config_merges_legacy_portal_values_into_vein(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import load_auto_case_config

    config_path = _write_auto_config(tmp_path, {"artery": [11], "vein": [22], "portal": [33, 44]})

    config = load_auto_case_config(config_path)

    assert config.vessel_label_values == {"artery": (11,), "vein": (22, 33, 44)}


def test_auto_case_config_resolves_totalsegmentator_cache_directory(tmp_path):
    from ct_vascular_resampling.auto_preprocessing import load_auto_case_config

    config_path = _write_auto_config(
        tmp_path,
        {"artery": [1], "vein": [2, 3]},
        totalsegmentator={"cache_directory": "cache/totalsegmentator"},
    )

    config = load_auto_case_config(config_path)

    assert config.totalsegmentator_cache_directory == (tmp_path / "cache" / "totalsegmentator").resolve()


@pytest.mark.parametrize(
    "label_values",
    [
        pytest.param({"artery": [1], "vein": [2], "organ": [3]}, id="unknown-key"),
        pytest.param({"artery": [], "vein": [2]}, id="empty-list"),
        pytest.param({"artery": [-1], "vein": [2]}, id="negative-value"),
        pytest.param({"artery": [1], "vein": [1]}, id="artery-vein-overlap"),
        pytest.param({"artery": [1], "vein": [2], "portal": [1]}, id="artery-portal-overlap"),
        pytest.param({"artery": [1], "vein": [2], "portal": [2]}, id="vein-portal-overlap"),
    ],
)
def test_auto_case_config_rejects_invalid_vessel_label_values(tmp_path, label_values):
    from ct_vascular_resampling.auto_preprocessing import load_auto_case_config

    config_path = _write_auto_config(tmp_path, label_values)

    with pytest.raises(ValueError, match="vessel_label_values|血管标签"):
        load_auto_case_config(config_path)


def test_auto_case_config_resolves_relative_config_paths_to_absolute(tmp_path, monkeypatch):
    from ct_vascular_resampling.auto_preprocessing import load_auto_case_config

    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    config_path = workspace / "configs" / "auto_case.yaml"
    config_path.write_text(
        """
case_id: auto_case
ct_path: ../input/ct.nrrd
vascular_segmentation_path: ../input/vessels.nrrd
registration_module_path: ../registration/2021.py
output_root: ../output
vessel_label_values:
  artery: [11]
  vein: [22]
  portal: [33]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)

    config = load_auto_case_config(Path("configs/auto_case.yaml"))

    assert config.registration_module_path == (workspace / "registration" / "2021.py").resolve()


def test_prepare_auto_case_reuses_valid_configured_cache_without_subprocess(tmp_path, monkeypatch):
    import ct_vascular_resampling.auto_preprocessing as auto_preprocessing

    cache_directory = tmp_path / "totalsegmentator_cache"
    _write_totalsegmentator_masks(cache_directory)
    config = _auto_case_config(tmp_path, cache_directory)

    def unexpected_run(*args, **kwargs):
        pytest.fail("有效 TotalSegmentator 缓存不应调用 subprocess")

    monkeypatch.setattr(auto_preprocessing.subprocess, "run", unexpected_run)

    result = auto_preprocessing.prepare_auto_case(config)

    assert result == tmp_path / "output" / "auto_case" / "preprocessing" / "case_preprocessed.yaml"
    manifest = yaml.safe_load(
        (tmp_path / "output" / "auto_case" / "preprocessing" / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = manifest["provenance"]
    assert provenance["cache_reused"] is True
    assert provenance["cache_directory"] == str(cache_directory.resolve())
    assert provenance["command_executed"] is False
    assert provenance["command"][provenance["command"].index("-o") + 1] == str(cache_directory.resolve())


def test_prepare_auto_case_runs_totalsegmentator_when_default_cache_is_missing(tmp_path, monkeypatch):
    import ct_vascular_resampling.auto_preprocessing as auto_preprocessing

    config = _auto_case_config(tmp_path, None)
    commands: list[list[str]] = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        _write_totalsegmentator_masks(Path(command[command.index("-o") + 1]))

    monkeypatch.setattr(auto_preprocessing.subprocess, "run", fake_run)

    auto_preprocessing.prepare_auto_case(config)

    expected_cache = tmp_path / "output" / "auto_case" / "preprocessing" / "totalsegmentator"
    assert len(commands) == 1
    assert Path(commands[0][commands[0].index("-o") + 1]) == expected_cache
    assert Path(commands[0][commands[0].index("-i") + 1]).is_file()
    manifest = yaml.safe_load((expected_cache.parent / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    assert provenance["cache_reused"] is False
    assert provenance["cache_directory"] == str(expected_cache)
    assert provenance["command_executed"] is True
    assert provenance["command"] == commands[0]


@pytest.mark.parametrize("invalid_kind", ["empty", "negative", "nan", "geometry"])
def test_prepare_auto_case_regenerates_invalid_cache(tmp_path, monkeypatch, invalid_kind):
    import ct_vascular_resampling.auto_preprocessing as auto_preprocessing

    cache_directory = tmp_path / "invalid_cache"
    _write_totalsegmentator_masks(cache_directory)
    invalid_values = np.zeros((8, 8, 8), dtype=np.float32)
    if invalid_kind == "negative":
        invalid_values[2, 2, 2] = -1.0
    elif invalid_kind == "nan":
        invalid_values[2, 2, 2] = np.nan
    invalid_mask = _image(invalid_values) if invalid_kind != "geometry" else _mask()
    if invalid_kind == "geometry":
        invalid_mask.SetOrigin((11.0, 20.0, 30.0))
    sitk.WriteImage(
        invalid_mask,
        str(cache_directory / f"{auto_preprocessing.AUTO_ORGAN_IDS[0]}.nrrd"),
        useCompression=True,
    )
    config = _auto_case_config(tmp_path, cache_directory)
    commands: list[list[str]] = []

    def fake_run(command, check):
        assert check is True
        commands.append(command)
        _write_totalsegmentator_masks(cache_directory)

    def fake_write_auto_preprocessed_case(**kwargs):
        assert set(kwargs["organ_masks"]) == set(auto_preprocessing.AUTO_ORGAN_IDS)
        return {"case_config_path": str(tmp_path / "case_preprocessed.yaml")}

    monkeypatch.setattr(auto_preprocessing.subprocess, "run", fake_run)
    monkeypatch.setattr(auto_preprocessing, "write_auto_preprocessed_case", fake_write_auto_preprocessed_case)

    auto_preprocessing.prepare_auto_case(config)

    assert len(commands) == 1
