from __future__ import annotations

import importlib

import pytest

from ct_vascular_resampling.config import REQUIRED_ORGAN_IDS, load_case_config


DEFAULT_VESSEL_MODELS = """  - id: portal_tree
    path: portal.stl
    label: portal
    color: [255, 0, 255]
  - id: hepatic_tree
    path: hepatic.ply
    label: hepatic
    color: [0, 188, 212]"""


def _case_yaml(organ_models: str, vessel_models: str = DEFAULT_VESSEL_MODELS) -> str:
    return f"""case_id: demo
ct_path: ct.nrrd
output_root: output
organ_models:
{organ_models}
vessel_models:
{vessel_models}
registration_module_path: registration_2021.py
"""


def test_case_config_resolves_relative_paths_and_uses_confirmed_defaults(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(_case_yaml(organ_models), encoding="utf-8")

    config = load_case_config(config_path)

    assert config.case_id == "demo"
    assert config.ct_path == tmp_path / "ct.nrrd"
    assert config.output_root == tmp_path / "output"
    assert set(config.organ_models) == set(REQUIRED_ORGAN_IDS)
    assert config.square.side_length_mm == 100.0
    assert config.ct.output_resolution == 300
    assert config.runtime.seed == 0
    assert config.filtering.black_ratio_limit == 0.50
    assert [model.label for model in config.vessel_models] == ["portal", "hepatic"]


def test_case_config_rejects_missing_source_algorithm_organ(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS if name != "gallbladder")
    config_path = tmp_path / "case.yaml"
    config_path.write_text(_case_yaml(organ_models), encoding="utf-8")

    with pytest.raises(ValueError, match="gallbladder"):
        load_case_config(config_path)


def test_case_config_rejects_invalid_black_line_ratio(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + "\nfiltering:\n  line_min_diagonal_fraction: 1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line_min_diagonal_fraction"):
        load_case_config(config_path)


def test_case_config_accepts_artery_and_vein_vessel_pair(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    vessels = """  - id: artery_tree
    path: models/artery_tree.ply
    label: artery
    color: [255, 82, 0]
  - id: vein_tree
    path: models/vein_tree.ply
    label: vein
    color: [0, 188, 212]"""
    config_path = tmp_path / "case.yaml"
    config_path.write_text(_case_yaml(organ_models, vessels), encoding="utf-8")

    config = load_case_config(config_path)

    assert [model.label for model in config.vessel_models] == ["artery", "vein"]


def test_case_config_loads_resampling_backend_controls(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models)
        + "\nruntime:\n  backend: gpu\n  gpu_device: 2\n  gpu_batch_size: 16\n",
        encoding="utf-8",
    )

    config = load_case_config(config_path)

    assert config.runtime.backend == "gpu"
    assert config.runtime.gpu_device == 2
    assert config.runtime.gpu_batch_size == 16


def test_case_config_accepts_an_explicit_dicom_series_uid(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + "\ndicom_series_uid: 1.2.840.113619.2.55.3\n",
        encoding="utf-8",
    )

    config = load_case_config(config_path)

    assert config.dicom_series_uid == "1.2.840.113619.2.55.3"


def test_rejected_audit_config_resolves_dicom_and_output_paths(tmp_path):
    module = importlib.import_module("ct_vascular_resampling.rejected_audit")
    config_path = tmp_path / "audit.yaml"
    config_path.write_text(
        """ct_path: dicom
dicom_series_uid: 1.2.3
rejected_jsonl: case/rejected/rejected.jsonl
output_directory: case/rejected/diagnostics
filtering:
  black_threshold: 50
  black_ratio_limit: 0.5
  line_min_diagonal_fraction: 0.6
representative_limit_per_cause: 30
""",
        encoding="utf-8",
    )

    config = module.load_rejected_audit_config(config_path)

    assert config.ct_path == tmp_path / "dicom"
    assert config.dicom_series_uid == "1.2.3"
    assert config.rejected_jsonl == tmp_path / "case/rejected/rejected.jsonl"
    assert config.output_directory == tmp_path / "case/rejected/diagnostics"
    assert config.filtering.black_ratio_limit == 0.5
    assert config.representative_limit_per_cause == 30
