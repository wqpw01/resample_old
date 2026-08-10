from __future__ import annotations

import importlib

import pytest

from ct_vascular_resampling.config import ORGAN_BOUNDARY_MODEL_IDS, REQUIRED_ORGAN_IDS, load_case_config


DEFAULT_VESSEL_MODELS = """  - id: portal_tree
    path: portal.stl
    label: portal
    color: [255, 0, 255]
  - id: hepatic_tree
    path: hepatic.ply
    label: hepatic
    color: [0, 188, 212]"""


def test_organ_boundary_models_include_dual_role_vessels():
    assert len(ORGAN_BOUNDARY_MODEL_IDS) == 14
    assert ORGAN_BOUNDARY_MODEL_IDS["aorta"] == "aorta"
    assert ORGAN_BOUNDARY_MODEL_IDS["inferior_vena_cava"] == "inferior_vena_cava"
    assert ORGAN_BOUNDARY_MODEL_IDS["portal_vein"] == "portal_vein_and_splenic_vein"
    assert "bile_duct" not in ORGAN_BOUNDARY_MODEL_IDS
    assert "common_bile_duct" not in ORGAN_BOUNDARY_MODEL_IDS


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
    assert config.geometry.input_coordinate_system == "LPS"
    assert config.geometry.canonical_coordinate_system == "RAS"
    assert config.sampling.ray_length_mm == 100.0
    assert config.sampling.ray_batch_size == 2048
    assert config.sampling.minimum_spacing_mm == 10.0
    assert config.sampling.centerline_max_terminal_spur_mm == 5.0
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


def test_case_config_rejects_obsolete_optional_pose_deduplication_switch(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + "\nsquare:\n  deduplicate_degenerate_edge_angles: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="deduplicate_degenerate_edge_angles|不支持"):
        load_case_config(config_path)


def test_case_config_rejects_obsolete_sampling_keys_instead_of_silently_using_defaults(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + "\nsampling:\n  stomach_search_distance_mm: 10.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stomach_search_distance_mm|不支持"):
        load_case_config(config_path)


def test_case_config_loads_manual_duodenum_centerline_endpoints_in_ras(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models)
        + """
sampling:
  duodenum_centerline_endpoint_hints_ras_mm:
    proximal: [19.0, 24.0, 700.0]
    distal: [-33.0, 1.0, 664.0]
  duodenum_centerline_endpoint_match_tolerance_mm: 1.0
""",
        encoding="utf-8",
    )

    config = load_case_config(config_path)

    assert config.sampling.duodenum_centerline_endpoint_hints_ras_mm == (
        (19.0, 24.0, 700.0),
        (-33.0, 1.0, 664.0),
    )
    assert config.sampling.duodenum_centerline_endpoint_match_tolerance_mm == 1.0


@pytest.mark.parametrize(
    ("sampling_yaml", "message"),
    [
        ("duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [1, 2, 3]", "distal"),
        (
            "duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [1, 2]\n    distal: [3, 4, 5]",
            "proximal",
        ),
        (
            "duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [.nan, 2, 3]\n    distal: [3, 4, 5]",
            "有限",
        ),
        (
            "duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [1, 2, 3]\n    distal: [3, 4, 5]\n"
            "duodenum_centerline_endpoint_match_tolerance_mm: 0",
            "endpoint_match_tolerance_mm",
        ),
        ("duodenum_centerline_endpoint_match_tolerance_mm: 0.5", "endpoint_hints"),
    ],
)
def test_case_config_rejects_invalid_manual_duodenum_centerline_configuration(
    tmp_path, sampling_yaml, message
):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    indented = "\n  ".join(sampling_yaml.splitlines())
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + f"\nsampling:\n  {indented}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_case_config(config_path)


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
