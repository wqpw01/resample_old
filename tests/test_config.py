from __future__ import annotations

import copy
import pytest
import yaml

from ct_vascular_resampling.config import ORGAN_BOUNDARY_MODEL_IDS, REQUIRED_ORGAN_IDS, load_case_config


DEFAULT_VESSEL_MODELS = """  - id: artery_tree
    path: artery.stl
    label: artery
    color: [255, 82, 0]
  - id: vein_tree
    path: vein.ply
    label: vein
    color: [0, 188, 212]"""


def test_organ_boundary_models_include_dual_role_vessels():
    assert len(ORGAN_BOUNDARY_MODEL_IDS) == 14
    assert ORGAN_BOUNDARY_MODEL_IDS["aorta"] == "aorta"
    assert ORGAN_BOUNDARY_MODEL_IDS["inferior_vena_cava"] == "inferior_vena_cava"
    assert ORGAN_BOUNDARY_MODEL_IDS["portal_vein"] == "portal_vein_and_splenic_vein"
    assert "bile_duct" not in ORGAN_BOUNDARY_MODEL_IDS
    assert "common_bile_duct" not in ORGAN_BOUNDARY_MODEL_IDS


def _case_yaml(
    organ_models: str,
    vessel_models: str = DEFAULT_VESSEL_MODELS,
    manual_segmentation: dict | None = None,
) -> str:
    base = f"""case_id: demo
ct_path: ct.nrrd
output_root: output
organ_models:
{organ_models}
vessel_models:
{vessel_models}
"""
    return base + yaml.safe_dump(
        {"manual_segmentation": manual_segmentation or _valid_manual_segmentation()},
        allow_unicode=True,
        sort_keys=False,
    )


def _valid_manual_segmentation() -> dict:
    return {
        "path": "labels/EUS-main-organ.seg.nrrd",
        "organ_label_values": {
            "spleen": [1],
            "kidney_right": [2],
            "kidney_left": [3],
            "gallbladder": [4],
            "esophagus": [5],
            "liver": [6],
            "stomach": [7],
            "aorta": [8],
            "inferior_vena_cava": [9],
            "pancreas": [11],
            "adrenal_gland_right": [12],
            "adrenal_gland_left": [13],
            "duodenum": [14],
            "portal_vein": [23, 26, 33, 34, 35, 36, 37],
        },
        "eus_vessel_label_values": {
            "aorta": [8],
            "inferior_vena_cava": [9],
            "portal_vein": [26, 33, 34, 35, 36, 37],
        },
        "eus_vessel_colors": {
            "aorta": [255, 0, 0],
            "inferior_vena_cava": [0, 0, 255],
            "portal_vein": [170, 85, 255],
        },
    }


def _write_manual_case(tmp_path, manual_segmentation: dict):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(_case_yaml(organ_models, manual_segmentation=manual_segmentation), encoding="utf-8")
    return config_path


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
    assert config.filtering.black_ratio_limit == 0.60
    assert [model.label for model in config.vessel_models] == ["artery", "vein"]
    assert config.manual_segmentation is not None


@pytest.mark.parametrize("invalid", ["true", "1.9", '"7"'])
def test_case_config_rejects_non_integer_sampling_point_count(tmp_path, invalid):
    organ_models = "\n".join(
        f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS
    )
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models)
        + f"\nsampling:\n  point_counts:\n    stomach: {invalid}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="point_counts.stomach.*整数"):
        load_case_config(config_path)


def test_case_config_loads_strict_manual_segmentation_mode(tmp_path):
    config = load_case_config(_write_manual_case(tmp_path, _valid_manual_segmentation()))

    manual = config.manual_segmentation
    assert manual is not None
    assert manual.path == tmp_path / "labels/EUS-main-organ.seg.nrrd"
    assert manual.organ_label_values["portal_vein"] == (23, 26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_label_values["portal_vein"] == (26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_colors == {
        "aorta": (255, 0, 0),
        "inferior_vena_cava": (0, 0, 255),
        "portal_vein": (170, 85, 255),
    }


def _missing_organ(mapping: dict) -> None:
    mapping["organ_label_values"].pop("spleen")


def _unknown_vessel(mapping: dict) -> None:
    mapping["eus_vessel_label_values"]["superior_mesenteric_vein"] = [26]


def _empty_label_values(mapping: dict) -> None:
    mapping["organ_label_values"]["spleen"] = []


def _duplicate_organ_label(mapping: dict) -> None:
    mapping["organ_label_values"]["spleen"] = [2]


def _boolean_label(mapping: dict) -> None:
    mapping["organ_label_values"]["spleen"] = [True]


def _non_integer_label(mapping: dict) -> None:
    mapping["organ_label_values"]["spleen"] = [1.5]


def _invalid_color(mapping: dict) -> None:
    mapping["eus_vessel_colors"]["aorta"] = [255, 0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_missing_organ, "spleen|缺少"),
        (_unknown_vessel, "superior_mesenteric_vein|不支持"),
        (_empty_label_values, "spleen|非空"),
        (_duplicate_organ_label, "重复|spleen|kidney_right"),
        (_boolean_label, "spleen|整数"),
        (_non_integer_label, "spleen|整数"),
        (_invalid_color, "aorta|color|颜色"),
    ],
)
def test_case_config_rejects_invalid_manual_segmentation_contract(tmp_path, mutation, message):
    manual = copy.deepcopy(_valid_manual_segmentation())
    mutation(manual)

    with pytest.raises(ValueError, match=message):
        load_case_config(_write_manual_case(tmp_path, manual))


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
        _case_yaml(organ_models) + "\ndicom_series_uid: 1.2.840.99999.1\n",
        encoding="utf-8",
    )

    config = load_case_config(config_path)

    assert config.dicom_series_uid == "1.2.840.99999.1"


@pytest.mark.parametrize(
    ("section", "unknown_key"),
    [
        (None, "unexpected_top_level"),
        ("geometry", "input_coordinate_systm"),
        ("ct", "output_resoluton"),
        ("filtering", "black_ratio_limt"),
        ("runtime", "gpu_batch_sze"),
        ("organ_models", "livre"),
    ],
)
def test_case_config_rejects_unknown_mapping_keys(tmp_path, section, unknown_key):
    organ_models = "\n".join(
        f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS
    )
    raw = yaml.safe_load(_case_yaml(organ_models))
    target = raw if section is None else raw.setdefault(section, {})
    target[unknown_key] = "unexpected"
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{unknown_key}|不支持"):
        load_case_config(config_path)


def test_case_config_rejects_unknown_vessel_item_key(tmp_path):
    organ_models = "\n".join(
        f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS
    )
    raw = yaml.safe_load(_case_yaml(organ_models))
    raw["vessel_models"][0]["colour"] = [255, 82, 0]
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="colour|不支持"):
        load_case_config(config_path)
