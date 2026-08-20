from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np
from PIL import Image
import pytest
import SimpleITK as sitk
import subprocess
import trimesh
from pathlib import Path
from dataclasses import replace

from ct_vascular_resampling.config import (
    CTConfig,
    CaseConfig,
    FilterConfig,
    GeometryConfig,
    ManualSegmentationConfig,
    ORGAN_BOUNDARY_IDS,
    RuntimeConfig,
    SamplingConfig,
    SquareConfig,
    VesselModel,
)
from ct_vascular_resampling.centerline import CenterlinePath, CenterlineSelectionAudit
from ct_vascular_resampling.contract import CORE_DESIGN_FILENAME, CORE_DESIGN_SHA256
from ct_vascular_resampling.ct_resampling import CTVolume, diagnose_square_fov
from ct_vascular_resampling.gallery import GalleryWriter
from ct_vascular_resampling.label_resampling import CpuLabelBackend
import ct_vascular_resampling.pipeline as pipeline_module
from ct_vascular_resampling.pipeline import PreparedVessel, render_precomputed_square, run_case
from ct_vascular_resampling.resampling_backend import CachedCpuBackend
from ct_vascular_resampling.sampling import SamplingStatistics
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


_FORMAL_SETTINGS_VALIDATOR = pipeline_module._validate_formal_contract_settings


@pytest.fixture(autouse=True)
def _allow_reduced_synthetic_protocol(monkeypatch):
    """本模块的集成夹具使用 10 mm/20 px；正式参数门禁由独立测试覆盖。"""

    monkeypatch.setattr(pipeline_module, "_validate_formal_contract_settings", lambda _config: None)


def _manual_segmentation_config(path: Path) -> ManualSegmentationConfig:
    return ManualSegmentationConfig(
        path=path,
        organ_label_values={
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
        },
        eus_vessel_label_values={
            "aorta": (8,),
            "inferior_vena_cava": (9,),
            "portal_vein": (26, 33, 34, 35, 36, 37),
        },
        eus_vessel_colors={
            "aorta": (255, 0, 0),
            "inferior_vena_cava": (0, 0, 255),
            "portal_vein": (170, 85, 255),
        },
    )


def test_build_git_commit_uses_exported_archive_metadata_without_git(monkeypatch):
    exported_commit = "a" * 40
    monkeypatch.delenv("CT_VASCULAR_RESAMPLING_GIT_COMMIT", raising=False)
    monkeypatch.setattr(
        pipeline_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        ),
    )
    monkeypatch.setattr(pipeline_module, "_ARCHIVE_GIT_COMMIT", exported_commit, raising=False)
    pipeline_module._build_git_commit.cache_clear()
    try:
        assert pipeline_module._build_git_commit() == exported_commit
    finally:
        pipeline_module._build_git_commit.cache_clear()


def test_formal_design_amendment_matches_contract_sha256():
    amendment = Path(__file__).parents[1] / "docs" / CORE_DESIGN_FILENAME

    assert amendment.is_file()
    assert hashlib.sha256(amendment.read_bytes()).hexdigest() == CORE_DESIGN_SHA256


def test_manual_precomputed_square_adds_eus_outputs_without_changing_original_vessels(tmp_path):
    vessel_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    vessel_mesh.apply_translation((7.0, 7.0, 5.0))
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray(
            [[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]
        ),
    )
    vessels = [PreparedVessel("artery_tree", "artery", (255, 82, 0), vessel_mesh)]
    hu = np.full((20, 20), 40.0, dtype=np.float32)
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[7:12, 7:12] = 8
    labels[0:4, 15:19] = 9
    manual_writer = GalleryWriter(
        tmp_path / "manual",
        "case",
    )

    manual_status = render_precomputed_square(
        sample,
        hu,
        vessels,
        CTConfig(output_resolution=20),
        FilterConfig(),
        manual_writer,
        resampling_backend="cpu",
        label_plane=labels,
        manual_segmentation=_manual_segmentation_config(Path("unused.seg.nrrd")),
    )

    manual_record = json.loads((tmp_path / "manual/gallery/gallery.jsonl").read_text(encoding="utf-8"))
    assert manual_status == "gallery"
    assert [item["label"] for item in manual_record["features"]] == ["artery"]
    assert (tmp_path / "manual/gallery/boundary_only" / f"{sample.sample_id}.png").is_file()
    assert (tmp_path / "manual/gallery/ct_overlay" / f"{sample.sample_id}.png").is_file()
    assert manual_record["eus_vessel_labels"] == ["aorta", "inferior_vena_cava"]
    assert [item["label"] for item in manual_record["eus_vessel_features"]] == ["aorta"]
    with Image.open(tmp_path / "manual/gallery" / manual_record["eus_vessel_boundary_png"]) as image:
        colors = set(image.convert("RGB").getdata())
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors


@pytest.mark.parametrize(
    ("hu_value", "vessels", "expected_status"),
    [(-1000.0, ["mesh"], "rejected"), (40.0, [], "unindexed")],
)
def test_manual_label_analysis_is_skipped_outside_original_gallery(
    monkeypatch,
    tmp_path,
    hu_value,
    vessels,
    expected_status,
):
    sample = SquareSample(
        sample_id="sample",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray(
            [[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]
        ),
    )
    prepared = []
    if vessels:
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh.apply_translation((7.0, 7.0, 5.0))
        prepared = [PreparedVessel("artery_tree", "artery", (255, 0, 255), mesh)]
    writer = GalleryWriter(
        tmp_path / "case",
        "case",
    )

    def unexpected_analysis(*_args, **_kwargs):
        raise AssertionError("non-gallery samples must not analyze manual labels")

    monkeypatch.setattr(pipeline_module, "analyze_manual_label_plane", unexpected_analysis)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), hu_value, dtype=np.float32),
        prepared,
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        label_plane=np.zeros((20, 20), dtype=np.uint8),
        manual_segmentation=_manual_segmentation_config(Path("unused.seg.nrrd")),
    )

    assert status == expected_status


def test_rejected_precomputed_square_skips_vessel_intersection(monkeypatch, tmp_path):
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]),
    )
    vessel_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    writer = GalleryWriter(tmp_path / "case", "case")

    def unexpected_intersection(*_):
        raise AssertionError("rejected CT should not intersect vessel meshes")

    monkeypatch.setattr("ct_vascular_resampling.pipeline.intersect_mesh_with_square", unexpected_intersection)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), -1000.0, dtype=np.float32),
        [PreparedVessel("artery_tree", "artery", (255, 82, 0), vessel_mesh)],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        resampling_backend="cpu",
        label_plane=np.zeros((20, 20), dtype=np.uint8),
        manual_segmentation=_manual_segmentation_config(Path("unused.seg.nrrd")),
    )

    assert status == "rejected"


def test_unindexed_precomputed_square_skips_manual_label_analysis(monkeypatch, tmp_path):
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]),
    )
    writer = GalleryWriter(tmp_path / "case", "case")

    def unexpected_intersection(*_):
        raise AssertionError("unindexed samples must skip manual label analysis")

    monkeypatch.setattr("ct_vascular_resampling.pipeline.analyze_manual_label_plane", unexpected_intersection)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), 40.0, dtype=np.float32),
        [],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        resampling_backend="cpu",
        label_plane=np.zeros((20, 20), dtype=np.uint8),
        manual_segmentation=_manual_segmentation_config(Path("unused.seg.nrrd")),
    )

    assert status == "unindexed"
    record = json.loads((tmp_path / "case" / "unindexed" / "unindexed.jsonl").read_text(encoding="utf-8"))
    assert "organ_labels" not in record


def test_out_of_fov_square_writes_black_filled_ct_only(monkeypatch, tmp_path):
    image = sitk.GetImageFromArray(np.full((16, 16, 16), 40.0, dtype=np.float32))
    volume = CTVolume.from_sitk(image)
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([2.0, 2.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[-10.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [-10.0, 12.0, 5.0]]),
    )
    writer = GalleryWriter(tmp_path / "case", "case")

    def unexpected_processing(*_):
        raise AssertionError("excluded FOV samples must skip quality and vessel processing")

    monkeypatch.setattr("ct_vascular_resampling.pipeline.evaluate_ct_quality", unexpected_processing)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.intersect_mesh_with_square", unexpected_processing)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), 40.0, dtype=np.float32),
        [],
        CTConfig(output_resolution=20, fill_hu_value=40.0),
        FilterConfig(),
        writer,
        resampling_backend="cpu",
        volume=volume,
        label_plane=np.zeros((20, 20), dtype=np.uint8),
        manual_segmentation=_manual_segmentation_config(Path("unused.seg.nrrd")),
    )

    record = json.loads((tmp_path / "case" / "excluded_fov.jsonl").read_text(encoding="utf-8"))
    ct_path = tmp_path / "case" / "excluded_fov" / "ct" / "stomach-000000-x-00.png"
    with Image.open(ct_path) as image:
        pixels = np.asarray(image)
    diagnosis = diagnose_square_fov(volume, sample.vertices, resolution=20)
    assert status == "excluded_fov"
    assert record["fov_diagnostics"]["contains_ct_fov_exceedance"] is True
    assert record["ct_png"] == "ct/stomach-000000-x-00.png"
    assert record["resampling_backend"] == "cpu"
    assert pixels.ndim == 2
    assert np.all(pixels[diagnosis.out_of_bounds_mask] == 0)
    assert np.any(pixels[~diagnosis.out_of_bounds_mask] > 0)
    assert not (tmp_path / "case" / "gallery").exists()
    assert not (tmp_path / "case" / "unindexed").exists()
    assert not (tmp_path / "case" / "rejected").exists()


def _single_fov_case(tmp_path: Path) -> tuple[CaseConfig, SquareSample]:
    ct_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(np.full((16, 16, 16), 40.0, dtype=np.float32)), str(ct_path))
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh_path = tmp_path / "model.obj"
    mesh.export(mesh_path)
    label_path = tmp_path / "labels.nrrd"
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.uint8)),
        str(label_path),
    )
    organ_models = {
        name: mesh_path
        for name in (
            "adrenal_gland_left",
            "adrenal_gland_right",
            "aorta",
            "duodenum",
            "esophagus",
            "gallbladder",
            "inferior_vena_cava",
            "kidney_left",
            "kidney_right",
            "liver",
            "pancreas",
            "portal_vein_and_splenic_vein",
            "spleen",
            "stomach",
        )
    }
    config = CaseConfig(
        case_id="fov_case",
        ct_path=ct_path,
        output_root=tmp_path / "output",
        organ_models=organ_models,
        vessel_models=(
            VesselModel("artery_tree", mesh_path, "artery", (255, 82, 0)),
            VesselModel("vein_tree", mesh_path, "vein", (0, 188, 212)),
        ),
        sampling=SamplingConfig(
            point_counts={"stomach": 1, "liver": 1, "pancreas": 1, "duodenum_part1": 1, "duodenum_part2": 1, "esophagus": 1}
        ),
        square=SquareConfig(side_length_mm=10.0),
        ct=CTConfig(output_resolution=20),
        filtering=FilterConfig(),
        runtime=RuntimeConfig(seed=0, workers=1, backend="cpu"),
        manual_segmentation=_manual_segmentation_config(label_path),
        geometry=GeometryConfig(input_coordinate_system="RAS"),
    )
    outside_sample = SquareSample(
        sample_id="esophagus-000010-x-01",
        organ="esophagus",
        probe_point_world=np.asarray([1.0, 3.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[-1.0, 2.0, 5.0], [3.0, 2.0, 5.0], [3.0, 6.0, 5.0], [-1.0, 6.0, 5.0]]),
        source_region="esophagus",
        yaw_policy="standard",
        local_x_world=np.asarray([0.0, 0.0, 1.0]),
        local_y_world=np.asarray([1.0, 0.0, 0.0]),
        local_z_world=np.asarray([0.0, 1.0, 0.0]),
    )
    return config, outside_sample


def _manual_fov_case(tmp_path: Path) -> tuple[CaseConfig, SquareSample]:
    return _single_fov_case(tmp_path)


def _formalized_test_config(config: CaseConfig) -> CaseConfig:
    return replace(
        config,
        square=SquareConfig(),
        ct=CTConfig(),
        filtering=FilterConfig(),
    )


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        ("sampling", "ray_length_mm", 99.0),
        ("sampling", "ray_batch_size", 1024),
        ("sampling", "minimum_spacing_mm", 9.0),
        ("sampling", "centerline_voxel_pitch_mm", 2.0),
        ("sampling", "centerline_tangent_window_mm", 9.0),
        ("sampling", "centerline_max_terminal_spur_mm", 4.0),
        ("square", "side_length_mm", 99.0),
        ("ct", "output_resolution", 299),
        ("ct", "window_level", 41.0),
        ("ct", "window_width", 401.0),
        ("ct", "fill_hu_value", -999.0),
        ("filtering", "black_threshold", 49),
        ("filtering", "black_ratio_limit", 0.59),
        ("filtering", "line_min_diagonal_fraction", 0.69),
        ("filtering", "black_side_min_ratio", 0.89),
        ("filtering", "valid_side_max_black_ratio", 0.11),
        ("runtime", "seed", 1),
    ],
)
def test_formal_contract_rejects_changed_fixed_setting(tmp_path, section, field, invalid):
    config, _ = _single_fov_case(tmp_path)
    config = _formalized_test_config(config)
    config = replace(
        config,
        **{section: replace(getattr(config, section), **{field: invalid})},
    )

    with pytest.raises(ValueError, match=field):
        _FORMAL_SETTINGS_VALIDATOR(config)


def test_preflight_applies_formal_contract_settings(monkeypatch, tmp_path):
    config, _ = _single_fov_case(tmp_path)
    config = _formalized_test_config(config)
    config = replace(config, square=replace(config.square, side_length_mm=10.0))
    monkeypatch.setattr(
        pipeline_module,
        "_validate_formal_contract_settings",
        _FORMAL_SETTINGS_VALIDATOR,
    )

    with pytest.raises(ValueError, match="side_length_mm"):
        pipeline_module._preflight(config)


def test_formal_contract_rejects_nonpositive_point_count(tmp_path):
    config, _ = _single_fov_case(tmp_path)
    config = _formalized_test_config(config)
    config = replace(
        config,
        sampling=replace(
            config.sampling,
            point_counts={**config.sampling.point_counts, "stomach": 0},
        ),
    )

    with pytest.raises(ValueError, match="point_counts"):
        _FORMAL_SETTINGS_VALIDATOR(config)


def test_manual_protocol_records_segmentation_geometry_mappings_and_sources(tmp_path):
    config, _ = _manual_fov_case(tmp_path)
    config = replace(
        config,
        filtering=replace(config.filtering, black_ratio_limit=0.60),
    )

    protocol = pipeline_module._run_protocol_metadata(config, None)

    manual = protocol["manual_segmentation"]
    segmentation_bytes = config.manual_segmentation.path.read_bytes()
    assert manual["source"] == {
        "path": str(config.manual_segmentation.path.resolve()),
        "sha256": hashlib.sha256(segmentation_bytes).hexdigest(),
    }
    assert manual["geometry"] == {
        "input_coordinate_system": "RAS",
        "canonical_coordinate_system": "RAS",
        "size_xyz": [16, 16, 16],
        "spacing_xyz_mm": [1.0, 1.0, 1.0],
        "origin_ras_mm": [0.0, 0.0, 0.0],
        "direction_ras": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    assert manual["label_sampling"] == {
        "interpolation": "nearest",
        "interpolation_order": 0,
        "prefilter": False,
        "outside_label_value": 0,
    }
    assert manual["organ_label_values"] == {
        identifier: list(values)
        for identifier, values in config.manual_segmentation.organ_label_values.items()
    }
    assert manual["eus_vessel_label_values"] == {
        identifier: list(values)
        for identifier, values in config.manual_segmentation.eus_vessel_label_values.items()
    }
    assert manual["eus_vessel_colors"] == {
        identifier: list(values)
        for identifier, values in config.manual_segmentation.eus_vessel_colors.items()
    }
    assert manual["component_analysis"] == {
        "connectivity": 8,
        "complete_component_rule": "exclude_components_touching_any_image_edge",
    }
    assert manual["organ_presence_rule"] == "at_least_one_sampled_pixel"
    assert manual["organ_model_sources"] == protocol["input_provenance"]["organ_models"]
    assert manual["external_reconstructed_vessel_sources"] == protocol["input_provenance"][
        "vessel_models"
    ]
    for vessel_config, source in zip(
        config.vessel_models,
        manual["external_reconstructed_vessel_sources"],
        strict=True,
    ):
        assert source["sha256"] == hashlib.sha256(vessel_config.path.read_bytes()).hexdigest()
    assert protocol["quality_filtering"]["black_ratio_limit"] == 0.60


def test_manual_protocol_hash_changes_with_segmentation_contract_or_threshold(tmp_path):
    config, _ = _manual_fov_case(tmp_path)
    config = replace(
        config,
        filtering=replace(config.filtering, black_ratio_limit=0.60),
    )
    base_hash = pipeline_module._run_protocol_metadata(config, None)["resume_protocol_sha256"]
    manual = config.manual_segmentation

    changed_mapping = replace(
        config,
        manual_segmentation=replace(
            manual,
            organ_label_values={**manual.organ_label_values, "portal_vein": (26, 33, 34, 35, 36, 37)},
        ),
    )
    changed_color = replace(
        config,
        manual_segmentation=replace(
            manual,
            eus_vessel_colors={**manual.eus_vessel_colors, "aorta": (254, 0, 0)},
        ),
    )
    changed_threshold = replace(
        config,
        filtering=replace(config.filtering, black_ratio_limit=0.59),
    )
    for changed in (changed_mapping, changed_color, changed_threshold):
        assert pipeline_module._run_protocol_metadata(changed, None)["resume_protocol_sha256"] != base_hash

    image = sitk.ReadImage(str(manual.path))
    labels = sitk.GetArrayFromImage(image)
    labels[0, 0, 0] = 8
    changed_image = sitk.GetImageFromArray(labels)
    changed_image.CopyInformation(image)
    sitk.WriteImage(changed_image, str(manual.path))
    assert pipeline_module._run_protocol_metadata(config, None)["resume_protocol_sha256"] != base_hash


def test_manual_library_summary_counts_visible_labels_and_complete_features(
    monkeypatch,
    tmp_path,
):
    config, _ = _manual_fov_case(tmp_path)
    case_directory = config.output_root / config.case_id
    gallery = case_directory / "gallery"
    for directory in (
        "organ_vessel_boundary",
        "eus_vessel_boundary",
        "ct_eus_vessel_overlay",
    ):
        (gallery / directory).mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id, labels, eus_features, original_label in (
        (
            "first",
            ["aorta", "inferior_vena_cava"],
            [{"label": "aorta", "x_mm": 1.0, "y_mm": 2.0, "area_mm2": 3.0}],
            "artery",
        ),
        ("second", ["aorta"], [], "vein"),
    ):
        for directory in (
            "organ_vessel_boundary",
            "eus_vessel_boundary",
            "ct_eus_vessel_overlay",
        ):
            Image.new("RGB", (4, 4), "white").save(gallery / directory / f"{sample_id}.png")
        records.append(
            {
                "slice_id": sample_id,
                "status": "gallery",
                "features": [
                    {"label": original_label, "x_mm": 1.0, "y_mm": 2.0, "area_mm2": 3.0}
                ],
                "organ_metadata_schema_version": "eus-organ-metadata/v1",
                "organ_vessel_boundary_png": f"organ_vessel_boundary/{sample_id}.png",
                "organ_labels": [],
                "eus_candidate_organ_labels": [],
                "eus_vessel_metadata_schema_version": "eus-vessel-metadata/v1",
                "eus_vessel_labels": labels,
                "eus_vessel_features": eus_features,
                "eus_vessel_boundary_png": f"eus_vessel_boundary/{sample_id}.png",
                "ct_eus_vessel_overlay_png": f"ct_eus_vessel_overlay/{sample_id}.png",
            }
        )
    (gallery / "gallery.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [])
    run_case(config, steps=["index"])

    summary = json.loads((case_directory / "library_summary.json").read_text(encoding="utf-8"))
    assert summary["eus_vessel_label_counts"] == {
        "aorta": 2,
        "inferior_vena_cava": 1,
    }
    assert summary["eus_vessel_feature_counts"] == {"aorta": 1}
    assert summary["indexed_feature_count"] == 2
    assert summary["feature_label_counts"] == {"artery": 1, "vein": 1}
    assert summary["eus_vessel_colors"] == {
        "aorta": [255, 0, 0],
        "inferior_vena_cava": [0, 0, 255],
        "portal_vein": [170, 85, 255],
    }
    assert summary["eus_vessel_label_values"]["portal_vein"] == [26, 33, 34, 35, 36, 37]


def test_run_refuses_orphaned_metadata_before_render(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    case_directory = config.output_root / config.case_id
    metadata = pipeline_module._run_protocol_metadata(config, None)
    pipeline_module._write_run_metadata(case_directory, metadata)
    metadata_before = (case_directory / "run_metadata.json").read_bytes()
    with pytest.raises(ValueError, match="run_metadata|manifest|不完整"):
        run_case(config, steps=["render"], workers=1)

    assert (case_directory / "run_metadata.json").read_bytes() == metadata_before
    assert not (case_directory / "manifest.jsonl").exists()


@pytest.mark.parametrize("step", ["sample", "square", "index"])
def test_run_refuses_incompatible_protocol_before_non_render_writes(
    monkeypatch,
    tmp_path,
    step,
):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    run_case(config, steps=["render"], workers=1)
    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resume_protocol_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="运行协议|配置|构建|不一致"):
        run_case(config, steps=[step], workers=1)


def test_dry_run_skips_output_protocol_summaries(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline._pose_plan_summary",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry-run 不应生成姿态摘要")),
    )
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.build_sampling_point_plan",
        lambda *_: (_ for _ in ()).throw(AssertionError("dry-run 不应生成逐点输出计划")),
    )

    summary = run_case(config, dry_run=True)

    assert summary.dry_run is True
    assert summary.total_squares == 1


def test_manual_run_loads_labels_once_and_samples_same_square_batch_as_ct(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    label_load_count = 0
    original_load = pipeline_module.load_label_volume
    ct_vertices: list[np.ndarray] = []
    label_vertices: list[np.ndarray] = []
    original_ct_sample = CachedCpuBackend.sample_many
    original_label_sample = CpuLabelBackend.sample_many

    def record_load(*args, **kwargs):
        nonlocal label_load_count
        label_load_count += 1
        return original_load(*args, **kwargs)

    def record_ct(self, vertices_batch, resolution, fill_hu_value):
        ct_vertices.append(np.asarray(vertices_batch).copy())
        return original_ct_sample(self, vertices_batch, resolution, fill_hu_value)

    def record_labels(self, vertices_batch, resolution):
        label_vertices.append(np.asarray(vertices_batch).copy())
        return original_label_sample(self, vertices_batch, resolution)

    monkeypatch.setattr(pipeline_module, "load_label_volume", record_load)
    monkeypatch.setattr(CachedCpuBackend, "sample_many", record_ct)
    monkeypatch.setattr(CpuLabelBackend, "sample_many", record_labels)

    summary = run_case(config, steps=["render"], workers=1)

    assert summary.status_counts == {"excluded_fov": 1}
    assert label_load_count == 1
    assert len(ct_vertices) == len(label_vertices) == 1
    assert np.array_equal(ct_vertices[0], label_vertices[0])


def test_manual_run_rejects_label_geometry_before_writing_run_artifacts(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    mismatched = sitk.GetImageFromArray(np.zeros((15, 16, 16), dtype=np.uint8))
    sitk.WriteImage(mismatched, str(config.manual_segmentation.path))
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )

    with pytest.raises(ValueError, match="Size"):
        run_case(config, steps=["render"], workers=1)

    case_directory = config.output_root / config.case_id
    assert not (case_directory / "manifest.jsonl").exists()
    assert not (case_directory / "run_metadata.json").exists()


def _force_cpu_ct_backend(volume, **kwargs):
    return CachedCpuBackend(volume), {
        "requested_backend": kwargs["backend"],
        "selected_backend": "cpu",
        "gpu_device": kwargs["gpu_device"],
        "gpu_batch_size": kwargs["gpu_batch_size"],
        "fallback_reason": None,
        "coefficient_dtype": "float64",
    }


@pytest.mark.parametrize("requested_backend", ["auto", "gpu"])
def test_manual_run_handles_label_gpu_calibration_mismatch_by_runtime_policy(
    monkeypatch,
    tmp_path,
    requested_backend,
):
    config, outside_sample = _manual_fov_case(tmp_path)
    config = replace(config, runtime=replace(config.runtime, backend=requested_backend))
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    monkeypatch.setattr(pipeline_module, "create_sampling_backend", _force_cpu_ct_backend)

    def mismatched_label_backend(volume, **kwargs):
        reference = CpuLabelBackend(volume)

        class Mismatch:
            name = "gpu:0"

            def sample_many(self, vertices_batch, resolution):
                result = reference.sample_many(vertices_batch, resolution).copy()
                result[0, 0, 0] ^= np.uint8(1)
                return result

            def close(self):
                pass

        return Mismatch(), {
            "requested_backend": kwargs["backend"],
            "selected_backend": "gpu:0",
            "gpu_device": 0,
            "gpu_batch_size": 32,
            "fallback_reason": None,
            "interpolation": "nearest",
            "outside_label_value": 0,
        }

    monkeypatch.setattr(
        pipeline_module,
        "create_label_sampling_backend",
        mismatched_label_backend,
    )

    if requested_backend == "gpu":
        with pytest.raises(ValueError, match="标签|校验|CPU"):
            run_case(config, steps=["render"], workers=1)
        assert not (config.output_root / config.case_id / "manifest.jsonl").exists()
    else:
        summary = run_case(config, steps=["render"], workers=1)
        metadata = json.loads(
            (config.output_root / config.case_id / "run_metadata.json").read_text(encoding="utf-8")
        )
        assert summary.status_counts == {"excluded_fov": 1}
        assert metadata["label_sampling"]["selected_backend"] == "cpu"
        assert metadata["label_sampling"]["calibration"]["accepted"] is False
        assert metadata["label_sampling"]["fallback_reason"]


def test_manual_run_auto_falls_back_after_label_gpu_runtime_failure(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    config = replace(config, runtime=replace(config.runtime, backend="auto"))
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    monkeypatch.setattr(pipeline_module, "create_sampling_backend", _force_cpu_ct_backend)

    def failing_label_backend(volume, **kwargs):
        reference = CpuLabelBackend(volume)

        class FailsAfterCalibration:
            name = "gpu:0"

            def __init__(self):
                self.calls = 0

            def sample_many(self, vertices_batch, resolution):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("simulated label GPU runtime failure")
                return reference.sample_many(vertices_batch, resolution)

            def close(self):
                pass

        return FailsAfterCalibration(), {
            "requested_backend": kwargs["backend"],
            "selected_backend": "gpu:0",
            "gpu_device": 0,
            "gpu_batch_size": 32,
            "fallback_reason": None,
            "interpolation": "nearest",
            "outside_label_value": 0,
        }

    monkeypatch.setattr(
        pipeline_module,
        "create_label_sampling_backend",
        failing_label_backend,
    )

    summary = run_case(config, steps=["render"], workers=1)

    metadata = json.loads(
        (config.output_root / config.case_id / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert summary.status_counts == {"excluded_fov": 1}
    assert metadata["label_sampling"]["selected_backend"] == "cpu"
    assert "simulated label GPU runtime failure" in metadata["label_sampling"]["fallback_reason"]


def test_run_case_resamples_fov_square_and_records_exclusion(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    sample_many_calls: list[int] = []
    original_sample_many = CachedCpuBackend.sample_many

    def record_ct_interpolation(self, vertices_batch, resolution, fill_hu_value):
        sample_many_calls.append(len(vertices_batch))
        return original_sample_many(self, vertices_batch, resolution, fill_hu_value)

    monkeypatch.setattr(CachedCpuBackend, "sample_many", record_ct_interpolation)

    summary = run_case(config, steps=["render"], workers=1)

    case_directory = tmp_path / "output" / "fov_case"
    record = json.loads((case_directory / "excluded_fov.jsonl").read_text(encoding="utf-8"))
    metadata = json.loads((case_directory / "run_metadata.json").read_text(encoding="utf-8"))
    assert summary.status_counts == {"excluded_fov": 1}
    assert sample_many_calls == [1]
    assert record["fov_diagnostics"]["contains_ct_fov_exceedance"] is True
    assert record["resampling_backend"] == "cpu"
    assert (case_directory / "excluded_fov" / "ct" / "esophagus-000010-x-01.png").is_file()
    assert metadata["excluded_fov_count"] == 1
    assert metadata["run_state"] == "complete"
    assert metadata["completed_pose_count"] == 1
    assert not (case_directory / "rejected").exists()


def test_run_case_writes_metadata_before_first_render(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    metadata_seen_before_render: dict[str, object] = {}
    original_render = pipeline_module.render_precomputed_square

    def observe_checkpoint(*args, **kwargs):
        metadata_path = config.output_root / config.case_id / "run_metadata.json"
        metadata_seen_before_render.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "render_precomputed_square", observe_checkpoint)

    summary = run_case(config, steps=["render"], workers=1)

    assert metadata_seen_before_render["run_state"] == "running"
    assert metadata_seen_before_render["completed_pose_count"] == 0
    assert metadata_seen_before_render["status_counts"] == {}
    assert summary.total_squares == 1


def test_run_case_marks_metadata_interrupted_after_render_error(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(pipeline_module, "render_precomputed_square", fail_render)

    with pytest.raises(RuntimeError, match="simulated render failure"):
        run_case(config, steps=["render"], workers=1)

    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_state"] == "interrupted"
    assert metadata["completed_pose_count"] == 0
    assert metadata["status_counts"] == {}


def test_recover_interrupted_run_metadata_rebuilds_audited_protocol(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    run_case(config, steps=["render"], workers=1)
    case_directory = config.output_root / config.case_id
    metadata_path = case_directory / "run_metadata.json"
    interrupted_commit = json.loads(metadata_path.read_text(encoding="utf-8"))["build_git_commit"]
    metadata_path.unlink()
    manifest_before = (case_directory / "manifest.jsonl").read_bytes()
    ct_path = case_directory / "excluded_fov" / "ct" / f"{outside_sample.sample_id}.png"
    ct_before = ct_path.read_bytes()

    metadata = pipeline_module.recover_interrupted_run_metadata(
        config,
        expected_completed_count=1,
        reason="sighup",
        exit_code=129,
        recovered_at_utc="2026-08-08T10:43:29Z",
    )

    assert metadata["run_state"] == "interrupted"
    assert metadata["completed_pose_count"] == 1
    assert metadata["status_counts"] == {"excluded_fov": 1}
    assert metadata["recovery_history"] == [
        {
            "reason": "sighup",
            "exit_code": 129,
            "recovered_at_utc": "2026-08-08T10:43:29Z",
                "completed_pose_count": 1,
                "status_counts": {"excluded_fov": 1},
                "completed_build_git_commits": [interrupted_commit],
                "recovery_build_git_commit": interrupted_commit,
            }
        ]
    assert len(metadata["resume_protocol_sha256"]) == 64
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert (case_directory / "manifest.jsonl").read_bytes() == manifest_before
    assert ct_path.read_bytes() == ct_before


def test_recover_interrupted_run_metadata_refuses_existing_metadata(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    run_case(config, steps=["render"], workers=1)

    with pytest.raises(FileExistsError, match="run_metadata.json|元数据"):
        pipeline_module.recover_interrupted_run_metadata(
            config,
            expected_completed_count=1,
            reason="sighup",
            exit_code=129,
        )


def test_recover_interrupted_run_metadata_refuses_count_mismatch(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    run_case(config, steps=["render"], workers=1)
    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    metadata_path.unlink()

    with pytest.raises(ValueError, match="完成姿态数|预期|1|2"):
        pipeline_module.recover_interrupted_run_metadata(
            config,
            expected_completed_count=2,
            reason="sighup",
            exit_code=129,
        )
    assert not metadata_path.exists()


def test_recover_interrupted_run_metadata_refuses_changed_pose(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    run_case(config, steps=["render"], workers=1)
    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    metadata_path.unlink()
    changed_sample = replace(outside_sample, vertices=outside_sample.vertices + np.asarray([0.0, 0.0, 0.5]))
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [changed_sample])

    with pytest.raises(ValueError, match="姿态|几何|不一致"):
        pipeline_module.recover_interrupted_run_metadata(
            config,
            expected_completed_count=1,
            reason="sighup",
            exit_code=129,
        )
    assert not metadata_path.exists()


def test_recover_interrupted_run_metadata_does_not_repair_jsonl(monkeypatch, tmp_path):
    config, outside_sample = _single_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])
    run_case(config, steps=["render"], workers=1)
    case_directory = config.output_root / config.case_id
    metadata_path = case_directory / "run_metadata.json"
    metadata_path.unlink()
    root_manifest = case_directory / "manifest.jsonl"
    root_before = root_manifest.read_bytes()
    state_manifest = case_directory / "excluded_fov.jsonl"
    state_manifest.unlink()

    with pytest.raises(ValueError, match="状态清单"):
        pipeline_module.recover_interrupted_run_metadata(
            config,
            expected_completed_count=1,
            reason="sighup",
            exit_code=129,
        )
    assert root_manifest.read_bytes() == root_before
    assert not state_manifest.exists()
    assert not metadata_path.exists()


def test_recovered_metadata_resumes_only_pending_samples(monkeypatch, tmp_path):
    config, first_sample = _single_fov_case(tmp_path)
    second_sample = replace(first_sample, sample_id="esophagus-000011-x-01")
    samples = [first_sample, second_sample]
    interrupted_commit = "1" * 40
    recovery_commit = "2" * 40
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: samples)
    monkeypatch.setattr(pipeline_module, "_build_git_commit", lambda: interrupted_commit)
    original_render = pipeline_module.render_precomputed_square

    def interrupt_second(sample, *args, **kwargs):
        if sample.sample_id == second_sample.sample_id:
            raise RuntimeError("simulated interruption")
        return original_render(sample, *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "render_precomputed_square", interrupt_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_case(config, steps=["render"], workers=1)

    case_directory = config.output_root / config.case_id
    metadata_path = case_directory / "run_metadata.json"
    assert len((case_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    metadata_path.unlink()
    monkeypatch.setattr(pipeline_module, "_build_git_commit", lambda: recovery_commit)
    recovered_metadata = pipeline_module.recover_interrupted_run_metadata(
        config,
        expected_completed_count=1,
        reason="sighup",
        exit_code=129,
        recovered_at_utc="2026-08-08T10:43:29Z",
    )
    assert recovered_metadata["compatible_completed_build_git_commits"] == [interrupted_commit]
    assert recovered_metadata["recovery_history"][0]["completed_build_git_commits"] == [interrupted_commit]
    assert recovered_metadata["recovery_history"][0]["recovery_build_git_commit"] == recovery_commit
    monkeypatch.setattr(pipeline_module, "render_precomputed_square", original_render)

    summary = run_case(config, steps=["render"], workers=1)

    manifest_records = [
        json.loads(line)
        for line in (case_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    final_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert summary.status_counts == {"excluded_fov": 2}
    assert [record["slice_id"] for record in manifest_records] == [first_sample.sample_id, second_sample.sample_id]
    assert [record["build_git_commit"] for record in manifest_records] == [interrupted_commit, recovery_commit]
    assert final_metadata["run_state"] == "complete"
    assert final_metadata["completed_pose_count"] == 2
    assert final_metadata["compatible_completed_build_git_commits"] == [interrupted_commit]
    assert final_metadata["recovery_history"][0]["exit_code"] == 129


def test_changed_protocol_is_rejected_before_state_manifest_repair(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    run_case(config, steps=["render"], workers=1)
    case_directory = config.output_root / config.case_id
    state_manifest = case_directory / "excluded_fov.jsonl"
    state_manifest.unlink()
    root_before = (case_directory / "manifest.jsonl").read_bytes()
    changed = replace(
        config,
        manual_segmentation=replace(
            config.manual_segmentation,
            eus_vessel_colors={
                **config.manual_segmentation.eus_vessel_colors,
                "aorta": (254, 0, 0),
            },
        ),
    )

    with pytest.raises(ValueError, match="运行协议|配置|不一致"):
        run_case(changed, steps=["render"], workers=1)

    assert not state_manifest.exists()
    assert (case_directory / "manifest.jsonl").read_bytes() == root_before


def test_run_case_writes_manual_full_resampling_artifacts_and_gallery(monkeypatch, tmp_path):
    ct_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(np.full((32, 32, 32), 40.0, dtype=np.float32)), str(ct_path))
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((8.0, 8.0, 13.0))
    mesh_path = tmp_path / "model.obj"
    mesh.export(mesh_path)
    label_path = tmp_path / "labels.nrrd"
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((32, 32, 32), dtype=np.uint8)),
        str(label_path),
    )
    organ_models = {name: mesh_path for name in (
        "adrenal_gland_left", "adrenal_gland_right", "aorta", "duodenum", "esophagus", "gallbladder",
        "inferior_vena_cava", "kidney_left", "kidney_right", "liver", "pancreas", "portal_vein_and_splenic_vein", "spleen", "stomach",
    )}
    config = CaseConfig(
        case_id="case_001",
        ct_path=ct_path,
        output_root=tmp_path / "output",
        organ_models=organ_models,
        vessel_models=(
            VesselModel("artery_tree", mesh_path, "artery", (255, 82, 0)),
            VesselModel("vein_tree", mesh_path, "vein", (0, 188, 212)),
        ),
        sampling=SamplingConfig(
            point_counts={
                "stomach": 1,
                "liver": 1,
                "pancreas": 1,
                "duodenum_part1": 1,
                "duodenum_part2": 1,
                "esophagus": 1,
            },
            duodenum_centerline_endpoint_hints_ras_mm=(
                (19.0, 24.0, 700.0),
                (-33.0, 1.0, 664.0),
            ),
            duodenum_centerline_endpoint_match_tolerance_mm=1.0,
        ),
        square=SquareConfig(side_length_mm=10.0),
        ct=CTConfig(output_resolution=20),
        filtering=FilterConfig(),
        runtime=RuntimeConfig(seed=0, workers=2, backend="cpu"),
        manual_segmentation=_manual_segmentation_config(label_path),
        geometry=GeometryConfig(input_coordinate_system="RAS"),
    )
    manual_selection = CenterlineSelectionAudit(
        mode="manual_endpoint_hints",
        coordinate_system="RAS",
        configured_proximal_ras_mm=(19.0, 24.0, 700.0),
        configured_distal_ras_mm=(-33.0, 1.0, 664.0),
        matched_proximal_ras_mm=(19.0, 24.0, 700.0),
        matched_distal_ras_mm=(-33.0, 1.0, 664.0),
        proximal_match_error_mm=0.0,
        distal_match_error_mm=0.0,
        endpoint_match_tolerance_mm=1.0,
        path_point_count=166,
        path_length_mm=224.7410832840096,
        skeleton_point_count=190,
        endpoint_count=7,
        branchpoint_count=5,
        connected_component_count=1,
        excluded_node_count=24,
        excluded_endpoints_ras_mm=(
            (-35.0, 2.0, 661.0),
            (-31.0, 16.0, 655.0),
            (25.0, 1.0, 629.0),
            (44.0, 1.0, 674.0),
            (35.0, 16.0, 679.0),
        ),
        automatic_terminal_spur_pruning_applied=False,
    )
    manual_centerline = CenterlinePath(
        np.asarray([[19.0, 24.0, 700.0], [-33.0, 1.0, 664.0]]),
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        np.asarray([0.0, 224.7410832840096]),
        selection_audit=manual_selection,
    )
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[8.0, 8.0, 8.0]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            sampling_statistics={
                "stomach": SamplingStatistics(1, 10, 1, 10.0, None),
            },
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, 0.0, 0.0]),
            pancreas_special_x_limit=100.0,
            source_surface_audit={
                "enabled": True,
                "selection_rule": "largest_watertight_absolute_volume",
                "input_component_count": 1,
                "input_face_count": 12,
                "kept_face_count": 12,
                "discarded_face_count": 0,
                "selected_enclosed_volume_mm3": 8.0,
            },
        ),
        "duodenum": SurfaceSamples(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            centerline=manual_centerline,
        ),
    }
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: surfaces)
    cpu_batch_sizes: list[int] = []
    original_sample_many = CachedCpuBackend.sample_many

    def record_cpu_batch(self, vertices_batch, resolution, fill_hu_value):
        cpu_batch_sizes.append(len(vertices_batch))
        return original_sample_many(self, vertices_batch, resolution, fill_hu_value)

    monkeypatch.setattr(CachedCpuBackend, "sample_many", record_cpu_batch)

    dry = run_case(config, dry_run=True)
    assert dry.total_squares == 3211
    assert not (tmp_path / "output").exists()

    completed = run_case(config)

    assert completed.total_squares == 3211
    assert (tmp_path / "output" / "case_001" / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "squarePLY" / "Stomach-vertex.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "gallery" / "gallery.jsonl").is_file()
    metadata = json.loads((tmp_path / "output" / "case_001" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_backend"] == "cpu"
    assert metadata["total_squares"] == 3211
    assert metadata["coordinate_system"] == "RAS"
    assert metadata["core_design_filename"] == "core-design-amendment-20260819.md"
    assert metadata["core_design_sha256"] == CORE_DESIGN_SHA256
    assert metadata["base_core_design_filename"] == "基于目标器官的采样方法-20260813.docx"
    assert metadata["base_core_design_sha256"] == (
        "de56e7a1b984f925e97631b076d6b729e77575eb6513b4d57f3028818b7e71ca"
    )
    assert len(metadata["build_git_commit"]) == 40
    assert metadata["minimum_point_spacing_mm"] == 10.0
    assert metadata["sampling_point_plan"] == {
        "schema_version": "sampling-point-plan/v1",
        "organs": {
            "stomach": [
                {
                    "point_index": 0,
                    "probe_point_world": [8.0, 8.0, 8.0],
                    "input_normal_world": [0.0, 0.0, 1.0],
                    "source_region": "stomach",
                    "yaw_policy": "standard",
                    "target_ids": ["liver"],
                    "base_local_axes_world": {
                        "x": [0.0, 0.0, 1.0],
                        "y": [0.7071067811865476, 0.7071067811865476, 0.0],
                        "z": [-0.7071067811865476, 0.7071067811865476, 0.0],
                    },
                    "candidate_pose_count": 3211,
                }
            ],
            "liver": [],
            "pancreas": [],
            "duodenum": [],
            "esophagus": [],
        },
    }
    assert metadata["sampling_configuration"]["count_policy"] == (
        "upper_bound_preserve_outer_surface_and_minimum_spacing"
    )
    assert metadata["surface_sampling_audit"]["organs"]["stomach"] == {
        "source_surface": surfaces["stomach"].source_surface_audit,
        "regions": {
            "stomach": {
                "requested_count": 1,
                "candidate_count": 10,
                "actual_count": 1,
                "shortfall_count": 0,
                "minimum_spacing_mm": 10.0,
                "actual_minimum_distance_mm": None,
            }
        },
        "requested_count": 1,
        "actual_count": 1,
        "shortfall_count": 0,
    }
    assert metadata["sampling_configuration"]["duodenum_centerline_endpoint_hints_ras_mm"] == {
        "proximal": [19.0, 24.0, 700.0],
        "distal": [-33.0, 1.0, 664.0],
    }
    assert metadata["sampling_configuration"]["duodenum_centerline_endpoint_match_tolerance_mm"] == 1.0
    assert metadata["duodenum_centerline_selection"]["mode"] == "manual_endpoint_hints"
    assert metadata["duodenum_centerline_selection"]["matched_proximal_ras_mm"] == [19.0, 24.0, 700.0]
    assert metadata["duodenum_centerline_selection"]["path_point_count"] == 166
    assert metadata["duodenum_centerline_selection"]["automatic_terminal_spur_pruning_applied"] is False
    assert metadata["sampling_configuration"]["esophagus_extension_target_filter"] == (
        "original_and_translated_segments_independently"
    )
    assert metadata["pose_angles_degrees"]["roll"] == list(np.arange(-45.0, 46.0, 5.0))
    assert metadata["pose_angles_degrees"]["pitch"] == list(np.arange(-30.0, 31.0, 5.0))
    assert metadata["pose_angles_degrees"]["yaw"]["duodenum_bulb"] == list(
        np.arange(-120.0, 31.0, 5.0)
    )
    assert metadata["pose_angles_degrees"]["yaw"]["liver_region_two"] == list(
        np.arange(-60.0, 61.0, 5.0)
    )
    assert metadata["pose_convention"] == {
        "coordinate_frame": "local_right_handed",
        "matrix_order": "B @ Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "positive_yaw": "counterclockwise",
        "yaw_observer": "local_positive_z_looking_toward_probe",
        "rotation_center": "probe_at_square_bottom_edge_midpoint",
    }
    assert metadata["square_sampling"] == {
        "side_length_mm": 10.0,
        "output_resolution": [20, 20],
        "interpolation": "cubic_bspline",
        "interpolation_order": 3,
        "window_level_hu": 40.0,
        "window_width_hu": 400.0,
        "fill_hu_value": -1000.0,
    }
    assert metadata["quality_filtering"] == {
        "black_threshold": 50,
        "black_ratio_limit": 0.6,
        "line_min_diagonal_fraction": 0.7,
        "black_side_min_ratio": 0.9,
        "valid_side_max_black_ratio": 0.1,
    }
    assert metadata["fov_policy"] == {
        "vertex_rule": "any_square_vertex_outside_ct",
        "outside_status": "excluded_fov",
        "saved_artifacts": ["ct_png"],
        "out_of_bounds_png_value": 0,
    }
    assert metadata["eus_possible_organs"] == {
        "schema_version": "eus-possible-organs/v1",
        "sha256": "54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d",
        "organ_labels": [
            "adrenal_gland_left",
            "adrenal_gland_right",
            "aorta",
            "duodenum",
            "inferior_vena_cava",
            "kidney_left",
            "kidney_right",
            "liver",
            "pancreas",
            "portal_vein",
            "spleen",
        ],
        "excluded_organ_labels": ["bile_duct", "common_bile_duct"],
        "geometry_sources": {"portal_vein": "portal_vein_and_splenic_vein"},
    }
    assert metadata["completed_pose_count"] == 3211
    assert metadata["status_counts"] == completed.status_counts
    assert cpu_batch_sizes == [8] * 401 + [3]
    library_summary = json.loads((tmp_path / "output" / "case_001" / "library_summary.json").read_text(encoding="utf-8"))
    assert library_summary["case_id"] == "case_001"
    assert library_summary["indexed_feature_count"] == completed.indexed_feature_count
    assert library_summary["gallery_manifest"] == "gallery/gallery.jsonl"
    assert set(library_summary["organ_boundary_colors"]) == set(ORGAN_BOUNDARY_IDS)
    assert set(library_summary["organ_label_counts"]).issubset(set(ORGAN_BOUNDARY_IDS))
    gallery_records = [
        json.loads(line)
        for line in (tmp_path / "output" / "case_001" / "gallery" / "gallery.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_eus_counts = Counter(
        label
        for record in gallery_records
        for label in set(record["eus_candidate_organ_labels"])
    )
    assert library_summary["eus_candidate_organ_label_counts"] == dict(
        sorted(expected_eus_counts.items())
    )
    assert library_summary["eus_possible_organs"] == {
        "schema_version": "eus-possible-organs/v1",
        "sha256": "54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d",
        "organs": pipeline_module.load_eus_organ_catalog().to_record()["organs"],
        "geometry_sources": {"portal_vein": "portal_vein_and_splenic_vein"},
    }

    calls_after_first_run = list(cpu_batch_sizes)
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.create_sampling_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no-op resume must not initialize a backend")),
    )
    resumed = run_case(config)
    assert cpu_batch_sizes == calls_after_first_run
    assert sum(resumed.status_counts.values()) == 3211
    assert len((tmp_path / "output" / "case_001" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 3211

    rectangles_before_incompatible_resume = (
        tmp_path / "output" / "case_001" / "rectangles.ply"
    ).read_bytes()
    incompatible_config = replace(config, square=replace(config.square, side_length_mm=11.0))
    with pytest.raises(ValueError, match="运行协议|配置|不一致"):
        run_case(incompatible_config)
    assert (tmp_path / "output" / "case_001" / "rectangles.ply").read_bytes() == rectangles_before_incompatible_resume

    recolored_config = replace(
        config,
        vessel_models=(
            replace(config.vessel_models[0], color=(1, 2, 3)),
            config.vessel_models[1],
        ),
    )
    with pytest.raises(ValueError, match="运行协议|配置|不一致"):
        run_case(recolored_config)

    shifted_surfaces = {
        **surfaces,
        "stomach": replace(
            surfaces["stomach"],
            points=surfaces["stomach"].points + np.asarray([0.0, 0.0, 1.0]),
        )
    }
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: shifted_surfaces)
    with pytest.raises(ValueError, match="姿态|几何|不一致"):
        run_case(config)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: surfaces)

    original_mesh_bytes = mesh_path.read_bytes()
    mesh_path.write_bytes(original_mesh_bytes + b"\n# changed input\n")
    with pytest.raises(ValueError, match="输入|运行协议|不一致"):
        run_case(config)
    mesh_path.write_bytes(original_mesh_bytes)

    case_directory = tmp_path / "output" / "case_001"
    manifest_path = case_directory / "manifest.jsonl"
    existing_record = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    stale_record = {**existing_record, "slice_id": "stale-pose-id"}
    serialized_stale = json.dumps(stale_record) + "\n"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized_stale)
    state_paths = {
        "gallery": case_directory / "gallery" / "gallery.jsonl",
        "unindexed": case_directory / "unindexed" / "unindexed.jsonl",
        "rejected": case_directory / "rejected" / "rejected.jsonl",
        "excluded_fov": case_directory / "excluded_fov.jsonl",
    }
    with state_paths[stale_record["status"]].open("a", encoding="utf-8") as handle:
        handle.write(serialized_stale)
    with pytest.raises(ValueError, match="陈旧|额外|姿态集合|stale-pose-id"):
        run_case(config)

    class OffsetBackend:
        name = "gpu:0"

        def __init__(self, cpu_backend):
            self.cpu_backend = cpu_backend

        def sample_many(self, vertices_batch, resolution, fill_hu_value):
            return self.cpu_backend.sample_many(vertices_batch, resolution, fill_hu_value) + np.float32(0.002)

        def close(self):
            pass

    def create_offset_backend(volume, **_):
        return OffsetBackend(CachedCpuBackend(volume)), {
            "requested_backend": "auto",
            "selected_backend": "gpu:0",
            "gpu_device": 0,
            "gpu_batch_size": 32,
            "fallback_reason": None,
            "coefficient_dtype": "float64",
        }

    monkeypatch.setattr("ct_vascular_resampling.pipeline.create_sampling_backend", create_offset_backend)
    run_case(replace(config, case_id="gpu_validation", runtime=replace(config.runtime, backend="auto")))

    gpu_metadata = json.loads((tmp_path / "output" / "gpu_validation" / "run_metadata.json").read_text(encoding="utf-8"))
    gallery_record = json.loads(
        (tmp_path / "output" / "gpu_validation" / "gallery" / "gallery.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert gpu_metadata["selected_backend"] == "cpu"
    assert gpu_metadata["calibration"]["accepted"] is False
    assert gallery_record["resampling_backend"] == "cpu"

    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    empty_config = replace(
        config,
        case_id="empty_case",
        sampling=replace(config.sampling, duodenum_centerline_endpoint_hints_ras_mm=None),
    )
    empty_index = run_case(empty_config, steps=["index"])

    assert empty_index.indexed_feature_count == 0

    log_only_case = tmp_path / "output" / "log_only_case" / "logs"
    log_only_case.mkdir(parents=True)
    (log_only_case / "run.log").write_text("started\n", encoding="utf-8")
    log_only = run_case(replace(empty_config, case_id="log_only_case"), resume=False, steps=["sample"])

    assert log_only.total_squares == 0
    assert (log_only_case.parent / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
