from __future__ import annotations

from collections import Counter
import hashlib
import json
import numpy as np
from PIL import Image
import pytest
import SimpleITK as sitk
import trimesh
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

from ct_vascular_resampling.config import (
    CTConfig,
    CaseConfig,
    FilterConfig,
    GeometryConfig,
    ManualSegmentationConfig,
    RuntimeConfig,
    SamplingConfig,
    SquareConfig,
    VesselModel,
)
from ct_vascular_resampling.centerline import CenterlinePath, CenterlineSelectionAudit
from ct_vascular_resampling.ct_resampling import CTVolume, diagnose_square_fov
from ct_vascular_resampling.gallery import GalleryWriter
from ct_vascular_resampling.label_resampling import CpuLabelBackend
import ct_vascular_resampling.pipeline as pipeline_module
from ct_vascular_resampling.pipeline import PreparedVessel, render_precomputed_square, render_square_sample, run_case
from ct_vascular_resampling.resampling_backend import CachedCpuBackend
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


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


def test_gallery_organ_summary_counts_slices_and_rejects_legacy_records(tmp_path):
    gallery_directory = tmp_path / "gallery"
    combined_directory = gallery_directory / "organ_vessel_boundary"
    combined_directory.mkdir(parents=True)
    for sample_id in ("first", "second"):
        Image.new("RGB", (10, 10), "white").save(combined_directory / f"{sample_id}.png")
    records = [
        {
            "slice_id": "first",
            "status": "gallery",
            "organ_metadata_schema_version": "eus-organ-metadata/v1",
            "organ_vessel_boundary_png": "organ_vessel_boundary/first.png",
            "organ_labels": ["liver", "stomach"],
            "eus_candidate_organ_labels": ["liver"],
        },
        {
            "slice_id": "second",
            "status": "gallery",
            "organ_metadata_schema_version": "eus-organ-metadata/v1",
            "organ_vessel_boundary_png": "organ_vessel_boundary/second.png",
            "organ_labels": ["liver"],
            "eus_candidate_organ_labels": ["liver"],
        },
    ]
    manifest = gallery_directory / "gallery.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    organ_counts, eus_counts = pipeline_module._gallery_organ_label_counts(manifest)

    assert organ_counts == Counter({"liver": 2, "stomach": 1})
    assert eus_counts == Counter({"liver": 2})

    manifest.write_text(
        json.dumps({"slice_id": "legacy", "status": "gallery", "organ_labels": ["liver"]})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="旧版|schema"):
        pipeline_module._gallery_organ_label_counts(manifest)


def test_prepared_organs_use_canonical_labels_and_existing_model_paths(monkeypatch, tmp_path):
    mesh = trimesh.creation.box()
    organ_models = {
        identifier: tmp_path / f"{identifier}.ply"
        for identifier in pipeline_module.ORGAN_BOUNDARY_MODEL_IDS.values()
    }
    loaded_paths = []

    def load(path, *, input_coordinate_system):
        loaded_paths.append((path, input_coordinate_system))
        return SimpleNamespace(mesh=mesh.copy())

    monkeypatch.setattr(pipeline_module, "load_surface_mesh", load)
    config = SimpleNamespace(
        organ_models=organ_models,
        geometry=SimpleNamespace(input_coordinate_system="LPS"),
    )

    prepared = pipeline_module._load_prepared_organs(config)

    by_label = {item.label: item for item in prepared}
    assert len(prepared) == 14
    assert by_label["portal_vein"].identifier == "portal_vein_and_splenic_vein"
    assert loaded_paths.count((organ_models["portal_vein_and_splenic_vein"], "LPS")) == 1


def test_prepared_organ_and_vessel_layers_share_mesh_for_the_same_source(monkeypatch, tmp_path):
    shared_path = tmp_path / "portal_vein_and_splenic_vein.ply"
    organ_models = {
        identifier: tmp_path / f"{identifier}.ply"
        for identifier in pipeline_module.ORGAN_BOUNDARY_MODEL_IDS.values()
    }
    organ_models["portal_vein_and_splenic_vein"] = shared_path
    loaded_paths = []

    def load(path, *, input_coordinate_system):
        loaded_paths.append((path, input_coordinate_system))
        return SimpleNamespace(mesh=trimesh.creation.box())

    monkeypatch.setattr(pipeline_module, "load_surface_mesh", load)
    config = SimpleNamespace(
        organ_models=organ_models,
        vessel_models=(
            VesselModel("portal_tree", shared_path, "portal", (255, 0, 255)),
        ),
        geometry=SimpleNamespace(input_coordinate_system="LPS"),
    )
    mesh_cache = {}

    organs = pipeline_module._load_prepared_organs(config, mesh_cache)
    vessels = pipeline_module._load_prepared_vessels(config, mesh_cache)

    portal_organ = next(item for item in organs if item.label == "portal_vein")
    assert loaded_paths.count((shared_path, "LPS")) == 1
    assert portal_organ.mesh is vessels[0].mesh
    assert portal_organ.label == "portal_vein"
    assert vessels[0].label == "portal"


def test_single_square_resamples_ct_and_vessel_model_into_gallery(tmp_path):
    image = sitk.GetImageFromArray(np.full((16, 16, 16), 40.0, dtype=np.float32))
    volume = CTVolume.from_sitk(image)
    vessel_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    vessel_mesh.apply_translation((7.0, 7.0, 5.0))
    organ_mesh = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    organ_mesh.apply_translation((7.0, 7.0, 5.0))
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]),
        source_region="stomach",
        yaw_policy="standard",
        roll_degrees=-5.0,
        pitch_degrees=0.0,
        yaw_degrees=30.0,
        local_x_world=np.asarray([0.0, 1.0, 0.0]),
        local_y_world=np.asarray([1.0, 0.0, 0.0]),
        local_z_world=np.asarray([0.0, 0.0, -1.0]),
        target_ids=("liver",),
    )
    writer = GalleryWriter(tmp_path / "case", "case")

    status = render_square_sample(
        sample,
        volume,
        [PreparedVessel("portal_tree", "portal", (255, 0, 255), vessel_mesh)],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        organs=[pipeline_module.PreparedOrgan("liver", "liver", (140, 86, 75), organ_mesh, organ_mesh.bounds)],
    )

    assert status == "gallery"
    assert (tmp_path / "case" / "gallery" / "ct_overlay" / "stomach-000000-x-00.png").is_file()
    assert (tmp_path / "case" / "gallery" / "organ_vessel_boundary" / "stomach-000000-x-00.png").is_file()
    record = json.loads((tmp_path / "case" / "gallery" / "gallery.jsonl").read_text(encoding="utf-8"))
    assert record["organ_labels"] == ["liver"]
    assert record["coordinate_system"] == "RAS"
    assert record["core_design_sha256"] == "4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b"
    assert len(record["build_git_commit"]) == 40
    assert record["source_region"] == "stomach"
    assert record["angles_degrees"] == {"roll": -5.0, "pitch": 0.0, "yaw": 30.0}
    assert record["target_ids"] == ["liver"]


def test_precomputed_square_writes_the_same_gallery_artifacts(tmp_path):
    vessel_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    vessel_mesh.apply_translation((7.0, 7.0, 5.0))
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]),
    )
    writer = GalleryWriter(tmp_path / "case", "case")

    status = render_precomputed_square(
        sample,
        np.full((20, 20), 40.0, dtype=np.float32),
        [PreparedVessel("portal_tree", "portal", (255, 0, 255), vessel_mesh)],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        resampling_backend="cpu",
    )

    assert status == "gallery"
    record = json.loads((tmp_path / "case" / "gallery" / "gallery.jsonl").read_text(encoding="utf-8"))
    assert record["resampling_backend"] == "cpu"


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
    vessels = [PreparedVessel("portal_tree", "portal", (255, 0, 255), vessel_mesh)]
    hu = np.full((20, 20), 40.0, dtype=np.float32)
    labels = np.zeros((20, 20), dtype=np.uint8)
    labels[7:12, 7:12] = 8
    labels[0:4, 15:19] = 9
    legacy_writer = GalleryWriter(tmp_path / "legacy", "case")
    manual_writer = GalleryWriter(
        tmp_path / "manual",
        "case",
        manual_segmentation_enabled=True,
    )

    legacy_status = render_precomputed_square(
        sample,
        hu,
        vessels,
        CTConfig(output_resolution=20),
        FilterConfig(),
        legacy_writer,
        resampling_backend="cpu",
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

    legacy_record = json.loads((tmp_path / "legacy/gallery/gallery.jsonl").read_text(encoding="utf-8"))
    manual_record = json.loads((tmp_path / "manual/gallery/gallery.jsonl").read_text(encoding="utf-8"))
    assert legacy_status == manual_status == "gallery"
    assert manual_record["features"] == legacy_record["features"]
    for directory in ("boundary_only", "ct_overlay"):
        assert (
            tmp_path / "manual/gallery" / directory / f"{sample.sample_id}.png"
        ).read_bytes() == (
            tmp_path / "legacy/gallery" / directory / f"{sample.sample_id}.png"
        ).read_bytes()
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
        prepared = [PreparedVessel("portal_tree", "portal", (255, 0, 255), mesh)]
    writer = GalleryWriter(
        tmp_path / "case",
        "case",
        manual_segmentation_enabled=True,
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
    organ_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    writer = GalleryWriter(tmp_path / "case", "case")

    def unexpected_intersection(*_):
        raise AssertionError("rejected CT should not intersect vessel meshes")

    monkeypatch.setattr("ct_vascular_resampling.pipeline.intersect_mesh_with_square", unexpected_intersection)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), -1000.0, dtype=np.float32),
        [PreparedVessel("portal_tree", "portal", (255, 0, 255), vessel_mesh)],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        organs=[pipeline_module.PreparedOrgan("liver", "liver", (140, 86, 75), organ_mesh, organ_mesh.bounds)],
        resampling_backend="cpu",
    )

    assert status == "rejected"


def test_unindexed_precomputed_square_skips_organ_intersection(monkeypatch, tmp_path):
    sample = SquareSample(
        sample_id="stomach-000000-x-00",
        organ="stomach",
        probe_point_world=np.asarray([7.0, 7.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[2.0, 2.0, 5.0], [12.0, 2.0, 5.0], [12.0, 12.0, 5.0], [2.0, 12.0, 5.0]]),
    )
    organ_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    writer = GalleryWriter(tmp_path / "case", "case")

    def unexpected_intersection(*_):
        raise AssertionError("unindexed samples must skip organ mesh intersections")

    monkeypatch.setattr("ct_vascular_resampling.pipeline.intersect_mesh_with_square", unexpected_intersection)

    status = render_precomputed_square(
        sample,
        np.full((20, 20), 40.0, dtype=np.float32),
        [],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
        organs=[pipeline_module.PreparedOrgan("liver", "liver", (140, 86, 75), organ_mesh, organ_mesh.bounds)],
        resampling_backend="cpu",
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

    status = render_square_sample(
        sample,
        volume,
        [],
        CTConfig(output_resolution=20, fill_hu_value=40.0),
        FilterConfig(),
        writer,
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
        registration_module_path=tmp_path / "2021.py",
        sampling=SamplingConfig(
            point_counts={"stomach": 1, "liver": 1, "pancreas": 1, "duodenum_part1": 1, "duodenum_part2": 1, "esophagus": 1}
        ),
        square=SquareConfig(side_length_mm=10.0),
        ct=CTConfig(output_resolution=20),
        filtering=FilterConfig(),
        runtime=RuntimeConfig(seed=0, workers=1, backend="cpu"),
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
    config, sample = _single_fov_case(tmp_path)
    label_path = tmp_path / "labels.nrrd"
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.uint8)),
        str(label_path),
    )
    return replace(config, manual_segmentation=_manual_segmentation_config(label_path)), sample


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
    for sample_id, labels, features in (
        (
            "first",
            ["aorta", "inferior_vena_cava"],
            [{"label": "aorta", "x_mm": 1.0, "y_mm": 2.0, "area_mm2": 3.0}],
        ),
        ("second", ["aorta"], []),
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
                "organ_metadata_schema_version": "eus-organ-metadata/v1",
                "organ_vessel_boundary_png": f"organ_vessel_boundary/{sample_id}.png",
                "organ_labels": [],
                "eus_candidate_organ_labels": [],
                "eus_vessel_metadata_schema_version": "eus-vessel-metadata/v1",
                "eus_vessel_labels": labels,
                "eus_vessel_features": features,
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
    import ct_vascular_resampling.registration_adapter as registration_adapter

    monkeypatch.setattr(
        registration_adapter,
        "load_gallery_database",
        lambda *_args, **_kwargs: SimpleNamespace(features=[]),
    )
    run_case(config, steps=["index"])

    summary = json.loads((case_directory / "library_summary.json").read_text(encoding="utf-8"))
    assert summary["eus_vessel_label_counts"] == {
        "aorta": 2,
        "inferior_vena_cava": 1,
    }
    assert summary["eus_vessel_feature_counts"] == {"aorta": 1}
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


def test_manual_recovery_refuses_to_invent_missing_original_protocol(monkeypatch, tmp_path):
    config, outside_sample = _manual_fov_case(tmp_path)
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: {})
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.generate_square_samples",
        lambda *_: [outside_sample],
    )
    run_case(config, steps=["render"], workers=1)
    case_directory = config.output_root / config.case_id
    metadata_path = case_directory / "run_metadata.json"
    metadata_path.unlink()
    manifest_before = (case_directory / "manifest.jsonl").read_bytes()

    with pytest.raises(ValueError, match="手工分割|原始.*协议|无法.*重建|新输出"):
        pipeline_module.recover_interrupted_run_metadata(
            config,
            expected_completed_count=1,
            reason="sighup",
            exit_code=129,
            recovered_at_utc="2026-08-11T12:00:00Z",
        )

    assert not metadata_path.exists()
    assert (case_directory / "manifest.jsonl").read_bytes() == manifest_before


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


def test_run_case_writes_legacy_intermediates_and_gallery(monkeypatch, tmp_path):
    ct_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(np.full((32, 32, 32), 40.0, dtype=np.float32)), str(ct_path))
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation((8.0, 8.0, 13.0))
    mesh_path = tmp_path / "model.obj"
    mesh.export(mesh_path)
    (tmp_path / "2021.py").write_text(
        """
class VesselTriplet:
    def __init__(self, x, y, area, label=''): self.x, self.y, self.area, self.label = x, y, area, label
class FeatureVector:
    def __init__(self, triplets=None, pose=None): self.triplets, self.pose = triplets or [], pose
class ProbePose:
    def __init__(self, surface_point, rx, ry, rz, depth): self.surface_point = surface_point
class MultiLabelledCBIR:
    def __init__(self, database, search_range=2): self.database = database
class HMMPoseEstimator:
    def __init__(self, **kwargs): pass
""".strip(),
        encoding="utf-8",
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
            VesselModel("portal", mesh_path, "portal", (255, 0, 255)),
            VesselModel("hepatic", mesh_path, "hepatic", (0, 188, 212)),
        ),
        registration_module_path=tmp_path / "2021.py",
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
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, 0.0, 0.0]),
            pancreas_special_x_limit=100.0,
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
    assert dry.total_squares == 455
    assert not (tmp_path / "output").exists()

    completed = run_case(config)

    assert completed.total_squares == 455
    assert (tmp_path / "output" / "case_001" / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "squarePLY" / "Stomach-vertex.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "gallery" / "gallery.jsonl").is_file()
    metadata = json.loads((tmp_path / "output" / "case_001" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_backend"] == "cpu"
    assert metadata["total_squares"] == 455
    assert metadata["coordinate_system"] == "RAS"
    assert metadata["core_design_sha256"] == "4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b"
    assert len(metadata["build_git_commit"]) == 40
    assert metadata["minimum_point_spacing_mm"] == 10.0
    assert metadata["sampling_configuration"]["duodenum_centerline_endpoint_hints_ras_mm"] == {
        "proximal": [19.0, 24.0, 700.0],
        "distal": [-33.0, 1.0, 664.0],
    }
    assert metadata["sampling_configuration"]["duodenum_centerline_endpoint_match_tolerance_mm"] == 1.0
    assert metadata["duodenum_centerline_selection"]["mode"] == "manual_endpoint_hints"
    assert metadata["duodenum_centerline_selection"]["matched_proximal_ras_mm"] == [19.0, 24.0, 700.0]
    assert metadata["duodenum_centerline_selection"]["path_point_count"] == 166
    assert metadata["duodenum_centerline_selection"]["automatic_terminal_spur_pruning_applied"] is False
    assert metadata["pose_angles_degrees"]["roll"] == [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    assert metadata["pose_angles_degrees"]["pitch"] == [-10.0, -5.0, 0.0, 5.0, 10.0]
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
        "black_ratio_limit": 0.5,
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
    assert metadata["completed_pose_count"] == 455
    assert metadata["status_counts"] == completed.status_counts
    assert cpu_batch_sizes == [8] * 56 + [7]
    library_summary = json.loads((tmp_path / "output" / "case_001" / "library_summary.json").read_text(encoding="utf-8"))
    assert library_summary["case_id"] == "case_001"
    assert library_summary["indexed_feature_count"] == completed.indexed_feature_count
    assert library_summary["gallery_manifest"] == "gallery/gallery.jsonl"
    assert set(library_summary["organ_boundary_colors"]) == set(pipeline_module.ORGAN_BOUNDARY_IDS)
    assert set(library_summary["organ_label_counts"]).issubset(set(pipeline_module.ORGAN_BOUNDARY_IDS))
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
    assert any(count > 1 for count in expected_eus_counts.values())
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
    assert sum(resumed.status_counts.values()) == 455
    assert len((tmp_path / "output" / "case_001" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 455

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
