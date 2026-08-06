from __future__ import annotations

import json
import numpy as np
from PIL import Image
import pytest
import SimpleITK as sitk
import trimesh
from pathlib import Path
from dataclasses import replace

from ct_vascular_resampling.config import (
    CTConfig,
    CaseConfig,
    FilterConfig,
    GeometryConfig,
    RuntimeConfig,
    SamplingConfig,
    SquareConfig,
    VesselModel,
)
from ct_vascular_resampling.ct_resampling import CTVolume, diagnose_square_fov
from ct_vascular_resampling.gallery import GalleryWriter
import ct_vascular_resampling.pipeline as pipeline_module
from ct_vascular_resampling.pipeline import PreparedVessel, render_precomputed_square, render_square_sample, run_case
from ct_vascular_resampling.resampling_backend import CachedCpuBackend
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


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


def test_run_case_resamples_fov_square_and_records_exclusion(monkeypatch, tmp_path):
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
    assert not (case_directory / "rejected").exists()


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
        sampling=SamplingConfig(point_counts={"stomach": 1, "liver": 1, "pancreas": 1, "duodenum_part1": 1, "duodenum_part2": 1, "esophagus": 1}),
        square=SquareConfig(side_length_mm=10.0),
        ct=CTConfig(output_resolution=20),
        filtering=FilterConfig(),
        runtime=RuntimeConfig(seed=0, workers=2, backend="cpu"),
        geometry=GeometryConfig(input_coordinate_system="RAS"),
    )
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[8.0, 8.0, 8.0]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, 0.0, 0.0]),
            pancreas_special_x_limit=100.0,
        )
    }
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_, **__: surfaces)
    cpu_batch_sizes: list[int] = []
    original_sample_many = CachedCpuBackend.sample_many

    def record_cpu_batch(self, vertices_batch, resolution, fill_hu_value):
        cpu_batch_sizes.append(len(vertices_batch))
        return original_sample_many(self, vertices_batch, resolution, fill_hu_value)

    monkeypatch.setattr(CachedCpuBackend, "sample_many", record_cpu_batch)

    dry = run_case(config, dry_run=True)
    assert dry.total_squares == 117
    assert not (tmp_path / "output").exists()

    completed = run_case(config)

    assert completed.total_squares == 117
    assert (tmp_path / "output" / "case_001" / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "squarePLY" / "Stomach-vertex.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "gallery" / "gallery.jsonl").is_file()
    metadata = json.loads((tmp_path / "output" / "case_001" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_backend"] == "cpu"
    assert metadata["total_squares"] == 117
    assert metadata["coordinate_system"] == "RAS"
    assert metadata["core_design_sha256"] == "4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b"
    assert len(metadata["build_git_commit"]) == 40
    assert metadata["minimum_point_spacing_mm"] == 10.0
    assert metadata["pose_angles_degrees"]["roll"] == [-5.0, 0.0, 5.0]
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
    assert metadata["completed_pose_count"] == 117
    assert metadata["status_counts"] == completed.status_counts
    assert cpu_batch_sizes == [8] * 14 + [5]
    library_summary = json.loads((tmp_path / "output" / "case_001" / "library_summary.json").read_text(encoding="utf-8"))
    assert library_summary["case_id"] == "case_001"
    assert library_summary["indexed_feature_count"] == completed.indexed_feature_count
    assert library_summary["gallery_manifest"] == "gallery/gallery.jsonl"
    assert set(library_summary["organ_boundary_colors"]) == set(pipeline_module.ORGAN_BOUNDARY_IDS)
    assert set(library_summary["organ_label_counts"]).issubset(set(pipeline_module.ORGAN_BOUNDARY_IDS))

    calls_after_first_run = list(cpu_batch_sizes)
    monkeypatch.setattr(
        "ct_vascular_resampling.pipeline.create_sampling_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no-op resume must not initialize a backend")),
    )
    resumed = run_case(config)
    assert cpu_batch_sizes == calls_after_first_run
    assert sum(resumed.status_counts.values()) == 117
    assert len((tmp_path / "output" / "case_001" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 117

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
    empty_index = run_case(replace(config, case_id="empty_case"), steps=["index"])

    assert empty_index.indexed_feature_count == 0

    log_only_case = tmp_path / "output" / "log_only_case" / "logs"
    log_only_case.mkdir(parents=True)
    (log_only_case / "run.log").write_text("started\n", encoding="utf-8")
    log_only = run_case(replace(config, case_id="log_only_case"), resume=False, steps=["sample"])

    assert log_only.total_squares == 0
    assert (log_only_case.parent / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
