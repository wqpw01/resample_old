from __future__ import annotations

import json
import numpy as np
import SimpleITK as sitk
import trimesh
from pathlib import Path
from dataclasses import replace

from ct_vascular_resampling.config import (
    CTConfig,
    CaseConfig,
    FilterConfig,
    RuntimeConfig,
    SamplingConfig,
    SquareConfig,
    VesselModel,
)
from ct_vascular_resampling.ct_resampling import CTVolume
from ct_vascular_resampling.gallery import GalleryWriter
from ct_vascular_resampling.pipeline import PreparedVessel, render_precomputed_square, render_square_sample, run_case
from ct_vascular_resampling.resampling_backend import CachedCpuBackend
from ct_vascular_resampling.sampling_pipeline import SquareSample, SurfaceSamples


def test_single_square_resamples_ct_and_vessel_model_into_gallery(tmp_path):
    image = sitk.GetImageFromArray(np.full((16, 16, 16), 40.0, dtype=np.float32))
    volume = CTVolume.from_sitk(image)
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

    status = render_square_sample(
        sample,
        volume,
        [PreparedVessel("portal_tree", "portal", (255, 0, 255), vessel_mesh)],
        CTConfig(output_resolution=20),
        FilterConfig(),
        writer,
    )

    assert status == "gallery"
    assert (tmp_path / "case" / "gallery" / "ct_overlay" / "stomach-000000-x-00.png").is_file()


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
        resampling_backend="cpu",
    )

    assert status == "rejected"


def test_out_of_fov_square_is_excluded_without_ct_artifacts(tmp_path):
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

    status = render_square_sample(sample, volume, [], CTConfig(output_resolution=20), FilterConfig(), writer)

    record = json.loads((tmp_path / "case" / "excluded_fov.jsonl").read_text(encoding="utf-8"))
    assert status == "excluded_fov"
    assert record["fov_diagnostics"]["contains_ct_fov_exceedance"] is True
    assert not (tmp_path / "case" / "rejected").exists()


def test_run_case_excludes_fov_square_before_ct_interpolation(monkeypatch, tmp_path):
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
    )
    outside_sample = SquareSample(
        sample_id="esophagus-000010-x-01",
        organ="esophagus",
        probe_point_world=np.asarray([1.0, 3.0, 5.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        vertices=np.asarray([[-1.0, 2.0, 5.0], [3.0, 2.0, 5.0], [3.0, 6.0, 5.0], [-1.0, 6.0, 5.0]]),
    )
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_: {})
    monkeypatch.setattr("ct_vascular_resampling.pipeline.generate_square_samples", lambda *_: [outside_sample])

    def unexpected_ct_interpolation(*_):
        raise AssertionError("FOV exclusion must happen before CT interpolation")

    monkeypatch.setattr(CachedCpuBackend, "sample_many", unexpected_ct_interpolation)

    summary = run_case(config, steps=["render"], workers=1)

    case_directory = tmp_path / "output" / "fov_case"
    record = json.loads((case_directory / "excluded_fov.jsonl").read_text(encoding="utf-8"))
    assert summary.status_counts == {"excluded_fov": 1}
    assert record["fov_diagnostics"]["contains_ct_fov_exceedance"] is True
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
    )
    surfaces = {"stomach": SurfaceSamples(np.asarray([[8.0, 8.0, 8.0]]), np.asarray([[0.0, 0.0, 1.0]]))}
    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_: surfaces)
    cpu_batch_sizes: list[int] = []
    original_sample_many = CachedCpuBackend.sample_many

    def record_cpu_batch(self, vertices_batch, resolution, fill_hu_value):
        cpu_batch_sizes.append(len(vertices_batch))
        return original_sample_many(self, vertices_batch, resolution, fill_hu_value)

    monkeypatch.setattr(CachedCpuBackend, "sample_many", record_cpu_batch)

    dry = run_case(config, dry_run=True)
    assert dry.total_squares == 81
    assert not (tmp_path / "output").exists()

    completed = run_case(config)

    assert completed.total_squares == 81
    assert (tmp_path / "output" / "case_001" / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "squarePLY" / "Stomach-vertex.ply").is_file()
    assert (tmp_path / "output" / "case_001" / "gallery" / "gallery.jsonl").is_file()
    metadata = json.loads((tmp_path / "output" / "case_001" / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["selected_backend"] == "cpu"
    assert metadata["total_squares"] == 81
    assert cpu_batch_sizes == [1] * 81
    library_summary = json.loads((tmp_path / "output" / "case_001" / "library_summary.json").read_text(encoding="utf-8"))
    assert library_summary["case_id"] == "case_001"
    assert library_summary["indexed_feature_count"] == completed.indexed_feature_count
    assert library_summary["gallery_manifest"] == "gallery/gallery.jsonl"

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

    monkeypatch.setattr("ct_vascular_resampling.pipeline.sample_organs", lambda *_: {})
    empty_index = run_case(replace(config, case_id="empty_case"), steps=["index"])

    assert empty_index.indexed_feature_count == 0

    log_only_case = tmp_path / "output" / "log_only_case" / "logs"
    log_only_case.mkdir(parents=True)
    (log_only_case / "run.log").write_text("started\n", encoding="utf-8")
    log_only = run_case(replace(config, case_id="log_only_case"), resume=False, steps=["sample"])

    assert log_only.total_squares == 0
    assert (log_only_case.parent / "ResampledpointPLY" / "FPS-Stomach.ply").is_file()
