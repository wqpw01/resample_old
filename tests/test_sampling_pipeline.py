from __future__ import annotations

import numpy as np
import pytest
import trimesh

import ct_vascular_resampling.sampling_pipeline as sampling_pipeline_module
from ct_vascular_resampling.centerline import CenterlinePath
from ct_vascular_resampling.config import SamplingConfig, SquareConfig
from ct_vascular_resampling.sampling_pipeline import (
    SquareSample,
    SurfaceSamples,
    _candidate_pose_count,
    _deduplicate_exact_poses,
    _merge_unique,
    _sample,
    _valid_ordinary_indices,
    generate_square_samples,
    sample_organs,
)


def test_square_sample_expansion_uses_confirmed_axis_count_and_variant_count():
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
            pancreas_special_x_limit=10.0,
        ),
        "esophagus": SurfaceSamples(
            np.asarray([[0.0, 0.0, 20.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("esophagus",),
            target_ids=((),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
        ),
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    stomach = [sample for sample in samples if sample.organ == "stomach"]
    esophagus = [sample for sample in samples if sample.organ == "esophagus"]
    assert len(stomach) == 455
    assert len(esophagus) == 455
    assert stomach[0].sample_id.startswith("stomach-000000-")
    assert np.isclose(np.linalg.norm(stomach[0].vertices[1] - stomach[0].vertices[0]), 100.0)


def test_square_sample_expansion_returns_a_reiterable_pose_stream_instead_of_a_list():
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
            pancreas_special_x_limit=10.0,
        )
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    assert not isinstance(samples, list)
    assert len(samples) == 455
    first_ids = [sample.sample_id for sample in samples]
    second_ids = [sample.sample_id for sample in samples]
    assert first_ids == second_ids


def test_square_sample_expansion_uses_pancreas_special_yaw_policy():
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[20.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver", "pancreas"),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
            pancreas_special_x_limit=10.0,
        ),
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    stomach = [sample for sample in samples if sample.organ == "stomach"]
    assert len(stomach) == 1085
    assert {sample.yaw_policy for sample in stomach} == {"pancreas_special"}
    assert min(sample.yaw_degrees for sample in stomach) == -120.0
    assert max(sample.yaw_degrees for sample in stomach) == 30.0


def test_duodenum_pancreas_ray_hit_does_not_use_the_stomach_pancreas_special_yaw_policy():
    centerline_points = np.column_stack([np.zeros(21), np.zeros(21), np.arange(21, dtype=np.float64)])
    centerline = CenterlinePath(
        centerline_points,
        np.asarray([[0.0, 0.0, 1.0]] * len(centerline_points)),
        np.arange(len(centerline_points), dtype=np.float64),
    )
    surfaces = {
        "duodenum": SurfaceSamples(
            np.asarray([[2.0, 0.0, 10.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("duodenum_remainder",),
            target_ids=(("pancreas",),),
            centerline=centerline,
        )
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    assert len(samples) == 455
    assert {sample.yaw_policy for sample in samples} == {"standard"}


def test_exact_duplicate_pose_keeps_base_region_and_records_supplement_source():
    anchor = np.asarray([0.0, -1.0, 0.0])
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=anchor,
            pancreas_special_x_limit=10.0,
        ),
        "liver": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[-1.0, 0.0, 0.0]]),
            region_ids=("liver_supplement",),
            target_ids=((),),
            zero_plane_anchor_world=anchor,
        ),
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    assert len(samples) == 455
    assert {sample.organ for sample in samples} == {"stomach"}
    assert {sample.duplicate_source_regions for sample in samples} == {("liver_supplement",)}
    assert len({sample.sample_id for sample in samples}) == 455
    assert all("-r" in sample.sample_id and "-p" in sample.sample_id and "-y" in sample.sample_id for sample in samples)


def test_candidate_pose_count_is_derived_from_roll_pitch_and_yaw_arrays(monkeypatch):
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
            pancreas_special_x_limit=10.0,
        )
    }
    monkeypatch.setattr(sampling_pipeline_module, "ROLL_ANGLES_DEGREES", (-10.0, 0.0, 10.0))
    monkeypatch.setattr(sampling_pipeline_module, "PITCH_ANGLES_DEGREES", (-5.0, 5.0))
    assert _candidate_pose_count(surfaces) == 3 * 2 * 13


def test_pose_deduplication_does_not_merge_through_a_tolerance_chain():
    samples = [
        SquareSample(
            sample_id=f"stomach-{index}",
            organ="stomach",
            probe_point_world=np.zeros(3),
            input_normal_world=np.asarray([1.0, 0.0, 0.0]),
            vertices=np.full((4, 3), offset),
            source_region=f"source-{index}",
        )
        for index, offset in enumerate((0.0, 0.75e-9, 1.5e-9))
    ]

    retained = _deduplicate_exact_poses(samples)

    assert [sample.sample_id for sample in retained] == ["stomach-0", "stomach-2"]
    assert retained[0].duplicate_source_regions == ("source-1",)


def test_degenerate_ordinary_frame_candidate_is_removed_before_fps():
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    indices = _valid_ordinary_indices(points, normals, np.zeros(3), reverse_normal=False)

    assert np.array_equal(indices, [1])


def test_sampling_region_without_legal_candidates_fails_instead_of_emitting_an_empty_region():
    with pytest.raises(ValueError, match="stomach.*合法候选"):
        _sample(
            np.empty((0, 3)),
            np.empty((0, 3)),
            count=1,
            seed=0,
            minimum_spacing_mm=10.0,
            region_id="stomach",
        )


def test_liver_region_provenance_survives_candidate_merge_and_fps():
    normal = np.asarray([[0.0, 0.0, -1.0]])
    region_one = (np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]), np.vstack([normal, normal]))
    region_two = (np.asarray([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]]), np.vstack([normal, normal]))

    points, normals, region_ids = _merge_unique(
        (*region_one, "liver_region_one"),
        (*region_two, "liver_region_two"),
    )
    sampled = _sample(
        points,
        normals,
        count=3,
        seed=0,
        minimum_spacing_mm=10.0,
        region_id="liver",
        region_ids=region_ids,
    )

    assert region_ids == (
        "liver_region_one",
        "liver_region_one+liver_region_two",
        "liver_region_two",
    )
    assert set(sampled.region_ids) == set(region_ids)


def test_full_organ_mesh_directory_runs_all_five_source_sampling_rules(monkeypatch, tmp_path):
    def export(name, center, scale=(1.0, 1.0, 1.0)):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        mesh.apply_scale(scale)
        mesh.apply_translation(center)
        path = tmp_path / f"{name}.obj"
        mesh.export(path)
        return path

    paths = {
        "stomach": export("stomach", (0.0, 0.0, 0.0)),
        "liver": export("liver", (50.0, 50.0, 0.0), (50.0, 50.0, 10.0)),
        "pancreas": export("pancreas", (0.0, 0.0, 0.0)),
        "duodenum": export("duodenum", (10.0, 0.0, 0.0)),
        "esophagus": export("esophagus", (0.0, 0.0, 0.0)),
        "gallbladder": export("gallbladder", (100.0, 0.0, 0.0)),
        "aorta": export("aorta", (-20.0, 0.0, 0.0)),
        "adrenal_gland_right": export("adrenal_gland_right", (5.0, 0.0, 1.0)),
        "adrenal_gland_left": export("adrenal_gland_left", (5.0, 0.0, 0.0)),
        "inferior_vena_cava": export("inferior_vena_cava", (5.0, 0.0, 0.0)),
        "kidney_left": export("kidney_left", (5.0, 0.0, 0.0)),
        "kidney_right": export("kidney_right", (5.0, 0.0, 0.0)),
        "portal_vein_and_splenic_vein": export("portal_vein_and_splenic_vein", (5.0, 0.0, 0.0)),
        "spleen": export("spleen", (5.0, 0.0, 0.0)),
    }
    settings = SamplingConfig(
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
    )
    centerline_points = np.column_stack([np.full(31, 10.0), np.zeros(31), np.arange(-15.0, 16.0)])
    centerline = CenterlinePath(
        centerline_points,
        np.asarray([[0.0, 0.0, 1.0]] * len(centerline_points)),
        np.arange(len(centerline_points), dtype=np.float64),
    )
    captured = {}

    def record_centerline(*_args, **kwargs):
        captured.update(kwargs)
        return centerline

    monkeypatch.setattr("ct_vascular_resampling.sampling_pipeline.extract_duodenum_centerline", record_centerline)

    samples = sample_organs(paths, settings, seed=0, input_coordinate_system="RAS")

    assert set(samples) == {"stomach", "liver", "pancreas", "duodenum", "esophagus"}
    assert all(len(value.points) == len(value.normals) and len(value.points) > 0 for value in samples.values())
    assert samples["stomach"].sampling_statistics["stomach"].minimum_spacing_mm == 10.0
    assert set(samples["duodenum"].sampling_statistics) == {"duodenum_bulb", "duodenum_remainder"}
    assert captured["endpoint_hints_ras_mm"] == settings.duodenum_centerline_endpoint_hints_ras_mm
    assert captured["endpoint_match_tolerance_mm"] == 1.0
    for organ in ("stomach", "duodenum", "esophagus"):
        assert all(target_ids for target_ids in samples[organ].target_ids)
