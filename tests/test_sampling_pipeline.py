from __future__ import annotations

import numpy as np
import pytest
import trimesh

import ct_vascular_resampling.sampling_pipeline as sampling_pipeline_module
from ct_vascular_resampling.centerline import CenterlinePath
from ct_vascular_resampling.config import SamplingConfig, SquareConfig
from ct_vascular_resampling.sampling_pipeline import (
    build_sampling_point_plan,
    SquareSample,
    SurfaceSamples,
    _candidate_pose_count,
    _assert_minimum_spacing,
    _assert_samples_on_allowed_surface,
    _deduplicate_exact_poses,
    _merge_unique,
    _sample,
    _valid_ordinary_indices,
    generate_square_samples,
    sample_organs,
)


def test_surface_vertex_postcondition_rejects_a_point_not_on_the_allowed_outer_surface():
    allowed = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    allowed_normals = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    _assert_samples_on_allowed_surface(
        np.asarray([[10.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0]]),
        allowed,
        allowed_normals,
        "liver",
    )
    with pytest.raises(ValueError, match="liver.*主外表面"):
        _assert_samples_on_allowed_surface(
            np.asarray([[9.999, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            allowed,
            allowed_normals,
            "liver",
        )


def test_surface_postcondition_rejects_an_inward_normal_at_an_allowed_vertex():
    with pytest.raises(ValueError, match="liver.*法线"):
        _assert_samples_on_allowed_surface(
            np.asarray([[10.0, 0.0, 0.0]]),
            np.asarray([[-1.0, 0.0, 0.0]]),
            np.asarray([[10.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            "liver",
        )


def test_organ_postcondition_rejects_cross_region_spacing_violation():
    _assert_minimum_spacing(
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        10.0,
        "duodenum",
    )
    with pytest.raises(ValueError, match="duodenum.*10"):
        _assert_minimum_spacing(
            np.asarray([[0.0, 0.0, 0.0], [9.9, 0.0, 0.0]]),
            10.0,
            "duodenum",
        )


def test_sample_rejects_when_fixed_points_exclude_every_candidate():
    with pytest.raises(ValueError, match="duodenum_remainder.*合法候选"):
        _sample(
            np.asarray([[0.2, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            count=1,
            seed=0,
            minimum_spacing_mm=10.0,
            region_id="duodenum_remainder",
            fixed_points=np.asarray([[0.0, 0.0, 0.0]]),
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
    assert len(stomach) == 3211
    assert len(esophagus) == 3211
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
    assert len(samples) == 3211
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
    assert len(stomach) == 7657
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

    assert len(samples) == 3211
    assert {sample.yaw_policy for sample in samples} == {"standard"}


@pytest.mark.parametrize("region_id", ["liver_region_two", "liver_region_one+liver_region_two"])
def test_liver_region_two_membership_uses_formal_sixty_degree_yaw(region_id):
    surfaces = {
        "liver": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[-1.0, 0.0, 0.0]]),
            region_ids=(region_id,),
            target_ids=((),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
        )
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    assert len(samples) == 6175
    assert {sample.yaw_policy for sample in samples} == {"liver_region_two"}
    assert min(sample.yaw_degrees for sample in samples) == -60.0
    assert max(sample.yaw_degrees for sample in samples) == 60.0


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
            region_ids=("liver_region_one",),
            target_ids=((),),
            zero_plane_anchor_world=anchor,
        ),
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    assert len(samples) == 3211
    assert {sample.organ for sample in samples} == {"stomach"}
    assert {sample.duplicate_source_regions for sample in samples} == {("liver_region_one",)}
    assert {
        sample.duplicate_source_pose_ids for sample in samples
    } == {
        (
            sample.sample_id.replace("stomach", "liver", 1),
        )
        for sample in samples
    }
    assert len({sample.sample_id for sample in samples}) == 3211
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


def test_pose_planning_does_not_retain_every_candidate_id(monkeypatch):
    class TrackedId:
        live = 0
        peak = 0

        def __init__(self, value):
            self.value = value
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def candidates(*_args):
        for index in range(100):
            yield SquareSample(
                sample_id=TrackedId(index),
                organ="stomach",
                probe_point_world=np.zeros(3),
                input_normal_world=np.asarray([1.0, 0.0, 0.0]),
                vertices=np.full((4, 3), float(index)),
                source_region="stomach",
            )

    monkeypatch.setattr(sampling_pipeline_module, "_candidate_pose_count", lambda *_: 100)
    monkeypatch.setattr(sampling_pipeline_module, "_iter_pose_candidates", candidates)

    generate_square_samples({}, SquareConfig())

    assert TrackedId.peak < 10


def test_sampling_point_plan_records_geometry_policy_and_contract_pose_count():
    surfaces = {
        "liver": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[-1.0, 0.0, 0.0]]),
            region_ids=("liver_region_two",),
            target_ids=((),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
        )
    }

    plan = build_sampling_point_plan(surfaces)

    assert plan == {
        "schema_version": "sampling-point-plan/v1",
        "organs": {
            "stomach": [],
            "liver": [
                {
                    "point_index": 0,
                    "probe_point_world": [0.0, 0.0, 0.0],
                    "input_normal_world": [-1.0, 0.0, 0.0],
                    "source_region": "liver_region_two",
                    "yaw_policy": "liver_region_two",
                    "target_ids": [],
                    "base_local_axes_world": {
                        "x": [1.0, 0.0, 0.0],
                        "y": [0.0, 1.0, 0.0],
                        "z": [0.0, 0.0, 1.0],
                    },
                    "candidate_pose_count": 6175,
                }
            ],
            "pancreas": [],
            "duodenum": [],
            "esophagus": [],
        },
    }


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
    assert retained[0].duplicate_source_pose_ids == ("stomach-1",)


def test_pose_deduplication_records_duplicate_sources_in_protocol_order(monkeypatch):
    class ReversedPairTree:
        def __init__(self, _values):
            pass

        def query_pairs(self, **_kwargs):
            return np.asarray([[0, 2], [0, 1]], dtype=np.int64)

    monkeypatch.setattr(sampling_pipeline_module, "cKDTree", ReversedPairTree)
    samples = [
        SquareSample(
            sample_id=f"stomach-{index}",
            organ="stomach",
            probe_point_world=np.zeros(3),
            input_normal_world=np.asarray([1.0, 0.0, 0.0]),
            vertices=np.zeros((4, 3)),
            source_region=f"source-{index}",
        )
        for index in range(3)
    ]

    retained = _deduplicate_exact_poses(samples)

    assert retained[0].duplicate_source_pose_ids == ("stomach-1", "stomach-2")
    assert retained[0].duplicate_source_regions == ("source-1", "source-2")


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
        minimum_spacing_mm=0.1,
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
    ray_origins: list[np.ndarray] = []
    mesh_load_modes: dict[str, list[bool]] = {}

    def record_centerline(*_args, **kwargs):
        captured.update(kwargs)
        return centerline

    original_load = sampling_pipeline_module.load_surface_mesh

    def record_mesh_load(path, **kwargs):
        mesh_load_modes.setdefault(path.stem, []).append(
            bool(kwargs.get("main_outer_surface_only", False))
        )
        return original_load(path, **kwargs)

    monkeypatch.setattr("ct_vascular_resampling.sampling_pipeline.extract_duodenum_centerline", record_centerline)
    monkeypatch.setattr(sampling_pipeline_module, "load_surface_mesh", record_mesh_load)
    original_filter = sampling_pipeline_module.filter_points_by_target_rays

    def record_ray_origins(points, *args, **kwargs):
        ray_origins.append(np.asarray(points, dtype=np.float64).copy())
        return original_filter(points, *args, **kwargs)

    monkeypatch.setattr(sampling_pipeline_module, "filter_points_by_target_rays", record_ray_origins)

    samples = sample_organs(paths, settings, seed=0, input_coordinate_system="RAS")

    assert set(samples) == {"stomach", "liver", "pancreas", "duodenum", "esophagus"}
    assert all(len(value.points) == len(value.normals) and len(value.points) > 0 for value in samples.values())
    assert samples["stomach"].sampling_statistics["stomach"].minimum_spacing_mm == 0.1
    assert set(samples["duodenum"].sampling_statistics) == {"duodenum_bulb", "duodenum_remainder"}
    assert captured["endpoint_hints_ras_mm"] == settings.duodenum_centerline_endpoint_hints_ras_mm
    assert captured["endpoint_match_tolerance_mm"] == 1.0
    assert len(ray_origins) == 4
    for organ in sampling_pipeline_module.ORGAN_ORDER:
        assert mesh_load_modes[organ] == [False, True]
    assert mesh_load_modes["spleen"] == [False]
    assert np.array_equal(ray_origins[-2][:, :2], ray_origins[-1][:, :2])
    assert np.all(ray_origins[-2][:, 2] > ray_origins[-1][:, 2])
    for organ in ("stomach", "duodenum", "esophagus"):
        assert all(target_ids for target_ids in samples[organ].target_ids)
