from __future__ import annotations

import numpy as np
import trimesh

from ct_vascular_resampling.config import SamplingConfig, SquareConfig
from ct_vascular_resampling.sampling_pipeline import SurfaceSamples, generate_square_samples, sample_organs


def test_square_sample_expansion_uses_confirmed_axis_count_and_variant_count():
    surfaces = {
        "stomach": SurfaceSamples(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
        "esophagus": SurfaceSamples(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
    }

    samples = generate_square_samples(surfaces, SquareConfig())

    stomach = [sample for sample in samples if sample.organ == "stomach"]
    esophagus = [sample for sample in samples if sample.organ == "esophagus"]
    assert len(stomach) == 81
    assert len(esophagus) == 27
    assert stomach[0].sample_id == "stomach-000000-x-00"
    assert np.isclose(np.linalg.norm(stomach[0].vertices[1] - stomach[0].vertices[0]), 100.0)


def test_square_sample_expansion_keeps_non_degenerate_edge_angle_variants_when_deduplication_enabled():
    surfaces = {
        "stomach": SurfaceSamples(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
        "esophagus": SurfaceSamples(np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[0.0, 0.0, 1.0]])),
    }

    samples = generate_square_samples(surfaces, SquareConfig(deduplicate_degenerate_edge_angles=True))

    stomach = [sample for sample in samples if sample.organ == "stomach"]
    esophagus = [sample for sample in samples if sample.organ == "esophagus"]
    assert len(stomach) == 81
    assert len(esophagus) == 27
    expected_variants = [f"{index:02d}" for index in range(27)]
    assert [sample.sample_id.rsplit("-", 1)[1] for sample in stomach[:27]] == expected_variants
    assert [sample.sample_id.rsplit("-", 1)[1] for sample in esophagus] == expected_variants


def test_full_organ_mesh_directory_runs_all_five_source_sampling_rules(tmp_path):
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
        "adrenal_gland_right": export("adrenal_gland_right", (5.0, 0.0, 0.0)),
        "adrenal_gland_left": export("adrenal_gland_left", (5.0, 0.0, 0.0)),
        "inferior_vena_cava": export("inferior_vena_cava", (5.0, 0.0, 0.0)),
        "kidney_left": export("kidney_left", (5.0, 0.0, 0.0)),
        "kidney_right": export("kidney_right", (5.0, 0.0, 0.0)),
        "portal_vein_and_splenic_vein": export("portal_vein_and_splenic_vein", (5.0, 0.0, 0.0)),
        "spleen": export("spleen", (5.0, 0.0, 0.0)),
    }
    settings = SamplingConfig(point_counts={"stomach": 1, "liver": 1, "pancreas": 1, "duodenum_part1": 1, "duodenum_part2": 1, "esophagus": 1})

    samples = sample_organs(paths, settings, seed=0, input_coordinate_system="RAS")

    assert set(samples) == {"stomach", "liver", "pancreas", "duodenum", "esophagus"}
    assert all(len(value.points) == len(value.normals) and len(value.points) > 0 for value in samples.values())
