from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import numpy as np
import pytest
import trimesh

from ct_vascular_resampling.contract import (
    BASE_CORE_DESIGN_FILENAME,
    BASE_CORE_DESIGN_SHA256,
    CORE_DESIGN_FILENAME,
    CORE_DESIGN_SHA256,
    LIVER_REGION_TWO_YAW_ANGLES_DEGREES,
    PITCH_ANGLES_DEGREES,
    ROLL_ANGLES_DEGREES,
    SPECIAL_YAW_ANGLES_DEGREES,
    STANDARD_YAW_ANGLES_DEGREES,
)
from ct_vascular_resampling.artifacts import write_surface_samples_ply
from ct_vascular_resampling.centerline import extract_duodenum_centerline
from ct_vascular_resampling.config import SquareConfig
from ct_vascular_resampling.output_audit import (
    _audit_sampling_point_ply,
    _pose_plan_summary,
    _validate_point_plan_against_input_surfaces,
    audit_output,
)
from ct_vascular_resampling.protocol import resume_protocol_sha256
from ct_vascular_resampling.mesh_io import load_surface_mesh
from ct_vascular_resampling.sampling import extreme_plateau_centroid
from ct_vascular_resampling.sampling_pipeline import (
    SurfaceSamples,
    base_local_frame,
    build_sampling_point_plan,
    generate_square_samples,
)
from ct_vascular_resampling.squares import LocalFrame, generate_pose_variant


CORE_HASH = "d" * 64
BUILD_COMMIT = "a" * 40


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _set_resume_protocol_hash(metadata: dict) -> None:
    metadata["resume_protocol_sha256"] = resume_protocol_sha256(metadata)


def _rewrite_manifest_record(case: Path, index: int, update) -> list[dict]:
    manifest_path = case / "manifest.jsonl"
    records = _read_jsonl(manifest_path)
    update(records[index])
    _write_jsonl(manifest_path, records)
    first = records[0]
    _write_jsonl(case / "gallery" / "gallery.jsonl", [first])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    return records


def _set_liver_policy_mismatch(record: dict) -> None:
    record["yaw_policy"] = "standard"
    record["angles_degrees"]["yaw"] = 0.0


def _valid_output(tmp_path: Path) -> Path:
    case = tmp_path / "case_2"
    gallery = case / "gallery"
    image = Image.new("RGB", (3, 3), "white")
    vessel = image.copy()
    vessel.putpixel((1, 1), (255, 0, 0))
    for directory, rendered in {
        "ct": image,
        "boundary_only": image,
        "ct_overlay": image,
        "organ_vessel_boundary": image,
        "eus_vessel_boundary": vessel,
        "ct_eus_vessel_overlay": vessel,
    }.items():
        destination = gallery / directory / "sample.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination)

    record = {
        "slice_id": "stomach-000000-rp000-pp000-yp000",
        "status": "gallery",
        "organ": "stomach",
        "coordinate_system": "RAS",
        "core_design_sha256": CORE_HASH,
        "build_git_commit": BUILD_COMMIT,
        "source_region": "stomach",
        "yaw_policy": "standard",
        "angles_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "ct_png": "ct/sample.png",
        "boundary_only_png": "boundary_only/sample.png",
        "ct_overlay_png": "ct_overlay/sample.png",
        "organ_vessel_boundary_png": "organ_vessel_boundary/sample.png",
        "organ_metadata_schema_version": "eus-organ-metadata/v1",
        "eus_vessel_boundary_png": "eus_vessel_boundary/sample.png",
        "ct_eus_vessel_overlay_png": "ct_eus_vessel_overlay/sample.png",
        "eus_vessel_metadata_schema_version": "eus-vessel-metadata/v1",
        "organ_labels": ["aorta"],
        "eus_candidate_organ_labels": ["aorta"],
        "eus_vessel_labels": ["aorta"],
        "eus_vessel_features": [
            {"label": "aorta", "x_mm": 1.0, "y_mm": 1.0, "area_mm2": 1.0}
        ],
        "features": [],
    }
    _write_jsonl(case / "manifest.jsonl", [record])
    _write_jsonl(gallery / "gallery.jsonl", [record])
    _write_json(
        case / "run_metadata.json",
        {
            "run_state": "complete",
            "total_squares": 1,
            "completed_pose_count": 1,
            "status_counts": {"gallery": 1},
            "core_design_filename": "基于目标器官的采样方法-20260813.docx",
            "core_design_sha256": CORE_HASH,
            "build_git_commit": BUILD_COMMIT,
            "pose_angles_degrees": {
                "roll": [0.0],
                "pitch": [0.0],
                "yaw": {"standard": [0.0], "duodenum_bulb": [0.0], "pancreas_special": [0.0]},
            },
            "quality_filtering": {"black_ratio_limit": 0.6},
            "manual_segmentation": {
                "eus_vessel_colors": {
                    "aorta": [255, 0, 0],
                    "inferior_vena_cava": [0, 0, 255],
                    "portal_vein": [170, 85, 255],
                }
            },
        },
    )
    _write_json(
        case / "library_summary.json",
        {
            "indexed_feature_count": 1,
            "organ_label_counts": {"aorta": 1},
            "eus_candidate_organ_label_counts": {"aorta": 1},
            "eus_vessel_label_counts": {"aorta": 1},
            "eus_vessel_feature_counts": {"aorta": 1},
            "eus_vessel_colors": {
                "aorta": [255, 0, 0],
                "inferior_vena_cava": [0, 0, 255],
                "portal_vein": [170, 85, 255],
            },
        },
    )
    return case


def _make_current_design_output(
    case: Path,
    *,
    manual_centerline: bool = False,
    esophagus_extra_component: bool = False,
) -> None:
    gallery_record = json.loads(
        (case / "gallery" / "gallery.jsonl").read_text(encoding="utf-8")
    )
    input_directory = case / "input_models"
    input_directory.mkdir(parents=True, exist_ok=True)
    centers = {
        "stomach": (-10.0, -10.0, -10.0),
        "liver": (10.0, 10.0, 10.0),
        "pancreas": (40.0, 0.0, 0.0),
        "duodenum": (0.0, 60.0, 20.0),
        "esophagus": (80.0, 0.0, 0.0),
    }
    full_meshes = {}
    source_meshes = {}
    input_provenance = {}
    for organ, center in centers.items():
        extents = (20.0, 20.0, 60.0) if organ == "duodenum" else (20.0, 20.0, 20.0)
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        if organ == "esophagus" and esophagus_extra_component:
            extra = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
            extra.apply_translation((50.0, 30.0, -40.0))
            mesh = trimesh.util.concatenate([mesh, extra])
        path = input_directory / f"{organ}.ply"
        mesh.export(path)
        full_meshes[organ] = load_surface_mesh(path)
        source_meshes[organ] = load_surface_mesh(path, main_outer_surface_only=True)
        input_provenance[organ] = {
            "path": str(path.resolve()),
            "kind": "file",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def surface_vertex(organ: str, expected: tuple[float, float, float]):
        source = source_meshes[organ]
        index = int(np.argmin(np.linalg.norm(source.vertices - expected, axis=1)))
        assert np.allclose(source.vertices[index], expected, rtol=0.0, atol=1e-12)
        return source.vertices[index].copy(), source.vertex_normals[index].copy()

    stomach_point, stomach_normal = surface_vertex("stomach", (0.0, 0.0, 0.0))
    liver_one_point, liver_one_normal = surface_vertex("liver", (0.0, 0.0, 0.0))
    liver_two_point, liver_two_normal = surface_vertex("liver", (20.0, 20.0, 20.0))
    pancreas_point, pancreas_normal = surface_vertex("pancreas", (30.0, -10.0, -10.0))
    duodenum_bulb_point, duodenum_bulb_normal = surface_vertex(
        "duodenum", (-10.0, 50.0, -10.0)
    )
    duodenum_remainder_point, duodenum_remainder_normal = surface_vertex(
        "duodenum", (10.0, 70.0, 50.0)
    )
    esophagus_point, esophagus_normal = surface_vertex("esophagus", (90.0, 10.0, 10.0))

    anchor = extreme_plateau_centroid(
        full_meshes["esophagus"].vertices,
        axis=2,
        maximum=False,
    )
    endpoint_hints = (
        ((0.0, 60.0, 0.0), (0.0, 60.0, 40.0))
        if manual_centerline
        else None
    )
    centerline = extract_duodenum_centerline(
        full_meshes["duodenum"].mesh,
        full_meshes["stomach"].vertices,
        voxel_pitch_mm=1.0,
        tangent_window_mm=10.0,
        max_terminal_spur_mm=5.0,
        endpoint_hints_ras_mm=endpoint_hints,
        endpoint_match_tolerance_mm=1.0,
    )
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([stomach_point]),
            np.asarray([stomach_normal]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=anchor,
            pancreas_special_x_limit=10.0,
        ),
        "liver": SurfaceSamples(
            np.asarray([liver_one_point, liver_two_point]),
            np.asarray([liver_one_normal, liver_two_normal]),
            region_ids=("liver_region_one", "liver_region_two"),
            target_ids=((), ()),
            zero_plane_anchor_world=anchor,
        ),
        "pancreas": SurfaceSamples(
            np.asarray([pancreas_point]),
            np.asarray([pancreas_normal]),
            region_ids=("pancreas",),
            target_ids=((),),
            zero_plane_anchor_world=anchor,
            pancreas_special_x_limit=100.0,
        ),
        "duodenum": SurfaceSamples(
            np.asarray([duodenum_bulb_point, duodenum_remainder_point]),
            np.asarray([duodenum_bulb_normal, duodenum_remainder_normal]),
            region_ids=("duodenum_bulb", "duodenum_remainder"),
            target_ids=((), ()),
            centerline=centerline,
        ),
        "esophagus": SurfaceSamples(
            np.asarray([esophagus_point]),
            np.asarray([esophagus_normal]),
            region_ids=("esophagus",),
            target_ids=((),),
            zero_plane_anchor_world=anchor,
        ),
    }
    samples = generate_square_samples(surfaces, SquareConfig())
    records = []
    for index, sample in enumerate(samples):
        record = dict(gallery_record)
        record.update(
            {
                "slice_id": sample.sample_id,
                "status": "gallery" if index == 0 else "excluded_fov",
                "organ": sample.organ,
                "core_design_sha256": CORE_DESIGN_SHA256,
                "source_region": sample.source_region,
                "yaw_policy": sample.yaw_policy,
                "probe_point_world": [float(value) for value in sample.probe_point_world],
                "input_normal_world": [float(value) for value in sample.input_normal_world],
                "square_vertices_world": [
                    [float(value) for value in vertex] for vertex in sample.vertices
                ],
                "angles_degrees": {
                    "roll": float(sample.roll_degrees),
                    "pitch": float(sample.pitch_degrees),
                    "yaw": float(sample.yaw_degrees),
                },
                "local_axes_world": {
                    "x": [float(value) for value in sample.local_x_world],
                    "y": [float(value) for value in sample.local_y_world],
                    "z": [float(value) for value in sample.local_z_world],
                },
                "target_ids": list(sample.target_ids),
                "duplicate_source_regions": list(sample.duplicate_source_regions),
                "duplicate_source_pose_ids": list(sample.duplicate_source_pose_ids),
            }
        )
        records.append(record)
    _write_jsonl(case / "manifest.jsonl", records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])

    requested_regions = {
        "stomach": {"stomach": 1},
        "liver": {"liver": 2},
        "pancreas": {"pancreas": 1},
        "duodenum": {"duodenum_bulb": 1, "duodenum_remainder": 1},
        "esophagus": {"esophagus": 1},
    }
    organs = {}
    for organ, surface in surfaces.items():
        organs[organ] = {
            "source_surface": source_meshes[organ].surface_audit.to_record(),
            "regions": {
                region: {
                    "requested_count": count,
                    "candidate_count": count,
                    "actual_count": count,
                    "shortfall_count": 0,
                    "minimum_spacing_mm": 10.0,
                    "actual_minimum_distance_mm": None,
                }
                for region, count in requested_regions[organ].items()
            },
            "requested_count": sum(requested_regions[organ].values()),
            "actual_count": len(surface.points),
            "shortfall_count": 0,
        }
        write_surface_samples_ply(
            case / "ResampledpointPLY" / f"FPS-{organ.capitalize()}.ply",
            surface,
        )

    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "core_design_filename": CORE_DESIGN_FILENAME,
            "core_design_sha256": CORE_DESIGN_SHA256,
            "coordinate_system": "RAS",
            "input_coordinate_system": "RAS",
            "input_provenance": {"organ_models": input_provenance},
            "base_core_design_filename": BASE_CORE_DESIGN_FILENAME,
            "base_core_design_sha256": BASE_CORE_DESIGN_SHA256,
            "sampling_configuration": {
                "point_counts": {
                    "stomach": 1,
                    "liver": 2,
                    "pancreas": 1,
                    "duodenum_part1": 1,
                    "duodenum_part2": 1,
                    "esophagus": 1,
                },
                "count_policy": "upper_bound_preserve_outer_surface_and_minimum_spacing",
                "source_surface_policy": "largest_watertight_absolute_volume",
                "esophagus_extension_target_filter": (
                    "original_and_translated_segments_independently"
                ),
                "ray_length_mm": 100.0,
                "ray_batch_size": 2048,
                "minimum_spacing_mm": 10.0,
                "centerline_voxel_pitch_mm": 1.0,
                "centerline_tangent_window_mm": 10.0,
                "centerline_max_terminal_spur_mm": 5.0,
                "duodenum_centerline_endpoint_hints_ras_mm": (
                    {
                        "proximal": list(endpoint_hints[0]),
                        "distal": list(endpoint_hints[1]),
                    }
                    if endpoint_hints is not None
                    else None
                ),
                "duodenum_centerline_endpoint_match_tolerance_mm": (
                    1.0 if endpoint_hints is not None else None
                ),
                "seed": 0,
            },
            "duodenum_centerline_selection": (
                centerline.selection_audit.to_record()
                if centerline.selection_audit is not None
                else None
            ),
            "minimum_point_spacing_mm": 10.0,
            "square_sampling": {
                "side_length_mm": 100.0,
                "output_resolution": [300, 300],
                "interpolation": "cubic_bspline",
                "interpolation_order": 3,
                "window_level_hu": 40.0,
                "window_width_hu": 400.0,
                "fill_hu_value": -1000.0,
            },
            "quality_filtering": {
                "black_threshold": 50,
                "black_ratio_limit": 0.6,
                "line_min_diagonal_fraction": 0.7,
                "black_side_min_ratio": 0.9,
                "valid_side_max_black_ratio": 0.1,
            },
            "pose_convention": {
                "coordinate_frame": "local_right_handed",
                "matrix_order": "B @ Rz(yaw) @ Ry(pitch) @ Rx(roll)",
                "positive_yaw": "counterclockwise",
                "yaw_observer": "local_positive_z_looking_toward_probe",
                "rotation_center": "probe_at_square_bottom_edge_midpoint",
            },
            "fov_policy": {
                "vertex_rule": "any_square_vertex_outside_ct",
                "outside_status": "excluded_fov",
                "saved_artifacts": ["ct_png"],
                "out_of_bounds_png_value": 0,
            },
            "eus_possible_organs": {},
            "manual_segmentation": {},
            "surface_sampling_audit": {
                "outer_surface_required": True,
                "minimum_spacing_preserved_on_shortfall": True,
                "organs": organs,
            },
            "pose_angles_degrees": {
                "roll": list(ROLL_ANGLES_DEGREES),
                "pitch": list(PITCH_ANGLES_DEGREES),
                "yaw": {
                    "standard": list(STANDARD_YAW_ANGLES_DEGREES),
                    "duodenum_bulb": list(SPECIAL_YAW_ANGLES_DEGREES),
                    "pancreas_special": list(SPECIAL_YAW_ANGLES_DEGREES),
                    "liver_region_two": list(LIVER_REGION_TWO_YAW_ANGLES_DEGREES),
                },
            },
            "sampling_point_plan": build_sampling_point_plan(surfaces),
            "total_squares": len(records),
            "completed_pose_count": len(records),
            "status_counts": {"excluded_fov": len(records) - 1, "gallery": 1},
        }
    )
    metadata["pose_plan"] = _pose_plan_summary(
        [json.loads(line) for line in (case / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    )
    _set_resume_protocol_hash(metadata)
    _write_json(metadata_path, metadata)


def test_audit_output_accepts_consistent_structure_and_pixels(tmp_path):
    report = audit_output(
        _valid_output(tmp_path),
        check_pixels=True,
        expected_core_design_sha256=CORE_HASH,
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["manifest_lines"] == 1
    assert report["gallery_lines"] == 1
    assert report["pixel_audit"]["frames_with_color"] == {"aorta": 1}


def test_audit_output_accepts_complete_library_with_zero_gallery_records(tmp_path):
    case = _valid_output(tmp_path)
    record = _read_jsonl(case / "manifest.jsonl")[0]
    record["status"] = "unindexed"
    _write_jsonl(case / "manifest.jsonl", [record])
    _write_jsonl(case / "unindexed" / "unindexed.jsonl", [record])
    gallery_manifest = case / "gallery" / "gallery.jsonl"
    gallery_manifest.unlink()
    for image_path in (case / "gallery").rglob("*.png"):
        image_path.unlink()
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["status_counts"] = {"unindexed": 1}
    _write_json(case / "run_metadata.json", metadata)
    summary = json.loads((case / "library_summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "gallery_manifest_exists": False,
            "indexed_feature_count": 0,
            "organ_label_counts": {},
            "eus_candidate_organ_label_counts": {},
            "eus_vessel_label_counts": {},
            "eus_vessel_feature_counts": {},
        }
    )
    _write_json(case / "library_summary.json", summary)

    report = audit_output(case, expected_core_design_sha256=CORE_HASH)

    assert report["errors"] == []
    assert report["passed"] is True
    assert report["gallery_lines"] == 0


def test_audit_output_rejects_unknown_design_hash_by_default(tmp_path):
    report = audit_output(_valid_output(tmp_path))

    assert report["passed"] is False
    assert any("SHA-256" in error for error in report["errors"])


def test_audit_output_accepts_formal_liver_region_two_yaw_policy(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=None,
        expected_duodenum_centerline_endpoint_match_tolerance_mm=None,
    )

    assert report["passed"] is True
    assert report["yaw_policy_counts"]["liver_region_two"] == 6175
    assert report["yaw_policy_counts"]["duodenum_bulb"] == 7657
    records = _read_jsonl(case / "manifest.jsonl")
    assert sum(len(record["duplicate_source_pose_ids"]) for record in records) == 3211


def test_current_design_audit_accepts_reconstructed_manual_duodenum_centerline(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, manual_centerline=True)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=(
            (0.0, 60.0, 0.0),
            (0.0, 60.0, 40.0),
        ),
        expected_duodenum_centerline_endpoint_match_tolerance_mm=1.0,
    )

    assert report["passed"] is True
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["duodenum_centerline_selection"]["mode"] == "manual_endpoint_hints"


def test_current_design_audit_requires_external_manual_duodenum_endpoints(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, manual_centerline=True)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=None,
        expected_duodenum_centerline_endpoint_match_tolerance_mm=None,
    )

    assert report["passed"] is False
    assert any("外部" in error and "端点" in error for error in report["errors"])


def test_current_design_audit_rejects_manual_to_automatic_centerline_downgrade(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, manual_centerline=True)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sampling = metadata["sampling_configuration"]
    sampling["duodenum_centerline_endpoint_hints_ras_mm"] = None
    sampling["duodenum_centerline_endpoint_match_tolerance_mm"] = None
    metadata["duodenum_centerline_selection"] = None
    _set_resume_protocol_hash(metadata)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("外部病例配置" in error for error in report["errors"])


def test_current_design_audit_rejects_self_consistent_reversed_manual_centerline(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, manual_centerline=True)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sampling = metadata["sampling_configuration"]
    original_hints = (
        tuple(sampling["duodenum_centerline_endpoint_hints_ras_mm"]["proximal"]),
        tuple(sampling["duodenum_centerline_endpoint_hints_ras_mm"]["distal"]),
    )
    reversed_hints = (original_hints[1], original_hints[0])
    sampling["duodenum_centerline_endpoint_hints_ras_mm"] = {
        "proximal": list(reversed_hints[0]),
        "distal": list(reversed_hints[1]),
    }
    provenance = metadata["input_provenance"]["organ_models"]
    duodenum_mesh = load_surface_mesh(Path(provenance["duodenum"]["path"]))
    stomach_mesh = load_surface_mesh(Path(provenance["stomach"]["path"]))
    reversed_centerline = extract_duodenum_centerline(
        duodenum_mesh.mesh,
        stomach_mesh.vertices,
        voxel_pitch_mm=1.0,
        tangent_window_mm=10.0,
        max_terminal_spur_mm=5.0,
        endpoint_hints_ras_mm=reversed_hints,
        endpoint_match_tolerance_mm=1.0,
    )
    metadata["duodenum_centerline_selection"] = reversed_centerline.selection_audit.to_record()
    frames: dict[int, LocalFrame] = {}
    for record in metadata["sampling_point_plan"]["organs"]["duodenum"]:
        frame = base_local_frame(
            "duodenum",
            np.asarray(record["probe_point_world"], dtype=np.float64),
            np.asarray(record["input_normal_world"], dtype=np.float64),
            esophagus_anchor=None,
            centerline=reversed_centerline,
        )
        frames[record["point_index"]] = frame
        record["base_local_axes_world"] = {
            "x": frame.x_axis.tolist(),
            "y": frame.y_axis.tolist(),
            "z": frame.z_axis.tolist(),
        }

    records = _read_jsonl(case / "manifest.jsonl")
    changed = 0
    for record in records:
        if record["organ"] != "duodenum":
            continue
        point_index = int(record["slice_id"].split("-")[1])
        angles = record["angles_degrees"]
        variant = generate_pose_variant(
            np.asarray(record["probe_point_world"], dtype=np.float64),
            frames[point_index],
            100.0,
            float(angles["roll"]),
            float(angles["pitch"]),
            float(angles["yaw"]),
        )
        record["square_vertices_world"] = variant.vertices.tolist()
        record["local_axes_world"] = {
            "x": variant.local_frame.x_axis.tolist(),
            "y": variant.local_frame.y_axis.tolist(),
            "z": variant.local_frame.z_axis.tolist(),
        }
        changed += 1
    assert changed == 10868
    _write_jsonl(case / "manifest.jsonl", records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    metadata["pose_plan"] = _pose_plan_summary(records)
    _set_resume_protocol_hash(metadata)
    _write_json(metadata_path, metadata)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=original_hints,
        expected_duodenum_centerline_endpoint_match_tolerance_mm=1.0,
    )

    assert report["passed"] is False
    assert any("端点与外部病例配置不一致" in error for error in report["errors"])


def test_current_design_audit_rejects_invalid_resume_protocol_hash(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["resume_protocol_sha256"] = "0" * 64
    _write_json(metadata_path, metadata)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=None,
        expected_duodenum_centerline_endpoint_match_tolerance_mm=None,
    )

    assert report["passed"] is False
    assert any("resume_protocol_sha256" in error for error in report["errors"])


def test_current_design_audit_accepts_recovered_records_from_compatible_commit(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    old_commit = "b" * 40
    records = _read_jsonl(case / "manifest.jsonl")
    records[0]["build_git_commit"] = old_commit
    _write_jsonl(case / "manifest.jsonl", records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["compatible_completed_build_git_commits"] = [old_commit]
    metadata["recovery_history"] = [
        {
            "completed_build_git_commits": [old_commit],
            "recovery_build_git_commit": BUILD_COMMIT,
        }
    ]
    _write_json(metadata_path, metadata)

    report = audit_output(
        case,
        expected_duodenum_centerline_endpoint_hints_ras_mm=None,
        expected_duodenum_centerline_endpoint_match_tolerance_mm=None,
    )

    assert report["passed"] is True
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("pose_convention", "pose_convention"),
        ("fov_policy", "fov_policy"),
        ("esophagus_extension_target_filter", "esophagus_extension_target_filter"),
    ],
)
def test_current_design_audit_rejects_self_consistent_fixed_protocol_tampering(
    tmp_path,
    field,
    expected_error,
):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if field == "pose_convention":
        metadata[field]["matrix_order"] = "tampered"
    elif field == "fov_policy":
        metadata[field]["vertex_rule"] = "tampered"
    else:
        metadata["sampling_configuration"][field] = "tampered"
    _set_resume_protocol_hash(metadata)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any(expected_error in error for error in report["errors"])


def test_current_design_source_audit_rejects_main_shell_esophagus_anchor(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, esophagus_extra_component=True)
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    esophagus = metadata["sampling_point_plan"]["organs"]["esophagus"][0]
    esophagus_path = Path(metadata["input_provenance"]["organ_models"]["esophagus"]["path"])
    main_esophagus = load_surface_mesh(esophagus_path, main_outer_surface_only=True)
    wrong_anchor = extreme_plateau_centroid(
        main_esophagus.vertices,
        axis=2,
        maximum=False,
    )
    wrong_frame = base_local_frame(
        "esophagus",
        np.asarray(esophagus["probe_point_world"], dtype=np.float64),
        np.asarray(esophagus["input_normal_world"], dtype=np.float64),
        esophagus_anchor=wrong_anchor,
        centerline=None,
    )
    esophagus["base_local_axes_world"] = {
        "x": wrong_frame.x_axis.tolist(),
        "y": wrong_frame.y_axis.tolist(),
        "z": wrong_frame.z_axis.tolist(),
    }
    point_plan = {
        (organ, record["point_index"]): record
        for organ, records in metadata["sampling_point_plan"]["organs"].items()
        for record in records
    }

    errors = _validate_point_plan_against_input_surfaces(metadata, point_plan)

    assert any("base_local_axes_world" in error for error in errors)


def test_current_design_source_audit_rejects_changed_centerline_selection(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case, manual_centerline=True)
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["duodenum_centerline_selection"]["path_point_count"] += 1
    point_plan = {
        (organ, record["point_index"]): record
        for organ, records in metadata["sampling_point_plan"]["organs"].items()
        for record in records
    }

    errors = _validate_point_plan_against_input_surfaces(metadata, point_plan)

    assert any("duodenum_centerline_selection" in error for error in errors)


@pytest.mark.parametrize("mutation", ["translate", "reverse_normal"])
def test_current_design_source_audit_rejects_point_plan_off_real_outer_surface(
    tmp_path,
    mutation,
):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    esophagus = metadata["sampling_point_plan"]["organs"]["esophagus"][0]
    if mutation == "translate":
        esophagus["probe_point_world"][0] += 1000.0
    else:
        esophagus["input_normal_world"] = [
            -value for value in esophagus["input_normal_world"]
        ]
    point_plan = {
        (organ, record["point_index"]): record
        for organ, records in metadata["sampling_point_plan"]["organs"].items()
        for record in records
    }

    errors = _validate_point_plan_against_input_surfaces(metadata, point_plan)

    assert any("真实主外壳" in error for error in errors)


def test_current_design_audit_rejects_self_consistent_rotated_esophagus_zero_plane(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    point_plan = metadata["sampling_point_plan"]["organs"]["esophagus"][0]
    base_axes = point_plan["base_local_axes_world"]
    rotated_frame = LocalFrame(
        np.asarray(base_axes["x"], dtype=np.float64),
        np.asarray(base_axes["z"], dtype=np.float64),
        -np.asarray(base_axes["y"], dtype=np.float64),
    )
    point_plan["base_local_axes_world"] = {
        "x": rotated_frame.x_axis.tolist(),
        "y": rotated_frame.y_axis.tolist(),
        "z": rotated_frame.z_axis.tolist(),
    }

    records = _read_jsonl(case / "manifest.jsonl")
    changed = 0
    for record in records:
        if record["organ"] != "esophagus":
            continue
        angles = record["angles_degrees"]
        variant = generate_pose_variant(
            np.asarray(record["probe_point_world"], dtype=np.float64),
            rotated_frame,
            100.0,
            float(angles["roll"]),
            float(angles["pitch"]),
            float(angles["yaw"]),
        )
        record["square_vertices_world"] = variant.vertices.tolist()
        record["local_axes_world"] = {
            "x": variant.local_frame.x_axis.tolist(),
            "y": variant.local_frame.y_axis.tolist(),
            "z": variant.local_frame.z_axis.tolist(),
        }
        changed += 1
    assert changed == 3211
    _write_jsonl(case / "manifest.jsonl", records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    metadata["pose_plan"] = _pose_plan_summary(records)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("base_local_axes_world" in error for error in report["errors"])


def test_current_design_source_audit_rejects_rotated_duodenum_zero_plane(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    duodenum = metadata["sampling_point_plan"]["organs"]["duodenum"][0]
    base_axes = duodenum["base_local_axes_world"]
    duodenum["base_local_axes_world"] = {
        "x": base_axes["x"],
        "y": base_axes["z"],
        "z": [-value for value in base_axes["y"]],
    }
    point_plan = {
        (organ, record["point_index"]): record
        for organ, records in metadata["sampling_point_plan"]["organs"].items()
        for record in records
    }

    errors = _validate_point_plan_against_input_surfaces(metadata, point_plan)

    assert any("base_local_axes_world" in error for error in errors)


def test_current_design_source_audit_rejects_self_consistent_nearby_normal(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata = json.loads((case / "run_metadata.json").read_text(encoding="utf-8"))
    esophagus = metadata["sampling_point_plan"]["organs"]["esophagus"][0]
    normal = np.asarray(esophagus["input_normal_world"], dtype=np.float64)
    tangent = np.cross(normal, np.asarray([1.0, 0.0, 0.0]))
    tangent /= np.linalg.norm(tangent)
    changed_normal = normal + tangent * 1e-5
    changed_normal /= np.linalg.norm(changed_normal)
    esophagus_path = Path(metadata["input_provenance"]["organ_models"]["esophagus"]["path"])
    full_esophagus = load_surface_mesh(esophagus_path)
    anchor = extreme_plateau_centroid(
        full_esophagus.vertices,
        axis=2,
        maximum=False,
    )
    frame = base_local_frame(
        "esophagus",
        np.asarray(esophagus["probe_point_world"], dtype=np.float64),
        changed_normal,
        esophagus_anchor=anchor,
        centerline=None,
    )
    esophagus["input_normal_world"] = changed_normal.tolist()
    esophagus["base_local_axes_world"] = {
        "x": frame.x_axis.tolist(),
        "y": frame.y_axis.tolist(),
        "z": frame.z_axis.tolist(),
    }
    point_plan = {
        (organ, record["point_index"]): record
        for organ, records in metadata["sampling_point_plan"]["organs"].items()
        for record in records
    }

    errors = _validate_point_plan_against_input_surfaces(metadata, point_plan)

    assert any("input_normal_world" in error for error in errors)


def test_current_design_sampling_ply_audit_rejects_float_properties(tmp_path):
    path = tmp_path / "case" / "ResampledpointPLY" / "FPS-Liver.ply"
    write_surface_samples_ply(
        path,
        SurfaceSamples(
            np.asarray([[1.0, 2.0, 700.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
        ),
    )
    path.write_text(
        path.read_text(encoding="utf-8").replace("property double", "property float"),
        encoding="utf-8",
    )

    audit, _ = _audit_sampling_point_ply(tmp_path / "case", minimum_spacing_mm=10.0)

    assert any("double" in error for error in audit["errors"])


def test_current_design_audit_rejects_reversed_p030_source_priority(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    liver_point = metadata["sampling_point_plan"]["organs"]["liver"][0]
    records = _read_jsonl(case / "manifest.jsonl")
    reversed_count = 0
    for record in records:
        if record["organ"] != "stomach" or not record["duplicate_source_pose_ids"]:
            continue
        stomach_id = record["slice_id"]
        record.update(
            {
                "slice_id": record["duplicate_source_pose_ids"][0],
                "organ": "liver",
                "probe_point_world": liver_point["probe_point_world"],
                "input_normal_world": liver_point["input_normal_world"],
                "source_region": liver_point["source_region"],
                "yaw_policy": liver_point["yaw_policy"],
                "target_ids": liver_point["target_ids"],
                "duplicate_source_regions": ["stomach"],
                "duplicate_source_pose_ids": [stomach_id],
            }
        )
        reversed_count += 1
    assert reversed_count == 3211
    _write_jsonl(case / "manifest.jsonl", records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    metadata["pose_plan"] = _pose_plan_summary(records)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("duplicate_source_priority" in error for error in report["errors"])


def test_current_design_audit_rejects_self_declared_truncated_liver_yaw_set(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pose_angles_degrees"]["yaw"]["liver_region_two"] = [60.0]
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("pose_angles_degrees" in error for error in report["errors"])


def test_current_design_audit_rejects_liver_region_policy_mismatch(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    source_records = _read_jsonl(case / "manifest.jsonl")
    liver_index = next(
        index
        for index, record in enumerate(source_records)
        if record["source_region"] == "liver_region_two"
    )
    records = _rewrite_manifest_record(
        case,
        liver_index,
        _set_liver_policy_mismatch,
    )
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pose_plan"] = _pose_plan_summary(records)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("source_region_yaw_policy" in error for error in report["errors"])


def test_current_design_audit_rejects_surface_point_count_mismatch(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["surface_sampling_audit"]["organs"]["liver"]["actual_count"] = 3
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("surface_sampling_audit" in error for error in report["errors"])


def test_current_design_audit_rejects_zero_required_point_request(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sampling_configuration"]["point_counts"]["stomach"] = 0
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("point_counts" in error and "正整数" in error for error in report["errors"])


def test_current_design_audit_rejects_zero_required_region_candidates(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["surface_sampling_audit"]["organs"]["pancreas"]["regions"]["pancreas"][
        "candidate_count"
    ] = 0
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("pancreas" in error and "计数不一致" in error for error in report["errors"])


def test_current_design_audit_rejects_duplicate_region_evidence(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)

    def duplicate_region(record: dict) -> None:
        record["duplicate_source_regions"] = ["liver_region_one", "liver_region_one"]

    _rewrite_manifest_record(case, 0, duplicate_region)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("duplicate_source_regions" in error for error in report["errors"])


def test_current_design_audit_rejects_manifest_pose_plan_mismatch(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    manifest_path = case / "manifest.jsonl"
    records = _read_jsonl(manifest_path)
    records[0]["angles_degrees"]["yaw"] = 55.0
    _write_jsonl(manifest_path, records)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("pose_plan" in error for error in report["errors"])


def test_current_design_audit_rejects_sampling_ply_below_ten_mm(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    write_surface_samples_ply(
        case / "ResampledpointPLY" / "FPS-Liver.ply",
        SurfaceSamples(
            np.asarray([[1.0, 2.0, 3.0], [6.0, 2.0, 3.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
    )
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    liver = metadata["surface_sampling_audit"]["organs"]["liver"]
    liver.update({"requested_count": 2, "actual_count": 2, "shortfall_count": 0})
    liver["regions"]["liver"].update(
        {
            "requested_count": 2,
            "candidate_count": 2,
            "actual_count": 2,
            "shortfall_count": 0,
        }
    )
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("最小间距" in error for error in report["errors"])


def test_current_design_audit_rejects_sampling_ply_not_matching_point_plan(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    write_surface_samples_ply(
        case / "ResampledpointPLY" / "FPS-Liver.ply",
        SurfaceSamples(
            np.asarray([[999.0, 999.0, 999.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
        ),
    )

    report = audit_output(case)

    assert report["passed"] is False
    assert any("sampling_point_plan" in error or "采样点计划" in error for error in report["errors"])


def test_current_design_audit_rejects_incomplete_point_angle_grid_even_if_summary_matches(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    manifest_path = case / "manifest.jsonl"
    record = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    _write_jsonl(manifest_path, [record])
    _write_jsonl(case / "gallery" / "gallery.jsonl", [record])
    (case / "excluded_fov.jsonl").unlink(missing_ok=True)
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "total_squares": 1,
            "completed_pose_count": 1,
            "status_counts": {"gallery": 1},
            "pose_plan": _pose_plan_summary([record]),
        }
    )
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("角度覆盖" in error or "sampling_point_plan" in error for error in report["errors"])


def test_current_design_audit_rejects_state_manifest_record_content_mismatch(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    gallery_path = case / "gallery" / "gallery.jsonl"
    record = json.loads(gallery_path.read_text(encoding="utf-8").splitlines()[0])
    record["features"] = [{"label": "changed-only-in-state"}]
    _write_jsonl(gallery_path, [record])

    report = audit_output(case)

    assert report["passed"] is False
    assert any("记录内容" in error or "内容摘要" in error for error in report["errors"])


def test_current_design_audit_rejects_false_exact_deduplication_evidence(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    manifest_path = case / "manifest.jsonl"
    records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    removed = records.pop(1)
    records[0]["duplicate_source_pose_ids"] = [removed["slice_id"]]
    _write_jsonl(manifest_path, records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    _write_jsonl(case / "excluded_fov.jsonl", records[1:])
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "total_squares": len(records),
            "completed_pose_count": len(records),
            "status_counts": {"excluded_fov": len(records) - 1, "gallery": 1},
            "pose_plan": _pose_plan_summary(records),
        }
    )
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("精确去重" in error or "duplicate_source_pose" in error for error in report["errors"])


def test_current_design_audit_rejects_retained_pose_geometry_tampering(tmp_path):
    case = _valid_output(tmp_path)
    _make_current_design_output(case)
    manifest_path = case / "manifest.jsonl"
    records = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["square_vertices_world"][0][0] += 7.0
    _write_jsonl(manifest_path, records)
    _write_jsonl(case / "gallery" / "gallery.jsonl", records[:1])
    metadata_path = case / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pose_plan"] = _pose_plan_summary(records)
    _write_json(metadata_path, metadata)

    report = audit_output(case)

    assert report["passed"] is False
    assert any("pose_geometry" in error or "姿态几何" in error for error in report["errors"])


def test_audit_output_rejects_feature_without_boundary_color(tmp_path):
    case = _valid_output(tmp_path)
    Image.new("RGB", (3, 3), "white").save(
        case / "gallery" / "eus_vessel_boundary" / "sample.png"
    )

    report = audit_output(
        case,
        check_pixels=True,
        expected_core_design_sha256=CORE_HASH,
    )

    assert report["passed"] is False
    assert report["pixel_audit"]["feature_without_color"] == {"aorta": 1}
    assert any("feature_without_color" in error for error in report["errors"])


def test_audit_cli_writes_report_and_returns_nonzero_for_failure(tmp_path):
    case = _valid_output(tmp_path)
    Image.new("RGB", (3, 3), "white").save(
        case / "gallery" / "eus_vessel_boundary" / "sample.png"
    )
    report_path = tmp_path / "audit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_resampling_output.py",
            str(case),
            "--report",
            str(report_path),
            "--check-pixels",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False
    assert not (tmp_path / ".audit.json.tmp").exists()
