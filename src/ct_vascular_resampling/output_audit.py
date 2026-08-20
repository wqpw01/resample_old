"""对已完成重采样库执行只读结构与 EUS 血管像素审计。"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from .centerline import CenterlinePath, extract_duodenum_centerline
from .config import EUS_VESSEL_IDS, ORGAN_BOUNDARY_IDS
from .contract import (
    BASE_CORE_DESIGN_FILENAME,
    BASE_CORE_DESIGN_SHA256,
    BLACK_RATIO_LIMIT,
    BLACK_SIDE_MIN_RATIO,
    BLACK_THRESHOLD,
    CENTERLINE_MAX_TERMINAL_SPUR_MM,
    CENTERLINE_TANGENT_WINDOW_MM,
    CENTERLINE_VOXEL_PITCH_MM,
    CORE_DESIGN_FILENAME,
    CORE_DESIGN_SHA256,
    ESOPHAGUS_EXTENSION_TARGET_FILTER,
    FILL_HU_VALUE,
    FOV_POLICY,
    LIVER_REGION_TWO_YAW_ANGLES_DEGREES,
    LINE_MIN_DIAGONAL_FRACTION,
    MINIMUM_POINT_SPACING_MM,
    OUTPUT_RESOLUTION,
    POSE_CONVENTION,
    PITCH_ANGLES_DEGREES,
    RAY_BATCH_SIZE,
    RAY_LENGTH_MM,
    ROLL_ANGLES_DEGREES,
    SAMPLING_SEED,
    SQUARE_SIDE_LENGTH_MM,
    SPECIAL_YAW_ANGLES_DEGREES,
    STANDARD_YAW_ANGLES_DEGREES,
    VALID_SIDE_MAX_BLACK_RATIO,
    WINDOW_LEVEL_HU,
    WINDOW_WIDTH_HU,
)
from .pose_plan import OrderIndependentDigest, summarize_pose_entries
from .protocol import resume_protocol_sha256
from .mesh_io import load_surface_mesh
from .sampling import extreme_plateau_centroid, filter_esophagus_valid_segment
from .sampling_pipeline import (
    _assert_samples_on_allowed_surface,
    base_local_frame,
    pose_sample_id,
    source_priority,
)
from .squares import LocalFrame, PoseVariant, generate_pose_variant, validate_pose_protocol


_GALLERY_IMAGE_FIELDS = (
    "ct_png",
    "boundary_only_png",
    "ct_overlay_png",
    "organ_vessel_boundary_png",
    "eus_vessel_boundary_png",
    "ct_eus_vessel_overlay_png",
)
_GALLERY_REQUIRED_FIELDS = frozenset(
    {
        "slice_id",
        "status",
        "organ",
        "coordinate_system",
        "core_design_sha256",
        "build_git_commit",
        "source_region",
        "yaw_policy",
        "angles_degrees",
        "features",
        "organ_metadata_schema_version",
        "organ_labels",
        "eus_candidate_organ_labels",
        "eus_vessel_metadata_schema_version",
        "eus_vessel_labels",
        "eus_vessel_features",
        *_GALLERY_IMAGE_FIELDS,
    }
)
_STATUS_PATHS = {
    "gallery": Path("gallery/gallery.jsonl"),
    "unindexed": Path("unindexed/unindexed.jsonl"),
    "rejected": Path("rejected/rejected.jsonl"),
    "excluded_fov": Path("excluded_fov.jsonl"),
}
_SAMPLING_ORGANS = ("stomach", "liver", "pancreas", "duodenum", "esophagus")
_REQUIRED_SAMPLING_REGIONS = {
    "stomach": {"stomach": "stomach"},
    "liver": {"liver": "liver"},
    "pancreas": {"pancreas": "pancreas"},
    "duodenum": {
        "duodenum_bulb": "duodenum_part1",
        "duodenum_remainder": "duodenum_part2",
    },
    "esophagus": {"esophagus": "esophagus"},
}
_CURRENT_POSE_ANGLES = {
    "roll": list(ROLL_ANGLES_DEGREES),
    "pitch": list(PITCH_ANGLES_DEGREES),
    "yaw": {
        "standard": list(STANDARD_YAW_ANGLES_DEGREES),
        "duodenum_bulb": list(SPECIAL_YAW_ANGLES_DEGREES),
        "pancreas_special": list(SPECIAL_YAW_ANGLES_DEGREES),
        "liver_region_two": list(LIVER_REGION_TWO_YAW_ANGLES_DEGREES),
    },
}
_POSE_ID = re.compile(
    r"^(stomach|liver|pancreas|duodenum|esophagus)-(\d{6})"
    r"-r([mpz]\d{3})-p([mpz]\d{3})-y([mpz]\d{3})$"
)
_POINT_COVERAGE_SCHEMA_VERSION = "sampling-point-coverage/v1"
_MANIFEST_RECORD_SCHEMA_VERSION = "manifest-record-set/v1"
_EXPECTED_ENDPOINTS_UNSET = object()


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path} 第 {line_number} 行为空")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须为 JSON 对象")
            yield line_number, record


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 JSON 文件 {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件顶层必须为对象: {path}")
    return value


def _safe_gallery_path(gallery: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    destination = (gallery / relative).resolve()
    try:
        destination.relative_to(gallery.resolve())
    except ValueError:
        return None
    return destination


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((key, int(value)) for key, value in counter.items() if value))


def _append_aggregate_error(
    errors: list[str],
    name: str,
    values: Counter[str] | dict[str, int],
) -> None:
    normalized = _counter_dict(Counter(values))
    if normalized:
        errors.append(f"{name}: {normalized}")


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_vector3(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    if any(
        isinstance(component, bool)
        or not isinstance(component, (int, float))
        or not np.isfinite(component)
        for component in value
    ):
        return None
    return np.asarray(value, dtype=np.float64)


def _point_key_from_pose_id(value: object) -> tuple[str, int] | None:
    identity = _pose_identity_from_pose_id(value)
    if identity is None:
        return None
    return identity[0], identity[1]


def _angle_from_token(token: str) -> float | None:
    magnitude = int(token[1:])
    if token[0] == "z":
        return 0.0 if magnitude == 0 else None
    return float(-magnitude if token[0] == "m" else magnitude)


def _pose_identity_from_pose_id(
    value: object,
) -> tuple[str, int, float, float, float] | None:
    if not isinstance(value, str):
        return None
    match = _POSE_ID.fullmatch(value)
    if match is None:
        return None
    angles = tuple(_angle_from_token(match.group(index)) for index in (3, 4, 5))
    if any(angle is None for angle in angles):
        return None
    organ = match.group(1)
    point_index = int(match.group(2))
    roll, pitch, yaw = angles
    if value != pose_sample_id(organ, point_index, roll, pitch, yaw):
        return None
    return organ, point_index, roll, pitch, yaw


def _finite_square_vertices(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    vertices = [_finite_vector3(vertex) for vertex in value]
    if any(vertex is None for vertex in vertices):
        return None
    return np.asarray(vertices, dtype=np.float64)


def _valid_local_axes(value: object) -> bool:
    return _local_axes_matrix(value) is not None


def _local_axes_matrix(value: object) -> np.ndarray | None:
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        return None
    vectors = [_finite_vector3(value[axis]) for axis in ("x", "y", "z")]
    if any(vector is None for vector in vectors):
        return None
    matrix = np.column_stack(vectors)
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=1e-8) or np.linalg.det(
        matrix
    ) <= 0.0:
        return None
    return matrix


def _finite_pose_angles(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, dict) or set(value) != {"roll", "pitch", "yaw"}:
        return None
    if any(
        isinstance(value[axis], bool)
        or not isinstance(value[axis], (int, float))
        or not np.isfinite(value[axis])
        for axis in ("roll", "pitch", "yaw")
    ):
        return None
    return float(value["roll"]), float(value["pitch"]), float(value["yaw"])


def _finite_endpoint_hints(
    value: object,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    result: list[tuple[float, float, float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 3 or any(
            isinstance(component, bool)
            or not isinstance(component, (int, float))
            or not np.isfinite(component)
            for component in point
        ):
            return None
        result.append(tuple(float(component) for component in point))
    return result[0], result[1]


def _expected_pose_variant(
    point_plan: dict[str, object],
    base_frame: LocalFrame | None,
    side_length_mm: float | None,
    angles: tuple[float, float, float],
) -> PoseVariant | None:
    if side_length_mm is None or base_frame is None:
        return None
    point = _finite_vector3(point_plan.get("probe_point_world"))
    if point is None:
        return None
    try:
        return generate_pose_variant(
            point,
            base_frame,
            side_length_mm,
            *angles,
        )
    except ValueError:
        return None


def _validate_current_surface_audit(
    metadata: dict,
    sampled_point_counts: dict[str, int],
) -> list[str]:
    audit = metadata.get("surface_sampling_audit")
    if not isinstance(audit, dict):
        return ["当前设计缺少 surface_sampling_audit"]
    if audit.get("outer_surface_required") is not True:
        return ["surface_sampling_audit.outer_surface_required 不是 true"]
    if audit.get("minimum_spacing_preserved_on_shortfall") is not True:
        return ["surface_sampling_audit 未声明数量不足时保持最小间距"]
    organs = audit.get("organs")
    if not isinstance(organs, dict) or set(organs) != set(_SAMPLING_ORGANS):
        return ["surface_sampling_audit.organs 必须完整包含五个采样器官"]

    sampling = metadata.get("sampling_configuration")
    point_counts = sampling.get("point_counts") if isinstance(sampling, dict) else None
    required_point_count_keys = {
        "stomach",
        "liver",
        "pancreas",
        "duodenum_part1",
        "duodenum_part2",
        "esophagus",
    }
    if (
        not isinstance(point_counts, dict)
        or set(point_counts) != required_point_count_keys
        or any(not _is_nonnegative_int(value) or value == 0 for value in point_counts.values())
    ):
        return ["sampling_configuration.point_counts 必须完整记录六个正整数请求数量"]
    requested_by_organ = {
        "stomach": point_counts["stomach"],
        "liver": point_counts["liver"],
        "pancreas": point_counts["pancreas"],
        "duodenum": point_counts["duodenum_part1"] + point_counts["duodenum_part2"],
        "esophagus": point_counts["esophagus"],
    }

    errors: list[str] = []
    for organ in _SAMPLING_ORGANS:
        record = organs[organ]
        if not isinstance(record, dict):
            errors.append(f"surface_sampling_audit.{organ} 不是对象")
            continue
        requested = record.get("requested_count")
        actual = record.get("actual_count")
        shortfall = record.get("shortfall_count")
        if not all(_is_nonnegative_int(value) for value in (requested, actual, shortfall)):
            errors.append(f"surface_sampling_audit.{organ} 计数无效")
            continue
        if actual > requested or shortfall != requested - actual:
            errors.append(f"surface_sampling_audit.{organ} 请求/实际/短缺计数不一致")
        if actual == 0:
            errors.append(f"surface_sampling_audit.{organ}.actual_count 不得为零")
        if requested != requested_by_organ[organ]:
            errors.append(
                f"surface_sampling_audit.{organ}.requested_count 与 sampling 配置不一致"
            )
        if actual != sampled_point_counts.get(organ):
            errors.append(f"surface_sampling_audit.{organ}.actual_count 与采样点 PLY 不一致")

        regions = record.get("regions")
        required_regions = _REQUIRED_SAMPLING_REGIONS[organ]
        if not isinstance(regions, dict) or set(regions) != set(required_regions):
            errors.append(
                f"surface_sampling_audit.{organ}.regions 必须为 "
                f"{sorted(required_regions)}"
            )
        else:
            region_requested = 0
            region_actual = 0
            for region, statistics in regions.items():
                if not isinstance(region, str) or not isinstance(statistics, dict):
                    errors.append(f"surface_sampling_audit.{organ}.regions 条目无效")
                    continue
                values = tuple(
                    statistics.get(key)
                    for key in (
                        "requested_count",
                        "candidate_count",
                        "actual_count",
                        "shortfall_count",
                    )
                )
                if not all(_is_nonnegative_int(value) for value in values):
                    errors.append(f"surface_sampling_audit.{organ}.{region} 计数无效")
                    continue
                region_request, candidate, region_count, region_shortfall = values
                configured_request = point_counts[required_regions[region]]
                if (
                    region_count > region_request
                    or region_count > candidate
                    or region_shortfall != region_request - region_count
                    or region_request != configured_request
                    or candidate == 0
                    or region_count == 0
                ):
                    errors.append(f"surface_sampling_audit.{organ}.{region} 计数不一致")
                region_requested += region_request
                region_actual += region_count
            if region_requested != requested or region_actual != actual:
                errors.append(f"surface_sampling_audit.{organ}.regions 汇总计数不一致")

        source = record.get("source_surface")
        required_source_fields = {
            "input_component_count",
            "input_face_count",
            "kept_face_count",
            "discarded_face_count",
        }
        if (
            not isinstance(source, dict)
            or source.get("enabled") is not True
            or source.get("selection_rule") != "largest_watertight_absolute_volume"
            or not required_source_fields.issubset(source)
        ):
            errors.append(f"surface_sampling_audit.{organ}.source_surface 无效")
            continue
        component_count = source["input_component_count"]
        input_faces = source["input_face_count"]
        kept_faces = source["kept_face_count"]
        discarded_faces = source["discarded_face_count"]
        volume = source.get("selected_enclosed_volume_mm3")
        if (
            not all(
                _is_nonnegative_int(value)
                for value in (component_count, input_faces, kept_faces, discarded_faces)
            )
            or component_count < 1
            or kept_faces < 1
            or input_faces != kept_faces + discarded_faces
            or isinstance(volume, bool)
            or not isinstance(volume, (int, float))
            or not np.isfinite(volume)
            or volume <= 0.0
        ):
            errors.append(f"surface_sampling_audit.{organ}.source_surface 几何统计无效")
    return errors


def _validate_current_design_metadata(metadata: dict) -> list[str]:
    expected_identity = {
        "core_design_filename": CORE_DESIGN_FILENAME,
        "core_design_sha256": CORE_DESIGN_SHA256,
        "base_core_design_filename": BASE_CORE_DESIGN_FILENAME,
        "base_core_design_sha256": BASE_CORE_DESIGN_SHA256,
    }
    errors = [
        f"当前设计身份字段 {key} 不一致"
        for key, value in expected_identity.items()
        if metadata.get(key) != value
    ]
    if metadata.get("pose_angles_degrees") != _CURRENT_POSE_ANGLES:
        errors.append("当前设计 pose_angles_degrees 与合同完整角集合不一致")
    sampling = metadata.get("sampling_configuration")
    if not isinstance(sampling, dict) or sampling.get("count_policy") != (
        "upper_bound_preserve_outer_surface_and_minimum_spacing"
    ):
        errors.append("当前设计 sampling_configuration.count_policy 不一致")
    if not isinstance(sampling, dict) or sampling.get("source_surface_policy") != (
        "largest_watertight_absolute_volume"
    ):
        errors.append("当前设计 sampling_configuration.source_surface_policy 不一致")
    if not isinstance(sampling, dict) or sampling.get("minimum_spacing_mm") != 10.0:
        errors.append("当前设计 sampling_configuration.minimum_spacing_mm 不是 10 mm")
    expected_sampling = {
        "ray_length_mm": RAY_LENGTH_MM,
        "ray_batch_size": RAY_BATCH_SIZE,
        "minimum_spacing_mm": MINIMUM_POINT_SPACING_MM,
        "centerline_voxel_pitch_mm": CENTERLINE_VOXEL_PITCH_MM,
        "centerline_tangent_window_mm": CENTERLINE_TANGENT_WINDOW_MM,
        "centerline_max_terminal_spur_mm": CENTERLINE_MAX_TERMINAL_SPUR_MM,
    }
    if isinstance(sampling, dict):
        for key, expected in expected_sampling.items():
            if sampling.get(key) != expected:
                errors.append(f"当前设计 sampling_configuration.{key} 不是合同值 {expected}")
        if sampling.get("esophagus_extension_target_filter") != (
            ESOPHAGUS_EXTENSION_TARGET_FILTER
        ):
            errors.append(
                "当前设计 sampling_configuration.esophagus_extension_target_filter 不一致"
            )
    if metadata.get("minimum_point_spacing_mm") != MINIMUM_POINT_SPACING_MM:
        errors.append("当前设计 minimum_point_spacing_mm 不是 10 mm")
    if not isinstance(sampling, dict) or sampling.get("seed") != SAMPLING_SEED:
        errors.append(f"当前设计 sampling_configuration.seed 不是合同值 {SAMPLING_SEED}")
    expected_square = {
        "side_length_mm": SQUARE_SIDE_LENGTH_MM,
        "output_resolution": [OUTPUT_RESOLUTION, OUTPUT_RESOLUTION],
        "interpolation": "cubic_bspline",
        "interpolation_order": 3,
        "window_level_hu": WINDOW_LEVEL_HU,
        "window_width_hu": WINDOW_WIDTH_HU,
        "fill_hu_value": FILL_HU_VALUE,
    }
    square_sampling = metadata.get("square_sampling")
    if not isinstance(square_sampling, dict):
        errors.append("当前设计 square_sampling 缺失或无效")
    else:
        for key, expected in expected_square.items():
            if square_sampling.get(key) != expected:
                errors.append(f"当前设计 square_sampling.{key} 不是合同值 {expected!r}")
    expected_quality = {
        "black_threshold": BLACK_THRESHOLD,
        "black_ratio_limit": BLACK_RATIO_LIMIT,
        "line_min_diagonal_fraction": LINE_MIN_DIAGONAL_FRACTION,
        "black_side_min_ratio": BLACK_SIDE_MIN_RATIO,
        "valid_side_max_black_ratio": VALID_SIDE_MAX_BLACK_RATIO,
    }
    quality_filtering = metadata.get("quality_filtering")
    if not isinstance(quality_filtering, dict):
        errors.append("当前设计 quality_filtering 缺失或无效")
    else:
        for key, expected in expected_quality.items():
            if quality_filtering.get(key) != expected:
                errors.append(f"当前设计 quality_filtering.{key} 不是合同值 {expected}")
    if metadata.get("pose_convention") != POSE_CONVENTION:
        errors.append("当前设计 pose_convention 与合同不一致")
    if metadata.get("fov_policy") != FOV_POLICY:
        errors.append("当前设计 fov_policy 与合同不一致")
    return errors


def _pose_plan_summary(records: Iterable[dict]) -> dict[str, int | str]:
    def entries() -> Iterator[dict[str, object]]:
        for record in records:
            angles = record.get("angles_degrees", {})
            probe = record.get("probe_point_world", [])
            input_normal = record.get("input_normal_world", [])
            vertices = record.get("square_vertices_world", [])
            yield {
                "slice_id": record.get("slice_id"),
                "organ": record.get("organ"),
                "probe_point_world": [float(value) for value in probe],
                "input_normal_world": [float(value) for value in input_normal],
                "square_vertices_world": [
                    [float(value) for value in vertex] for vertex in vertices
                ],
                "source_region": record.get("source_region"),
                "yaw_policy": record.get("yaw_policy"),
                "angles_degrees": {
                    "roll": float(angles.get("roll")),
                    "pitch": float(angles.get("pitch")),
                    "yaw": float(angles.get("yaw")),
                },
            }

    return summarize_pose_entries(entries())


def _audit_sampling_point_ply(
    case: Path,
    minimum_spacing_mm: float,
) -> tuple[dict[str, object], dict[str, tuple[np.ndarray, np.ndarray]]]:
    counts: dict[str, int] = {}
    minimum_distances: dict[str, float | None] = {}
    geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    errors: list[str] = []
    for organ in _SAMPLING_ORGANS:
        path = case / "ResampledpointPLY" / f"FPS-{organ.capitalize()}.ply"
        if not path.is_file():
            errors.append(f"缺少采样点文件: {path.relative_to(case)}")
            continue
        try:
            loaded = trimesh.load(path, process=False)
            raw = loaded.metadata["_ply_raw"]["vertex"]
            count = int(raw["length"])
            data = raw.get("data", {})
            if any(
                np.asarray(data[axis]).dtype.kind != "f"
                or np.asarray(data[axis]).dtype.itemsize != 8
                for axis in ("x", "y", "z", "nx", "ny", "nz")
            ):
                errors.append(f"采样点文件必须使用 double 坐标和法线: {path.relative_to(case)}")
            points = np.column_stack(
                [np.asarray(data[axis], dtype=np.float64).reshape(-1) for axis in ("x", "y", "z")]
            ) if count else np.empty((0, 3), dtype=np.float64)
            normals = np.column_stack(
                [np.asarray(data[axis], dtype=np.float64).reshape(-1) for axis in ("nx", "ny", "nz")]
            ) if count else np.empty((0, 3), dtype=np.float64)
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"采样点文件无效 {path.relative_to(case)}: {error}")
            continue
        if points.shape != (count, 3) or normals.shape != (count, 3):
            errors.append(f"采样点文件字段数量不一致: {path.relative_to(case)}")
            continue
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
            errors.append(f"采样点文件包含非有限数值: {path.relative_to(case)}")
            continue
        if count and np.any(np.abs(np.linalg.norm(normals, axis=1) - 1.0) > 2e-5):
            errors.append(f"采样点文件法线不是单位向量: {path.relative_to(case)}")
        actual_minimum: float | None = None
        if count >= 2:
            actual_minimum = float(np.min(cKDTree(points).query(points, k=2)[0][:, 1]))
            if actual_minimum < minimum_spacing_mm - 5e-6:
                errors.append(
                    f"{organ} 采样点 PLY 最小间距 {actual_minimum:.9g} mm 小于 {minimum_spacing_mm:g} mm"
                )
        counts[organ] = count
        minimum_distances[organ] = actual_minimum
        geometry[organ] = (points, normals)
    return (
        {
            "counts": counts,
            "minimum_distances_mm": minimum_distances,
            "errors": errors,
        },
        geometry,
    )


def _validate_sampling_point_plan(
    metadata: dict,
    ply_geometry: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[
    dict[tuple[str, int], dict[str, object]],
    dict[tuple[str, int], OrderIndependentDigest],
    list[str],
]:
    raw_plan = metadata.get("sampling_point_plan")
    if not isinstance(raw_plan, dict) or raw_plan.get("schema_version") != (
        "sampling-point-plan/v1"
    ):
        return {}, {}, ["当前设计缺少有效 sampling_point_plan/v1"]
    raw_organs = raw_plan.get("organs")
    if not isinstance(raw_organs, dict) or set(raw_organs) != set(_SAMPLING_ORGANS):
        return {}, {}, ["sampling_point_plan.organs 必须完整包含五个采样器官"]

    points_by_key: dict[tuple[str, int], dict[str, object]] = {}
    expected_coverage: dict[tuple[str, int], OrderIndependentDigest] = {}
    errors: list[str] = []
    axis_pose_count = len(ROLL_ANGLES_DEGREES) * len(PITCH_ANGLES_DEGREES)
    for organ in _SAMPLING_ORGANS:
        records = raw_organs.get(organ)
        if not isinstance(records, list):
            errors.append(f"sampling_point_plan.{organ} 必须是列表")
            continue
        ply = ply_geometry.get(organ)
        if ply is None:
            errors.append(f"sampling_point_plan.{organ} 缺少可比对的采样点 PLY")
        elif len(records) != len(ply[0]):
            errors.append(f"sampling_point_plan.{organ} 与采样点 PLY 数量不一致")

        for expected_index, record in enumerate(records):
            if not isinstance(record, dict) or record.get("point_index") != expected_index:
                errors.append(f"sampling_point_plan.{organ}[{expected_index}] 点序号无效")
                continue
            point = _finite_vector3(record.get("probe_point_world"))
            normal = _finite_vector3(record.get("input_normal_world"))
            source_region = record.get("source_region")
            yaw_policy = record.get("yaw_policy")
            target_ids = record.get("target_ids")
            base_axes = record.get("base_local_axes_world")
            if point is None or normal is None:
                errors.append(f"sampling_point_plan.{organ}[{expected_index}] 坐标或法线无效")
                continue
            if abs(float(np.linalg.norm(normal)) - 1.0) > 1e-8:
                errors.append(f"sampling_point_plan.{organ}[{expected_index}] 法线不是单位向量")
            if not isinstance(target_ids, list) or any(
                not isinstance(value, str) for value in target_ids
            ) or len(target_ids) != len(set(target_ids)):
                errors.append(f"sampling_point_plan.{organ}[{expected_index}].target_ids 无效")
                target_ids = []
            if not _valid_local_axes(base_axes):
                errors.append(
                    f"sampling_point_plan.{organ}[{expected_index}].base_local_axes_world 无效"
                )
                continue
            try:
                validate_pose_protocol(
                    organ,
                    source_region,
                    yaw_policy,
                    target_ids=target_ids,
                )
            except ValueError as error:
                errors.append(f"sampling_point_plan.{organ}[{expected_index}]: {error}")
                continue
            expected_count = axis_pose_count * len(_CURRENT_POSE_ANGLES["yaw"][str(yaw_policy)])
            if record.get("candidate_pose_count") != expected_count:
                errors.append(
                    f"sampling_point_plan.{organ}[{expected_index}].candidate_pose_count "
                    f"不是合同值 {expected_count}"
                )

            if ply is not None and expected_index < len(ply[0]):
                if not np.allclose(ply[0][expected_index], point, rtol=0.0, atol=2e-6):
                    errors.append(
                        f"sampling_point_plan.{organ}[{expected_index}] 与采样点 PLY 坐标不一致"
                    )
                if not np.allclose(ply[1][expected_index], normal, rtol=0.0, atol=2e-6):
                    errors.append(
                        f"sampling_point_plan.{organ}[{expected_index}] 与采样点 PLY 法线不一致"
                    )

            key = (organ, expected_index)
            points_by_key[key] = record
            coverage = OrderIndependentDigest(_POINT_COVERAGE_SCHEMA_VERSION)
            for yaw in _CURRENT_POSE_ANGLES["yaw"][str(yaw_policy)]:
                for pitch in PITCH_ANGLES_DEGREES:
                    for roll in ROLL_ANGLES_DEGREES:
                        coverage.update(
                            pose_sample_id(organ, expected_index, roll, pitch, yaw)
                        )
            expected_coverage[key] = coverage
    return points_by_key, expected_coverage, errors


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _allowed_build_git_commits(metadata: dict) -> tuple[set[str], list[str]]:
    current = metadata.get("build_git_commit")
    errors: list[str] = []
    if not _is_git_commit(current):
        return set(), ["run_metadata.build_git_commit 不是 40 位小写十六进制提交"]

    compatible = metadata.get("compatible_completed_build_git_commits", [])
    if (
        not isinstance(compatible, list)
        or any(not _is_git_commit(value) for value in compatible)
        or compatible != sorted(set(compatible))
    ):
        return {str(current)}, ["compatible_completed_build_git_commits 无效"]

    history = metadata.get("recovery_history")
    history_commits: set[str] = set()
    if history is not None:
        if not isinstance(history, list) or not history:
            errors.append("recovery_history 必须是非空列表")
        else:
            for index, record in enumerate(history):
                completed = (
                    record.get("completed_build_git_commits")
                    if isinstance(record, dict)
                    else None
                )
                recovery_commit = (
                    record.get("recovery_build_git_commit")
                    if isinstance(record, dict)
                    else None
                )
                if (
                    not isinstance(completed, list)
                    or any(not _is_git_commit(value) for value in completed)
                    or completed != sorted(set(completed))
                    or not _is_git_commit(recovery_commit)
                ):
                    errors.append(f"recovery_history[{index}] 的构建提交证据无效")
                    continue
                history_commits.update(completed)
    if set(compatible) != history_commits:
        errors.append(
            "compatible_completed_build_git_commits 与 recovery_history 证据不一致"
        )
    return {str(current), *compatible}, errors


def _reconstruct_input_base_frames(
    metadata: dict,
    point_plan: dict[tuple[str, int], dict[str, object]],
    *,
    expected_duodenum_centerline_endpoint_hints_ras_mm: object = _EXPECTED_ENDPOINTS_UNSET,
    expected_duodenum_centerline_endpoint_match_tolerance_mm: object = _EXPECTED_ENDPOINTS_UNSET,
) -> tuple[dict[tuple[str, int], LocalFrame], list[str]]:
    provenance = metadata.get("input_provenance")
    organ_models = provenance.get("organ_models") if isinstance(provenance, dict) else None
    if not isinstance(organ_models, dict):
        return {}, ["当前设计缺少 input_provenance.organ_models，无法复核真实主外壳"]
    input_coordinate_system = metadata.get("input_coordinate_system")
    if input_coordinate_system not in {"LPS", "RAS"}:
        return {}, ["当前设计 input_coordinate_system 必须为 LPS 或 RAS"]

    errors: list[str] = []
    paths: dict[str, Path] = {}
    for organ in _SAMPLING_ORGANS:
        source = organ_models.get(organ)
        if (
            not isinstance(source, dict)
            or source.get("kind") != "file"
            or not isinstance(source.get("path"), str)
            or not isinstance(source.get("sha256"), str)
        ):
            errors.append(f"input_provenance.organ_models.{organ} 无效")
            continue
        path = Path(source["path"]).resolve()
        if not path.is_file():
            errors.append(f"主外壳输入文件不存在: {organ}")
            continue
        try:
            actual_hash = _sha256_file(path)
        except OSError as error:
            errors.append(f"无法读取主外壳输入 {organ}: {error}")
            continue
        if actual_hash != source["sha256"]:
            errors.append(f"主外壳输入 SHA-256 不一致: {organ}")
            continue
        paths[organ] = path
    if set(paths) != set(_SAMPLING_ORGANS):
        return {}, errors

    try:
        full_meshes = {
            organ: load_surface_mesh(
                paths[organ],
                input_coordinate_system=str(input_coordinate_system),
            )
            for organ in _SAMPLING_ORGANS
        }
        source_meshes = {
            organ: load_surface_mesh(
                paths[organ],
                input_coordinate_system=str(input_coordinate_system),
                main_outer_surface_only=True,
            )
            for organ in _SAMPLING_ORGANS
        }
    except (OSError, ValueError) as error:
        return {}, [*errors, f"无法从输入溯源重建主外壳: {error}"]

    declared_surface_audit = metadata.get("surface_sampling_audit")
    declared_organs = (
        declared_surface_audit.get("organs", {})
        if isinstance(declared_surface_audit, dict)
        else {}
    )
    for organ, mesh in source_meshes.items():
        actual_audit = mesh.surface_audit.to_record() if mesh.surface_audit is not None else None
        declared = declared_organs.get(organ) if isinstance(declared_organs, dict) else None
        if not isinstance(declared, dict) or declared.get("source_surface") != actual_audit:
            errors.append(f"surface_sampling_audit.{organ}.source_surface 与真实输入不一致")

    allowed_vertices = {
        organ: mesh.vertices for organ, mesh in source_meshes.items()
    }
    allowed_normals = {
        organ: mesh.vertex_normals for organ, mesh in source_meshes.items()
    }
    frame_vertices = dict(allowed_vertices)
    frame_normals = dict(allowed_normals)
    try:
        esophagus_points, esophagus_normals = filter_esophagus_valid_segment(
            source_meshes["esophagus"].vertices,
            source_meshes["esophagus"].vertex_normals,
            full_meshes["liver"].vertices,
        )
        translated = esophagus_points.copy()
        translated[:, 2] -= float(np.ptp(esophagus_points[:, 2]))
        allowed_vertices["esophagus"] = np.vstack(
            [allowed_vertices["esophagus"], translated]
        )
        allowed_normals["esophagus"] = np.vstack(
            [allowed_normals["esophagus"], esophagus_normals]
        )
        frame_vertices["esophagus"] = np.vstack([esophagus_points, translated])
        frame_normals["esophagus"] = np.vstack(
            [esophagus_normals, esophagus_normals]
        )
    except ValueError as error:
        errors.append(f"无法重建食管允许外表面: {error}")

    sampling = metadata.get("sampling_configuration")
    hints_record = (
        sampling.get("duodenum_centerline_endpoint_hints_ras_mm")
        if isinstance(sampling, dict)
        else None
    )
    tolerance_record = (
        sampling.get("duodenum_centerline_endpoint_match_tolerance_mm")
        if isinstance(sampling, dict)
        else None
    )
    endpoint_hints = None
    endpoint_tolerance = 1.0
    if not isinstance(sampling, dict) or (
        "duodenum_centerline_endpoint_hints_ras_mm" not in sampling
        or "duodenum_centerline_endpoint_match_tolerance_mm" not in sampling
    ):
        errors.append("sampling_configuration 缺少十二指肠中心线端点配置")
    elif hints_record is None:
        if tolerance_record is not None:
            errors.append("十二指肠中心线无端点提示时匹配容差必须为 null")
    elif not isinstance(hints_record, dict) or set(hints_record) != {"proximal", "distal"}:
        errors.append("十二指肠中心线端点提示无效")
    else:
        proximal = _finite_vector3(hints_record["proximal"])
        distal = _finite_vector3(hints_record["distal"])
        if proximal is None or distal is None:
            errors.append("十二指肠中心线端点提示必须是有限三维坐标")
        elif (
            isinstance(tolerance_record, bool)
            or not isinstance(tolerance_record, (int, float))
            or not np.isfinite(tolerance_record)
            or tolerance_record <= 0.0
        ):
            errors.append("十二指肠中心线端点匹配容差必须是有限正数")
        else:
            endpoint_hints = (
                tuple(float(value) for value in proximal),
                tuple(float(value) for value in distal),
            )
            endpoint_tolerance = float(tolerance_record)

    expected_endpoint_hints = (
        None
        if expected_duodenum_centerline_endpoint_hints_ras_mm is None
        else _finite_endpoint_hints(
            expected_duodenum_centerline_endpoint_hints_ras_mm
        )
    )
    if hints_record is not None:
        if expected_duodenum_centerline_endpoint_hints_ras_mm is _EXPECTED_ENDPOINTS_UNSET:
            errors.append("人工十二指肠中心线审计必须提供外部病例配置端点")
        elif expected_endpoint_hints is None:
            errors.append("外部病例配置未提供有效的人工十二指肠中心线端点")
        elif endpoint_hints != expected_endpoint_hints:
            errors.append("人工十二指肠中心线端点与外部病例配置不一致")
        if expected_duodenum_centerline_endpoint_match_tolerance_mm is _EXPECTED_ENDPOINTS_UNSET:
            errors.append("人工十二指肠中心线审计必须提供外部病例配置匹配容差")
        elif (
            isinstance(expected_duodenum_centerline_endpoint_match_tolerance_mm, bool)
            or not isinstance(
                expected_duodenum_centerline_endpoint_match_tolerance_mm,
                (int, float),
            )
            or not np.isfinite(expected_duodenum_centerline_endpoint_match_tolerance_mm)
            or float(expected_duodenum_centerline_endpoint_match_tolerance_mm) <= 0.0
        ):
            errors.append("外部病例配置的十二指肠中心线匹配容差无效")
        elif endpoint_hints is not None and endpoint_tolerance != float(
            expected_duodenum_centerline_endpoint_match_tolerance_mm
        ):
            errors.append("十二指肠中心线匹配容差与外部病例配置不一致")
    elif (
        expected_duodenum_centerline_endpoint_hints_ras_mm is not _EXPECTED_ENDPOINTS_UNSET
        and expected_duodenum_centerline_endpoint_hints_ras_mm is not None
    ):
        if expected_endpoint_hints is None:
            errors.append("外部病例配置的十二指肠中心线端点无效")
        else:
            errors.append("运行元数据缺少外部病例配置指定的人工十二指肠中心线端点")
    elif (
        expected_duodenum_centerline_endpoint_match_tolerance_mm
        is not _EXPECTED_ENDPOINTS_UNSET
        and expected_duodenum_centerline_endpoint_match_tolerance_mm is not None
    ):
        errors.append("自动十二指肠中心线的外部匹配容差必须为空")

    esophagus_anchor: np.ndarray | None = None
    centerline: CenterlinePath | None = None
    try:
        esophagus_anchor = extreme_plateau_centroid(
            full_meshes["esophagus"].vertices,
            axis=2,
            maximum=False,
        )
        centerline = extract_duodenum_centerline(
            full_meshes["duodenum"].mesh,
            full_meshes["stomach"].vertices,
            voxel_pitch_mm=CENTERLINE_VOXEL_PITCH_MM,
            tangent_window_mm=CENTERLINE_TANGENT_WINDOW_MM,
            max_terminal_spur_mm=CENTERLINE_MAX_TERMINAL_SPUR_MM,
            endpoint_hints_ras_mm=endpoint_hints,
            endpoint_match_tolerance_mm=endpoint_tolerance,
        )
    except ValueError as error:
        errors.append(f"无法从真实输入重建局部坐标基准: {error}")
    if centerline is not None:
        expected_selection = (
            centerline.selection_audit.to_record()
            if centerline.selection_audit is not None
            else None
        )
        if metadata.get("duodenum_centerline_selection") != expected_selection:
            errors.append("duodenum_centerline_selection 与真实输入重建结果不一致")

    base_frames: dict[tuple[str, int], LocalFrame] = {}
    for organ in _SAMPLING_ORGANS:
        records = [
            point_plan[(organ, index)]
            for index in range(sum(key[0] == organ for key in point_plan))
            if (organ, index) in point_plan
        ]
        if not records:
            errors.append(f"sampling_point_plan.{organ} 不得为空")
            continue
        points = np.asarray([record["probe_point_world"] for record in records], dtype=np.float64)
        normals = np.asarray(
            [record["input_normal_world"] for record in records],
            dtype=np.float64,
        )
        try:
            _assert_samples_on_allowed_surface(
                points,
                normals,
                allowed_vertices[organ],
                allowed_normals[organ],
                organ,
            )
        except ValueError as error:
            errors.append(f"sampling_point_plan 与真实主外壳不一致: {error}")
        frame_tree = cKDTree(frame_vertices[organ])
        for record in records:
            point_index = int(record["point_index"])
            point = np.asarray(record["probe_point_world"], dtype=np.float64)
            declared_normal = np.asarray(
                record["input_normal_world"],
                dtype=np.float64,
            )
            matches = frame_tree.query_ball_point(point, 1e-9)
            if not matches:
                errors.append(
                    f"sampling_point_plan.{organ}[{point_index}].input_normal_world "
                    "无法映射到真实输入顶点"
                )
                continue
            reference_normal = np.asarray(
                frame_normals[organ][min(matches)],
                dtype=np.float64,
            )
            reference_normal /= np.linalg.norm(reference_normal)
            if not np.allclose(
                declared_normal,
                reference_normal,
                rtol=0.0,
                atol=1e-10,
            ):
                errors.append(
                    f"sampling_point_plan.{organ}[{point_index}].input_normal_world "
                    "与真实输入外向法线不一致"
                )
            try:
                frame = base_local_frame(
                    organ,
                    point,
                    reference_normal,
                    esophagus_anchor=esophagus_anchor,
                    centerline=centerline,
                )
            except ValueError as error:
                errors.append(
                    f"sampling_point_plan.{organ}[{point_index}].base_local_axes_world "
                    f"无法从真实输入重建: {error}"
                )
                continue
            base_frames[(organ, point_index)] = frame
            expected_axes = np.column_stack([frame.x_axis, frame.y_axis, frame.z_axis])
            declared_axes = _local_axes_matrix(record.get("base_local_axes_world"))
            if declared_axes is None or not np.allclose(
                declared_axes,
                expected_axes,
                rtol=0.0,
                atol=1e-10,
            ):
                errors.append(
                    f"sampling_point_plan.{organ}[{point_index}].base_local_axes_world "
                    "与真实输入重建结果不一致"
                )
    return base_frames, errors


def _validate_point_plan_against_input_surfaces(
    metadata: dict,
    point_plan: dict[tuple[str, int], dict[str, object]],
    *,
    expected_duodenum_centerline_endpoint_hints_ras_mm: object = _EXPECTED_ENDPOINTS_UNSET,
    expected_duodenum_centerline_endpoint_match_tolerance_mm: object = _EXPECTED_ENDPOINTS_UNSET,
) -> list[str]:
    _, errors = _reconstruct_input_base_frames(
        metadata,
        point_plan,
        expected_duodenum_centerline_endpoint_hints_ras_mm=(
            expected_duodenum_centerline_endpoint_hints_ras_mm
        ),
        expected_duodenum_centerline_endpoint_match_tolerance_mm=(
            expected_duodenum_centerline_endpoint_match_tolerance_mm
        ),
    )
    return errors


def _audit_pixels(
    gallery_directory: Path,
    gallery_manifest: Path,
    colors: dict[str, tuple[int, int, int]],
) -> dict[str, object]:
    decoded_frames = 0
    color_pixels: Counter[str] = Counter()
    frames_with_color: Counter[str] = Counter()
    label_without_color: Counter[str] = Counter()
    color_without_label: Counter[str] = Counter()
    feature_without_color: Counter[str] = Counter()
    feature_without_label: Counter[str] = Counter()
    open_visible_without_feature: Counter[str] = Counter()
    unknown_color_frames = 0
    decode_errors: Counter[str] = Counter()
    white = np.asarray((255, 255, 255), dtype=np.uint8)

    for line_number, record in _iter_jsonl(gallery_manifest):
        relative = record.get("eus_vessel_boundary_png")
        image_path = _safe_gallery_path(gallery_directory, relative)
        if image_path is None or not image_path.is_file():
            decode_errors["missing_or_unsafe_path"] += 1
            continue
        try:
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError):
            decode_errors["invalid_png"] += 1
            continue
        decoded_frames += 1
        labels = set(record.get("eus_vessel_labels", []))
        feature_labels = {
            feature.get("label")
            for feature in record.get("eus_vessel_features", [])
            if isinstance(feature, dict)
        }
        known_mask = np.all(pixels == white, axis=2)
        present_colors: set[str] = set()
        for label, color in colors.items():
            mask = np.all(pixels == np.asarray(color, dtype=np.uint8), axis=2)
            count = int(np.count_nonzero(mask))
            known_mask |= mask
            if count:
                present_colors.add(label)
                color_pixels[label] += count
                frames_with_color[label] += 1
        if np.any(~known_mask):
            unknown_color_frames += 1
        for label in labels - present_colors:
            label_without_color[str(label)] += 1
        for label in present_colors - labels:
            color_without_label[label] += 1
        for label in feature_labels - present_colors:
            feature_without_color[str(label)] += 1
        for label in feature_labels - labels:
            feature_without_label[str(label)] += 1
        for label in labels - feature_labels:
            open_visible_without_feature[str(label)] += 1

    return {
        "decoded_frames": decoded_frames,
        "color_pixels": _counter_dict(color_pixels),
        "frames_with_color": _counter_dict(frames_with_color),
        "label_without_color": _counter_dict(label_without_color),
        "color_without_label": _counter_dict(color_without_label),
        "feature_without_color": _counter_dict(feature_without_color),
        "feature_without_label": _counter_dict(feature_without_label),
        "open_visible_without_feature": _counter_dict(open_visible_without_feature),
        "border_visible_without_feature": _counter_dict(open_visible_without_feature),
        "unknown_color_frames": unknown_color_frames,
        "decode_errors": _counter_dict(decode_errors),
    }


def audit_output(
    case_directory: str | Path,
    *,
    check_pixels: bool = False,
    expected_core_design_sha256: str | None = CORE_DESIGN_SHA256,
    expected_build_git_commit: str | None = None,
    expected_duodenum_centerline_endpoint_hints_ras_mm: object = _EXPECTED_ENDPOINTS_UNSET,
    expected_duodenum_centerline_endpoint_match_tolerance_mm: object = _EXPECTED_ENDPOINTS_UNSET,
) -> dict[str, object]:
    """流式审计完成的病例输出；除读取外不修改病例目录。"""

    case = Path(case_directory).resolve()
    errors: list[str] = []
    for required in ("manifest.jsonl", "run_metadata.json", "library_summary.json"):
        if not (case / required).is_file():
            errors.append(f"缺少必需文件: {required}")
    gallery_manifest = case / "gallery" / "gallery.jsonl"
    if errors:
        return {"passed": False, "errors": errors, "case_directory": str(case)}

    try:
        metadata = _read_json(case / "run_metadata.json")
        summary = _read_json(case / "library_summary.json")
    except ValueError as error:
        return {"passed": False, "errors": [str(error)], "case_directory": str(case)}

    current_design = metadata.get("core_design_sha256") == CORE_DESIGN_SHA256
    allowed_build_git_commits, build_commit_errors = _allowed_build_git_commits(metadata)
    errors.extend(build_commit_errors)
    sampling_point_audit: dict[str, object] | None = None
    point_plan: dict[tuple[str, int], dict[str, object]] = {}
    input_base_frames: dict[tuple[str, int], LocalFrame] = {}
    expected_point_coverage: dict[tuple[str, int], OrderIndependentDigest] = {}
    side_length_mm: float | None = None
    if current_design:
        if (
            expected_duodenum_centerline_endpoint_hints_ras_mm
            is _EXPECTED_ENDPOINTS_UNSET
            or expected_duodenum_centerline_endpoint_match_tolerance_mm
            is _EXPECTED_ENDPOINTS_UNSET
        ):
            errors.append(
                "当前正式设计审计必须提供外部病例配置；自动中心线须显式提供空端点和空容差"
            )
        errors.extend(_validate_current_design_metadata(metadata))
        try:
            actual_resume_protocol_sha256 = resume_protocol_sha256(metadata)
        except (TypeError, ValueError) as error:
            errors.append(f"resume_protocol_sha256 无法复算: {error}")
        else:
            if metadata.get("resume_protocol_sha256") != actual_resume_protocol_sha256:
                errors.append("resume_protocol_sha256 与当前运行协议字段不一致")
        square_sampling = metadata.get("square_sampling")
        raw_side_length = (
            square_sampling.get("side_length_mm") if isinstance(square_sampling, dict) else None
        )
        if (
            isinstance(raw_side_length, bool)
            or not isinstance(raw_side_length, (int, float))
            or not np.isfinite(raw_side_length)
            or raw_side_length <= 0.0
        ):
            errors.append("当前设计 square_sampling.side_length_mm 无效")
        else:
            side_length_mm = float(raw_side_length)
        sampling_point_audit, ply_geometry = _audit_sampling_point_ply(
            case,
            minimum_spacing_mm=10.0,
        )
        errors.extend(str(error) for error in sampling_point_audit["errors"])
        errors.extend(
            _validate_current_surface_audit(
                metadata,
                sampled_point_counts=dict(sampling_point_audit["counts"]),
            )
        )
        point_plan, expected_point_coverage, point_plan_errors = _validate_sampling_point_plan(
            metadata,
            ply_geometry,
        )
        errors.extend(point_plan_errors)
        if point_plan:
            input_base_frames, input_errors = _reconstruct_input_base_frames(
                metadata,
                point_plan,
                expected_duodenum_centerline_endpoint_hints_ras_mm=(
                    expected_duodenum_centerline_endpoint_hints_ras_mm
                ),
                expected_duodenum_centerline_endpoint_match_tolerance_mm=(
                    expected_duodenum_centerline_endpoint_match_tolerance_mm
                ),
            )
            errors.extend(input_errors)

    manifest_lines = 0
    duplicate_slice_ids = 0
    missing_slice_ids = 0
    status_counts: Counter[str] = Counter()
    yaw_policy_counts: Counter[str] = Counter()
    seen_slice_ids: set[str] = set()
    pose_protocol_mismatches: Counter[str] = Counter()
    actual_point_coverage = {
        key: OrderIndependentDigest(_POINT_COVERAGE_SCHEMA_VERSION) for key in point_plan
    }
    duplicate_source_pose_ids: set[str] = set()
    duplicate_source_pose_id_count = 0
    root_record_digests = {
        status: OrderIndependentDigest(_MANIFEST_RECORD_SCHEMA_VERSION) for status in _STATUS_PATHS
    }
    metadata_angles = metadata.get("pose_angles_degrees", {})
    if current_design:
        expected_roll = set(ROLL_ANGLES_DEGREES)
        expected_pitch = set(PITCH_ANGLES_DEGREES)
        expected_yaw = _CURRENT_POSE_ANGLES["yaw"]
    else:
        expected_roll = set(metadata_angles.get("roll", [])) if isinstance(metadata_angles, dict) else set()
        expected_pitch = set(metadata_angles.get("pitch", [])) if isinstance(metadata_angles, dict) else set()
        expected_yaw = metadata_angles.get("yaw", {}) if isinstance(metadata_angles, dict) else {}

    try:
        for _, record in _iter_jsonl(case / "manifest.jsonl"):
            manifest_lines += 1
            slice_id = record.get("slice_id")
            if not isinstance(slice_id, str) or not slice_id:
                missing_slice_ids += 1
            elif slice_id in seen_slice_ids:
                duplicate_slice_ids += 1
            else:
                seen_slice_ids.add(slice_id)
            status = record.get("status")
            if not isinstance(status, str) or status not in _STATUS_PATHS:
                pose_protocol_mismatches["invalid_status"] += 1
            else:
                status_counts[str(status)] += 1
                root_record_digests[str(status)].update(record)
            policy = record.get("yaw_policy")
            angles = record.get("angles_degrees")
            if not isinstance(policy, str) or policy not in {
                "standard",
                "duodenum_bulb",
                "pancreas_special",
                "liver_region_two",
            }:
                pose_protocol_mismatches["yaw_policy"] += 1
            else:
                yaw_policy_counts[str(policy)] += 1
            if current_design:
                organ = record.get("organ")
                source_region = record.get("source_region")
                probe = record.get("probe_point_world")
                input_normal = record.get("input_normal_world")
                target_ids = record.get("target_ids")
                probe_vector = _finite_vector3(probe)
                input_normal_vector = _finite_vector3(input_normal)
                if (
                    organ not in _SAMPLING_ORGANS
                    or probe_vector is None
                    or input_normal_vector is None
                ):
                    pose_protocol_mismatches["probe_point_world"] += 1
                if (
                    not isinstance(target_ids, list)
                    or any(not isinstance(value, str) for value in target_ids)
                    or len(target_ids) != len(set(target_ids))
                ):
                    pose_protocol_mismatches["target_ids"] += 1
                try:
                    validate_pose_protocol(
                        organ,
                        source_region,
                        policy,
                        target_ids=target_ids,
                        angles_degrees=angles,
                    )
                except ValueError:
                    pose_protocol_mismatches["source_region_yaw_policy"] += 1
                if not _valid_local_axes(record.get("local_axes_world")):
                    pose_protocol_mismatches["local_axes_world"] += 1

                point_key = _point_key_from_pose_id(slice_id)
                planned = point_plan.get(point_key) if point_key is not None else None
                pose_angles = _finite_pose_angles(angles)
                if planned is None:
                    pose_protocol_mismatches["sampling_point_plan"] += 1
                else:
                    actual_point_coverage[point_key].update(slice_id)
                    if probe_vector is None or input_normal_vector is None or (
                        organ != point_key[0]
                        or source_region != planned.get("source_region")
                        or policy != planned.get("yaw_policy")
                        or target_ids != planned.get("target_ids")
                        or not np.allclose(
                            probe_vector,
                            np.asarray(planned["probe_point_world"], dtype=np.float64),
                            rtol=0.0,
                            atol=1e-12,
                        )
                        or not np.allclose(
                            input_normal_vector,
                            np.asarray(planned["input_normal_world"], dtype=np.float64),
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ):
                        pose_protocol_mismatches["sampling_point_plan"] += 1
                    if pose_angles is not None:
                        expected_slice_id = pose_sample_id(
                            point_key[0],
                            point_key[1],
                            *pose_angles,
                        )
                        if slice_id != expected_slice_id:
                            pose_protocol_mismatches["slice_id_angles"] += 1
                        expected_variant = _expected_pose_variant(
                            planned,
                            input_base_frames.get(point_key),
                            side_length_mm,
                            pose_angles,
                        )
                        retained_vertices = _finite_square_vertices(
                            record.get("square_vertices_world")
                        )
                        retained_axes = _local_axes_matrix(record.get("local_axes_world"))
                        expected_axes = (
                            None
                            if expected_variant is None
                            else np.column_stack(
                                [
                                    expected_variant.local_frame.x_axis,
                                    expected_variant.local_frame.y_axis,
                                    expected_variant.local_frame.z_axis,
                                ]
                            )
                        )
                        if (
                            retained_vertices is None
                            or retained_axes is None
                            or expected_variant is None
                            or expected_axes is None
                            or not np.allclose(
                                retained_vertices,
                                expected_variant.vertices,
                                rtol=0.0,
                                atol=1e-9,
                            )
                            or not np.allclose(
                                retained_axes,
                                expected_axes,
                                rtol=0.0,
                                atol=1e-10,
                            )
                        ):
                            pose_protocol_mismatches["pose_geometry"] += 1

                raw_duplicate_ids = record.get("duplicate_source_pose_ids")
                raw_duplicate_regions = record.get("duplicate_source_regions")
                if not isinstance(raw_duplicate_ids, list) or any(
                    not isinstance(value, str) for value in raw_duplicate_ids
                ):
                    pose_protocol_mismatches["duplicate_source_pose_ids"] += 1
                    raw_duplicate_ids = []
                if len(raw_duplicate_ids) != len(set(raw_duplicate_ids)):
                    pose_protocol_mismatches["duplicate_source_pose_ids"] += 1
                if not isinstance(raw_duplicate_regions, list) or any(
                    not isinstance(value, str) for value in raw_duplicate_regions
                ):
                    pose_protocol_mismatches["duplicate_source_regions"] += 1
                    raw_duplicate_regions = []
                elif len(raw_duplicate_regions) != len(set(raw_duplicate_regions)):
                    pose_protocol_mismatches["duplicate_source_regions"] += 1
                duplicate_regions_from_plan: list[str] = []
                retained_vertices = _finite_square_vertices(
                    record.get("square_vertices_world")
                )
                for duplicate_id in raw_duplicate_ids:
                    duplicate_source_pose_id_count += 1
                    if duplicate_id in duplicate_source_pose_ids:
                        pose_protocol_mismatches["duplicate_source_pose_ids"] += 1
                    duplicate_source_pose_ids.add(duplicate_id)
                    duplicate_identity = _pose_identity_from_pose_id(duplicate_id)
                    duplicate_key = (
                        (duplicate_identity[0], duplicate_identity[1])
                        if duplicate_identity is not None
                        else None
                    )
                    duplicate_plan = (
                        point_plan.get(duplicate_key) if duplicate_key is not None else None
                    )
                    if duplicate_plan is None:
                        pose_protocol_mismatches["duplicate_source_pose_ids"] += 1
                        continue
                    actual_point_coverage[duplicate_key].update(duplicate_id)
                    expected_duplicate = _expected_pose_variant(
                        duplicate_plan,
                        input_base_frames.get(duplicate_key),
                        side_length_mm,
                        (
                            duplicate_identity[2],
                            duplicate_identity[3],
                            duplicate_identity[4],
                        ),
                    )
                    if (
                        retained_vertices is None
                        or expected_duplicate is None
                        or not np.allclose(
                            retained_vertices,
                            expected_duplicate.vertices,
                            rtol=0.0,
                            atol=1e-9,
                        )
                    ):
                        pose_protocol_mismatches["duplicate_source_pose_geometry"] += 1
                    duplicate_region = duplicate_plan.get("source_region")
                    if source_priority(str(organ), str(source_region)) > source_priority(
                        duplicate_key[0],
                        str(duplicate_region),
                    ):
                        pose_protocol_mismatches["duplicate_source_priority"] += 1
                    if duplicate_region != source_region and isinstance(duplicate_region, str):
                        duplicate_regions_from_plan.append(duplicate_region)
                if set(raw_duplicate_regions) != set(duplicate_regions_from_plan):
                    pose_protocol_mismatches["duplicate_source_regions"] += 1
            if not isinstance(angles, dict) or set(angles) != {"roll", "pitch", "yaw"}:
                pose_protocol_mismatches["angles_shape"] += 1
            elif any(
                isinstance(angles[axis], bool)
                or not isinstance(angles[axis], (int, float))
                or not np.isfinite(angles[axis])
                for axis in ("roll", "pitch", "yaw")
            ):
                pose_protocol_mismatches["angles_value"] += 1
            else:
                if angles["roll"] not in expected_roll:
                    pose_protocol_mismatches["roll"] += 1
                if angles["pitch"] not in expected_pitch:
                    pose_protocol_mismatches["pitch"] += 1
                policy_yaw = (
                    set(expected_yaw.get(policy, []))
                    if isinstance(expected_yaw, dict) and isinstance(policy, str)
                    else set()
                )
                if angles["yaw"] not in policy_yaw:
                    pose_protocol_mismatches["yaw"] += 1
            if record.get("coordinate_system") != "RAS":
                pose_protocol_mismatches["coordinate_system"] += 1
            if record.get("core_design_sha256") != metadata.get("core_design_sha256"):
                pose_protocol_mismatches["core_design_sha256"] += 1
            if record.get("build_git_commit") not in allowed_build_git_commits:
                pose_protocol_mismatches["build_git_commit"] += 1
    except ValueError as error:
        errors.append(str(error))

    state_manifest_lines: dict[str, int] = {}
    state_slice_ids: set[str] = set()
    state_duplicate_slice_ids = 0
    state_status_mismatches = 0
    state_record_digests = {
        status: OrderIndependentDigest(_MANIFEST_RECORD_SCHEMA_VERSION) for status in _STATUS_PATHS
    }
    for status, relative in _STATUS_PATHS.items():
        path = case / relative
        count = 0
        if not path.is_file():
            if status_counts[status]:
                errors.append(f"状态 {status} 有 {status_counts[status]} 条但缺少 {relative}")
            state_manifest_lines[status] = 0
            continue
        try:
            for _, record in _iter_jsonl(path):
                count += 1
                if record.get("status") != status:
                    state_status_mismatches += 1
                state_record_digests[status].update(record)
                slice_id = record.get("slice_id")
                if not isinstance(slice_id, str) or slice_id in state_slice_ids:
                    state_duplicate_slice_ids += 1
                else:
                    state_slice_ids.add(slice_id)
        except ValueError as error:
            errors.append(str(error))
        state_manifest_lines[status] = count
        if count != status_counts[status]:
            errors.append(
                f"状态 {status} 清单行数 {count} 与根清单计数 {status_counts[status]} 不一致"
            )
        if state_record_digests[status].to_record() != root_record_digests[status].to_record():
            errors.append(f"根清单与状态 {status} 清单的记录内容摘要不一致")
    if state_slice_ids != seen_slice_ids:
        errors.append(
            "状态清单与根清单 slice_id 集合不一致: "
            f"state_only={len(state_slice_ids - seen_slice_ids)}, "
            f"root_only={len(seen_slice_ids - state_slice_ids)}"
        )
    overlap_with_retained = duplicate_source_pose_ids & seen_slice_ids
    if overlap_with_retained:
        errors.append(
            f"被精确去重的姿态 ID 同时存在于根清单: {len(overlap_with_retained)}"
        )

    gallery_lines = 0
    missing_fields: Counter[str] = Counter()
    missing_files: Counter[str] = Counter()
    organ_label_counts: Counter[str] = Counter()
    candidate_label_counts: Counter[str] = Counter()
    eus_vessel_label_counts: Counter[str] = Counter()
    eus_vessel_feature_counts: Counter[str] = Counter()
    invalid_gallery_values: Counter[str] = Counter()
    if gallery_manifest.is_file():
        try:
            for _, record in _iter_jsonl(gallery_manifest):
                gallery_lines += 1
                for field in _GALLERY_REQUIRED_FIELDS - record.keys():
                    missing_fields[field] += 1
                for field in _GALLERY_IMAGE_FIELDS:
                    destination = _safe_gallery_path(case / "gallery", record.get(field))
                    if destination is None or not destination.is_file():
                        missing_files[field] += 1
                organ_labels = record.get("organ_labels", [])
                candidate_labels = record.get("eus_candidate_organ_labels", [])
                vessel_labels = record.get("eus_vessel_labels", [])
                features = record.get("eus_vessel_features", [])
                if not isinstance(organ_labels, list) or any(
                    label not in ORGAN_BOUNDARY_IDS for label in organ_labels
                ):
                    invalid_gallery_values["organ_labels"] += 1
                    organ_labels = []
                if not isinstance(candidate_labels, list):
                    invalid_gallery_values["eus_candidate_organ_labels"] += 1
                    candidate_labels = []
                if not isinstance(vessel_labels, list) or any(
                    label not in EUS_VESSEL_IDS for label in vessel_labels
                ):
                    invalid_gallery_values["eus_vessel_labels"] += 1
                    vessel_labels = []
                if not isinstance(features, list):
                    invalid_gallery_values["eus_vessel_features"] += 1
                    features = []
                organ_label_counts.update(organ_labels)
                candidate_label_counts.update(candidate_labels)
                eus_vessel_label_counts.update(vessel_labels)
                for feature in features:
                    if not isinstance(feature, dict) or feature.get("label") not in EUS_VESSEL_IDS:
                        invalid_gallery_values["eus_vessel_features.item"] += 1
                    else:
                        eus_vessel_feature_counts[str(feature["label"])] += 1
        except ValueError as error:
            errors.append(str(error))

    directory_png_counts: dict[str, int] = {}
    for field in _GALLERY_IMAGE_FIELDS:
        directory = case / "gallery" / field.removesuffix("_png")
        count = sum(1 for _ in directory.glob("*.png")) if directory.is_dir() else 0
        directory_png_counts[field.removesuffix("_png")] = count
        if count != gallery_lines:
            errors.append(f"Gallery 目录 {directory.name} PNG 数 {count} != {gallery_lines}")

    temporary_file_count = sum(
        1
        for path in case.rglob("*")
        if path.is_file() and (path.name.endswith(".tmp") or ".tmp." in path.name)
    )
    if metadata.get("run_state") != "complete":
        errors.append(f"run_state 不是 complete: {metadata.get('run_state')!r}")
    if metadata.get("total_squares") != manifest_lines:
        errors.append(f"total_squares {metadata.get('total_squares')} != {manifest_lines}")
    if metadata.get("completed_pose_count") != manifest_lines:
        errors.append(f"completed_pose_count {metadata.get('completed_pose_count')} != {manifest_lines}")
    if metadata.get("status_counts") != _counter_dict(status_counts):
        errors.append("run_metadata.status_counts 与根清单流式统计不一致")
    if metadata.get("quality_filtering", {}).get("black_ratio_limit") != BLACK_RATIO_LIMIT:
        errors.append(f"black_ratio_limit 不是 {BLACK_RATIO_LIMIT}")
    if expected_core_design_sha256 is not None and metadata.get(
        "core_design_sha256"
    ) != expected_core_design_sha256:
        errors.append("核心设计 SHA-256 与命令行期望值不一致")
    if expected_build_git_commit is not None and metadata.get(
        "build_git_commit"
    ) != expected_build_git_commit:
        errors.append("构建提交与命令行期望值不一致")
    if gallery_lines != status_counts["gallery"]:
        errors.append(f"gallery 行数 {gallery_lines} != 根清单 Gallery 数 {status_counts['gallery']}")
    if summary.get("indexed_feature_count") != gallery_lines:
        errors.append("library_summary.indexed_feature_count 与 Gallery 行数不一致")
    for key, actual in (
        ("organ_label_counts", organ_label_counts),
        ("eus_candidate_organ_label_counts", candidate_label_counts),
        ("eus_vessel_label_counts", eus_vessel_label_counts),
        ("eus_vessel_feature_counts", eus_vessel_feature_counts),
    ):
        if summary.get(key, {}) != _counter_dict(actual):
            errors.append(f"library_summary.{key} 与 Gallery 流式统计不一致")
    if duplicate_slice_ids:
        errors.append(f"根清单重复 slice_id: {duplicate_slice_ids}")
    if missing_slice_ids:
        errors.append(f"根清单缺失 slice_id: {missing_slice_ids}")
    if state_duplicate_slice_ids:
        errors.append(f"状态清单重复或缺失 slice_id: {state_duplicate_slice_ids}")
    if state_status_mismatches:
        errors.append(f"状态清单 status 不一致: {state_status_mismatches}")
    if temporary_file_count:
        errors.append(f"残留临时文件: {temporary_file_count}")
    if current_design:
        coverage_mismatches = [
            key
            for key, expected in expected_point_coverage.items()
            if actual_point_coverage[key].to_record() != expected.to_record()
        ]
        if coverage_mismatches:
            errors.append(
                "sampling_point_plan 逐采样点角度覆盖不完整或重复: "
                f"{len(coverage_mismatches)} 个点"
            )
        expected_candidate_count = sum(
            int(expected.to_record()["count"]) for expected in expected_point_coverage.values()
        )
        if expected_candidate_count != manifest_lines + duplicate_source_pose_id_count:
            errors.append(
                "sampling_point_plan 候选姿态总数与清单及精确去重记录不一致: "
                f"expected={expected_candidate_count}, "
                f"actual={manifest_lines + duplicate_source_pose_id_count}"
            )
        expected_pose_plan = metadata.get("pose_plan")
        try:
            actual_pose_plan = _pose_plan_summary(
                record for _, record in _iter_jsonl(case / "manifest.jsonl")
            )
        except (TypeError, ValueError) as error:
            actual_pose_plan = None
            errors.append(f"manifest.jsonl 无法生成 pose_plan 摘要: {error}")
        if expected_pose_plan != actual_pose_plan:
            errors.append(
                "pose_plan 与 manifest.jsonl 不一致: "
                f"expected={expected_pose_plan!r}, actual={actual_pose_plan!r}"
            )
    _append_aggregate_error(errors, "pose_protocol_mismatches", pose_protocol_mismatches)
    _append_aggregate_error(errors, "missing_fields", missing_fields)
    _append_aggregate_error(errors, "missing_files", missing_files)
    _append_aggregate_error(errors, "invalid_gallery_values", invalid_gallery_values)

    pixel_audit: dict[str, object] | None = None
    if check_pixels:
        raw_colors = metadata.get("manual_segmentation", {}).get("eus_vessel_colors", {})
        colors = {
            label: tuple(int(value) for value in raw_colors.get(label, []))
            for label in EUS_VESSEL_IDS
        }
        if any(len(color) != 3 for color in colors.values()):
            errors.append("run_metadata 缺少完整三类 EUS 血管颜色")
        else:
            pixel_audit = _audit_pixels(case / "gallery", gallery_manifest, colors)
            if pixel_audit["decoded_frames"] != gallery_lines:
                errors.append(
                    f"pixel decoded_frames {pixel_audit['decoded_frames']} != {gallery_lines}"
                )
            for key in (
                "label_without_color",
                "color_without_label",
                "feature_without_color",
                "feature_without_label",
                "decode_errors",
            ):
                if pixel_audit[key]:
                    errors.append(f"{key}: {pixel_audit[key]}")
            if pixel_audit["unknown_color_frames"]:
                errors.append(f"unknown_color_frames: {pixel_audit['unknown_color_frames']}")

    return {
        "passed": not errors,
        "errors": errors,
        "case_directory": str(case),
        "manifest_lines": manifest_lines,
        "status_counts": _counter_dict(status_counts),
        "yaw_policy_counts": _counter_dict(yaw_policy_counts),
        "duplicate_slice_ids": duplicate_slice_ids,
        "gallery_lines": gallery_lines,
        "directory_png_counts": directory_png_counts,
        "missing_fields": _counter_dict(missing_fields),
        "missing_files": _counter_dict(missing_files),
        "organ_label_counts": _counter_dict(organ_label_counts),
        "eus_candidate_organ_label_counts": _counter_dict(candidate_label_counts),
        "eus_vessel_label_counts": _counter_dict(eus_vessel_label_counts),
        "eus_vessel_feature_counts": _counter_dict(eus_vessel_feature_counts),
        "temporary_file_count": temporary_file_count,
        "state_manifest_lines": state_manifest_lines,
        "core_design_filename": metadata.get("core_design_filename"),
        "core_design_sha256": metadata.get("core_design_sha256"),
        "build_git_commit": metadata.get("build_git_commit"),
        "pose_angles_degrees": metadata.get("pose_angles_degrees"),
        "black_ratio_limit": metadata.get("quality_filtering", {}).get("black_ratio_limit"),
        "ct_backend": metadata.get("selected_backend"),
        "label_sampling": metadata.get("label_sampling"),
        "sampling_point_audit": sampling_point_audit,
        "pixel_audit": pixel_audit,
    }
