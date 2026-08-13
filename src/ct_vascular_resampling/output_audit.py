"""对已完成重采样库执行只读结构与 EUS 血管像素审计。"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .config import EUS_VESSEL_IDS, ORGAN_BOUNDARY_IDS


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
    expected_core_design_sha256: str | None = None,
    expected_build_git_commit: str | None = None,
) -> dict[str, object]:
    """流式审计完成的病例输出；除读取外不修改病例目录。"""

    case = Path(case_directory).resolve()
    errors: list[str] = []
    for required in ("manifest.jsonl", "run_metadata.json", "library_summary.json"):
        if not (case / required).is_file():
            errors.append(f"缺少必需文件: {required}")
    gallery_manifest = case / "gallery" / "gallery.jsonl"
    if not gallery_manifest.is_file():
        errors.append("缺少必需文件: gallery/gallery.jsonl")
    if errors:
        return {"passed": False, "errors": errors, "case_directory": str(case)}

    try:
        metadata = _read_json(case / "run_metadata.json")
        summary = _read_json(case / "library_summary.json")
    except ValueError as error:
        return {"passed": False, "errors": [str(error)], "case_directory": str(case)}

    manifest_lines = 0
    duplicate_slice_ids = 0
    missing_slice_ids = 0
    status_counts: Counter[str] = Counter()
    yaw_policy_counts: Counter[str] = Counter()
    seen_slice_ids: set[str] = set()
    pose_protocol_mismatches: Counter[str] = Counter()
    metadata_angles = metadata.get("pose_angles_degrees", {})
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
            if status not in _STATUS_PATHS:
                pose_protocol_mismatches["invalid_status"] += 1
            else:
                status_counts[str(status)] += 1
            policy = record.get("yaw_policy")
            angles = record.get("angles_degrees")
            if policy not in {"standard", "duodenum_bulb", "pancreas_special"}:
                pose_protocol_mismatches["yaw_policy"] += 1
            else:
                yaw_policy_counts[str(policy)] += 1
            if not isinstance(angles, dict) or set(angles) != {"roll", "pitch", "yaw"}:
                pose_protocol_mismatches["angles_shape"] += 1
            else:
                if angles["roll"] not in expected_roll:
                    pose_protocol_mismatches["roll"] += 1
                if angles["pitch"] not in expected_pitch:
                    pose_protocol_mismatches["pitch"] += 1
                policy_yaw = set(expected_yaw.get(policy, [])) if isinstance(expected_yaw, dict) else set()
                if angles["yaw"] not in policy_yaw:
                    pose_protocol_mismatches["yaw"] += 1
            if record.get("coordinate_system") != "RAS":
                pose_protocol_mismatches["coordinate_system"] += 1
            if record.get("core_design_sha256") != metadata.get("core_design_sha256"):
                pose_protocol_mismatches["core_design_sha256"] += 1
            if record.get("build_git_commit") != metadata.get("build_git_commit"):
                pose_protocol_mismatches["build_git_commit"] += 1
    except ValueError as error:
        errors.append(str(error))

    state_manifest_lines: dict[str, int] = {}
    state_slice_ids: set[str] = set()
    state_duplicate_slice_ids = 0
    state_status_mismatches = 0
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
    if state_slice_ids != seen_slice_ids:
        errors.append(
            "状态清单与根清单 slice_id 集合不一致: "
            f"state_only={len(state_slice_ids - seen_slice_ids)}, "
            f"root_only={len(seen_slice_ids - state_slice_ids)}"
        )

    gallery_lines = 0
    missing_fields: Counter[str] = Counter()
    missing_files: Counter[str] = Counter()
    organ_label_counts: Counter[str] = Counter()
    candidate_label_counts: Counter[str] = Counter()
    eus_vessel_label_counts: Counter[str] = Counter()
    eus_vessel_feature_counts: Counter[str] = Counter()
    invalid_gallery_values: Counter[str] = Counter()
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
    if metadata.get("quality_filtering", {}).get("black_ratio_limit") != 0.6:
        errors.append("black_ratio_limit 不是 0.6")
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
        "pixel_audit": pixel_audit,
    }
