"""运行恢复协议的稳定字段集合与规范摘要。"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json


RESUME_PROTOCOL_FIELDS = (
    "coordinate_system",
    "input_coordinate_system",
    "core_design_filename",
    "core_design_sha256",
    "base_core_design_filename",
    "base_core_design_sha256",
    "build_git_commit",
    "input_provenance",
    "sampling_configuration",
    "duodenum_centerline_selection",
    "minimum_point_spacing_mm",
    "surface_sampling_audit",
    "pose_plan",
    "sampling_point_plan",
    "pose_angles_degrees",
    "pose_convention",
    "square_sampling",
    "quality_filtering",
    "fov_policy",
    "eus_possible_organs",
    "manual_segmentation",
)


def resume_protocol_sha256(metadata: Mapping[str, object]) -> str:
    """仅对确定运行与恢复兼容性的协议字段计算规范 SHA-256。"""

    missing = [field for field in RESUME_PROTOCOL_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"运行协议缺少字段: {', '.join(missing)}")
    protocol = {field: metadata[field] for field in RESUME_PROTOCOL_FIELDS}
    canonical = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
