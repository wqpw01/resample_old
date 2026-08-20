"""按核心设计构造局部坐标和 100 mm 方形姿态。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centerline import CenterlinePath
from .contract import (
    PITCH_ANGLES_DEGREES,
    ROLL_ANGLES_DEGREES,
    LIVER_REGION_TWO_YAW_ANGLES_DEGREES,
    SPECIAL_YAW_ANGLES_DEGREES,
    STANDARD_YAW_ANGLES_DEGREES,
)


STANDARD_YAW = "standard"
DUODENUM_BULB_YAW = "duodenum_bulb"
PANCREAS_SPECIAL_YAW = "pancreas_special"
LIVER_REGION_TWO_YAW = "liver_region_two"
YAW_ANGLES_DEGREES = {
    STANDARD_YAW: STANDARD_YAW_ANGLES_DEGREES,
    DUODENUM_BULB_YAW: SPECIAL_YAW_ANGLES_DEGREES,
    PANCREAS_SPECIAL_YAW: SPECIAL_YAW_ANGLES_DEGREES,
    LIVER_REGION_TWO_YAW: LIVER_REGION_TWO_YAW_ANGLES_DEGREES,
}

_SOURCE_REGIONS_BY_ORGAN = {
    "stomach": {"stomach"},
    "liver": {
        "liver_region_one",
        "liver_region_two",
        "liver_region_one+liver_region_two",
    },
    "pancreas": {"pancreas"},
    "duodenum": {"duodenum_bulb", "duodenum_remainder"},
    "esophagus": {"esophagus"},
}


@dataclass(frozen=True)
class LocalFrame:
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray


@dataclass(frozen=True)
class PoseVariant:
    vertices: np.ndarray
    local_frame: LocalFrame
    roll_degrees: float
    pitch_degrees: float
    yaw_degrees: float


def validate_pose_protocol(
    organ: object,
    source_region: object,
    yaw_policy: object,
    *,
    target_ids: object = (),
    angles_degrees: object | None = None,
) -> None:
    """校验器官、采样区域、偏航策略及合同角集合的一致性。"""

    if not isinstance(organ, str) or organ not in _SOURCE_REGIONS_BY_ORGAN:
        raise ValueError(f"不支持的采样器官: {organ!r}")
    if not isinstance(source_region, str) or source_region not in _SOURCE_REGIONS_BY_ORGAN[organ]:
        raise ValueError(f"{organ} 的采样区域无效: {source_region!r}")
    if not isinstance(yaw_policy, str) or yaw_policy not in YAW_ANGLES_DEGREES:
        raise ValueError(f"不支持的 yaw_policy: {yaw_policy!r}")

    expected_policy: str | None = None
    if organ == "liver":
        expected_policy = (
            LIVER_REGION_TWO_YAW
            if "liver_region_two" in str(source_region).split("+")
            else STANDARD_YAW
        )
    elif organ == "duodenum":
        expected_policy = (
            DUODENUM_BULB_YAW if source_region == "duodenum_bulb" else STANDARD_YAW
        )
    elif organ == "esophagus":
        expected_policy = STANDARD_YAW
    elif organ in {"stomach", "pancreas"}:
        if yaw_policy not in {STANDARD_YAW, PANCREAS_SPECIAL_YAW}:
            raise ValueError(f"{organ} 的采样区域与 yaw_policy 不一致")

    if expected_policy is not None and yaw_policy != expected_policy:
        raise ValueError(f"{organ} 的采样区域与 yaw_policy 不一致")
    if organ == "stomach" and yaw_policy == PANCREAS_SPECIAL_YAW:
        if not isinstance(target_ids, (list, tuple)) or "pancreas" not in target_ids:
            raise ValueError("stomach 的 pancreas_special yaw_policy 缺少 pancreas 目标命中")

    if angles_degrees is None:
        return
    if not isinstance(angles_degrees, dict) or set(angles_degrees) != {"roll", "pitch", "yaw"}:
        raise ValueError("角度必须完整包含 roll、pitch、yaw")
    expected = {
        "roll": ROLL_ANGLES_DEGREES,
        "pitch": PITCH_ANGLES_DEGREES,
        "yaw": YAW_ANGLES_DEGREES[str(yaw_policy)],
    }
    for axis, allowed in expected.items():
        value = angles_degrees[axis]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ) or not np.isfinite(value):
            raise ValueError(f"{axis} 角度必须是有限数值")
        if float(value) not in allowed:
            raise ValueError(f"{axis} 角度 {value!r} 不在当前合同集合中")


def _xyz(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是三个有限数值")
    return result


def _unit(value: np.ndarray, name: str) -> np.ndarray:
    result = _xyz(value, name)
    magnitude = float(np.linalg.norm(result))
    if magnitude < 1e-8:
        raise ValueError(f"{name} 不能为零向量")
    return result / magnitude


def _validated_frame(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> LocalFrame:
    x = _unit(x_axis, "局部 x 轴")
    y = _unit(y_axis - np.dot(y_axis, x) * x, "局部 y 轴")
    z = _unit(np.cross(x, y), "局部 z 轴")
    expected_z = _unit(z_axis, "预期局部 z 轴")
    if np.dot(z, expected_z) < 1.0 - 1e-8:
        raise ValueError("局部坐标轴不满足右手正交关系")
    return LocalFrame(x, y, z)


def ordinary_local_frame(point: np.ndarray, forward: np.ndarray, esophagus_anchor: np.ndarray) -> LocalFrame:
    probe = _xyz(point, "point")
    anchor = _xyz(esophagus_anchor, "esophagus_anchor")
    x_axis = _unit(forward, "探头前进方向")
    reference = probe - anchor
    y_axis = reference - np.dot(reference, x_axis) * x_axis
    y_axis = _unit(y_axis, "食管极点参考方向")
    z_axis = _unit(np.cross(x_axis, y_axis), "普通 0 度面法向")
    return _validated_frame(x_axis, y_axis, z_axis)


def duodenum_local_frame(point: np.ndarray, outward_normal: np.ndarray, centerline: CenterlinePath) -> LocalFrame:
    probe = _xyz(point, "point")
    normal = _unit(outward_normal, "十二指肠外法线")
    if len(centerline.points) == 0 or centerline.points.shape != centerline.tangents.shape:
        raise ValueError("十二指肠中心线无效")
    nearest_index = int(np.argmin(np.linalg.norm(centerline.points - probe, axis=1)))
    radial = probe - centerline.points[nearest_index]
    tangent = _unit(centerline.tangents[nearest_index], "中心线切向")
    z_axis = _unit(np.cross(tangent, radial), "十二指肠 0 度面法向")
    x_axis = _unit(normal - np.dot(normal, z_axis) * z_axis, "投影后的十二指肠外法线")
    if np.dot(x_axis, normal) <= 0.0:
        raise ValueError("十二指肠前进方向与外法线不一致")
    y_axis = _unit(np.cross(z_axis, x_axis), "十二指肠局部 y 轴")
    return _validated_frame(x_axis, y_axis, z_axis)


def _rotation_matrix(roll_degrees: float, pitch_degrees: float, yaw_degrees: float) -> np.ndarray:
    roll, pitch, yaw = np.radians([roll_degrees, pitch_degrees, yaw_degrees])
    rx = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]]
    )
    ry = np.asarray(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
    )
    rz = np.asarray(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    return rz @ ry @ rx


def _validated_frame_matrix(frame: LocalFrame) -> np.ndarray:
    base = np.column_stack([frame.x_axis, frame.y_axis, frame.z_axis])
    if not np.allclose(base.T @ base, np.eye(3), rtol=0.0, atol=1e-8) or np.linalg.det(
        base
    ) < 0.0:
        raise ValueError("局部坐标必须是右手正交系")
    return base


def _pose_variant(
    probe: np.ndarray,
    base: np.ndarray,
    side_length_mm: float,
    roll_degrees: float,
    pitch_degrees: float,
    yaw_degrees: float,
) -> PoseVariant:
    rotated = base @ _rotation_matrix(roll_degrees, pitch_degrees, yaw_degrees)
    rotated_frame = LocalFrame(rotated[:, 0], rotated[:, 1], rotated[:, 2])
    half_side = side_length_mm / 2.0
    bottom_left = probe - rotated_frame.y_axis * half_side
    bottom_right = probe + rotated_frame.y_axis * half_side
    top_right = bottom_right + rotated_frame.x_axis * side_length_mm
    top_left = bottom_left + rotated_frame.x_axis * side_length_mm
    return PoseVariant(
        np.asarray([bottom_left, bottom_right, top_right, top_left]),
        rotated_frame,
        roll_degrees,
        pitch_degrees,
        yaw_degrees,
    )


def generate_pose_variant(
    point: np.ndarray,
    frame: LocalFrame,
    side_length_mm: float,
    roll_degrees: float,
    pitch_degrees: float,
    yaw_degrees: float,
) -> PoseVariant:
    """从未旋转局部坐标直接重建一个指定角度姿态。"""

    probe = _xyz(point, "point")
    if side_length_mm <= 0.0:
        raise ValueError("side_length_mm 必须大于零")
    return _pose_variant(
        probe,
        _validated_frame_matrix(frame),
        side_length_mm,
        roll_degrees,
        pitch_degrees,
        yaw_degrees,
    )


def generate_pose_variants(
    point: np.ndarray,
    frame: LocalFrame,
    side_length_mm: float,
    yaw_policy: str,
) -> list[PoseVariant]:
    probe = _xyz(point, "point")
    if side_length_mm <= 0.0:
        raise ValueError("side_length_mm 必须大于零")
    yaw_angles = YAW_ANGLES_DEGREES.get(yaw_policy)
    if yaw_angles is None:
        raise ValueError(f"不支持的偏航策略: {yaw_policy}")
    base = _validated_frame_matrix(frame)
    variants: list[PoseVariant] = []
    for yaw_degrees in yaw_angles:
        for pitch_degrees in PITCH_ANGLES_DEGREES:
            for roll_degrees in ROLL_ANGLES_DEGREES:
                variants.append(
                    _pose_variant(
                        probe,
                        base,
                        side_length_mm,
                        roll_degrees,
                        pitch_degrees,
                        yaw_degrees,
                    )
                )
    return variants
