"""正式重采样采样点与零度基准面的可视化导出。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re

import numpy as np


POINT_TOLERANCE_MM = 1e-5
AXIS_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ZeroPlaneRecord:
    slice_id: str
    organ: str
    point_index: int
    probe: np.ndarray
    vertices: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    input_normal: np.ndarray


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是形状 {shape} 的有限数值")
    result.setflags(write=False)
    return result


def _is_zero_pose(record: Mapping[str, object]) -> bool:
    angles = record.get("angles_degrees")
    if not isinstance(angles, Mapping) or set(angles) != {"roll", "pitch", "yaw"}:
        raise ValueError("angles_degrees 必须包含 roll、pitch、yaw")
    try:
        values = tuple(float(angles[name]) for name in ("roll", "pitch", "yaw"))
    except (TypeError, ValueError) as error:
        raise ValueError("angles_degrees 必须是有限数值") from error
    if not np.all(np.isfinite(values)):
        raise ValueError("angles_degrees 必须是有限数值")
    return values == (0.0, 0.0, 0.0)


def _point_index(slice_id: str, organ: str) -> int:
    matched = re.fullmatch(rf"{re.escape(organ)}-(\d{{6}})-.+", slice_id)
    if matched is None:
        raise ValueError(f"slice_id 与器官或点索引不一致: {slice_id}")
    return int(matched.group(1))


def _validated_record(record: Mapping[str, object]) -> ZeroPlaneRecord:
    organ = record.get("organ")
    slice_id = record.get("slice_id")
    if not isinstance(organ, str) or not organ:
        raise ValueError("organ 必须是非空字符串")
    if not isinstance(slice_id, str) or not slice_id:
        raise ValueError("slice_id 必须是非空字符串")
    if record.get("coordinate_system") != "RAS":
        raise ValueError(f"{slice_id} 的坐标系不是 RAS")

    probe = _finite_array(record.get("probe_point_world"), (3,), "probe_point_world")
    vertices = _finite_array(
        record.get("square_vertices_world"), (4, 3), "square_vertices_world"
    )
    axes = record.get("local_axes_world")
    if not isinstance(axes, Mapping) or set(axes) != {"x", "y", "z"}:
        raise ValueError("local_axes_world 必须包含 x、y、z")
    x_axis = _finite_array(axes["x"], (3,), "local_axes_world.x")
    y_axis = _finite_array(axes["y"], (3,), "local_axes_world.y")
    z_axis = _finite_array(axes["z"], (3,), "local_axes_world.z")
    input_normal = _finite_array(
        record.get("input_normal_world"), (3,), "input_normal_world"
    )

    axis_matrix = np.column_stack((x_axis, y_axis, z_axis))
    if not np.allclose(
        axis_matrix.T @ axis_matrix,
        np.eye(3),
        rtol=0.0,
        atol=AXIS_TOLERANCE,
    ):
        raise ValueError(f"{slice_id} 的局部轴不是单位正交坐标系")
    if not np.allclose(
        np.cross(x_axis, y_axis),
        z_axis,
        rtol=0.0,
        atol=AXIS_TOLERANCE,
    ):
        raise ValueError(f"{slice_id} 的局部轴不满足右手关系")

    edge_vectors = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    if not np.allclose(
        edge_lengths,
        np.full(4, 100.0),
        rtol=0.0,
        atol=POINT_TOLERANCE_MM,
    ):
        raise ValueError(f"{slice_id} 不是边长 100 mm 正方形")
    if not np.allclose(
        (vertices[0] + vertices[1]) / 2.0,
        probe,
        rtol=0.0,
        atol=POINT_TOLERANCE_MM,
    ):
        raise ValueError(f"{slice_id} 的探头不在零度面底边中点")
    if not (
        np.allclose(vertices[1] - vertices[0], y_axis * 100.0, rtol=0.0, atol=POINT_TOLERANCE_MM)
        and np.allclose(vertices[3] - vertices[0], x_axis * 100.0, rtol=0.0, atol=POINT_TOLERANCE_MM)
        and np.allclose(vertices[2], vertices[1] + x_axis * 100.0, rtol=0.0, atol=POINT_TOLERANCE_MM)
    ):
        raise ValueError(f"{slice_id} 的四顶点与局部轴不一致，不能构成 100 mm 正方形")

    return ZeroPlaneRecord(
        slice_id=slice_id,
        organ=organ,
        point_index=_point_index(slice_id, organ),
        probe=probe,
        vertices=vertices,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        input_normal=input_normal,
    )


def select_zero_planes(
    records: Iterable[Mapping[str, object]],
    expected_counts: Mapping[str, int],
) -> list[ZeroPlaneRecord]:
    """筛选零姿态记录并验证每个采样点恰好对应一个实际基准面。"""

    expected = dict(expected_counts)
    if not expected or any(
        not isinstance(organ, str)
        or not organ
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        for organ, count in expected.items()
    ):
        raise ValueError("expected_counts 必须是非空的正整数器官计数")

    selected: list[ZeroPlaneRecord] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        if not _is_zero_pose(record):
            continue
        validated = _validated_record(record)
        if validated.organ not in expected:
            raise ValueError(f"出现预期之外的零度面器官: {validated.organ}")
        key = (validated.organ, validated.point_index)
        if key in seen:
            raise ValueError(f"零度面采样点重复: {validated.organ}/{validated.point_index}")
        seen.add(key)
        selected.append(validated)

    actual = Counter(item.organ for item in selected)
    if actual != Counter(expected):
        raise ValueError(
            f"零度面逐器官计数不一致: actual={dict(actual)}, expected={expected}"
        )
    organ_order = {organ: index for index, organ in enumerate(expected)}
    return sorted(selected, key=lambda item: (organ_order[item.organ], item.point_index))
