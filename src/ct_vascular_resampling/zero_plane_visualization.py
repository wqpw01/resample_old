"""正式重采样采样点与零度基准面的可视化导出。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re

import numpy as np


POINT_TOLERANCE_MM = 1e-5
AXIS_TOLERANCE = 1e-8

ORGAN_COLORS: dict[str, tuple[int, int, int]] = {
    "stomach": (228, 87, 86),
    "liver": (76, 120, 168),
    "pancreas": (242, 207, 91),
    "duodenum": (84, 162, 75),
    "esophagus": (178, 121, 162),
}


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _color(organ: str) -> tuple[int, int, int]:
    try:
        return ORGAN_COLORS[organ]
    except KeyError as error:
        raise ValueError(f"没有为器官 {organ!r} 定义可视化颜色") from error


def _validate_provenance(provenance: Mapping[str, object]) -> dict[str, str]:
    expected_lengths = {
        "source_manifest_sha256": 64,
        "core_design_sha256": 64,
        "build_git_commit": 40,
    }
    result: dict[str, str] = {}
    for key, length in expected_lengths.items():
        value = provenance.get(key)
        if (
            not isinstance(value, str)
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"provenance.{key} 必须是 {length} 位十六进制字符串")
        result[key] = value.lower()
    return result


def _record_payload(record: ZeroPlaneRecord) -> dict[str, object]:
    return {
        "slice_id": record.slice_id,
        "organ": record.organ,
        "point_index": record.point_index,
        "probe_point_world": record.probe.tolist(),
        "input_normal_world": record.input_normal.tolist(),
        "local_axes_world": {
            "x": record.x_axis.tolist(),
            "y": record.y_axis.tolist(),
            "z": record.z_axis.tolist(),
        },
        "vertices_world": record.vertices.tolist(),
    }


def _points_ply(records: list[ZeroPlaneRecord], organ_ids: Mapping[str, int]) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(records)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property ushort organ_id",
        "end_header",
    ]
    for record in records:
        red, green, blue = _color(record.organ)
        values = (*record.probe, *record.input_normal)
        lines.append(
            " ".join(f"{float(value):.9f}" for value in values)
            + f" {red} {green} {blue} {organ_ids[record.organ]}"
        )
    return "\n".join(lines) + "\n"


def _edges_ply(records: list[ZeroPlaneRecord], organ_ids: Mapping[str, int]) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(records) * 4}",
        "property float x",
        "property float y",
        "property float z",
        f"element edge {len(records) * 4}",
        "property int vertex1",
        "property int vertex2",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property ushort organ_id",
        "end_header",
    ]
    for record in records:
        lines.extend(" ".join(f"{float(value):.9f}" for value in vertex) for vertex in record.vertices)
    for record_index, record in enumerate(records):
        red, green, blue = _color(record.organ)
        offset = record_index * 4
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            lines.append(
                f"{offset + first} {offset + second} {red} {green} {blue} {organ_ids[record.organ]}"
            )
    return "\n".join(lines) + "\n"


def _faces_ply(records: list[ZeroPlaneRecord], organ_ids: Mapping[str, int]) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(records) * 4}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(records)}",
        "property list uchar int vertex_indices",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property ushort organ_id",
        "end_header",
    ]
    for record in records:
        lines.extend(" ".join(f"{float(value):.9f}" for value in vertex) for vertex in record.vertices)
    for record_index, record in enumerate(records):
        red, green, blue = _color(record.organ)
        offset = record_index * 4
        lines.append(
            f"4 {offset} {offset + 1} {offset + 2} {offset + 3} "
            f"{red} {green} {blue} {organ_ids[record.organ]}"
        )
    return "\n".join(lines) + "\n"


def _csv_text(records: list[ZeroPlaneRecord]) -> str:
    fixed = ["slice_id", "organ", "point_index"]
    vector_fields = [
        "probe",
        "normal",
        "local_x",
        "local_y",
        "local_z",
        "v0",
        "v1",
        "v2",
        "v3",
    ]
    axes = ("x_mm", "y_mm", "z_mm")
    fields = fixed + [f"{name}_{axis}" for name in vector_fields for axis in axes]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        values = {
            "probe": record.probe,
            "normal": record.input_normal,
            "local_x": record.x_axis,
            "local_y": record.y_axis,
            "local_z": record.z_axis,
            "v0": record.vertices[0],
            "v1": record.vertices[1],
            "v2": record.vertices[2],
            "v3": record.vertices[3],
        }
        row: dict[str, object] = {
            "slice_id": record.slice_id,
            "organ": record.organ,
            "point_index": record.point_index,
        }
        for name, vector in values.items():
            for axis, value in zip(axes, vector, strict=True):
                row[f"{name}_{axis}"] = f"{float(value):.9f}"
        writer.writerow(row)
    return stream.getvalue()


def write_structured_exports(
    records: Iterable[ZeroPlaneRecord],
    output_directory: str | Path,
    provenance: Mapping[str, object],
) -> None:
    """写出可由常见三维工具和表格程序读取的零度面结构化文件。"""

    values = list(records)
    if not values:
        raise ValueError("不能导出空的零度面记录")
    source = _validate_provenance(provenance)
    organs = list(dict.fromkeys(record.organ for record in values))
    organ_ids = {organ: index for index, organ in enumerate(organs)}
    counts = Counter(record.organ for record in values)
    destination = Path(output_directory)
    _atomic_text(destination / "sampling_points.ply", _points_ply(values, organ_ids))
    _atomic_text(destination / "zero_planes_edges.ply", _edges_ply(values, organ_ids))
    _atomic_text(destination / "zero_planes_faces.ply", _faces_ply(values, organ_ids))
    _atomic_text(destination / "sampling_points_zero_planes.csv", _csv_text(values))
    payload = {
        "schema_version": "zero-plane-visualization/v1",
        "coordinate_system": "RAS",
        "unit": "mm",
        "record_count": len(values),
        "organ_counts": dict(sorted(counts.items())),
        "organ_colors_rgb": {organ: list(_color(organ)) for organ in organs},
        "provenance": source,
        "records": [_record_payload(record) for record in values],
    }
    _atomic_text(
        destination / "sampling_points_zero_planes.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
