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
import shutil
from typing import Any
import hashlib

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

TARGET_ORGAN_COUNTS: dict[str, int] = {
    "stomach": 118,
    "liver": 162,
    "pancreas": 37,
    "duodenum": 53,
    "esophagus": 30,
}

LEGACY_ORGAN_NAMES: dict[str, str] = {
    "stomach": "Stomach",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "duodenum": "Duodenum",
    "esophagus": "Esophagus",
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


def _mesh_arrays(mesh: Any, maximum_faces: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("器官网格顶点必须是有限的 Nx3 数组")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("器官网格必须由三角面组成")
    if len(faces) > maximum_faces:
        generator = np.random.default_rng(seed)
        faces = faces[np.sort(generator.choice(len(faces), maximum_faces, replace=False))]
    return vertices, faces


def _css_color(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _plane_mesh(records: list[ZeroPlaneRecord]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.concatenate([record.vertices for record in records], axis=0)
    faces: list[tuple[int, int, int]] = []
    for index in range(len(records)):
        offset = index * 4
        faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _line_coordinates(segments: Iterable[np.ndarray]) -> tuple[list[float | None], ...]:
    x_values: list[float | None] = []
    y_values: list[float | None] = []
    z_values: list[float | None] = []
    for segment in segments:
        for point in segment:
            x_values.append(float(point[0]))
            y_values.append(float(point[1]))
            z_values.append(float(point[2]))
        x_values.append(None)
        y_values.append(None)
        z_values.append(None)
    return x_values, y_values, z_values


def render_interactive_html(
    records: Iterable[ZeroPlaneRecord],
    organ_meshes: Mapping[str, Any],
    output_path: str | Path,
) -> None:
    """生成内嵌 Plotly 的离线三维交互页面。"""

    import plotly.graph_objects as go
    import plotly.io as pio

    values = list(records)
    if not values:
        raise ValueError("不能渲染空的零度面记录")
    organs = list(dict.fromkeys(record.organ for record in values))
    if set(organ_meshes) != set(organs):
        raise ValueError("器官网格集合必须与零度面器官集合完全一致")

    figure = go.Figure()
    roles: list[str] = []
    axis_styles = (
        ("x", (214, 39, 40)),
        ("y", (44, 160, 44)),
        ("z", (31, 119, 180)),
    )
    for organ_index, organ in enumerate(organs):
        organ_records = [record for record in values if record.organ == organ]
        color = _color(organ)
        vertices, faces = _mesh_arrays(organ_meshes[organ], 12_000, organ_index + 20260815)
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=f"{organ} organ",
                legendgroup=organ,
                legendgrouptitle_text=organ,
                color=_css_color(color),
                opacity=0.12,
                hoverinfo="skip",
                showlegend=True,
            )
        )
        roles.append("organ_mesh")

        probes = np.asarray([record.probe for record in organ_records])
        hover = [
            f"{record.slice_id}<br>RAS: {record.probe[0]:.2f}, {record.probe[1]:.2f}, {record.probe[2]:.2f} mm"
            for record in organ_records
        ]
        figure.add_trace(
            go.Scatter3d(
                x=probes[:, 0],
                y=probes[:, 1],
                z=probes[:, 2],
                mode="markers",
                marker={"size": 4.5, "color": _css_color(color), "line": {"width": 0.5, "color": "black"}},
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name=f"{organ} sample points",
                legendgroup=organ,
                showlegend=True,
            )
        )
        roles.append("points")

        plane_vertices, plane_faces = _plane_mesh(organ_records)
        figure.add_trace(
            go.Mesh3d(
                x=plane_vertices[:, 0],
                y=plane_vertices[:, 1],
                z=plane_vertices[:, 2],
                i=plane_faces[:, 0],
                j=plane_faces[:, 1],
                k=plane_faces[:, 2],
                name=f"{organ} zero planes",
                legendgroup=organ,
                color=_css_color(color),
                opacity=0.055,
                hoverinfo="skip",
                showlegend=True,
            )
        )
        roles.append("planes")

        edges = [
            record.vertices[[first, second]]
            for record in organ_records
            for first, second in ((0, 1), (1, 2), (2, 3), (3, 0))
        ]
        edge_x, edge_y, edge_z = _line_coordinates(edges)
        figure.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": _css_color(color), "width": 1.2},
                opacity=0.32,
                hoverinfo="skip",
                legendgroup=organ,
                showlegend=False,
            )
        )
        roles.append("planes")

        for axis_name, axis_color in axis_styles:
            segments = []
            for record in organ_records:
                axis = getattr(record, f"{axis_name}_axis")
                segments.append(np.vstack((record.probe, record.probe + axis * 8.0)))
            axis_x, axis_y, axis_z = _line_coordinates(segments)
            figure.add_trace(
                go.Scatter3d(
                    x=axis_x,
                    y=axis_y,
                    z=axis_z,
                    mode="lines",
                    line={"color": _css_color(axis_color), "width": 2.0},
                    opacity=0.65,
                    hoverinfo="skip",
                    legendgroup=organ,
                    showlegend=False,
                )
            )
            roles.append("axes")

    def visibility(visible_roles: set[str]) -> list[bool]:
        return [role in visible_roles for role in roles]

    buttons = [
        ("All", {"organ_mesh", "points", "planes", "axes"}),
        ("Points only", {"points"}),
        ("Points + zero planes", {"points", "planes", "axes"}),
        ("Hide organ meshes", {"points", "planes", "axes"}),
    ]
    figure.update_layout(
        title={"text": "Case 2 Sampling Points And Zero-Degree Reference Planes", "x": 0.5},
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "R (+) / L (-) [mm]",
            "yaxis_title": "A (+) / P (-) [mm]",
            "zaxis_title": "S (+) / I (-) [mm]",
            "camera": {"eye": {"x": 1.45, "y": -1.55, "z": 1.05}},
        },
        legend={"groupclick": "togglegroup", "itemsizing": "constant"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.08,
                "buttons": [
                    {"label": label, "method": "update", "args": [{"visible": visibility(role_set)}]}
                    for label, role_set in buttons
                ],
            }
        ],
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        pio.write_html(
            figure,
            file=str(temporary),
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
            config={"displaylogo": False, "responsive": True},
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _world_bounds(
    records: list[ZeroPlaneRecord], organ_meshes: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.concatenate([record.vertices for record in records], axis=0)]
    arrays.extend(np.asarray(mesh.vertices, dtype=np.float64) for mesh in organ_meshes.values())
    all_points = np.concatenate(arrays, axis=0)
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.max(upper - lower)) / 2.0
    return center - radius, center + radius


def render_static_views(
    records: Iterable[ZeroPlaneRecord],
    organ_meshes: Mapping[str, Any],
    output_directory: str | Path,
) -> None:
    """生成共享世界坐标范围的等距、轴位、冠状位和矢状位 PNG。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    values = list(records)
    if not values:
        raise ValueError("不能渲染空的零度面记录")
    organs = list(dict.fromkeys(record.organ for record in values))
    if set(organ_meshes) != set(organs):
        raise ValueError("器官网格集合必须与零度面器官集合完全一致")
    lower, upper = _world_bounds(values, organ_meshes)
    views = {
        "isometric": (25.0, -55.0),
        "axial": (90.0, -90.0),
        "coronal": (0.0, -90.0),
        "sagittal": (0.0, 0.0),
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    for view_name, (elevation, azimuth) in views.items():
        figure = plt.figure(figsize=(12, 9), dpi=140, facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        for organ_index, organ in enumerate(organs):
            color = np.asarray(_color(organ), dtype=np.float64) / 255.0
            mesh_vertices, mesh_faces = _mesh_arrays(
                organ_meshes[organ], 5_000, organ_index + 20260815
            )
            axis.add_collection3d(
                Poly3DCollection(
                    mesh_vertices[mesh_faces],
                    facecolors=[(*color, 0.075)],
                    edgecolors="none",
                )
            )
            organ_records = [record for record in values if record.organ == organ]
            axis.add_collection3d(
                Poly3DCollection(
                    [record.vertices for record in organ_records],
                    facecolors=[(*color, 0.022)],
                    edgecolors=[(*color, 0.16)],
                    linewidths=0.35,
                )
            )
            probes = np.asarray([record.probe for record in organ_records])
            axis.scatter(
                probes[:, 0],
                probes[:, 1],
                probes[:, 2],
                s=14,
                c=[color],
                edgecolors="black",
                linewidths=0.25,
                depthshade=False,
            )
            for axis_name, axis_color in (
                ("x", (0.84, 0.15, 0.16, 0.38)),
                ("y", (0.17, 0.63, 0.17, 0.38)),
                ("z", (0.12, 0.47, 0.71, 0.38)),
            ):
                segments = [
                    np.vstack(
                        (
                            record.probe,
                            record.probe + getattr(record, f"{axis_name}_axis") * 8.0,
                        )
                    )
                    for record in organ_records
                ]
                axis.add_collection3d(
                    Line3DCollection(segments, colors=[axis_color], linewidths=0.55)
                )

        axis.set_xlim(float(lower[0]), float(upper[0]))
        axis.set_ylim(float(lower[1]), float(upper[1]))
        axis.set_zlim(float(lower[2]), float(upper[2]))
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_xlabel("R (+) / L (-) [mm]")
        axis.set_ylabel("A (+) / P (-) [mm]")
        axis.set_zlabel("S (+) / I (-) [mm]")
        axis.set_title(f"Sampling Points And Zero-Degree Planes — {view_name.title()}")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=np.asarray(_color(organ)) / 255.0,
                markeredgecolor="black",
                markersize=7,
                label=organ,
            )
            for organ in organs
        ]
        axis.legend(handles=handles, loc="upper right", framealpha=0.92)
        axis.grid(True, linewidth=0.35, alpha=0.35)
        figure.tight_layout()
        output = destination / f"sampling_points_zero_planes_{view_name}.png"
        temporary = output.with_name(f".{output.name}.tmp.png")
        try:
            figure.savefig(temporary, format="png", bbox_inches="tight", facecolor="white")
            os.replace(temporary, output)
        finally:
            plt.close(figure)
            temporary.unlink(missing_ok=True)


def load_zero_record_jsonl(path: str | Path) -> list[dict[str, object]]:
    """读取服务器端已经筛选好的零度记录 JSONL。"""

    source = Path(path)
    records: list[dict[str, object]] = []
    try:
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"第 {line_number} 行不是 JSON 对象")
                records.append(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"零度记录 JSONL 第 {error.lineno} 行损坏: {error.msg}") from error
    if not records:
        raise ValueError("零度记录 JSONL 为空")
    return records


def read_surface_samples_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """读取旧流程 FPS ASCII PLY 中的点和法向量。"""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError(f"{source} 不是 ASCII PLY 1.0")
        vertex_count: int | None = None
        properties: list[str] = []
        in_vertex = False
        for raw_line in stream:
            line = raw_line.strip()
            if line == "end_header":
                break
            if line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) == 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"{source} 的 vertex property 无效")
                properties.append(parts[2])
        else:
            raise ValueError(f"{source} 缺少 end_header")
        required = ("x", "y", "z", "nx", "ny", "nz")
        if vertex_count is None or any(name not in properties for name in required):
            raise ValueError(f"{source} 缺少点数或 xyz/nxyz 属性")
        indices = [properties.index(name) for name in required]
        values: list[list[float]] = []
        for row_index in range(vertex_count):
            line = stream.readline()
            if not line:
                raise ValueError(f"{source} 的顶点数据少于声明的 {vertex_count} 行")
            parts = line.split()
            if len(parts) < len(properties):
                raise ValueError(f"{source} 第 {row_index + 1} 个顶点字段不足")
            values.append([float(parts[index]) for index in indices])
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (vertex_count, 6) or not np.all(np.isfinite(array)):
        raise ValueError(f"{source} 包含无效点或法向量")
    return array[:, :3], array[:, 3:]


def _validate_surface_samples(
    records: list[ZeroPlaneRecord], sample_ply_directory: Path
) -> None:
    for organ in TARGET_ORGAN_COUNTS:
        organ_records = [record for record in records if record.organ == organ]
        source = sample_ply_directory / f"FPS-{LEGACY_ORGAN_NAMES[organ]}.ply"
        if not source.is_file():
            raise ValueError(f"缺少采样点 PLY: {source}")
        points, normals = read_surface_samples_ply(source)
        expected_points = np.asarray([record.probe for record in organ_records])
        expected_normals = np.asarray([record.input_normal for record in organ_records])
        if points.shape != expected_points.shape or not np.allclose(
            points, expected_points, rtol=0.0, atol=POINT_TOLERANCE_MM
        ):
            raise ValueError(f"{organ} 的 FPS 点与零度记录不一致")
        if normals.shape != expected_normals.shape or not np.allclose(
            normals, expected_normals, rtol=0.0, atol=POINT_TOLERANCE_MM
        ):
            raise ValueError(f"{organ} 的 FPS 法向量与零度记录不一致")


def _load_organ_meshes(directory: Path) -> dict[str, Any]:
    import trimesh

    meshes: dict[str, Any] = {}
    for organ in TARGET_ORGAN_COUNTS:
        source = directory / f"{organ}.ply"
        if not source.is_file():
            raise ValueError(f"缺少目标器官网格: {source}")
        loaded = trimesh.load(source, process=False, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            raise ValueError(f"目标器官网格无效: {source}")
        _mesh_arrays(loaded, max(1, len(loaded.faces)), 0)
        meshes[organ] = loaded
    return meshes


def _read_run_metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 run_metadata.json: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("run_metadata.json 顶层必须是对象")
    if value.get("run_state") != "complete":
        raise ValueError("run_metadata.run_state 不是 complete")
    if value.get("total_squares") != 1_431_118:
        raise ValueError("run_metadata.total_squares 不是 1431118")
    return value


def _readme_text(
    records: list[ZeroPlaneRecord], provenance: Mapping[str, str]
) -> str:
    counts = Counter(record.organ for record in records)
    color_lines = [
        f"- {organ}: {counts[organ]} 个，RGB {list(_color(organ))}"
        for organ in TARGET_ORGAN_COUNTS
    ]
    return "\n".join(
        [
            "病例 2 采样点与零度基准面可视化",
            "=================================",
            "",
            "坐标与几何",
            "----------",
            "- 坐标系：RAS 患者世界坐标；单位：毫米。",
            "- 零度面直接来自正式 manifest 中 roll=0、pitch=0、yaw=0 的记录。",
            "- 每个切面为 100 mm x 100 mm；探头/采样点位于方形底边中点。",
            "- 局部 +x 指向切面深度，+y 沿底边，+z=x×y，满足右手关系。",
            "- squarePLY 与 rectangles.ply 包含全旋转切面，本交付未将其误作零度面。",
            "",
            "器官颜色与数量",
            "--------------",
            *color_lines,
            "",
            "文件说明",
            "--------",
            "- sampling_points_zero_planes_interactive.html：离线交互三维视图。",
            "- sampling_points_zero_planes_*.png：等距、轴位、冠状位和矢状位视图。",
            "- sampling_points.ply：400 个采样点，含输入法向量、RGB 和器官编号。",
            "- zero_planes_edges.ply：400 个零度面的边框。",
            "- zero_planes_faces.ply：400 个零度面的四边形面。",
            "- sampling_points_zero_planes.csv/json：完整点、局部轴和四顶点数据。",
            "- target_organ_meshes：五个目标器官网格，作为空间参照。",
            "",
            "打开方式",
            "--------",
            "- HTML 可直接用浏览器离线打开；右上角图例可按器官显隐。",
            "- PLY 可在 3D Slicer、MeshLab 或 CloudCompare 中打开。",
            "- 在 3D Slicer 中必须保持 RAS 世界坐标，不要自动居中各个文件。",
            "",
            "来源追溯",
            "--------",
            f"- 根 manifest SHA-256：{provenance['source_manifest_sha256']}",
            f"- 核心设计 SHA-256：{provenance['core_design_sha256']}",
            f"- 生成重采样结果的 Git commit：{provenance['build_git_commit']}",
            "",
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_sha256_manifest(directory: Path) -> None:
    output = directory / "SHA256SUMS.txt"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(directory).as_posix()}"
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != output
    ]
    _atomic_text(output, "\n".join(lines) + "\n")


def export_visualization_bundle(
    *,
    zero_records_jsonl: str | Path,
    sample_ply_directory: str | Path,
    organ_mesh_directory: str | Path,
    run_metadata_path: str | Path,
    source_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, object]:
    """验证正式结果输入并一次性发布完整本地可视化交付目录。"""

    destination = Path(output_directory).resolve()
    if destination.exists():
        raise ValueError(f"输出目录已经存在，为避免覆盖请先确认: {destination}")
    staging = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        raw_records = load_zero_record_jsonl(zero_records_jsonl)
        records = select_zero_planes(raw_records, TARGET_ORGAN_COUNTS)
        _validate_surface_samples(records, Path(sample_ply_directory))
        metadata = _read_run_metadata(Path(run_metadata_path))
        provenance = _validate_provenance(
            {
                "source_manifest_sha256": source_manifest_sha256,
                "core_design_sha256": metadata.get("core_design_sha256"),
                "build_git_commit": metadata.get("build_git_commit"),
            }
        )
        meshes = _load_organ_meshes(Path(organ_mesh_directory))
        staging.mkdir(parents=True)
        write_structured_exports(records, staging, provenance)
        render_interactive_html(
            records, meshes, staging / "sampling_points_zero_planes_interactive.html"
        )
        render_static_views(records, meshes, staging)
        mesh_output = staging / "target_organ_meshes"
        mesh_output.mkdir()
        for organ in TARGET_ORGAN_COUNTS:
            shutil.copy2(Path(organ_mesh_directory) / f"{organ}.ply", mesh_output / f"{organ}.ply")
        _atomic_text(staging / "README_中文.txt", _readme_text(records, provenance))
        _write_sha256_manifest(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_directory": str(destination),
        "record_count": len(records),
        "organ_counts": dict(TARGET_ORGAN_COUNTS),
        "source_manifest_sha256": provenance["source_manifest_sha256"],
    }
