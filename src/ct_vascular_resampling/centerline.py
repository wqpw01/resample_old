"""十二指肠中心线路径与局部切向。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import trimesh


@dataclass(frozen=True)
class CenterlineSelectionAudit:
    mode: str
    coordinate_system: str
    configured_proximal_ras_mm: tuple[float, float, float] | None
    configured_distal_ras_mm: tuple[float, float, float] | None
    matched_proximal_ras_mm: tuple[float, float, float] | None
    matched_distal_ras_mm: tuple[float, float, float] | None
    proximal_match_error_mm: float | None
    distal_match_error_mm: float | None
    endpoint_match_tolerance_mm: float | None
    path_point_count: int
    path_length_mm: float
    skeleton_point_count: int
    endpoint_count: int
    branchpoint_count: int
    connected_component_count: int
    excluded_node_count: int
    excluded_endpoints_ras_mm: tuple[tuple[float, float, float], ...]
    automatic_terminal_spur_pruning_applied: bool

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "coordinate_system": self.coordinate_system,
            "configured_proximal_ras_mm": (
                list(self.configured_proximal_ras_mm) if self.configured_proximal_ras_mm else None
            ),
            "configured_distal_ras_mm": (
                list(self.configured_distal_ras_mm) if self.configured_distal_ras_mm else None
            ),
            "matched_proximal_ras_mm": list(self.matched_proximal_ras_mm) if self.matched_proximal_ras_mm else None,
            "matched_distal_ras_mm": list(self.matched_distal_ras_mm) if self.matched_distal_ras_mm else None,
            "proximal_match_error_mm": self.proximal_match_error_mm,
            "distal_match_error_mm": self.distal_match_error_mm,
            "endpoint_match_tolerance_mm": self.endpoint_match_tolerance_mm,
            "path_point_count": self.path_point_count,
            "path_length_mm": self.path_length_mm,
            "skeleton_point_count": self.skeleton_point_count,
            "endpoint_count": self.endpoint_count,
            "branchpoint_count": self.branchpoint_count,
            "connected_component_count": self.connected_component_count,
            "excluded_node_count": self.excluded_node_count,
            "excluded_endpoints_ras_mm": [list(point) for point in self.excluded_endpoints_ras_mm],
            "automatic_terminal_spur_pruning_applied": self.automatic_terminal_spur_pruning_applied,
        }


@dataclass(frozen=True)
class CenterlinePath:
    points: np.ndarray
    tangents: np.ndarray
    cumulative_length_mm: np.ndarray
    pruned_terminal_spur_lengths_mm: tuple[float, ...] = ()
    selection_audit: CenterlineSelectionAudit | None = None


def _skeleton_adjacency(indices: np.ndarray) -> list[list[int]]:
    index_by_coordinate = {tuple(value): index for index, value in enumerate(indices)}
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    return [
        [
            index_by_coordinate[candidate]
            for offset in offsets
            if (candidate := tuple(value + offset)) in index_by_coordinate
        ]
        for value in indices
    ]


def _connected_components(adjacency: list[list[int]]) -> tuple[frozenset[int], ...]:
    unvisited = set(range(len(adjacency)))
    components: list[frozenset[int]] = []
    while unvisited:
        component: set[int] = set()
        stack = [next(iter(unvisited))]
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            unvisited.discard(current)
            stack.extend(adjacency[current])
        components.append(frozenset(component))
    return tuple(components)


def select_skeleton_indices_by_endpoint_hints(
    skeleton: np.ndarray,
    skeleton_points_world: np.ndarray,
    *,
    proximal_hint_ras_mm: np.ndarray,
    distal_hint_ras_mm: np.ndarray,
    match_tolerance_mm: float,
) -> tuple[np.ndarray, CenterlineSelectionAudit]:
    """在未剪枝的树状骨架中复现人工确认的 RAS 端点路径。"""

    values = np.asarray(skeleton, dtype=bool)
    if values.ndim != 3 or not np.any(values):
        raise ValueError("人工端点中心线骨架必须是非空三维数组")
    indices = np.argwhere(values)
    world_points = np.asarray(skeleton_points_world, dtype=np.float64)
    proximal_hint = np.asarray(proximal_hint_ras_mm, dtype=np.float64)
    distal_hint = np.asarray(distal_hint_ras_mm, dtype=np.float64)
    if world_points.shape != (len(indices), 3):
        raise ValueError("骨架 RAS 世界坐标必须与骨架点一一对应")
    if proximal_hint.shape != (3,) or distal_hint.shape != (3,):
        raise ValueError("人工中心线近端和远端提示必须是三个 RAS 坐标")
    if not np.all(np.isfinite(world_points)) or not np.all(np.isfinite(proximal_hint)) or not np.all(
        np.isfinite(distal_hint)
    ):
        raise ValueError("人工中心线骨架点和端点提示必须是有限 RAS 坐标")
    tolerance = float(match_tolerance_mm)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("人工中心线端点匹配容差必须是有限正数")

    adjacency = _skeleton_adjacency(indices)
    components = _connected_components(adjacency)
    if len(components) != 1:
        raise ValueError(f"人工端点中心线骨架必须单连通，实际连通分量为 {len(components)}")
    edge_count = sum(len(neighbors) for neighbors in adjacency) // 2
    if edge_count != len(indices) - 1:
        raise ValueError(
            f"人工端点中心线骨架必须是无环树，节点 {len(indices)}、边 {edge_count}"
        )
    endpoints = [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    if len(endpoints) < 2:
        raise ValueError(f"人工端点中心线骨架至少需要 2 个端点，实际为 {len(endpoints)}")
    endpoint_points = world_points[endpoints]

    def match_endpoint(hint: np.ndarray, role: str) -> tuple[int, float]:
        distances = np.linalg.norm(endpoint_points - hint, axis=1)
        order = np.argsort(distances, kind="stable")
        if len(order) > 1 and np.isclose(distances[order[0]], distances[order[1]], rtol=0.0, atol=1e-9):
            raise ValueError(f"{role}提示无法唯一匹配骨架端点")
        error = float(distances[order[0]])
        if error > tolerance + 1e-9:
            raise ValueError(f"{role}提示匹配误差 {error:.6f} mm 超过 {tolerance:.6f} mm")
        return endpoints[int(order[0])], error

    proximal, proximal_error = match_endpoint(proximal_hint, "近端")
    distal, distal_error = match_endpoint(distal_hint, "远端")
    if proximal == distal:
        raise ValueError("人工中心线近端和远端匹配到同一个端点")

    parent: dict[int, int | None] = {proximal: None}
    stack = [proximal]
    while stack and distal not in parent:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in parent:
                parent[neighbor] = current
                stack.append(neighbor)
    if distal not in parent:
        raise ValueError("人工中心线近端与远端之间不存在路径")
    ordered = [distal]
    while ordered[-1] != proximal:
        previous = parent[ordered[-1]]
        if previous is None:
            raise ValueError("人工中心线路径未到达配置近端")
        ordered.append(previous)
    ordered.reverse()
    selected_world = world_points[ordered]
    path_length = float(np.sum(np.linalg.norm(np.diff(selected_world, axis=0), axis=1)))
    excluded_endpoints = tuple(
        sorted(
            (
                tuple(float(value) for value in world_points[index])
                for index in endpoints
                if index not in {proximal, distal}
            )
        )
    )
    audit = CenterlineSelectionAudit(
        mode="manual_endpoint_hints",
        coordinate_system="RAS",
        configured_proximal_ras_mm=tuple(float(value) for value in proximal_hint),
        configured_distal_ras_mm=tuple(float(value) for value in distal_hint),
        matched_proximal_ras_mm=tuple(float(value) for value in world_points[proximal]),
        matched_distal_ras_mm=tuple(float(value) for value in world_points[distal]),
        proximal_match_error_mm=proximal_error,
        distal_match_error_mm=distal_error,
        endpoint_match_tolerance_mm=tolerance,
        path_point_count=len(ordered),
        path_length_mm=path_length,
        skeleton_point_count=len(indices),
        endpoint_count=len(endpoints),
        branchpoint_count=sum(len(neighbors) > 2 for neighbors in adjacency),
        connected_component_count=len(components),
        excluded_node_count=len(indices) - len(ordered),
        excluded_endpoints_ras_mm=excluded_endpoints,
        automatic_terminal_spur_pruning_applied=False,
    )
    return indices[ordered], audit


def _order_skeleton_indices(
    skeleton: np.ndarray,
    proximal_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float,
    max_terminal_spur_mm: float,
) -> tuple[np.ndarray, tuple[float, ...]]:
    values = np.asarray(skeleton, dtype=bool)
    references = np.asarray(proximal_reference_points, dtype=np.float64)
    if values.ndim != 3 or not np.any(values):
        raise ValueError("中心线骨架必须是非空三维数组")
    if references.ndim != 2 or references.shape[1] != 3 or len(references) == 0:
        raise ValueError("近端参考点必须是非空 N×3 数组")
    if voxel_pitch_mm <= 0.0 or max_terminal_spur_mm < 0.0:
        raise ValueError("体素间距必须大于零且毛刺阈值不能为负数")
    indices = np.argwhere(values)
    adjacency = _skeleton_adjacency(indices)
    active = set(range(len(indices)))

    def neighbors(index: int) -> list[int]:
        return [neighbor for neighbor in adjacency[index] if neighbor in active]

    visited: set[int] = set()
    stack = [0]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(adjacency[current])
    if len(visited) != len(indices):
        raise ValueError("中心线骨架不是单连通路径")
    pruned_lengths: list[float] = []
    while any(len(neighbors(index)) > 2 for index in active):
        terminal_branches: list[tuple[float, tuple[int, ...]]] = []
        endpoints = [index for index in active if len(neighbors(index)) == 1]
        for endpoint in endpoints:
            branch = [endpoint]
            previous: int | None = None
            current = endpoint
            length = 0.0
            while True:
                following = [neighbor for neighbor in neighbors(current) if neighbor != previous]
                if not following:
                    break
                if len(following) != 1:
                    break
                next_index = following[0]
                length += float(np.linalg.norm(indices[next_index] - indices[current]) * voxel_pitch_mm)
                if len(neighbors(next_index)) > 2:
                    terminal_branches.append((length, tuple(branch)))
                    break
                branch.append(next_index)
                previous, current = current, next_index
        if not terminal_branches:
            raise ValueError("中心线骨架包含无法清理的分叉")
        terminal_branches.sort(key=lambda value: value[0])
        shortest_length, shortest_branch = terminal_branches[0]
        if shortest_length > max_terminal_spur_mm + 1e-9:
            raise ValueError(
                f"中心线骨架包含分叉，最短终末支 {shortest_length:.6f} mm 超过 {max_terminal_spur_mm:.6f} mm"
            )
        if len(terminal_branches) > 1 and np.isclose(
            terminal_branches[1][0], shortest_length, rtol=0.0, atol=1e-9
        ):
            raise ValueError("中心线骨架包含无法唯一判定的等长短分叉")
        active.difference_update(shortest_branch)
        pruned_lengths.append(shortest_length)
    endpoints = [index for index in active if len(neighbors(index)) == 1]
    if len(endpoints) != 2:
        raise ValueError(f"中心线骨架必须恰好有 2 个端点，实际为 {len(endpoints)}")
    endpoint_points = indices[endpoints].astype(np.float64)
    distances, _ = cKDTree(references).query(endpoint_points, k=1)
    current = endpoints[int(np.argmin(distances))]
    ordered: list[int] = []
    previous: int | None = None
    while True:
        ordered.append(current)
        following = [neighbor for neighbor in neighbors(current) if neighbor != previous]
        if not following:
            break
        if len(following) != 1:
            raise ValueError("中心线骨架包含分叉")
        previous, current = current, following[0]
    if len(ordered) != len(active):
        raise ValueError("中心线骨架路径未覆盖全部体素")
    return indices[ordered], tuple(pruned_lengths)


def order_skeleton_indices(
    skeleton: np.ndarray,
    proximal_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float = 1.0,
    max_terminal_spur_mm: float = 5.0,
) -> np.ndarray:
    ordered, _ = _order_skeleton_indices(
        skeleton,
        proximal_reference_points,
        voxel_pitch_mm=voxel_pitch_mm,
        max_terminal_spur_mm=max_terminal_spur_mm,
    )
    return ordered


def _cumulative_lengths(points: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segments < 1e-8):
        raise ValueError("中心线路径包含重复点")
    return np.concatenate([[0.0], np.cumsum(segments)])


def _point_at_arclength(points: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    target = float(np.clip(distance, 0.0, cumulative[-1]))
    upper = int(np.searchsorted(cumulative, target, side="right"))
    if upper == 0:
        return points[0]
    if upper >= len(points):
        return points[-1]
    lower = upper - 1
    fraction = (target - cumulative[lower]) / (cumulative[upper] - cumulative[lower])
    return points[lower] + fraction * (points[upper] - points[lower])


def centerline_tangents(points: np.ndarray, window_mm: float) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("中心线路径必须是至少两个点的 N×3 数组")
    if window_mm <= 0.0:
        raise ValueError("window_mm 必须大于零")
    cumulative = _cumulative_lengths(values)
    if cumulative[-1] < window_mm:
        raise ValueError("中心线长度小于切向估计窗口")
    half_window = window_mm / 2.0
    tangents: list[np.ndarray] = []
    for distance in cumulative:
        if distance < half_window:
            left_distance, right_distance = 0.0, window_mm
        elif cumulative[-1] - distance < half_window:
            left_distance, right_distance = cumulative[-1] - window_mm, cumulative[-1]
        else:
            left_distance, right_distance = distance - half_window, distance + half_window
        vector = _point_at_arclength(values, cumulative, right_distance) - _point_at_arclength(
            values, cumulative, left_distance
        )
        magnitude = float(np.linalg.norm(vector))
        if magnitude < 1e-8:
            raise ValueError("中心线切向量退化")
        tangents.append(vector / magnitude)
    return np.asarray(tangents, dtype=np.float64)


def extract_duodenum_centerline(
    mesh: trimesh.Trimesh,
    stomach_reference_points: np.ndarray,
    *,
    voxel_pitch_mm: float,
    tangent_window_mm: float,
    max_terminal_spur_mm: float = 5.0,
    endpoint_hints_ras_mm: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    endpoint_match_tolerance_mm: float = 1.0,
) -> CenterlinePath:
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("十二指肠必须是非空三角网格")
    if voxel_pitch_mm <= 0.0:
        raise ValueError("voxel_pitch_mm 必须大于零")
    references = np.asarray(stomach_reference_points, dtype=np.float64)
    if references.ndim != 2 or references.shape[1] != 3 or len(references) == 0:
        raise ValueError("胃参考点必须是非空 N×3 数组")
    voxels = mesh.voxelized(pitch=voxel_pitch_mm).fill()
    skeleton = skeletonize(np.asarray(voxels.matrix, dtype=bool))
    selection_audit = None
    if endpoint_hints_ras_mm is None:
        reference_indices = voxels.points_to_indices(references)
        ordered_indices, pruned_lengths = _order_skeleton_indices(
            skeleton,
            reference_indices,
            voxel_pitch_mm=voxel_pitch_mm,
            max_terminal_spur_mm=max_terminal_spur_mm,
        )
    else:
        skeleton_indices = np.argwhere(skeleton)
        skeleton_points_world = np.asarray(voxels.indices_to_points(skeleton_indices), dtype=np.float64)
        ordered_indices, selection_audit = select_skeleton_indices_by_endpoint_hints(
            skeleton,
            skeleton_points_world,
            proximal_hint_ras_mm=np.asarray(endpoint_hints_ras_mm[0], dtype=np.float64),
            distal_hint_ras_mm=np.asarray(endpoint_hints_ras_mm[1], dtype=np.float64),
            match_tolerance_mm=endpoint_match_tolerance_mm,
        )
        pruned_lengths = ()
    points = np.asarray(voxels.indices_to_points(ordered_indices), dtype=np.float64)
    cumulative = _cumulative_lengths(points)
    tangents = centerline_tangents(points, tangent_window_mm)
    return CenterlinePath(points, tangents, cumulative, pruned_lengths, selection_audit)
