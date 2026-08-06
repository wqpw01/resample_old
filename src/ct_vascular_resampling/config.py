"""病例 YAML 配置与固定算法默认值。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_ORGAN_IDS = (
    "adrenal_gland_left",
    "adrenal_gland_right",
    "aorta",
    "duodenum",
    "esophagus",
    "gallbladder",
    "inferior_vena_cava",
    "kidney_left",
    "kidney_right",
    "liver",
    "pancreas",
    "portal_vein_and_splenic_vein",
    "spleen",
    "stomach",
)
DEFAULT_POINT_COUNTS = {
    "stomach": 1000,
    "liver": 500,
    "pancreas": 500,
    "duodenum_part1": 500,
    "duodenum_part2": 500,
    "esophagus": 200,
}
DEFAULT_SQUARE_SPECS = {
    "stomach": (False, ("x", "y", "z")),
    "liver": (True, ("x", "y", "z")),
    "pancreas": (True, ("x", "y", "z")),
    "duodenum": (False, ("x", "y", "z")),
    "esophagus": (False, ("x",)),
}
VALID_VESSEL_LABEL_PAIRS = (frozenset({"portal", "hepatic"}), frozenset({"artery", "vein"}))
DEFAULT_VESSEL_COLORS = {
    "portal": (255, 0, 255),
    "hepatic": (0, 188, 212),
    "artery": (255, 82, 0),
    "vein": (0, 188, 212),
}
ORGAN_BOUNDARY_IDS = (
    "adrenal_gland_left",
    "adrenal_gland_right",
    "duodenum",
    "esophagus",
    "gallbladder",
    "kidney_left",
    "kidney_right",
    "liver",
    "pancreas",
    "spleen",
    "stomach",
)
DEFAULT_ORGAN_COLORS = {
    "adrenal_gland_left": (31, 119, 180),
    "adrenal_gland_right": (174, 199, 232),
    "duodenum": (44, 160, 44),
    "esophagus": (152, 223, 138),
    "gallbladder": (188, 189, 34),
    "kidney_left": (214, 39, 40),
    "kidney_right": (255, 152, 150),
    "liver": (140, 86, 75),
    "pancreas": (148, 103, 189),
    "spleen": (227, 119, 194),
    "stomach": (127, 127, 127),
}


@dataclass(frozen=True)
class VesselModel:
    identifier: str
    path: Path
    label: str
    color: tuple[int, int, int]


@dataclass(frozen=True)
class SamplingConfig:
    point_counts: dict[str, int]
    ray_length_mm: float = 100.0
    minimum_spacing_mm: float = 10.0
    centerline_voxel_pitch_mm: float = 1.0
    centerline_tangent_window_mm: float = 10.0
    centerline_max_terminal_spur_mm: float = 5.0


@dataclass(frozen=True)
class SquareConfig:
    side_length_mm: float = 100.0
    specs: dict[str, tuple[bool, tuple[str, ...]]] | None = None
    deduplicate_degenerate_edge_angles: bool = False

    def spec_for(self, organ: str) -> tuple[bool, tuple[str, ...]]:
        values = (self.specs or DEFAULT_SQUARE_SPECS).get(organ)
        if values is None:
            raise ValueError(f"没有 {organ} 的方形采样配置")
        return values


@dataclass(frozen=True)
class CTConfig:
    output_resolution: int = 300
    window_level: float = 40.0
    window_width: float = 400.0
    fill_hu_value: float = -1000.0


@dataclass(frozen=True)
class FilterConfig:
    black_threshold: int = 50
    black_ratio_limit: float = 0.50
    line_min_diagonal_fraction: float = 0.70
    black_side_min_ratio: float = 0.90
    valid_side_max_black_ratio: float = 0.10


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 0
    workers: int = 8
    backend: str = "auto"
    gpu_device: int = 0
    gpu_batch_size: int = 32


@dataclass(frozen=True)
class GeometryConfig:
    input_coordinate_system: str = "LPS"
    canonical_coordinate_system: str = "RAS"


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    ct_path: Path
    output_root: Path
    organ_models: dict[str, Path]
    vessel_models: tuple[VesselModel, ...]
    registration_module_path: Path
    sampling: SamplingConfig
    square: SquareConfig
    ct: CTConfig
    filtering: FilterConfig
    runtime: RuntimeConfig
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    dicom_series_uid: str | None = None


def _as_path(value: Any, config_directory: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} 必须是非空路径字符串")
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_directory / path


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 YAML 映射")
    return value


def _number(value: Any, field: str, default: float) -> float:
    result = default if value is None else float(value)
    if not result > 0:
        raise ValueError(f"{field} 必须大于零")
    return result


def _integer(value: Any, field: str, default: int, minimum: int = 1) -> int:
    result = default if value is None else int(value)
    if result < minimum:
        raise ValueError(f"{field} 必须不小于 {minimum}")
    return result


def _ratio(value: Any, field: str, default: float, allow_zero: bool = True) -> float:
    result = default if value is None else float(value)
    lower_valid = result >= 0.0 if allow_zero else result > 0.0
    if not lower_valid or result > 1.0:
        raise ValueError(f"{field} 必须在 0-1 内")
    return result


def _boolean(value: Any, field: str, default: bool) -> bool:
    result = default if value is None else value
    if not isinstance(result, bool):
        raise ValueError(f"{field} 必须是布尔值")
    return result


def _color(value: Any, identifier: str) -> tuple[int, int, int]:
    if value is None:
        return DEFAULT_VESSEL_COLORS[identifier]
    if not isinstance(value, list) or len(value) != 3 or any(not isinstance(item, int) or item < 0 or item > 255 for item in value):
        raise ValueError(f"血管模型 {identifier} 的 color 必须是三个 0-255 整数")
    return tuple(value)


def _load_vessels(raw: Any, config_directory: Path) -> tuple[VesselModel, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("vessel_models 必须是非空 YAML 列表")
    vessels: list[VesselModel] = []
    labels: set[str] = set()
    ids: set[str] = set()
    for item in raw:
        values = _mapping(item, "vessel_models 项")
        identifier = values.get("id")
        label = values.get("label")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("每个 vessel_models 项必须提供 id")
        if identifier in ids:
            raise ValueError(f"血管模型 id 重复: {identifier}")
        if label not in DEFAULT_VESSEL_COLORS:
            raise ValueError(f"血管模型 {identifier} 的 label 必须是 portal、hepatic、artery 或 vein")
        if label in labels:
            raise ValueError(f"血管标签重复: {label}")
        ids.add(identifier)
        labels.add(label)
        vessels.append(
            VesselModel(
                identifier=identifier,
                path=_as_path(values.get("path"), config_directory, f"vessel_models.{identifier}.path"),
                label=label,
                color=_color(values.get("color"), label),
            )
        )
    if frozenset(labels) not in VALID_VESSEL_LABEL_PAIRS:
        raise ValueError("vessel_models 必须同时提供 portal/hepatic 或 artery/vein 网格")
    return tuple(vessels)


def _load_sampling(raw: Any) -> SamplingConfig:
    values = _mapping(raw or {}, "sampling")
    counts = dict(DEFAULT_POINT_COUNTS)
    supplied = values.get("point_counts", {})
    if not isinstance(supplied, dict):
        raise ValueError("sampling.point_counts 必须是 YAML 映射")
    for name, value in supplied.items():
        if name not in counts:
            raise ValueError(f"不支持的采样点数量配置: {name}")
        counts[name] = _integer(value, f"sampling.point_counts.{name}", counts[name])
    return SamplingConfig(
        point_counts=counts,
        ray_length_mm=_number(values.get("ray_length_mm"), "sampling.ray_length_mm", 100.0),
        minimum_spacing_mm=_number(values.get("minimum_spacing_mm"), "sampling.minimum_spacing_mm", 10.0),
        centerline_voxel_pitch_mm=_number(
            values.get("centerline_voxel_pitch_mm"), "sampling.centerline_voxel_pitch_mm", 1.0
        ),
        centerline_tangent_window_mm=_number(
            values.get("centerline_tangent_window_mm"), "sampling.centerline_tangent_window_mm", 10.0
        ),
        centerline_max_terminal_spur_mm=_number(
            values.get("centerline_max_terminal_spur_mm"), "sampling.centerline_max_terminal_spur_mm", 5.0
        ),
    )


def _load_geometry(raw: Any) -> GeometryConfig:
    values = _mapping(raw or {}, "geometry")
    input_coordinate_system = str(values.get("input_coordinate_system", "LPS")).upper()
    canonical_coordinate_system = str(values.get("canonical_coordinate_system", "RAS")).upper()
    if input_coordinate_system not in {"LPS", "RAS"}:
        raise ValueError("geometry.input_coordinate_system 必须是 LPS 或 RAS")
    if canonical_coordinate_system != "RAS":
        raise ValueError("geometry.canonical_coordinate_system 必须是 RAS")
    return GeometryConfig(input_coordinate_system, canonical_coordinate_system)


def _load_square(raw: Any) -> SquareConfig:
    values = _mapping(raw or {}, "square")
    side_length = _number(values.get("side_length_mm"), "square.side_length_mm", 100.0)
    return SquareConfig(
        side_length_mm=side_length,
        specs=dict(DEFAULT_SQUARE_SPECS),
        deduplicate_degenerate_edge_angles=_boolean(
            values.get("deduplicate_degenerate_edge_angles"),
            "square.deduplicate_degenerate_edge_angles",
            False,
        ),
    )


def _load_ct(raw: Any) -> CTConfig:
    values = _mapping(raw or {}, "ct")
    return CTConfig(
        output_resolution=_integer(values.get("output_resolution"), "ct.output_resolution", 300),
        window_level=float(values.get("window_level", 40.0)),
        window_width=_number(values.get("window_width"), "ct.window_width", 400.0),
        fill_hu_value=float(values.get("fill_hu_value", -1000.0)),
    )


def _load_filter(raw: Any) -> FilterConfig:
    values = _mapping(raw or {}, "filtering")
    black_threshold = _integer(values.get("black_threshold"), "filtering.black_threshold", 50, minimum=0)
    if black_threshold > 255:
        raise ValueError("filtering.black_threshold 必须在 0-255 内")
    return FilterConfig(
        black_threshold=black_threshold,
        black_ratio_limit=_ratio(values.get("black_ratio_limit"), "filtering.black_ratio_limit", 0.50),
        line_min_diagonal_fraction=_ratio(
            values.get("line_min_diagonal_fraction"), "filtering.line_min_diagonal_fraction", 0.70, allow_zero=False
        ),
        black_side_min_ratio=_ratio(values.get("black_side_min_ratio"), "filtering.black_side_min_ratio", 0.90),
        valid_side_max_black_ratio=_ratio(
            values.get("valid_side_max_black_ratio"), "filtering.valid_side_max_black_ratio", 0.10),
    )


def _load_runtime(raw: Any) -> RuntimeConfig:
    values = _mapping(raw or {}, "runtime")
    backend = values.get("backend", "auto")
    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("runtime.backend 必须是 auto、gpu 或 cpu")
    return RuntimeConfig(
        seed=int(values.get("seed", 0)),
        workers=_integer(values.get("workers"), "runtime.workers", 8),
        backend=backend,
        gpu_device=_integer(values.get("gpu_device"), "runtime.gpu_device", 0, minimum=0),
        gpu_batch_size=_integer(values.get("gpu_batch_size"), "runtime.gpu_batch_size", 32),
    )


def load_case_config(path: str | Path) -> CaseConfig:
    """读取一个无 P/N/D 的 CT 血管重采样病例。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = _mapping(raw, "病例配置")
    config_directory = config_path.parent
    case_id = values.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id 必须是非空字符串")
    organs_raw = _mapping(values.get("organ_models"), "organ_models")
    missing = sorted(set(REQUIRED_ORGAN_IDS) - set(organs_raw))
    if missing:
        raise ValueError(f"organ_models 缺少源算法必需模型: {', '.join(missing)}")
    organ_models = {
        name: _as_path(organs_raw[name], config_directory, f"organ_models.{name}") for name in REQUIRED_ORGAN_IDS
    }
    dicom_series_uid = values.get("dicom_series_uid")
    if dicom_series_uid is not None and (not isinstance(dicom_series_uid, str) or not dicom_series_uid):
        raise ValueError("dicom_series_uid 必须是非空字符串")
    return CaseConfig(
        case_id=case_id,
        ct_path=_as_path(values.get("ct_path"), config_directory, "ct_path"),
        output_root=_as_path(values.get("output_root"), config_directory, "output_root"),
        organ_models=organ_models,
        vessel_models=_load_vessels(values.get("vessel_models"), config_directory),
        registration_module_path=_as_path(values.get("registration_module_path"), config_directory, "registration_module_path"),
        sampling=_load_sampling(values.get("sampling")),
        square=_load_square(values.get("square")),
        ct=_load_ct(values.get("ct")),
        filtering=_load_filter(values.get("filtering")),
        runtime=_load_runtime(values.get("runtime")),
        geometry=_load_geometry(values.get("geometry")),
        dicom_series_uid=dicom_series_uid,
    )
