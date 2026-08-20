"""病例 YAML 配置与固定算法默认值。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

from .contract import (
    BLACK_RATIO_LIMIT,
    BLACK_SIDE_MIN_RATIO,
    BLACK_THRESHOLD,
    CENTERLINE_MAX_TERMINAL_SPUR_MM,
    CENTERLINE_TANGENT_WINDOW_MM,
    CENTERLINE_VOXEL_PITCH_MM,
    FILL_HU_VALUE,
    LINE_MIN_DIAGONAL_FRACTION,
    MINIMUM_POINT_SPACING_MM,
    OUTPUT_RESOLUTION,
    RAY_BATCH_SIZE,
    RAY_LENGTH_MM,
    SAMPLING_SEED,
    SQUARE_SIDE_LENGTH_MM,
    VALID_SIDE_MAX_BLACK_RATIO,
    WINDOW_LEVEL_HU,
    WINDOW_WIDTH_HU,
)


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
VALID_VESSEL_LABEL_PAIR = frozenset({"artery", "vein"})
DEFAULT_VESSEL_COLORS = {
    "artery": (255, 82, 0),
    "vein": (0, 188, 212),
}
ORGAN_BOUNDARY_MODEL_IDS = {
    "adrenal_gland_left": "adrenal_gland_left",
    "adrenal_gland_right": "adrenal_gland_right",
    "aorta": "aorta",
    "duodenum": "duodenum",
    "esophagus": "esophagus",
    "gallbladder": "gallbladder",
    "inferior_vena_cava": "inferior_vena_cava",
    "kidney_left": "kidney_left",
    "kidney_right": "kidney_right",
    "liver": "liver",
    "pancreas": "pancreas",
    "portal_vein": "portal_vein_and_splenic_vein",
    "spleen": "spleen",
    "stomach": "stomach",
}
ORGAN_BOUNDARY_IDS = tuple(ORGAN_BOUNDARY_MODEL_IDS)
EUS_VESSEL_IDS = ("aorta", "inferior_vena_cava", "portal_vein")
DEFAULT_ORGAN_COLORS = {
    "adrenal_gland_left": (31, 119, 180),
    "adrenal_gland_right": (174, 199, 232),
    "aorta": DEFAULT_VESSEL_COLORS["artery"],
    "duodenum": (44, 160, 44),
    "esophagus": (152, 223, 138),
    "gallbladder": (188, 189, 34),
    "inferior_vena_cava": DEFAULT_VESSEL_COLORS["vein"],
    "kidney_left": (214, 39, 40),
    "kidney_right": (255, 152, 150),
    "liver": (140, 86, 75),
    "pancreas": (148, 103, 189),
    "portal_vein": DEFAULT_VESSEL_COLORS["vein"],
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
    ray_length_mm: float = RAY_LENGTH_MM
    ray_batch_size: int = RAY_BATCH_SIZE
    minimum_spacing_mm: float = MINIMUM_POINT_SPACING_MM
    centerline_voxel_pitch_mm: float = CENTERLINE_VOXEL_PITCH_MM
    centerline_tangent_window_mm: float = CENTERLINE_TANGENT_WINDOW_MM
    centerline_max_terminal_spur_mm: float = CENTERLINE_MAX_TERMINAL_SPUR_MM
    duodenum_centerline_endpoint_hints_ras_mm: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    duodenum_centerline_endpoint_match_tolerance_mm: float = 1.0


@dataclass(frozen=True)
class SquareConfig:
    side_length_mm: float = SQUARE_SIDE_LENGTH_MM


@dataclass(frozen=True)
class CTConfig:
    output_resolution: int = OUTPUT_RESOLUTION
    window_level: float = WINDOW_LEVEL_HU
    window_width: float = WINDOW_WIDTH_HU
    fill_hu_value: float = FILL_HU_VALUE


@dataclass(frozen=True)
class FilterConfig:
    black_threshold: int = BLACK_THRESHOLD
    black_ratio_limit: float = BLACK_RATIO_LIMIT
    line_min_diagonal_fraction: float = LINE_MIN_DIAGONAL_FRACTION
    black_side_min_ratio: float = BLACK_SIDE_MIN_RATIO
    valid_side_max_black_ratio: float = VALID_SIDE_MAX_BLACK_RATIO


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = SAMPLING_SEED
    workers: int = 8
    backend: str = "auto"
    gpu_device: int = 0
    gpu_batch_size: int = 32


@dataclass(frozen=True)
class GeometryConfig:
    input_coordinate_system: str = "LPS"
    canonical_coordinate_system: str = "RAS"


@dataclass(frozen=True)
class ManualSegmentationConfig:
    path: Path
    organ_label_values: dict[str, tuple[int, ...]]
    eus_vessel_label_values: dict[str, tuple[int, ...]]
    eus_vessel_colors: dict[str, tuple[int, int, int]]


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    ct_path: Path
    output_root: Path
    organ_models: dict[str, Path]
    vessel_models: tuple[VesselModel, ...]
    sampling: SamplingConfig
    square: SquareConfig
    ct: CTConfig
    filtering: FilterConfig
    runtime: RuntimeConfig
    manual_segmentation: ManualSegmentationConfig
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


def _reject_unexpected_keys(
    values: dict[str, Any],
    supported: set[str],
    field: str,
) -> None:
    unexpected = sorted(set(values) - supported)
    if unexpected:
        raise ValueError(f"{field} 包含不支持的配置: {', '.join(unexpected)}")


def _finite_number(value: Any, field: str, default: float) -> float:
    source = default if value is None else value
    if isinstance(source, bool) or not isinstance(source, (int, float)):
        raise ValueError(f"{field} 必须是有限数值")
    result = float(source)
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数值")
    return result


def _number(value: Any, field: str, default: float) -> float:
    result = _finite_number(value, field, default)
    if not result > 0:
        raise ValueError(f"{field} 必须大于零")
    return result


def _finite_vector3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} 必须是三个有限数值")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field} 必须是三个有限数值")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} 必须是三个有限数值")
    return result


def _integer(value: Any, field: str, default: int, minimum: int = 1) -> int:
    source = default if value is None else value
    if isinstance(source, bool) or not isinstance(source, int):
        raise ValueError(f"{field} 必须是整数")
    result = source
    if result < minimum:
        raise ValueError(f"{field} 必须不小于 {minimum}")
    return result


def _ratio(value: Any, field: str, default: float, allow_zero: bool = True) -> float:
    result = _finite_number(value, field, default)
    lower_valid = result >= 0.0 if allow_zero else result > 0.0
    if not lower_valid or result > 1.0:
        raise ValueError(f"{field} 必须在 0-1 内")
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
        _reject_unexpected_keys(values, {"id", "path", "label", "color"}, "vessel_models 项")
        identifier = values.get("id")
        label = values.get("label")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("每个 vessel_models 项必须提供 id")
        if identifier in ids:
            raise ValueError(f"血管模型 id 重复: {identifier}")
        if label not in {"artery", "vein"}:
            raise ValueError(f"血管模型 {identifier} 的 label 必须是 artery 或 vein")
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
    if frozenset(labels) != VALID_VESSEL_LABEL_PAIR:
        raise ValueError("vessel_models 必须同时提供 artery 和 vein 网格")
    return tuple(vessels)


def _load_sampling(raw: Any) -> SamplingConfig:
    values = _mapping(raw or {}, "sampling")
    supported = {
        "point_counts",
        "ray_length_mm",
        "ray_batch_size",
        "minimum_spacing_mm",
        "centerline_voxel_pitch_mm",
        "centerline_tangent_window_mm",
        "centerline_max_terminal_spur_mm",
        "duodenum_centerline_endpoint_hints_ras_mm",
        "duodenum_centerline_endpoint_match_tolerance_mm",
    }
    unexpected = set(values) - supported
    if unexpected:
        raise ValueError(f"sampling 包含不支持的配置: {', '.join(sorted(unexpected))}")
    counts = dict(DEFAULT_POINT_COUNTS)
    supplied = values.get("point_counts", {})
    if not isinstance(supplied, dict):
        raise ValueError("sampling.point_counts 必须是 YAML 映射")
    for name, value in supplied.items():
        if name not in counts:
            raise ValueError(f"不支持的采样点数量配置: {name}")
        counts[name] = _integer(value, f"sampling.point_counts.{name}", counts[name])
    hints_raw = values.get("duodenum_centerline_endpoint_hints_ras_mm")
    tolerance_supplied = "duodenum_centerline_endpoint_match_tolerance_mm" in values
    if hints_raw is None and tolerance_supplied:
        raise ValueError(
            "sampling.duodenum_centerline_endpoint_match_tolerance_mm 只能与 "
            "sampling.duodenum_centerline_endpoint_hints_ras_mm 同时配置"
        )
    endpoint_hints = None
    endpoint_tolerance = 1.0
    if hints_raw is not None:
        hints = _mapping(hints_raw, "sampling.duodenum_centerline_endpoint_hints_ras_mm")
        unexpected_hints = set(hints) - {"proximal", "distal"}
        missing_hints = {"proximal", "distal"} - set(hints)
        if missing_hints:
            raise ValueError(
                "sampling.duodenum_centerline_endpoint_hints_ras_mm 缺少配置: "
                + ", ".join(sorted(missing_hints))
            )
        if unexpected_hints:
            raise ValueError(
                "sampling.duodenum_centerline_endpoint_hints_ras_mm 包含不支持的配置: "
                + ", ".join(sorted(unexpected_hints))
            )
        endpoint_hints = (
            _finite_vector3(
                hints["proximal"],
                "sampling.duodenum_centerline_endpoint_hints_ras_mm.proximal",
            ),
            _finite_vector3(
                hints["distal"],
                "sampling.duodenum_centerline_endpoint_hints_ras_mm.distal",
            ),
        )
        endpoint_tolerance = _number(
            values.get("duodenum_centerline_endpoint_match_tolerance_mm"),
            "sampling.duodenum_centerline_endpoint_match_tolerance_mm",
            1.0,
        )
        if not math.isfinite(endpoint_tolerance):
            raise ValueError("sampling.duodenum_centerline_endpoint_match_tolerance_mm 必须是有限数值")
    return SamplingConfig(
        point_counts=counts,
        ray_length_mm=_number(
            values.get("ray_length_mm"),
            "sampling.ray_length_mm",
            RAY_LENGTH_MM,
        ),
        ray_batch_size=_integer(
            values.get("ray_batch_size"),
            "sampling.ray_batch_size",
            RAY_BATCH_SIZE,
        ),
        minimum_spacing_mm=_number(
            values.get("minimum_spacing_mm"),
            "sampling.minimum_spacing_mm",
            MINIMUM_POINT_SPACING_MM,
        ),
        centerline_voxel_pitch_mm=_number(
            values.get("centerline_voxel_pitch_mm"),
            "sampling.centerline_voxel_pitch_mm",
            CENTERLINE_VOXEL_PITCH_MM,
        ),
        centerline_tangent_window_mm=_number(
            values.get("centerline_tangent_window_mm"),
            "sampling.centerline_tangent_window_mm",
            CENTERLINE_TANGENT_WINDOW_MM,
        ),
        centerline_max_terminal_spur_mm=_number(
            values.get("centerline_max_terminal_spur_mm"),
            "sampling.centerline_max_terminal_spur_mm",
            CENTERLINE_MAX_TERMINAL_SPUR_MM,
        ),
        duodenum_centerline_endpoint_hints_ras_mm=endpoint_hints,
        duodenum_centerline_endpoint_match_tolerance_mm=endpoint_tolerance,
    )


def _load_geometry(raw: Any) -> GeometryConfig:
    values = _mapping(raw or {}, "geometry")
    _reject_unexpected_keys(
        values,
        {"input_coordinate_system", "canonical_coordinate_system"},
        "geometry",
    )
    input_coordinate_system = str(values.get("input_coordinate_system", "LPS")).upper()
    canonical_coordinate_system = str(values.get("canonical_coordinate_system", "RAS")).upper()
    if input_coordinate_system not in {"LPS", "RAS"}:
        raise ValueError("geometry.input_coordinate_system 必须是 LPS 或 RAS")
    if canonical_coordinate_system != "RAS":
        raise ValueError("geometry.canonical_coordinate_system 必须是 RAS")
    return GeometryConfig(input_coordinate_system, canonical_coordinate_system)


def _load_square(raw: Any) -> SquareConfig:
    values = _mapping(raw or {}, "square")
    unexpected = set(values) - {"side_length_mm"}
    if unexpected:
        raise ValueError(f"square 包含不支持的配置: {', '.join(sorted(unexpected))}")
    side_length = _number(
        values.get("side_length_mm"),
        "square.side_length_mm",
        SQUARE_SIDE_LENGTH_MM,
    )
    return SquareConfig(side_length_mm=side_length)


def _load_ct(raw: Any) -> CTConfig:
    values = _mapping(raw or {}, "ct")
    _reject_unexpected_keys(
        values,
        {"output_resolution", "window_level", "window_width", "fill_hu_value"},
        "ct",
    )
    return CTConfig(
        output_resolution=_integer(
            values.get("output_resolution"),
            "ct.output_resolution",
            OUTPUT_RESOLUTION,
        ),
        window_level=_finite_number(
            values.get("window_level"),
            "ct.window_level",
            WINDOW_LEVEL_HU,
        ),
        window_width=_number(
            values.get("window_width"),
            "ct.window_width",
            WINDOW_WIDTH_HU,
        ),
        fill_hu_value=_finite_number(
            values.get("fill_hu_value"),
            "ct.fill_hu_value",
            FILL_HU_VALUE,
        ),
    )


def _load_filter(raw: Any) -> FilterConfig:
    values = _mapping(raw or {}, "filtering")
    _reject_unexpected_keys(
        values,
        {
            "black_threshold",
            "black_ratio_limit",
            "line_min_diagonal_fraction",
            "black_side_min_ratio",
            "valid_side_max_black_ratio",
        },
        "filtering",
    )
    black_threshold = _integer(
        values.get("black_threshold"),
        "filtering.black_threshold",
        BLACK_THRESHOLD,
        minimum=0,
    )
    if black_threshold > 255:
        raise ValueError("filtering.black_threshold 必须在 0-255 内")
    return FilterConfig(
        black_threshold=black_threshold,
        black_ratio_limit=_ratio(
            values.get("black_ratio_limit"),
            "filtering.black_ratio_limit",
            BLACK_RATIO_LIMIT,
        ),
        line_min_diagonal_fraction=_ratio(
            values.get("line_min_diagonal_fraction"),
            "filtering.line_min_diagonal_fraction",
            LINE_MIN_DIAGONAL_FRACTION,
            allow_zero=False,
        ),
        black_side_min_ratio=_ratio(
            values.get("black_side_min_ratio"),
            "filtering.black_side_min_ratio",
            BLACK_SIDE_MIN_RATIO,
        ),
        valid_side_max_black_ratio=_ratio(
            values.get("valid_side_max_black_ratio"),
            "filtering.valid_side_max_black_ratio",
            VALID_SIDE_MAX_BLACK_RATIO,
        ),
    )


def _load_runtime(raw: Any) -> RuntimeConfig:
    values = _mapping(raw or {}, "runtime")
    _reject_unexpected_keys(
        values,
        {"seed", "workers", "backend", "gpu_device", "gpu_batch_size"},
        "runtime",
    )
    backend = values.get("backend", "auto")
    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("runtime.backend 必须是 auto、gpu 或 cpu")
    return RuntimeConfig(
        seed=_integer(values.get("seed"), "runtime.seed", SAMPLING_SEED, minimum=0),
        workers=_integer(values.get("workers"), "runtime.workers", 8),
        backend=backend,
        gpu_device=_integer(values.get("gpu_device"), "runtime.gpu_device", 0, minimum=0),
        gpu_batch_size=_integer(values.get("gpu_batch_size"), "runtime.gpu_batch_size", 32),
    )


def _require_exact_keys(values: dict[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(values))
    if missing:
        raise ValueError(f"{field} 缺少配置: {', '.join(missing)}")
    unexpected = sorted(set(values) - expected)
    if unexpected:
        raise ValueError(f"{field} 包含不支持的配置: {', '.join(unexpected)}")


def _load_label_values(raw: Any, expected: tuple[str, ...], field: str) -> dict[str, tuple[int, ...]]:
    values = _mapping(raw, field)
    _require_exact_keys(values, set(expected), field)
    result: dict[str, tuple[int, ...]] = {}
    owners: dict[int, str] = {}
    for identifier in expected:
        source = values[identifier]
        item_field = f"{field}.{identifier}"
        if not isinstance(source, list) or not source:
            raise ValueError(f"{item_field} 必须是非空整数列表")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in source):
            raise ValueError(f"{item_field} 必须是非空整数列表")
        if any(item < 0 or item > 255 for item in source):
            raise ValueError(f"{item_field} 标签值必须在 0-255 内")
        if len(set(source)) != len(source):
            raise ValueError(f"{item_field} 包含重复标签值")
        for label_value in source:
            previous = owners.get(label_value)
            if previous is not None:
                raise ValueError(
                    f"{field} 标签值 {label_value} 在 {previous} 与 {identifier} 中重复"
                )
            owners[label_value] = identifier
        result[identifier] = tuple(source)
    return result


def _load_manual_segmentation(raw: Any, config_directory: Path) -> ManualSegmentationConfig | None:
    if raw is None:
        return None
    values = _mapping(raw, "manual_segmentation")
    expected_fields = {
        "path",
        "organ_label_values",
        "eus_vessel_label_values",
        "eus_vessel_colors",
    }
    _require_exact_keys(values, expected_fields, "manual_segmentation")
    colors_raw = _mapping(values["eus_vessel_colors"], "manual_segmentation.eus_vessel_colors")
    _require_exact_keys(
        colors_raw,
        set(EUS_VESSEL_IDS),
        "manual_segmentation.eus_vessel_colors",
    )
    colors: dict[str, tuple[int, int, int]] = {}
    for identifier in EUS_VESSEL_IDS:
        source = colors_raw[identifier]
        if (
            not isinstance(source, list)
            or len(source) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > 255 for item in source)
        ):
            raise ValueError(
                f"manual_segmentation.eus_vessel_colors.{identifier} 颜色必须是三个 0-255 整数"
            )
        colors[identifier] = tuple(source)
    return ManualSegmentationConfig(
        path=_as_path(values["path"], config_directory, "manual_segmentation.path"),
        organ_label_values=_load_label_values(
            values["organ_label_values"],
            ORGAN_BOUNDARY_IDS,
            "manual_segmentation.organ_label_values",
        ),
        eus_vessel_label_values=_load_label_values(
            values["eus_vessel_label_values"],
            EUS_VESSEL_IDS,
            "manual_segmentation.eus_vessel_label_values",
        ),
        eus_vessel_colors=colors,
    )


def load_case_config(path: str | Path) -> CaseConfig:
    """读取一个无 P/N/D 的 CT 血管重采样病例。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = _mapping(raw, "病例配置")
    if "registration_module_path" in values:
        raise ValueError("registration_module_path 已移除；论文版内部生成 library_summary.json")
    _reject_unexpected_keys(
        values,
        {
            "case_id",
            "ct_path",
            "output_root",
            "organ_models",
            "vessel_models",
            "sampling",
            "square",
            "ct",
            "filtering",
            "runtime",
            "geometry",
            "dicom_series_uid",
            "manual_segmentation",
        },
        "病例配置",
    )
    config_directory = config_path.parent
    case_id = values.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id 必须是非空字符串")
    organs_raw = _mapping(values.get("organ_models"), "organ_models")
    _require_exact_keys(organs_raw, set(REQUIRED_ORGAN_IDS), "organ_models")
    organ_models = {
        name: _as_path(organs_raw[name], config_directory, f"organ_models.{name}") for name in REQUIRED_ORGAN_IDS
    }
    dicom_series_uid = values.get("dicom_series_uid")
    if dicom_series_uid is not None and (not isinstance(dicom_series_uid, str) or not dicom_series_uid):
        raise ValueError("dicom_series_uid 必须是非空字符串")
    manual_segmentation = _load_manual_segmentation(
        values.get("manual_segmentation"),
        config_directory,
    )
    if manual_segmentation is None:
        raise ValueError("论文版病例配置必须包含 manual_segmentation 手工分割输入")
    return CaseConfig(
        case_id=case_id,
        ct_path=_as_path(values.get("ct_path"), config_directory, "ct_path"),
        output_root=_as_path(values.get("output_root"), config_directory, "output_root"),
        organ_models=organ_models,
        vessel_models=_load_vessels(values.get("vessel_models"), config_directory),
        sampling=_load_sampling(values.get("sampling")),
        square=_load_square(values.get("square")),
        ct=_load_ct(values.get("ct")),
        filtering=_load_filter(values.get("filtering")),
        runtime=_load_runtime(values.get("runtime")),
        geometry=_load_geometry(values.get("geometry")),
        dicom_series_uid=dicom_series_uid,
        manual_segmentation=manual_segmentation,
    )
