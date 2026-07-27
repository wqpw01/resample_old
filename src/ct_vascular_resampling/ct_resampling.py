"""保留 CT 原生物理空间的斜方形重采样。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class CTVolume:
    """三维 CT 数组及其未修改的 ITK 物理空间元数据。"""

    data_zyx: np.ndarray
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction_xyz: np.ndarray

    @classmethod
    def from_sitk(cls, image: sitk.Image) -> "CTVolume":
        if image.GetDimension() != 3:
            raise ValueError("CT 图像必须是三维")
        data = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        if data.ndim != 3:
            raise ValueError("CT 图像必须转换为 z×y×x 数组")
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        if np.any(spacing <= 0.0) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError("CT spacing、origin 或 direction 无效")
        return cls(data_zyx=data, spacing_xyz=spacing, origin_xyz=origin, direction_xyz=direction)

    @property
    def physical_to_index_matrix(self) -> np.ndarray:
        """原生物理点到连续 x/y/z 体素索引的线性部分。"""

        return np.linalg.inv(self.direction_xyz @ np.diag(self.spacing_xyz))

    def world_to_continuous_indices(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("物理点坐标最后一维必须为 3")
        flat = points.reshape(-1, 3)
        indices = (flat - self.origin_xyz) @ self.physical_to_index_matrix.T
        return indices.reshape(points.shape)


@dataclass(frozen=True)
class SquareFovDiagnosis:
    """方形采样网格相对 CT 原始体积范围的精确诊断。"""

    out_of_bounds_mask: np.ndarray
    out_of_bounds_ratio: float
    face_out_of_bounds_ratios: dict[str, float]
    continuous_index_min_zyx: tuple[float, float, float]
    continuous_index_max_zyx: tuple[float, float, float]
    probe_point_inside_ct: bool | None

    @property
    def contains_ct_fov_exceedance(self) -> bool:
        return bool(np.any(self.out_of_bounds_mask))

    def to_record(self) -> dict[str, object]:
        return {
            "contains_ct_fov_exceedance": self.contains_ct_fov_exceedance,
            "out_of_bounds_pixel_count": int(np.count_nonzero(self.out_of_bounds_mask)),
            "out_of_bounds_ratio": self.out_of_bounds_ratio,
            "face_out_of_bounds_ratios": self.face_out_of_bounds_ratios,
            "continuous_index_min_zyx": list(self.continuous_index_min_zyx),
            "continuous_index_max_zyx": list(self.continuous_index_max_zyx),
            "probe_point_inside_ct": self.probe_point_inside_ct,
        }


def load_ct(path: str | Path, *, dicom_series_uid: str | None = None) -> CTVolume:
    """读取 NIfTI、NRRD 或指定 DICOM 序列，保持原始物理空间。"""

    source = Path(path)
    if source.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = tuple(reader.GetGDCMSeriesIDs(str(source)) or ())
        if not series_ids:
            raise ValueError(f"目录中没有可读取的 DICOM 序列: {source}")
        if dicom_series_uid is None:
            if len(series_ids) != 1:
                raise ValueError(f"DICOM 目录包含多个序列，必须指定 dicom_series_uid: {source}")
            selected_series_uid = series_ids[0]
        else:
            if dicom_series_uid not in series_ids:
                raise ValueError(f"指定的 DICOM Series UID 不存在: {dicom_series_uid}")
            selected_series_uid = dicom_series_uid
        filenames = reader.GetGDCMSeriesFileNames(str(source), selected_series_uid)
        if not filenames:
            raise ValueError(f"DICOM Series UID 没有切片文件: {selected_series_uid}")
        reader.SetFileNames(filenames)
        return CTVolume.from_sitk(reader.Execute())
    if not source.is_file():
        raise FileNotFoundError(f"CT 文件不存在: {source}")
    if dicom_series_uid is not None:
        raise ValueError("dicom_series_uid 仅适用于 DICOM 目录")
    name = source.name.lower()
    if not name.endswith((".nii", ".nii.gz", ".nrrd")):
        raise ValueError(f"仅支持 NIfTI 或 NRRD CT: {source}")
    return CTVolume.from_sitk(sitk.ReadImage(str(source)))


def sample_ct_square(
    volume: CTVolume,
    vertices_world: np.ndarray,
    resolution: int,
    fill_hu_value: float,
) -> np.ndarray:
    """在 V1,V2,V3,V4 定义的方形中执行参考管线的三次插值。"""

    coordinates = square_coordinates_zyx(volume, vertices_world, resolution)
    sampled = map_coordinates(
        volume.data_zyx,
        coordinates=coordinates,
        order=3,
        mode="constant",
        cval=float(fill_hu_value),
        prefilter=True,
    )
    return sampled.reshape(resolution, resolution).astype(np.float32, copy=False)


def square_coordinates_zyx(volume: CTVolume, vertices_world: np.ndarray, resolution: int) -> np.ndarray:
    """按参考方形公式生成 SciPy/CuPy 所需的 z/y/x 连续体素坐标。"""

    if resolution <= 1:
        raise ValueError("resolution 必须大于 1")
    vertices = np.asarray(vertices_world, dtype=np.float64)
    if vertices.shape != (4, 3):
        raise ValueError("vertices_world 必须是 4×3 数组")
    u_coords = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    v_coords = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(u_coords, v_coords)
    origin = vertices[0]
    u_vector = vertices[1] - origin
    v_vector = vertices[3] - origin
    world = origin + grid_u[..., None] * u_vector + grid_v[..., None] * v_vector
    xyz = volume.world_to_continuous_indices(world).reshape(-1, 3)
    return np.vstack([xyz[:, 2], xyz[:, 1], xyz[:, 0]])


def diagnose_square_fov(
    volume: CTVolume,
    vertices_world: np.ndarray,
    resolution: int,
    probe_point_world: np.ndarray | None = None,
) -> SquareFovDiagnosis:
    """标记重采样方形中会触发常量填充的 CT 范围外像素。"""

    coordinates = square_coordinates_zyx(volume, vertices_world, resolution).reshape(3, resolution, resolution)
    shape_zyx = np.asarray(volume.data_zyx.shape, dtype=np.float64).reshape(3, 1, 1)
    low = coordinates < 0.0
    high = coordinates > (shape_zyx - 1.0)
    mask = np.any(low | high, axis=0)
    probe_inside: bool | None = None
    if probe_point_world is not None:
        probe_xyz = volume.world_to_continuous_indices(np.asarray(probe_point_world, dtype=np.float64))
        probe_zyx = probe_xyz[[2, 1, 0]]
        probe_inside = bool(np.all(probe_zyx >= 0.0) and np.all(probe_zyx <= shape_zyx[:, 0, 0] - 1.0))
    return SquareFovDiagnosis(
        out_of_bounds_mask=mask,
        out_of_bounds_ratio=float(np.mean(mask)),
        face_out_of_bounds_ratios={
            "z_low": float(np.mean(low[0])),
            "z_high": float(np.mean(high[0])),
            "y_low": float(np.mean(low[1])),
            "y_high": float(np.mean(high[1])),
            "x_low": float(np.mean(low[2])),
            "x_high": float(np.mean(high[2])),
        },
        continuous_index_min_zyx=tuple(float(np.min(axis)) for axis in coordinates),
        continuous_index_max_zyx=tuple(float(np.max(axis)) for axis in coordinates),
        probe_point_inside_ct=probe_inside,
    )


def hu_to_grayscale(hu: np.ndarray, window_level: float, window_width: float) -> np.ndarray:
    """按参考窗位窗宽映射为 uint8 PNG 灰度。"""

    if window_width <= 0.0:
        raise ValueError("window_width 必须大于零")
    lower = window_level - window_width / 2.0
    clipped = np.clip(np.asarray(hu, dtype=np.float32), lower, lower + window_width)
    return ((clipped - lower) / window_width * 255.0).astype(np.uint8)
