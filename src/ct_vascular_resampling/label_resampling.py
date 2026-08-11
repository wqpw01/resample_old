"""与 CT 方形共享世界坐标网格的离散标签最近邻采样。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.ndimage import map_coordinates
import SimpleITK as sitk

from .coordinates import to_ras_direction, to_ras_points
from .ct_resampling import CTVolume, square_coordinates_zyx


@dataclass(frozen=True)
class LabelVolume:
    """三维离散标签及其规范 RAS 物理空间元数据。"""

    data_zyx: np.ndarray
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction_xyz: np.ndarray

    @classmethod
    def from_sitk(cls, image: sitk.Image, *, input_coordinate_system: str = "RAS") -> "LabelVolume":
        if image.GetDimension() != 3:
            raise ValueError("标签图必须是三维单标量图像")
        data = sitk.GetArrayFromImage(image)
        if data.ndim != 3 or not np.issubdtype(data.dtype, np.integer):
            raise ValueError("标签图必须是三维整数单标量图像")
        if data.size == 0 or int(np.min(data)) < 0 or int(np.max(data)) > 255:
            raise ValueError("标签值必须在 0-255 内并可表示为 uint8")
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        origin = to_ras_points(np.asarray(image.GetOrigin(), dtype=np.float64), input_coordinate_system)
        direction = to_ras_direction(
            np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
            input_coordinate_system,
        )
        if np.any(spacing <= 0.0) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
            raise ValueError("标签图 spacing、origin 或 direction 无效")
        return cls(data.astype(np.uint8, copy=False), spacing, origin, direction)

    @property
    def physical_to_index_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.direction_xyz @ np.diag(self.spacing_xyz))

    def world_to_continuous_indices(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("物理点坐标最后一维必须为 3")
        flat = points.reshape(-1, 3)
        indices = (flat - self.origin_xyz) @ self.physical_to_index_matrix.T
        return indices.reshape(points.shape)


def load_label_volume(
    path: str | Path,
    *,
    input_coordinate_system: str = "RAS",
) -> LabelVolume:
    """读取单个 NIfTI/NRRD 离散标签体并保留其物理空间。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"标签文件不存在: {source}")
    name = source.name.lower()
    if not name.endswith((".nii", ".nii.gz", ".nrrd")):
        raise ValueError(f"仅支持 NIfTI 或 NRRD 标签图: {source}")
    image = sitk.ReadImage(str(source))
    return LabelVolume.from_sitk(
        image,
        input_coordinate_system=input_coordinate_system,
    )


def validate_label_geometry(ct: CTVolume, labels: LabelVolume, *, atol_mm: float = 1e-6) -> None:
    """确认 CT 与标签图共享离散网格和规范 RAS 物理空间。"""

    if ct.data_zyx.shape != labels.data_zyx.shape:
        raise ValueError("CT 与标签图的 Size 不一致")
    if not np.allclose(ct.spacing_xyz, labels.spacing_xyz, atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与标签图的 Spacing 不一致")
    if not np.allclose(ct.origin_xyz, labels.origin_xyz, atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与标签图的 Origin 不一致")
    if not np.allclose(ct.direction_xyz, labels.direction_xyz, atol=1e-8, rtol=0.0):
        raise ValueError("CT 与标签图的 Direction 不一致")


class LabelSamplingBackend(Protocol):
    name: str

    def sample_many(self, vertices_batch: np.ndarray, resolution: int) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass
class CpuLabelBackend:
    volume: LabelVolume
    name: str = "cpu"

    def sample_many(self, vertices_batch: np.ndarray, resolution: int) -> np.ndarray:
        vertices = np.asarray(vertices_batch, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices_batch 必须是 N×4×3 数组")
        sampled: list[np.ndarray] = []
        for square in vertices:
            values = map_coordinates(
                self.volume.data_zyx,
                square_coordinates_zyx(self.volume, square, resolution),
                output=np.uint8,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            )
            sampled.append(values.reshape(resolution, resolution))
        if not sampled:
            return np.empty((0, resolution, resolution), dtype=np.uint8)
        return np.stack(sampled, axis=0)

    def close(self) -> None:
        """CPU 后端不持有外部资源。"""


def _load_cupy() -> tuple[Any, Any]:
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates
    except ImportError as error:
        raise ImportError("未安装可用的 CuPy 标签采样后端") from error
    return cp, cupy_map_coordinates


class CuPyLabelBackend:
    """在 GPU 上执行与 CPU 逐像素一致的最近邻标签采样。"""

    def __init__(
        self,
        volume: LabelVolume,
        gpu_device: int,
        gpu_batch_size: int,
        *,
        cupy_loader: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        cp, cupy_map_coordinates = (cupy_loader or _load_cupy)()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if gpu_device >= device_count:
            raise RuntimeError(f"GPU 设备 {gpu_device} 不可用，当前设备数: {device_count}")
        if gpu_batch_size < 1:
            raise ValueError("gpu_batch_size 必须大于零")
        self._cp = cp
        self._map_coordinates = cupy_map_coordinates
        self._device = cp.cuda.Device(gpu_device)
        self._gpu_batch_size = gpu_batch_size
        self._volume = volume
        with self._device:
            self._labels_zyx = cp.asarray(volume.data_zyx)
        self.name = f"gpu:{gpu_device}"

    def sample_many(self, vertices_batch: np.ndarray, resolution: int) -> np.ndarray:
        vertices = np.asarray(vertices_batch, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices_batch 必须是 N×4×3 数组")
        if not len(vertices):
            return np.empty((0, resolution, resolution), dtype=np.uint8)
        host_results: list[np.ndarray] = []
        with self._device:
            for start in range(0, len(vertices), self._gpu_batch_size):
                device_results = []
                for square in vertices[start : start + self._gpu_batch_size]:
                    coordinates = self._cp.asarray(
                        square_coordinates_zyx(self._volume, square, resolution)
                    )
                    sampled = self._map_coordinates(
                        self._labels_zyx,
                        coordinates,
                        output=self._cp.uint8,
                        order=0,
                        mode="constant",
                        cval=0,
                        prefilter=False,
                    )
                    device_results.append(sampled.reshape(resolution, resolution))
                host_results.extend(
                    self._cp.asnumpy(result).astype(np.uint8, copy=False)
                    for result in device_results
                )
        return np.stack(host_results, axis=0)

    def close(self) -> None:
        with self._device:
            self._labels_zyx = None


def create_label_sampling_backend(
    volume: LabelVolume,
    *,
    backend: str,
    gpu_device: int,
    gpu_batch_size: int,
) -> tuple[LabelSamplingBackend, dict[str, Any]]:
    """创建标签采样后端；auto 初始化失败时回退 CPU。"""

    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("backend 必须是 auto、gpu 或 cpu")
    cpu_backend = CpuLabelBackend(volume)
    metadata: dict[str, Any] = {
        "requested_backend": backend,
        "selected_backend": "cpu",
        "gpu_device": gpu_device,
        "gpu_batch_size": gpu_batch_size,
        "fallback_reason": None,
        "interpolation": "nearest",
        "outside_label_value": 0,
    }
    if backend == "cpu":
        return cpu_backend, metadata
    try:
        gpu_backend = CuPyLabelBackend(volume, gpu_device, gpu_batch_size)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if backend == "gpu":
            raise RuntimeError(f"GPU 标签后端初始化失败: {error}") from error
        metadata["fallback_reason"] = str(error)
        return cpu_backend, metadata
    metadata["selected_backend"] = gpu_backend.name
    metadata["cupy_version"] = gpu_backend._cp.__version__
    gpu_name = gpu_backend._cp.cuda.runtime.getDeviceProperties(gpu_device)["name"]
    metadata["gpu_name"] = gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name)
    return gpu_backend, metadata


def validate_label_backend_against_cpu(
    candidate: LabelSamplingBackend,
    cpu_backend: CpuLabelBackend,
    vertices_batch: np.ndarray,
    *,
    resolution: int,
) -> dict[str, Any]:
    """逐像素比较候选标签后端与 CPU 最近邻参考。"""

    vertices = np.asarray(vertices_batch, dtype=np.float64)
    expected = cpu_backend.sample_many(vertices, resolution)
    actual = candidate.sample_many(vertices, resolution)
    if actual.shape != expected.shape:
        raise ValueError(f"候选标签后端输出形状不一致: {actual.shape} != {expected.shape}")
    mismatched = int(np.count_nonzero(actual != expected))
    equal = bool(np.array_equal(actual, expected))
    return {
        "sample_count": int(len(vertices)),
        "mismatched_pixel_count": mismatched,
        "pixel_equal": equal,
        "accepted": equal,
    }
