"""可选 GPU 加速前的 CT 方形重采样后端。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from .ct_resampling import CTVolume, hu_to_grayscale, square_coordinates_zyx


class CTSamplingBackend(Protocol):
    """CT 方形插值后端的最小接口。"""

    name: str

    def sample_many(self, vertices_batch: np.ndarray, resolution: int, fill_hu_value: float) -> np.ndarray: ...

    def close(self) -> None: ...


def _load_cupy() -> tuple[Any, Any]:
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import map_coordinates as cupy_map_coordinates
    except ImportError as error:
        raise ImportError("未安装可用的 CuPy GPU 后端") from error
    return cp, cupy_map_coordinates


@dataclass
class CachedCpuBackend:
    """以与参考 SciPy 路径逐位一致的已缓存三次样条系数重采样。"""

    volume: CTVolume
    name: str = field(init=False, default="cpu")
    _coefficients_zyx: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._coefficients_zyx = spline_filter(
            self.volume.data_zyx,
            order=3,
            output=np.float64,
            mode="constant",
        )

    @property
    def coefficients_zyx(self) -> np.ndarray:
        """供 GPU 路径复用的参考精度三次样条系数。"""

        return self._coefficients_zyx

    def sample_many(self, vertices_batch: np.ndarray, resolution: int, fill_hu_value: float) -> np.ndarray:
        vertices = np.asarray(vertices_batch, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices_batch 必须是 N×4×3 数组")
        sampled_squares = []
        for square in vertices:
            sampled = map_coordinates(
                self._coefficients_zyx,
                square_coordinates_zyx(self.volume, square, resolution),
                output=np.float32,
                order=3,
                mode="constant",
                cval=float(fill_hu_value),
                prefilter=False,
            )
            sampled_squares.append(sampled.reshape(resolution, resolution))
        return np.stack(sampled_squares, axis=0) if sampled_squares else np.empty((0, resolution, resolution), dtype=np.float32)

    def close(self) -> None:
        """保留与 GPU 后端一致的资源释放接口。"""


class CuPyBackend:
    """使用已验证 CPU 样条系数的可选 CUDA 插值后端。"""

    name: str

    def __init__(
        self,
        cpu_backend: CachedCpuBackend,
        gpu_device: int,
        gpu_batch_size: int,
        *,
        cupy_loader: Callable[[], tuple[Any, Any]] | None = None,
    ) -> None:
        cp, cupy_map_coordinates = (cupy_loader or _load_cupy)()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if gpu_device >= device_count:
            raise RuntimeError(f"GPU 设备 {gpu_device} 不可用，当前设备数: {device_count}")
        self._cp = cp
        self._map_coordinates = cupy_map_coordinates
        self._device = cp.cuda.Device(gpu_device)
        self._gpu_batch_size = gpu_batch_size
        self._volume = cpu_backend.volume
        with self._device:
            self._coefficients_zyx = cp.asarray(cpu_backend.coefficients_zyx)
        self.name = f"gpu:{gpu_device}"

    def sample_many(self, vertices_batch: np.ndarray, resolution: int, fill_hu_value: float) -> np.ndarray:
        vertices = np.asarray(vertices_batch, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices_batch 必须是 N×4×3 数组")
        if not len(vertices):
            return np.empty((0, resolution, resolution), dtype=np.float32)
        host_results: list[np.ndarray] = []
        with self._device:
            for start in range(0, len(vertices), self._gpu_batch_size):
                device_results = []
                for square in vertices[start : start + self._gpu_batch_size]:
                    coordinates = self._cp.asarray(square_coordinates_zyx(self._volume, square, resolution))
                    sampled = self._map_coordinates(
                        self._coefficients_zyx,
                        coordinates,
                        output=self._cp.float32,
                        order=3,
                        mode="constant",
                        cval=float(fill_hu_value),
                        prefilter=False,
                    )
                    device_results.append(sampled.reshape(resolution, resolution))
                host_results.extend(self._cp.asnumpy(result) for result in device_results)
        return np.stack(host_results, axis=0)

    def close(self) -> None:
        with self._device:
            self._coefficients_zyx = None


def create_sampling_backend(
    volume: CTVolume,
    *,
    backend: str,
    gpu_device: int,
    gpu_batch_size: int,
) -> tuple[CTSamplingBackend, dict[str, Any]]:
    """创建指定 CT 后端；auto 仅在 GPU 初始化失败时回退 CPU。"""

    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError("backend 必须是 auto、gpu 或 cpu")
    cpu_backend = CachedCpuBackend(volume)
    metadata: dict[str, Any] = {
        "requested_backend": backend,
        "selected_backend": "cpu",
        "gpu_device": gpu_device,
        "gpu_batch_size": gpu_batch_size,
        "fallback_reason": None,
        "coefficient_dtype": str(cpu_backend.coefficients_zyx.dtype),
    }
    if backend == "cpu":
        return cpu_backend, metadata
    try:
        gpu_backend = CuPyBackend(cpu_backend, gpu_device, gpu_batch_size)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        if backend == "gpu":
            raise RuntimeError(f"GPU 后端初始化失败: {error}") from error
        metadata["fallback_reason"] = str(error)
        return cpu_backend, metadata
    metadata["selected_backend"] = gpu_backend.name
    metadata["cupy_version"] = gpu_backend._cp.__version__
    metadata["gpu_name"] = gpu_backend._cp.cuda.runtime.getDeviceProperties(gpu_device)["name"].decode()
    return gpu_backend, metadata


def validate_backend_against_cpu(
    candidate: CTSamplingBackend,
    cpu_backend: CachedCpuBackend,
    vertices_batch: np.ndarray,
    *,
    resolution: int,
    fill_hu_value: float,
    window_level: float,
    window_width: float,
) -> dict[str, Any]:
    """以实际方形比较候选后端与参考 CPU 插值结果。"""

    vertices = np.asarray(vertices_batch, dtype=np.float64)
    expected = cpu_backend.sample_many(vertices, resolution, fill_hu_value)
    actual = candidate.sample_many(vertices, resolution, fill_hu_value)
    if actual.shape != expected.shape:
        raise ValueError(f"候选后端输出形状不一致: {actual.shape} != {expected.shape}")
    max_abs_hu = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.size else 0.0
    grayscale_equal = all(
        np.array_equal(
            hu_to_grayscale(actual[index], window_level, window_width),
            hu_to_grayscale(expected[index], window_level, window_width),
        )
        for index in range(len(actual))
    )
    return {
        "sample_count": int(len(vertices)),
        "max_abs_hu": max_abs_hu,
        "grayscale_png_equal": grayscale_equal,
        "accepted": max_abs_hu <= 1e-3 and grayscale_equal,
    }
