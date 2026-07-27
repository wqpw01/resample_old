from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ct_vascular_resampling.ct_resampling import CTVolume, sample_ct_square, square_coordinates_zyx
from ct_vascular_resampling.resampling_backend import CachedCpuBackend, CuPyBackend, create_sampling_backend, validate_backend_against_cpu


def _physical_square(image: sitk.Image, indices: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    return np.asarray([image.TransformIndexToPhysicalPoint(index) for index in indices], dtype=np.float64)


def test_cached_cpu_backend_is_bitwise_equal_to_reference_square_sampling():
    data = np.arange(9 * 10 * 11, dtype=np.float32).reshape(9, 10, 11)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((1.7, 2.1, 2.5))
    image.SetOrigin((13.0, -5.0, 27.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    volume = CTVolume.from_sitk(image)
    in_bounds = _physical_square(image, ((1, 2, 3), (6, 2, 3), (6, 7, 3), (1, 7, 3)))
    out_of_bounds = in_bounds + np.asarray([100.0, 0.0, 0.0])
    vertices = np.stack([in_bounds, out_of_bounds])

    actual = CachedCpuBackend(volume).sample_many(vertices, resolution=7, fill_hu_value=-1000.0)
    expected = np.stack([sample_ct_square(volume, square, resolution=7, fill_hu_value=-1000.0) for square in vertices])

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_auto_backend_falls_back_to_cached_cpu_for_an_invalid_gpu_device():
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    volume = CTVolume.from_sitk(image)

    backend, metadata = create_sampling_backend(
        volume,
        backend="auto",
        gpu_device=1_000_000,
        gpu_batch_size=8,
    )

    assert isinstance(backend, CachedCpuBackend)
    assert metadata["requested_backend"] == "auto"
    assert metadata["selected_backend"] == "cpu"
    assert metadata["fallback_reason"]


def test_cupy_backend_uses_the_same_world_to_index_coordinates_as_cpu():
    class FakeDevice:
        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 1

        @staticmethod
        def getDeviceProperties(_):
            return {"name": b"fake-gpu"}

    class FakeCuPy:
        float32 = np.float32

        class cuda:
            runtime = FakeRuntime()
            Device = FakeDevice

        @staticmethod
        def asarray(value):
            return np.asarray(value)

        @staticmethod
        def asnumpy(value):
            return np.asarray(value)

    calls: list[np.ndarray] = []

    def fake_map_coordinates(_, coordinates, **__):
        calls.append(np.asarray(coordinates))
        return np.zeros(coordinates.shape[1], dtype=np.float32)

    image = sitk.GetImageFromArray(np.zeros((6, 6, 6), dtype=np.float32))
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    volume = CTVolume.from_sitk(image)
    vertices = _physical_square(image, ((1, 1, 1), (3, 1, 1), (3, 3, 1), (1, 3, 1)))

    backend = CuPyBackend(CachedCpuBackend(volume), 0, 8, cupy_loader=lambda: (FakeCuPy, fake_map_coordinates))
    backend.sample_many(vertices[None, ...], resolution=3, fill_hu_value=-1000.0)

    assert np.array_equal(calls, [square_coordinates_zyx(volume, vertices, resolution=3)])


def test_backend_validation_rejects_hu_values_outside_the_strict_tolerance():
    class OffsetBackend:
        name = "gpu:0"

        def __init__(self, reference: CachedCpuBackend):
            self.reference = reference

        def sample_many(self, vertices_batch, resolution, fill_hu_value):
            return self.reference.sample_many(vertices_batch, resolution, fill_hu_value) + np.float32(0.002)

        def close(self):
            pass

    image = sitk.GetImageFromArray(np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6))
    volume = CTVolume.from_sitk(image)
    vertices = _physical_square(image, ((1, 1, 1), (3, 1, 1), (3, 3, 1), (1, 3, 1)))
    cpu = CachedCpuBackend(volume)

    result = validate_backend_against_cpu(
        OffsetBackend(cpu),
        cpu,
        vertices[None, ...],
        resolution=5,
        fill_hu_value=-1000.0,
        window_level=40.0,
        window_width=400.0,
    )

    assert result["sample_count"] == 1
    assert result["max_abs_hu"] > 1e-3
    assert result["accepted"] is False
