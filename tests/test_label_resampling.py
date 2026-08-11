from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates as scipy_map_coordinates
import SimpleITK as sitk

from ct_vascular_resampling.ct_resampling import CTVolume, square_coordinates_zyx
from ct_vascular_resampling.label_resampling import (
    CpuLabelBackend,
    CuPyLabelBackend,
    LabelVolume,
    create_label_sampling_backend,
    validate_label_backend_against_cpu,
    validate_label_geometry,
)


def _image(
    values: np.ndarray,
    *,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sitk.Image:
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetOrigin(origin)
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _square(image: sitk.Image) -> np.ndarray:
    indices = ((0, 0, 1), (4, 0, 1), (4, 3, 1), (0, 3, 1))
    points_lps = np.asarray(
        [image.TransformIndexToPhysicalPoint(index) for index in indices],
        dtype=np.float64,
    )
    points_lps[:, :2] *= -1.0
    return points_lps


def test_label_volume_converts_lps_geometry_to_ras_without_copying_uint8_values():
    values = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image = _image(values, origin=(10.0, 20.0, 30.0))

    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")

    assert volume.data_zyx.dtype == np.uint8
    assert np.array_equal(volume.data_zyx, values)
    assert np.array_equal(volume.spacing_xyz, [1.5, 2.0, 2.5])
    assert np.array_equal(volume.origin_xyz, [-10.0, -20.0, 30.0])
    expected_direction = np.diag([-1.0, -1.0, 1.0]) @ np.asarray(image.GetDirection()).reshape(3, 3)
    assert np.array_equal(volume.direction_xyz, expected_direction)


def test_label_volume_rejects_non_integer_or_out_of_uint8_values():
    with pytest.raises(ValueError, match="整数|uint8"):
        LabelVolume.from_sitk(_image(np.ones((2, 2, 2), dtype=np.float32)), input_coordinate_system="LPS")

    with pytest.raises(ValueError, match="0-255|uint8"):
        LabelVolume.from_sitk(_image(np.full((2, 2, 2), 256, dtype=np.uint16)), input_coordinate_system="LPS")


def test_label_geometry_accepts_sub_micrometre_origin_difference():
    ct = CTVolume.from_sitk(
        _image(np.zeros((3, 4, 5), dtype=np.int16), origin=(10.0, 20.0, 30.0)),
        input_coordinate_system="LPS",
    )
    labels = LabelVolume.from_sitk(
        _image(np.zeros((3, 4, 5), dtype=np.uint8), origin=(10.0 + 5e-7, 20.0, 30.0)),
        input_coordinate_system="LPS",
    )

    validate_label_geometry(ct, labels)


@pytest.mark.parametrize("difference", ["size", "spacing", "origin", "direction"])
def test_label_geometry_rejects_any_grid_mismatch(difference):
    ct_image = _image(np.zeros((3, 4, 5), dtype=np.int16), origin=(10.0, 20.0, 30.0))
    label_image = _image(np.zeros((3, 4, 5), dtype=np.uint8), origin=(10.0, 20.0, 30.0))
    if difference == "size":
        label_image = _image(np.zeros((3, 4, 6), dtype=np.uint8), origin=(10.0, 20.0, 30.0))
    elif difference == "spacing":
        label_image.SetSpacing((1.5, 2.0, 2.500002))
    elif difference == "origin":
        label_image.SetOrigin((10.0 + 2e-6, 20.0, 30.0))
    else:
        label_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="Size|Spacing|Origin|Direction"):
        validate_label_geometry(
            CTVolume.from_sitk(ct_image, input_coordinate_system="LPS"),
            LabelVolume.from_sitk(label_image, input_coordinate_system="LPS"),
        )


def test_cpu_label_backend_samples_nearest_labels_and_fills_zero():
    labels = np.zeros((3, 4, 5), dtype=np.uint8)
    labels[1, 1, 1] = 8
    image = _image(labels)
    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")
    vertices = _square(image)

    sampled = CpuLabelBackend(volume).sample_many(vertices[None], resolution=5)
    outside = vertices + np.asarray([100.0, 0.0, 0.0])

    assert sampled.dtype == np.uint8
    assert sampled.shape == (1, 5, 5)
    assert 8 in sampled[0]
    assert np.all(CpuLabelBackend(volume).sample_many(outside[None], 5) == 0)


class _FakeDevice:
    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeRuntime:
    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def getDeviceProperties(_):
        return {"name": b"fake-gpu"}


class _FakeCuPy:
    uint8 = np.uint8
    __version__ = "fake"

    class cuda:
        runtime = _FakeRuntime()
        Device = _FakeDevice

    @staticmethod
    def asarray(value):
        return np.asarray(value)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)


def test_cupy_label_backend_uses_same_coordinates_and_is_pixel_equal_to_cpu():
    values = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image = _image(values)
    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")
    vertices = _square(image)[None]
    cpu = CpuLabelBackend(volume)
    sampled_coordinates: list[np.ndarray] = []

    def record_gpu_coordinates(values, coordinates, **kwargs):
        sampled_coordinates.append(np.asarray(coordinates))
        return scipy_map_coordinates(values, coordinates, **kwargs)

    gpu = CuPyLabelBackend(
        volume,
        gpu_device=0,
        gpu_batch_size=2,
        cupy_loader=lambda: (_FakeCuPy, record_gpu_coordinates),
    )

    actual = gpu.sample_many(vertices, resolution=5)
    expected = cpu.sample_many(vertices, resolution=5)

    assert np.array_equal(actual, expected)
    assert np.array_equal(
        sampled_coordinates[0],
        square_coordinates_zyx(volume, _square(image), resolution=5),
    )


def test_auto_label_backend_falls_back_to_cpu_but_forced_gpu_rejects():
    image = _image(np.zeros((3, 4, 5), dtype=np.uint8))
    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")

    backend, metadata = create_label_sampling_backend(
        volume,
        backend="auto",
        gpu_device=1_000_000,
        gpu_batch_size=2,
    )

    assert isinstance(backend, CpuLabelBackend)
    assert metadata["selected_backend"] == "cpu"
    assert metadata["fallback_reason"]
    with pytest.raises(RuntimeError, match="GPU"):
        create_label_sampling_backend(
            volume,
            backend="gpu",
            gpu_device=1_000_000,
            gpu_batch_size=2,
        )


def test_label_backend_validation_rejects_one_changed_pixel():
    image = _image(np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5))
    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")
    reference = CpuLabelBackend(volume)
    vertices = _square(image)[None]

    class ChangedPixelBackend:
        name = "gpu:0"

        def sample_many(self, vertices_batch, resolution):
            result = reference.sample_many(vertices_batch, resolution).copy()
            result[0, 0, 0] ^= np.uint8(1)
            return result

        def close(self):
            pass

    result = validate_label_backend_against_cpu(
        ChangedPixelBackend(),
        reference,
        vertices,
        resolution=5,
    )

    assert result == {
        "sample_count": 1,
        "mismatched_pixel_count": 1,
        "pixel_equal": False,
        "accepted": False,
    }
