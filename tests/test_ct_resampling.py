from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ct_vascular_resampling.ct_resampling import (
    CTVolume,
    diagnose_square_fov,
    hu_to_grayscale,
    load_ct,
    sample_ct_square,
    square_vertices_inside_ct,
)


def _physical_point(image: sitk.Image, index: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(image.TransformIndexToPhysicalPoint(index), dtype=np.float64)


def test_ct_square_sampling_uses_original_spacing_origin_and_direction():
    data = np.fromfunction(lambda z, y, x: x + 10.0 * y + 100.0 * z, (8, 8, 8), dtype=np.float32).astype(np.float32)
    image = sitk.GetImageFromArray(data)
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    volume = CTVolume.from_sitk(image)
    p0 = _physical_point(image, (1, 2, 3))
    p1 = _physical_point(image, (3, 2, 3))
    p3 = _physical_point(image, (1, 4, 3))
    p2 = _physical_point(image, (3, 4, 3))

    sampled = sample_ct_square(volume, np.asarray([p0, p1, p2, p3]), resolution=3, fill_hu_value=-1000.0)

    assert np.allclose(sampled, [[321.0, 322.0, 323.0], [331.0, 332.0, 333.0], [341.0, 342.0, 343.0]])


def test_windowing_matches_reference_ct_gray_scale_mapping():
    pixels = hu_to_grayscale(np.asarray([[40.0, -160.0, 240.0]], dtype=np.float32), window_level=40.0, window_width=400.0)

    assert np.array_equal(pixels, [[127, 0, 255]])


def test_load_ct_accepts_nifti_gz_name_with_dicom_uid_prefix(tmp_path):
    image = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.int16))
    path = tmp_path / "1.2.840.78.85.7.5.14904336.1746414291_Abdomen_+V_402.nii.gz"
    sitk.WriteImage(image, str(path))

    volume = load_ct(path)

    assert volume.data_zyx.shape == (2, 2, 2)


def test_load_ct_selects_the_requested_dicom_series(monkeypatch, tmp_path):
    image = sitk.GetImageFromArray(np.full((3, 2, 2), 40, dtype=np.int16))

    class Reader:
        def __init__(self):
            self.files = None

        def GetGDCMSeriesIDs(self, directory):
            assert directory == str(tmp_path)
            return ("other-series", "target-series")

        def GetGDCMSeriesFileNames(self, directory, series_uid):
            assert directory == str(tmp_path)
            assert series_uid == "target-series"
            return ("slice-001.dcm", "slice-002.dcm", "slice-003.dcm")

        def SetFileNames(self, files):
            self.files = tuple(files)

        def Execute(self):
            assert self.files == ("slice-001.dcm", "slice-002.dcm", "slice-003.dcm")
            return image

    monkeypatch.setattr("ct_vascular_resampling.ct_resampling.sitk.ImageSeriesReader", Reader)

    volume = load_ct(tmp_path, dicom_series_uid="target-series")

    assert volume.data_zyx.shape == (3, 2, 2)


def test_square_fov_diagnosis_marks_exact_pixels_outside_ct_volume():
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.int16))
    volume = CTVolume.from_sitk(image)
    vertices = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 3.0, 0.0],
            [-1.0, 3.0, 0.0],
        ]
    )

    diagnosis = diagnose_square_fov(volume, vertices, resolution=3, probe_point_world=np.asarray([0.0, 0.0, 0.0]))

    assert diagnosis.out_of_bounds_mask.shape == (3, 3)
    assert np.array_equal(diagnosis.out_of_bounds_mask, [[True, False, False]] * 3)
    assert diagnosis.out_of_bounds_ratio == 1.0 / 3.0
    assert diagnosis.probe_point_inside_ct is True
    assert diagnosis.face_out_of_bounds_ratios["x_low"] == 1.0 / 3.0


def test_square_vertices_inside_ct_uses_ct_physical_space_for_all_four_vertices():
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.int16))
    image.SetSpacing((2.0, 3.0, 4.0))
    image.SetOrigin((10.0, 20.0, 30.0))
    image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    volume = CTVolume.from_sitk(image)
    vertices = np.asarray(
        [
            _physical_point(image, (0, 0, 0)),
            _physical_point(image, (3, 0, 0)),
            _physical_point(image, (3, 3, 0)),
            _physical_point(image, (0, 3, 0)),
        ]
    )
    outside_vertices = vertices.copy()
    outside_vertices[3] = _physical_point(image, (0, 4, 0))

    assert square_vertices_inside_ct(volume, vertices) is True
    assert square_vertices_inside_ct(volume, outside_vertices) is False
