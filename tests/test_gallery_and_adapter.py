from __future__ import annotations

import json

import numpy as np
from PIL import Image
import pytest

from ct_vascular_resampling.gallery import GalleryWriter, write_rectangles_ply
from ct_vascular_resampling.geometry import frame_from_vertices
from ct_vascular_resampling.quality import QualityResult
from ct_vascular_resampling.registration_adapter import load_gallery_database
from ct_vascular_resampling.rendering import OrganLayer, VesselLayer, render_sample_images
from ct_vascular_resampling.geometry import SectionContour


def _frame():
    return frame_from_vertices(np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]))


def _rendered():
    contour = SectionContour(
        points_mm=np.asarray([[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]]),
        complete=True,
        centroid_mm=np.asarray([3.5, 3.5]),
        area_mm2=9.0,
    )
    return render_sample_images(
        np.full((20, 20), 127, dtype=np.uint8),
        10.0,
        10.0,
        [VesselLayer("portal_tree", "portal", (255, 0, 255), [contour])],
        organ_layers=[OrganLayer("liver", "liver", (140, 86, 75), [contour])],
    )


def test_gallery_writer_routes_featured_sample_to_gallery_with_compatible_record(tmp_path):
    writer = GalleryWriter(tmp_path / "case_001", case_id="case_001")
    status = writer.write_sample(
        sample_id="stomach-000001",
        organ="stomach",
        probe_point_world=np.asarray([1.0, 2.0, 3.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        frame=_frame(),
        rendered=_rendered(),
        quality=QualityResult(True, None, 0.0),
    )

    assert status == "gallery"
    assert (tmp_path / "case_001" / "gallery" / "ct" / "stomach-000001.png").is_file()
    assert (tmp_path / "case_001" / "gallery" / "organ_vessel_boundary" / "stomach-000001.png").is_file()
    record = json.loads((tmp_path / "case_001" / "gallery" / "gallery.jsonl").read_text(encoding="utf-8"))
    assert record["features"][0]["label"] == "portal"
    assert record["ct_png"] == "ct/stomach-000001.png"
    assert record["organ_vessel_boundary_png"] == "organ_vessel_boundary/stomach-000001.png"
    assert record["organ_labels"] == ["liver"]
    assert record["pixel_spacing_mm"] == [10.0 / 19.0, 10.0 / 19.0]
    assert writer.completed_status("stomach-000001") == "gallery"


def test_gallery_writer_routes_empty_and_rejected_samples_to_separate_directories(tmp_path):
    writer = GalleryWriter(tmp_path / "case_001", case_id="case_001")
    empty = render_sample_images(np.full((20, 20), 127, dtype=np.uint8), 10.0, 10.0, [])
    assert writer.write_sample("empty", "liver", np.zeros(3), np.array([0.0, 0.0, 1.0]), _frame(), empty, QualityResult(True, None, 0.0)) == "unindexed"
    assert writer.write_sample("bad", "liver", np.zeros(3), np.array([0.0, 0.0, 1.0]), _frame(), empty, QualityResult(False, "black_ratio", 0.31)) == "rejected"
    assert (tmp_path / "case_001" / "unindexed" / "unindexed.jsonl").is_file()
    assert (tmp_path / "case_001" / "rejected" / "rejected.jsonl").is_file()
    for status, sample_id in (("unindexed", "empty"), ("rejected", "bad")):
        record = json.loads((tmp_path / "case_001" / status / f"{status}.jsonl").read_text(encoding="utf-8"))
        assert "organ_vessel_boundary_png" not in record
        assert "organ_labels" not in record
        assert not (tmp_path / "case_001" / status / "organ_vessel_boundary" / f"{sample_id}.png").exists()


def test_gallery_writer_persists_fov_exclusion_with_ct_only(tmp_path):
    case_directory = tmp_path / "case_001"
    writer = GalleryWriter(case_directory, case_id="case_001")
    ct_image = Image.fromarray(np.full((20, 20), 91, dtype=np.uint8))

    status = writer.write_fov_exclusion(
        sample_id="esophagus-000010-x-01",
        organ="esophagus",
        probe_point_world=np.asarray([1.0, 2.0, 3.0]),
        input_normal_world=np.asarray([0.0, 0.0, 1.0]),
        frame=_frame(),
        fov_diagnostics={"contains_ct_fov_exceedance": True, "out_of_bounds_ratio": 0.62},
        ct_image=ct_image,
        resampling_backend="cpu",
    )

    assert status == "excluded_fov"
    assert writer.completed_status("esophagus-000010-x-01") == "excluded_fov"
    record = json.loads((case_directory / "excluded_fov.jsonl").read_text(encoding="utf-8"))
    ct_path = case_directory / "excluded_fov" / "ct" / "esophagus-000010-x-01.png"
    assert record["status"] == "excluded_fov"
    assert record["exclusion_reason"] == "ct_fov_exceeded"
    assert record["fov_diagnostics"]["out_of_bounds_ratio"] == 0.62
    assert record["ct_png"] == "ct/esophagus-000010-x-01.png"
    assert record["resampling_backend"] == "cpu"
    assert "boundary_only_png" not in record
    assert "ct_overlay_png" not in record
    assert "organ_vessel_boundary_png" not in record
    assert "organ_labels" not in record
    with Image.open(ct_path) as saved:
        assert saved.mode == "L"
        assert np.all(np.asarray(saved) == 91)
    assert GalleryWriter(case_directory, case_id="case_001").completed_status("esophagus-000010-x-01") == "excluded_fov"


def test_gallery_writer_rejects_resume_from_gallery_without_organ_artifacts(tmp_path):
    case_directory = tmp_path / "case_001"
    case_directory.mkdir()
    (case_directory / "manifest.jsonl").write_text(
        json.dumps({"slice_id": "old-gallery", "status": "gallery"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="旧版|organ_vessel_boundary"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_state_manifest_when_root_manifest_is_missing(tmp_path):
    case_directory = tmp_path / "case_001"
    gallery_directory = case_directory / "gallery"
    gallery_directory.mkdir(parents=True)
    (gallery_directory / "gallery.jsonl").write_text(
        json.dumps({"slice_id": "old-gallery", "status": "gallery"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest.jsonl|旧版"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_old_record_found_only_in_gallery_manifest(tmp_path):
    case_directory = tmp_path / "case_001"
    gallery_directory = case_directory / "gallery"
    gallery_directory.mkdir(parents=True)
    (case_directory / "manifest.jsonl").write_text(
        json.dumps({"slice_id": "known-unindexed", "status": "unindexed"}) + "\n",
        encoding="utf-8",
    )
    (gallery_directory / "gallery.jsonl").write_text(
        json.dumps({"slice_id": "old-gallery", "status": "gallery"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="旧版|organ_vessel_boundary"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_repairs_root_record_when_state_manifest_append_was_interrupted(tmp_path, monkeypatch):
    case_directory = tmp_path / "case_001"
    writer = GalleryWriter(case_directory, case_id="case_001")
    original_append = GalleryWriter._append_jsonl
    calls = 0

    def interrupt_second_append(path, record):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption after root manifest append")
        original_append(path, record)

    monkeypatch.setattr(GalleryWriter, "_append_jsonl", staticmethod(interrupt_second_append))
    with pytest.raises(OSError, match="simulated interruption"):
        writer.write_sample(
            "interrupted-gallery",
            "stomach",
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            _frame(),
            _rendered(),
            QualityResult(True, None, 0.0),
        )

    monkeypatch.setattr(GalleryWriter, "_append_jsonl", staticmethod(original_append))
    resumed = GalleryWriter(case_directory, case_id="case_001")

    assert resumed.completed_status("interrupted-gallery") == "gallery"
    gallery_records = [
        json.loads(line)
        for line in (case_directory / "gallery" / "gallery.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["slice_id"] for record in gallery_records] == ["interrupted-gallery"]
    root_records = [
        json.loads(line)
        for line in (case_directory / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["slice_id"] for record in root_records] == ["interrupted-gallery"]


@pytest.mark.parametrize(
    "organ_labels",
    [
        pytest.param(["liver", "liver"], id="duplicate"),
        pytest.param(["stomach", "liver"], id="unsorted"),
        pytest.param(["aorta"], id="unsupported"),
        pytest.param([1], id="non-string"),
    ],
)
def test_gallery_writer_rejects_invalid_organ_labels_on_resume(tmp_path, organ_labels):
    case_directory = tmp_path / "case_001"
    combined_path = case_directory / "gallery" / "organ_vessel_boundary" / "sample.png"
    combined_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(combined_path)
    record = {
        "slice_id": "sample",
        "status": "gallery",
        "organ_vessel_boundary_png": "organ_vessel_boundary/sample.png",
        "organ_labels": organ_labels,
    }
    serialized = json.dumps(record) + "\n"
    (case_directory / "manifest.jsonl").write_text(serialized, encoding="utf-8")
    (case_directory / "gallery" / "gallery.jsonl").write_text(serialized, encoding="utf-8")

    with pytest.raises(ValueError, match="organ_labels"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_state_record_missing_from_root_manifest(tmp_path):
    case_directory = tmp_path / "case_001"
    case_directory.mkdir()
    (case_directory / "manifest.jsonl").write_text(
        json.dumps({"slice_id": "known", "status": "unindexed"}) + "\n",
        encoding="utf-8",
    )
    unindexed_directory = case_directory / "unindexed"
    unindexed_directory.mkdir()
    (unindexed_directory / "unindexed.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"slice_id": "known", "status": "unindexed"}),
                json.dumps({"slice_id": "orphan", "status": "unindexed"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="根 manifest|orphan"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_duplicate_state_manifest_records(tmp_path):
    case_directory = tmp_path / "case_001"
    case_directory.mkdir()
    record = {"slice_id": "duplicate", "status": "unindexed"}
    serialized = json.dumps(record) + "\n"
    (case_directory / "manifest.jsonl").write_text(serialized, encoding="utf-8")
    unindexed_directory = case_directory / "unindexed"
    unindexed_directory.mkdir()
    (unindexed_directory / "unindexed.jsonl").write_text(serialized * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="重复|duplicate"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_slice_id_in_multiple_root_statuses(tmp_path):
    case_directory = tmp_path / "case_001"
    case_directory.mkdir()
    root_records = [
        {"slice_id": "conflict", "status": "unindexed"},
        {"slice_id": "conflict", "status": "rejected"},
    ]
    (case_directory / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in root_records),
        encoding="utf-8",
    )
    for record in root_records:
        status = record["status"]
        directory = case_directory / status
        directory.mkdir()
        (directory / f"{status}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重复|多个状态|conflict"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_rejects_mismatched_root_and_state_records(tmp_path):
    case_directory = tmp_path / "case_001"
    case_directory.mkdir()
    root_record = {"slice_id": "mismatch", "status": "unindexed", "organ": "liver"}
    state_record = {"slice_id": "mismatch", "status": "unindexed", "organ": "stomach"}
    (case_directory / "manifest.jsonl").write_text(json.dumps(root_record) + "\n", encoding="utf-8")
    unindexed_directory = case_directory / "unindexed"
    unindexed_directory.mkdir()
    (unindexed_directory / "unindexed.jsonl").write_text(json.dumps(state_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="内容不一致|mismatch"):
        GalleryWriter(case_directory, case_id="case_001")


def test_gallery_writer_preserves_combined_line_and_black_ratio_quality_evidence(tmp_path):
    writer = GalleryWriter(tmp_path / "case_001", case_id="case_001")
    empty = render_sample_images(np.full((20, 20), 127, dtype=np.uint8), 10.0, 10.0, [])

    status = writer.write_sample(
        "boundary",
        "liver",
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        _frame(),
        empty,
        QualityResult(
            False,
            "black_boundary_line",
            0.60,
            line_length_px=100.0,
            black_side_ratio=1.0,
            valid_side_black_ratio=0.0,
            line_segment_px=(10, 0, 10, 99),
            black_ratio_exceeded=True,
        ),
    )

    record = json.loads((tmp_path / "case_001" / "rejected" / "rejected.jsonl").read_text(encoding="utf-8"))
    assert status == "rejected"
    assert record["quality"]["reason"] == "black_boundary_line"
    assert record["quality"]["black_ratio_exceeded"] is True
    assert record["quality"]["line_segment_px"] == [10, 0, 10, 99]


def test_rectangle_ply_contains_four_vertices_per_frame(tmp_path):
    path = tmp_path / "rectangles.ply"

    write_rectangles_ply(path, [_frame()])

    assert "element vertex 4" in path.read_text(encoding="utf-8")


def _write_fake_2021(path):
    path.write_text(
        """
class VesselTriplet:
    def __init__(self, x, y, area, label=''):
        self.x, self.y, self.area, self.label = x, y, area, label
class FeatureVector:
    def __init__(self, triplets=None, pose=None):
        self.triplets, self.pose = triplets or [], pose
class ProbePose:
    def __init__(self, surface_point, rx, ry, rz, depth):
        self.surface_point, self.rx, self.ry, self.rz, self.depth = surface_point, rx, ry, rz, depth
class MultiLabelledCBIR:
    def __init__(self, database, search_range=2):
        self.database, self.search_range = database, search_range
class HMMPoseEstimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
""".strip(),
        encoding="utf-8",
    )


def test_registration_adapter_reads_gallery_jsonl_and_creates_cbir_database(tmp_path):
    writer = GalleryWriter(tmp_path / "case_001", case_id="case_001")
    writer.write_sample("sample", "stomach", np.zeros(3), np.array([0.0, 0.0, 1.0]), _frame(), _rendered(), QualityResult(True, None, 0.0))
    module_path = tmp_path / "2021.py"
    _write_fake_2021(module_path)

    database = load_gallery_database(tmp_path / "case_001" / "gallery", module_path)

    assert list(database.database) == ["portal:1"]
    assert database.features[0].triplets[0].area == 9.0
    assert database.create_cbir(search_range=4).search_range == 4
