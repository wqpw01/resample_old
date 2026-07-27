from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import SimpleITK as sitk


def _write_label_tar(directory: Path) -> None:
    stem = directory.name
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[2:4, 2:4] = 26  # Complete vein component.
    labels[5:7, 6:8] = 33  # Complete artery component.
    labels[0:2, 2:4] = 31  # Incomplete vein component touching the image edge.
    image_path = directory / "labels.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(labels), str(image_path))
    metadata = {
        "FileInfo": {"Width": 10, "Height": 10},
        "Models": {
            "ColorLabelTableModel": [
                {"ID": 26, "Desc": "门静脉（包括分支", "Color": [170, 85, 255, 255]},
                {"ID": 31, "Desc": "肝静脉", "Color": [0, 0, 255, 255]},
                {"ID": 33, "Desc": "腹主动脉", "Color": [255, 0, 0, 255]},
            ]
        }
    }
    tar_path = directory / f"{stem}_cropped_jpg_Label.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo(f"{stem}_cropped_jpg_Label.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        archive.add(image_path, arcname=f"{stem}_cropped_jpg_Label.nii.gz")


def _write_empty_label_tar(directory: Path) -> None:
    stem = directory.name
    metadata = {"FileInfo": {"Width": 10, "Height": 10}, "Models": {"ColorLabelTableModel": []}}
    tar_path = directory / f"{stem}_cropped_jpg_Label.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        info = tarfile.TarInfo(f"{stem}_cropped_jpg_Label.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def test_process_cropped_folder_extracts_complete_vessels_and_writes_adapter_record(tmp_path):
    from ct_vascular_resampling.cropped_retrieval import process_cropped_folder

    folder = tmp_path / "frame_00000273"
    folder.mkdir()
    _write_label_tar(folder)

    result = process_cropped_folder(folder)

    details = json.loads(result.feature_path.read_text(encoding="utf-8"))
    gallery_record = json.loads(result.gallery_path.read_text(encoding="utf-8"))
    assert [feature["label"] for feature in details["features"]] == ["vein", "artery"]
    assert [feature["label_id"] for feature in details["features"]] == [26, 33]
    assert details["skipped_components"][0]["reason"] == "touches_image_edge"
    assert details["features"][0]["area_px"] == 4
    assert details["features"][0]["x_mm"] == 2.5 * 100.0 / 9.0
    assert details["features"][0]["y_mm"] == 2.5 * 100.0 / 9.0
    assert gallery_record["features"] == [
        {key: feature[key] for key in ("label", "x_mm", "y_mm", "area_mm2")}
        for feature in details["features"]
    ]
    assert gallery_record["pose_coordinate_system"] == "synthetic_2d_10cm_crop"
    assert gallery_record["patient_world_pose"] is False
    assert result.label_white_path.is_file()


def test_process_cropped_root_keeps_retrieval_results_in_each_frame_folder(tmp_path):
    from ct_vascular_resampling.cropped_retrieval import process_cropped_root

    for name in ("frame_00000273", "frame_00000300"):
        folder = tmp_path / name
        folder.mkdir()
        _write_label_tar(folder)

    summary = process_cropped_root(tmp_path)

    assert summary.folder_count == 2
    assert summary.gallery_record_count == 2
    assert not (tmp_path / "retrieval_gallery.jsonl").exists()
    assert not (tmp_path / "retrieval_feature_summary.json").exists()
    for name in ("frame_00000273", "frame_00000300"):
        assert (tmp_path / name / f"{name}_cropped_retrieval_features.json").is_file()
        assert (tmp_path / name / f"{name}_cropped_gallery.jsonl").is_file()


def test_process_cropped_folder_records_json_only_tar_as_unindexed_empty_label(tmp_path):
    from ct_vascular_resampling.cropped_retrieval import process_cropped_folder

    folder = tmp_path / "frame_00003744"
    folder.mkdir()
    _write_empty_label_tar(folder)

    result = process_cropped_folder(folder)

    details = json.loads(result.feature_path.read_text(encoding="utf-8"))
    record = json.loads(result.gallery_path.read_text(encoding="utf-8"))
    assert details["label_source"] == "empty_label_json"
    assert details["features"] == []
    assert details["skipped_components"] == []
    assert record["status"] == "unindexed"
    assert record["features"] == []
    assert np.all(np.asarray(Image.open(result.label_white_path).convert("RGB")) == 255)
