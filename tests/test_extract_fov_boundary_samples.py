from __future__ import annotations

import csv
import json

from scripts.extract_fov_boundary_samples import extract_fov_boundary_samples


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_extract_fov_boundary_samples_copies_selected_assets_and_location_metadata(tmp_path):
    library_root = tmp_path / "case_2"
    rejected_root = library_root / "rejected"
    sample_id = "esophagus-000010-x-01"
    source_record = {
        "slice_id": sample_id,
        "organ": "esophagus",
        "probe_point_world": [15.0, 4.0, 777.0],
        "origin_world": [10.0, -36.0, 750.0],
        "center_world": [15.0, -23.0, 819.0],
        "square_vertices_world": [[10.0, -36.0, 750.0]] * 4,
        "quality": {"reason": "black_ratio", "black_ratio": 0.64},
        "ct_png": f"ct/{sample_id}.png",
        "ct_overlay_png": f"ct_overlay/{sample_id}.png",
        "boundary_only_png": f"boundary_only/{sample_id}.png",
    }
    ignored_record = {**source_record, "slice_id": "other", "quality": {"reason": "black_ratio", "black_ratio": 0.70}}
    _write_jsonl(rejected_root / "rejected.jsonl", [source_record, ignored_record])
    for relative_path in (source_record["ct_png"], source_record["ct_overlay_png"], source_record["boundary_only_png"]):
        path = rejected_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_path.encode("utf-8"))
    representative_root = rejected_root / "diagnostics" / "representatives" / "fov_boundary_aligned"
    representative_root.mkdir(parents=True)
    (representative_root / f"{sample_id}_oob_mask.png").write_bytes(b"mask")
    (representative_root / f"{sample_id}_overlay.png").write_bytes(b"overlay")
    audit_jsonl = rejected_root / "diagnostics" / "rejected_fov_audit.jsonl"
    _write_jsonl(
        audit_jsonl,
        [
            {
                "slice_id": sample_id,
                "original_quality": {"reason": "black_ratio"},
                "fov_diagnostics": {"cause": "fov_boundary_aligned", "out_of_bounds_ratio": 0.63},
            },
            {
                "slice_id": "other",
                "original_quality": {"reason": "black_ratio"},
                "fov_diagnostics": {"cause": "ct_fov_exceeded", "out_of_bounds_ratio": 0.70},
            },
        ],
    )

    destination = tmp_path / "extracted"
    summary = extract_fov_boundary_samples(library_root, audit_jsonl, destination, limit=5)

    assert summary.sample_ids == (sample_id,)
    sample_directory = destination / sample_id
    assert (sample_directory / "ct.png").read_bytes() == source_record["ct_png"].encode("utf-8")
    assert (sample_directory / "ct_overlay.png").is_file()
    assert (sample_directory / "boundary_only.png").is_file()
    assert (sample_directory / "fov_oob_mask.png").read_bytes() == b"mask"
    metadata = json.loads((sample_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_record"]["probe_point_world"] == [15.0, 4.0, 777.0]
    manifest = json.loads((destination / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["slice_id"] == sample_id
    with (destination / "locations.csv").open(newline="", encoding="utf-8") as handle:
        locations = list(csv.DictReader(handle))
    assert locations[0]["slice_id"] == sample_id
    assert locations[0]["fov_cause"] == "fov_boundary_aligned"
