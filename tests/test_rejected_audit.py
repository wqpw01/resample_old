from __future__ import annotations

import importlib
import json

import numpy as np
import SimpleITK as sitk
from PIL import Image

from ct_vascular_resampling.config import FilterConfig


def test_rejected_audit_writes_fov_cause_summary_and_representative_images(tmp_path):
    module = importlib.import_module("ct_vascular_resampling.rejected_audit")
    ct_path = tmp_path / "ct.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(np.full((1, 100, 100), 40, dtype=np.int16)), str(ct_path))
    rejected_root = tmp_path / "case" / "rejected"
    ct_directory = rejected_root / "ct"
    ct_directory.mkdir(parents=True)
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    pixels[:, :51] = 0
    Image.fromarray(pixels).save(ct_directory / "sample.png")
    vertices = [[-100.0, 0.0, 0.0], [99.0, 0.0, 0.0], [99.0, 99.0, 0.0], [-100.0, 99.0, 0.0]]
    rejected_jsonl = rejected_root / "rejected.jsonl"
    rejected_jsonl.write_text(
        json.dumps(
            {
                "slice_id": "sample",
                "status": "rejected",
                "organ": "stomach",
                "probe_point_world": [0.0, 0.0, 0.0],
                "square_vertices_world": vertices,
                "ct_png": "ct/sample.png",
                "quality": {"reason": "black_ratio"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = module.RejectedAuditConfig(
        ct_path=ct_path,
        dicom_series_uid=None,
        rejected_jsonl=rejected_jsonl,
        output_directory=rejected_root / "diagnostics",
        filtering=FilterConfig(),
        representative_limit_per_cause=1,
    )

    summary = module.run_rejected_audit(config)

    audit_record = json.loads((rejected_root / "diagnostics" / "rejected_fov_audit.jsonl").read_text(encoding="utf-8"))
    assert summary.sample_count == 1
    assert summary.cause_counts == {"fov_boundary_aligned": 1}
    assert audit_record["recomputed_quality"]["black_ratio_exceeded"] is True
    assert audit_record["fov_diagnostics"]["cause"] == "fov_boundary_aligned"
    assert (rejected_root / "diagnostics" / "representatives" / "fov_boundary_aligned" / "sample_overlay.png").is_file()
