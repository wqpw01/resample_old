"""对既有 rejected 图库进行 CT FOV 空间归因审计。"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml

from .config import FilterConfig, _as_path, _integer, _load_filter, _mapping
from .ct_resampling import load_ct
from .fov_diagnostics import RejectedFovAssessment, assess_rejected_fov
from .quality import QualityResult, evaluate_ct_quality


@dataclass(frozen=True)
class RejectedAuditConfig:
    ct_path: Path
    dicom_series_uid: str | None
    rejected_jsonl: Path
    output_directory: Path
    filtering: FilterConfig
    representative_limit_per_cause: int = 30


@dataclass(frozen=True)
class RejectedAuditSummary:
    sample_count: int
    cause_counts: dict[str, int]
    original_quality_reason_counts: dict[str, int]
    organ_counts: dict[str, int]


def load_rejected_audit_config(path: str | Path) -> RejectedAuditConfig:
    """读取既有 rejected 图库的独立审计配置。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = _mapping(yaml.safe_load(handle) or {}, "rejected 审计配置")
    series_uid = values.get("dicom_series_uid")
    if series_uid is not None and (not isinstance(series_uid, str) or not series_uid):
        raise ValueError("dicom_series_uid 必须是非空字符串")
    return RejectedAuditConfig(
        ct_path=_as_path(values.get("ct_path"), config_path.parent, "ct_path"),
        dicom_series_uid=series_uid,
        rejected_jsonl=_as_path(values.get("rejected_jsonl"), config_path.parent, "rejected_jsonl"),
        output_directory=_as_path(values.get("output_directory"), config_path.parent, "output_directory"),
        filtering=_load_filter(values.get("filtering")),
        representative_limit_per_cause=_integer(
            values.get("representative_limit_per_cause"),
            "representative_limit_per_cause",
            30,
            minimum=0,
        ),
    )


def _quality_record(quality: QualityResult) -> dict[str, object]:
    return {
        "accepted": quality.accepted,
        "reason": quality.reason,
        "black_ratio": quality.black_ratio,
        "line_length_px": quality.line_length_px,
        "black_side_ratio": quality.black_side_ratio,
        "valid_side_black_ratio": quality.valid_side_black_ratio,
        "line_segment_px": list(quality.line_segment_px) if quality.line_segment_px is not None else None,
    }


def _save_representative(
    root: Path,
    sample_id: str,
    pixels: np.ndarray,
    assessment: RejectedFovAssessment,
    quality: QualityResult,
) -> None:
    destination = root / assessment.cause
    destination.mkdir(parents=True, exist_ok=True)
    Image.fromarray((assessment.fov.out_of_bounds_mask * 255).astype(np.uint8)).save(destination / f"{sample_id}_oob_mask.png")
    base = np.asarray(pixels, dtype=np.uint8)
    overlay = np.repeat(base[..., None], 3, axis=2)
    overlay[assessment.fov.out_of_bounds_mask] = (255, 48, 48)
    if quality.line_segment_px is not None:
        x1, y1, x2, y2 = quality.line_segment_px
        cv2.line(overlay, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
    Image.fromarray(overlay).save(destination / f"{sample_id}_overlay.png")


def run_rejected_audit(config: RejectedAuditConfig) -> RejectedAuditSummary:
    """复算旧 rejected 切片的 FOV 指标，原清单与 PNG 均保持不变。"""

    if config.representative_limit_per_cause < 0:
        raise ValueError("representative_limit_per_cause 不能为负数")
    if not config.rejected_jsonl.is_file():
        raise FileNotFoundError(f"rejected JSONL 不存在: {config.rejected_jsonl}")
    volume = load_ct(config.ct_path, dicom_series_uid=config.dicom_series_uid)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    representatives_root = config.output_directory / "representatives"
    audit_path = config.output_directory / "rejected_fov_audit.jsonl"
    csv_path = config.output_directory / "summary.csv"
    summary_path = config.output_directory / "summary.json"
    cause_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    organ_counts: Counter[str] = Counter()
    representative_counts: Counter[str] = Counter()
    csv_rows: list[dict[str, object]] = []
    sample_count = 0
    temporary_audit = audit_path.with_name(f".{audit_path.name}.tmp")
    with config.rejected_jsonl.open("r", encoding="utf-8") as source, temporary_audit.open("w", encoding="utf-8", newline="\n") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "rejected":
                raise ValueError(f"第 {line_number} 行不是 rejected 样本")
            sample_id = str(record["slice_id"])
            image_path = config.rejected_jsonl.parent / str(record["ct_png"])
            if not image_path.is_file():
                raise FileNotFoundError(f"样本 CT PNG 不存在 ({sample_id}): {image_path}")
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("L"), dtype=np.uint8)
            quality = evaluate_ct_quality(pixels, config.filtering)
            assessment = assess_rejected_fov(
                volume,
                np.asarray(record["square_vertices_world"], dtype=np.float64),
                pixels,
                config.filtering,
                quality,
                probe_point_world=np.asarray(record["probe_point_world"], dtype=np.float64),
            )
            audit_record = {
                "slice_id": sample_id,
                "organ": record.get("organ"),
                "ct_png": record["ct_png"],
                "original_quality": record.get("quality", {}),
                "recomputed_quality": _quality_record(quality),
                "fov_diagnostics": assessment.to_record(),
            }
            destination.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
            sample_count += 1
            cause_counts[assessment.cause] += 1
            quality_counts[str(record.get("quality", {}).get("reason"))] += 1
            organ_counts[str(record.get("organ"))] += 1
            csv_rows.append(
                {
                    "slice_id": sample_id,
                    "organ": record.get("organ"),
                    "original_quality_reason": record.get("quality", {}).get("reason"),
                    "fov_cause": assessment.cause,
                    "contains_ct_fov_exceedance": assessment.fov.contains_ct_fov_exceedance,
                    "out_of_bounds_ratio": assessment.fov.out_of_bounds_ratio,
                    "black_out_of_bounds_overlap_ratio": assessment.black_out_of_bounds_overlap_ratio,
                    "boundary_line_oob_alignment_ratio": assessment.boundary_line_oob_alignment_ratio,
                }
            )
            if representative_counts[assessment.cause] < config.representative_limit_per_cause:
                _save_representative(representatives_root, sample_id, pixels, assessment, quality)
                representative_counts[assessment.cause] += 1
    temporary_audit.replace(audit_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]) if csv_rows else ["slice_id"])
        writer.writeheader()
        writer.writerows(csv_rows)
    summary = RejectedAuditSummary(
        sample_count=sample_count,
        cause_counts=dict(sorted(cause_counts.items())),
        original_quality_reason_counts=dict(sorted(quality_counts.items())),
        organ_counts=dict(sorted(organ_counts.items())),
    )
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": summary.sample_count,
                "cause_counts": summary.cause_counts,
                "original_quality_reason_counts": summary.original_quality_reason_counts,
                "organ_counts": summary.organ_counts,
                "ct_shape_zyx": list(volume.data_zyx.shape),
                "ct_spacing_xyz": [float(value) for value in volume.spacing_xyz],
                "ct_origin_xyz": [float(value) for value in volume.origin_xyz],
                "ct_direction_xyz": volume.direction_xyz.tolist(),
                "dicom_series_uid": config.dicom_series_uid,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
