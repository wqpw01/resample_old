"""提取历史 rejected 中由 CT FOV 边界造成的黑色直线样本。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class ExtractionSummary:
    sample_ids: tuple[str, ...]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL 不存在: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            records.append(value)
    return records


def _copy_required(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"待提取文件不存在: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.name


def _copy_optional(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.name


def _json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def extract_fov_boundary_samples(
    library_root: str | Path,
    audit_jsonl: str | Path,
    destination: str | Path,
    *,
    limit: int = 5,
) -> ExtractionSummary:
    """提取原判为 black_ratio 且审计显示 FOV 边界对齐的样本。"""

    if limit < 1:
        raise ValueError("limit 必须大于零")
    source_root = Path(library_root)
    rejected_root = source_root / "rejected"
    audit_path = Path(audit_jsonl)
    output_root = Path(destination)
    source_records = _read_jsonl(rejected_root / "rejected.jsonl")
    source_by_id = {str(record["slice_id"]): record for record in source_records}
    audit_records = _read_jsonl(audit_path)
    selected_audits = sorted(
        (
            record
            for record in audit_records
            if str(record.get("original_quality", {}).get("reason")) == "black_ratio"
            and str(record.get("fov_diagnostics", {}).get("cause")) == "fov_boundary_aligned"
        ),
        key=lambda record: str(record["slice_id"]),
    )[:limit]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, object]] = []
    csv_rows: list[dict[str, str]] = []
    selected_ids: list[str] = []
    representative_root = audit_path.parent / "representatives" / "fov_boundary_aligned"
    for audit_record in selected_audits:
        sample_id = str(audit_record["slice_id"])
        try:
            source_record = source_by_id[sample_id]
        except KeyError as error:
            raise ValueError(f"审计样本不在 rejected.jsonl 中: {sample_id}") from error
        sample_directory = output_root / sample_id
        sample_directory.mkdir(parents=True, exist_ok=True)
        copied_files = {
            "ct_png": _copy_required(rejected_root / str(source_record["ct_png"]), sample_directory / "ct.png"),
            "ct_overlay_png": _copy_required(
                rejected_root / str(source_record["ct_overlay_png"]), sample_directory / "ct_overlay.png"
            ),
            "boundary_only_png": _copy_required(
                rejected_root / str(source_record["boundary_only_png"]), sample_directory / "boundary_only.png"
            ),
        }
        optional_files = {
            "fov_oob_mask_png": _copy_optional(
                representative_root / f"{sample_id}_oob_mask.png", sample_directory / "fov_oob_mask.png"
            ),
            "fov_overlay_png": _copy_optional(
                representative_root / f"{sample_id}_overlay.png", sample_directory / "fov_overlay.png"
            ),
        }
        metadata = {
            "slice_id": sample_id,
            "selection_rule": {
                "original_quality_reason": "black_ratio",
                "fov_cause": "fov_boundary_aligned",
            },
            "source_record": source_record,
            "audit_record": audit_record,
            "copied_files": copied_files,
            "optional_files": optional_files,
        }
        (sample_directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fov = audit_record.get("fov_diagnostics", {})
        manifest_records.append(
            {
                "slice_id": sample_id,
                "organ": source_record.get("organ"),
                "sample_directory": sample_id,
                "metadata_json": f"{sample_id}/metadata.json",
                "quality_reason": source_record.get("quality", {}).get("reason"),
                "black_ratio": source_record.get("quality", {}).get("black_ratio"),
                "fov_cause": fov.get("cause"),
                "out_of_bounds_ratio": fov.get("out_of_bounds_ratio"),
            }
        )
        csv_rows.append(
            {
                "slice_id": sample_id,
                "organ": str(source_record.get("organ", "")),
                "probe_point_world": _json_cell(source_record.get("probe_point_world")),
                "origin_world": _json_cell(source_record.get("origin_world")),
                "center_world": _json_cell(source_record.get("center_world")),
                "square_vertices_world": _json_cell(source_record.get("square_vertices_world")),
                "quality_reason": str(source_record.get("quality", {}).get("reason", "")),
                "black_ratio": str(source_record.get("quality", {}).get("black_ratio", "")),
                "fov_cause": str(fov.get("cause", "")),
                "out_of_bounds_ratio": str(fov.get("out_of_bounds_ratio", "")),
                "face_out_of_bounds_ratios": _json_cell(fov.get("face_out_of_bounds_ratios")),
                "continuous_index_min_zyx": _json_cell(fov.get("continuous_index_min_zyx")),
                "continuous_index_max_zyx": _json_cell(fov.get("continuous_index_max_zyx")),
                "probe_point_inside_ct": str(fov.get("probe_point_inside_ct", "")),
            }
        )
        selected_ids.append(sample_id)
    (output_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in manifest_records), encoding="utf-8"
    )
    fields = [
        "slice_id",
        "organ",
        "probe_point_world",
        "origin_world",
        "center_world",
        "square_vertices_world",
        "quality_reason",
        "black_ratio",
        "fov_cause",
        "out_of_bounds_ratio",
        "face_out_of_bounds_ratios",
        "continuous_index_min_zyx",
        "continuous_index_max_zyx",
        "probe_point_inside_ct",
    ]
    with (output_root / "locations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    return ExtractionSummary(tuple(selected_ids))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 CT FOV 边界导致的 black_ratio 历史样本")
    parser.add_argument("--library-root", required=True, help="病例图库根目录，例如 case_2")
    parser.add_argument("--audit-jsonl", required=True, help="rejected_fov_audit.jsonl 路径")
    parser.add_argument("--destination", required=True, help="提取结果目录")
    parser.add_argument("--limit", type=int, default=5, help="最多提取样本数，默认 5")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = extract_fov_boundary_samples(args.library_root, args.audit_jsonl, args.destination, limit=args.limit)
    print(f"已提取 {len(summary.sample_ids)} 个样本: {', '.join(summary.sample_ids)}")


if __name__ == "__main__":
    main()
