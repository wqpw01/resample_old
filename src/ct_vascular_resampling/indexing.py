"""仓库内 Gallery 清单校验与论文级汇总索引。"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

from .config import DEFAULT_ORGAN_COLORS, ORGAN_BOUNDARY_IDS, ManualSegmentationConfig
from .eus_organs import load_eus_organ_catalog
from .gallery import validate_gallery_eus_vessel_metadata, validate_gallery_organ_metadata


def _iter_gallery_records(manifest: Path):
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"gallery 清单第 {line_number} 行不是有效 JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"gallery 清单第 {line_number} 行必须是 JSON 对象")
            yield line_number, record


def build_library_summary(
    *,
    case_id: str,
    case_directory: str | Path,
    manual_segmentation: ManualSegmentationConfig,
) -> dict[str, Any]:
    """校验 Gallery JSONL 并生成不依赖外部检索模块的汇总。"""

    case_path = Path(case_directory)
    manifest = case_path / "gallery" / "gallery.jsonl"
    organ_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    eus_label_counts: Counter[str] = Counter()
    eus_feature_counts: Counter[str] = Counter()
    feature_label_counts: Counter[str] = Counter()
    indexed_feature_count = 0
    catalog = load_eus_organ_catalog()

    if manifest.is_file():
        for line_number, record in _iter_gallery_records(manifest):
            try:
                if record.get("status") != "gallery":
                    raise ValueError("gallery.jsonl 记录状态必须为 gallery")
                validate_gallery_organ_metadata(record, manifest.parent, catalog)
                validate_gallery_eus_vessel_metadata(record, manifest.parent)
                features = record.get("features", [])
                if not isinstance(features, list):
                    raise ValueError("features 必须是列表")
                indexed_feature_count += int(bool(features))
                for feature in features:
                    if not isinstance(feature, dict) or set(feature) != {
                        "label",
                        "x_mm",
                        "y_mm",
                        "area_mm2",
                    }:
                        raise ValueError("features 项字段无效")
                    if feature["label"] not in {"artery", "vein"}:
                        raise ValueError("features.label 必须是 artery 或 vein")
                    for field in ("x_mm", "y_mm", "area_mm2"):
                        value = feature[field]
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            raise ValueError(f"features.{field} 必须是有限数值")
                    if float(feature["area_mm2"]) <= 0.0:
                        raise ValueError("features.area_mm2 必须是正数")
                    feature_label_counts.update([feature["label"]])
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"gallery 清单第 {line_number} 行校验失败: {error}") from error
            organ_counts.update(record["organ_labels"])
            candidate_counts.update(record["eus_candidate_organ_labels"])
            eus_label_counts.update(record["eus_vessel_labels"])
            eus_feature_counts.update(
                str(feature["label"]) for feature in record["eus_vessel_features"]
            )

    return {
        "case_id": case_id,
        "gallery_manifest": "gallery/gallery.jsonl",
        "gallery_manifest_exists": manifest.is_file(),
        "indexed_feature_count": indexed_feature_count,
        "feature_label_counts": dict(sorted(feature_label_counts.items())),
        "organ_label_counts": dict(sorted(organ_counts.items())),
        "eus_candidate_organ_label_counts": dict(sorted(candidate_counts.items())),
        "eus_possible_organs": catalog.to_record(),
        "organ_boundary_colors": {
            identifier: list(DEFAULT_ORGAN_COLORS[identifier])
            for identifier in ORGAN_BOUNDARY_IDS
        },
        "eus_vessel_label_counts": dict(sorted(eus_label_counts.items())),
        "eus_vessel_feature_counts": dict(sorted(eus_feature_counts.items())),
        "eus_vessel_colors": {
            identifier: list(color)
            for identifier, color in sorted(manual_segmentation.eus_vessel_colors.items())
        },
        "eus_vessel_label_values": {
            identifier: list(values)
            for identifier, values in sorted(manual_segmentation.eus_vessel_label_values.items())
        },
        "manual_organ_label_values": {
            identifier: list(values)
            for identifier, values in sorted(manual_segmentation.organ_label_values.items())
        },
    }
