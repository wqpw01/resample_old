#!/usr/bin/env python3
"""从手工三维标签精确重建独立的白底 EUS 血管填充图。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_vascular_resampling.label_resampling import CpuLabelBackend, load_label_volume


EXPECTED_SEGMENTATION_SHA256 = (
    "0b56268488411925d96bb070e25e72a0105a8502e87ffd349a9ba01cd32dc124"
)
RECORDS_FILENAME = "gallery_sample_100_unique_positions.jsonl"
OUTPUT_RESOLUTION = 300
EXPECTED_POSITION_COUNT = 100


@dataclass(frozen=True)
class VesselClassSpec:
    label_values: tuple[int, ...]
    color_rgb: tuple[int, int, int]
    chinese_name: str


CLASS_SPECS = {
    "aorta": VesselClassSpec((8,), (255, 0, 0), "腹主动脉"),
    "inferior_vena_cava": VesselClassSpec((9,), (0, 0, 255), "下腔静脉"),
    "portal_vein": VesselClassSpec(
        (26, 33, 34, 35, 36, 37),
        (170, 85, 255),
        "门静脉系",
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def render_filled_rgb(labels: np.ndarray, spec: VesselClassSpec) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("标签平面必须是二维数组")
    mask = np.isin(values, spec.label_values)
    rgb = np.full((*values.shape, 3), 255, dtype=np.uint8)
    rgb[mask] = spec.color_rgb
    return rgb


def validate_boundary_subset(
    boundary_mask: np.ndarray,
    source_mask: np.ndarray,
    path: Path,
) -> None:
    if boundary_mask.shape != source_mask.shape:
        raise ValueError(f"边界图尺寸不一致: {path}")
    outside_count = int(np.count_nonzero(boundary_mask & ~source_mask))
    if outside_count:
        raise ValueError(f"边界像素不属于重采样标签: {path}, count={outside_count}")


def validate_filename_sets(
    actual_by_class: dict[str, set[str]],
    expected_names: set[str],
) -> None:
    if len(expected_names) != EXPECTED_POSITION_COUNT:
        raise ValueError(
            f"JSONL 位置数不是 {EXPECTED_POSITION_COUNT}: {len(expected_names)}"
        )
    if set(actual_by_class) != set(CLASS_SPECS):
        raise ValueError("血管类别目录集合不一致")
    for identifier in CLASS_SPECS:
        actual = actual_by_class[identifier]
        if actual != expected_names:
            missing = sorted(expected_names - actual)[:5]
            unexpected = sorted(actual - expected_names)[:5]
            raise ValueError(
                f"文件名集合不一致: {identifier}, missing={missing}, "
                f"unexpected={unexpected}, count={len(actual)}"
            )


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"切面 JSONL 不存在: {path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 第 {line_number} 行无法解析: {error}") from error
            slice_id = record.get("slice_id")
            if not isinstance(slice_id, str) or not slice_id:
                raise ValueError(f"JSONL 第 {line_number} 行缺少有效 slice_id")
            if slice_id in seen_ids:
                raise ValueError(f"JSONL 包含重复 slice_id: {slice_id}")
            vertices = np.asarray(record.get("square_vertices_world"), dtype=np.float64)
            if vertices.shape != (4, 3) or not np.all(np.isfinite(vertices)):
                raise ValueError(f"JSONL 中方形顶点无效: {slice_id}")
            seen_ids.add(slice_id)
            records.append(record)
    if len(records) != EXPECTED_POSITION_COUNT:
        raise ValueError(
            f"JSONL 位置数不是 {EXPECTED_POSITION_COUNT}: {len(records)}"
        )
    return records


def _read_boundary_mask(path: Path, spec: VesselClassSpec) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION):
            raise ValueError(
                f"输入边界图必须是 {OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION} RGB: {path}"
            )
        values = np.asarray(image, dtype=np.uint8)
    white = np.all(values == 255, axis=2)
    color = np.all(values == spec.color_rgb, axis=2)
    invalid_count = int(np.count_nonzero(~(white | color)))
    if invalid_count:
        raise ValueError(f"输入边界图含非预期颜色: {path}, count={invalid_count}")
    return color


def _validate_inputs(
    input_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, set[str]]:
    expected_names = {f"{record['slice_id']}.png" for record in records}
    actual_by_class: dict[str, set[str]] = {}
    for identifier, spec in CLASS_SPECS.items():
        directory = input_root / identifier
        if not directory.is_dir():
            raise FileNotFoundError(f"血管类别目录不存在: {directory}")
        files = sorted(directory.glob("*.png"))
        actual_by_class[identifier] = {path.name for path in files}
        for path in files:
            _read_boundary_mask(path, spec)
    validate_filename_sets(actual_by_class, expected_names)
    return actual_by_class


def _validate_written_image(
    path: Path,
    expected_mask: np.ndarray,
    spec: VesselClassSpec,
) -> None:
    with Image.open(path) as image:
        if image.mode != "RGB" or image.size != (OUTPUT_RESOLUTION, OUTPUT_RESOLUTION):
            raise ValueError(f"输出图像格式无效: {path}")
        values = np.asarray(image, dtype=np.uint8)
    white = np.all(values == 255, axis=2)
    colored = np.all(values == spec.color_rgb, axis=2)
    if np.any(~(white | colored)):
        raise ValueError(f"输出图像含非预期颜色: {path}")
    if not np.array_equal(colored, expected_mask):
        mismatch = int(np.count_nonzero(colored != expected_mask))
        raise ValueError(f"输出填充掩膜与源标签不一致: {path}, count={mismatch}")


def _readme_text(segmentation_path: Path, pixel_totals: dict[str, int]) -> str:
    lines = [
        "三类 EUS 血管独立填充图说明",
        "",
        f"三维标签源: {segmentation_path}",
        f"三维标签 SHA-256: {EXPECTED_SEGMENTATION_SHA256}",
        "输入标签物理坐标系: LPS",
        "切面记录规范坐标系: RAS",
        f"输出分辨率: {OUTPUT_RESOLUTION}x{OUTPUT_RESOLUTION}",
        "标签插值: 最近邻（order=0，prefilter=False）",
        "填充规则: 填充切面内所有可见标签像素，包括触边、开放或被视野截断的区域。",
        "未执行平滑、膨胀、腐蚀、形态学闭合或轮廓 flood fill。",
        "",
        "类别映射:",
    ]
    for identifier, spec in CLASS_SPECS.items():
        lines.append(
            f"- {identifier} ({spec.chinese_name}): labels={list(spec.label_values)}, "
            f"RGB={spec.color_rgb}, filled_pixels={pixel_totals[identifier]}"
        )
    return "\n".join(lines) + "\n"


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_filled_images(
    *,
    input_root: str | Path,
    segmentation_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    source_root = Path(input_root).expanduser().resolve()
    segmentation = Path(segmentation_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"输入边界图目录不存在: {source_root}")
    if not segmentation.is_file():
        raise FileNotFoundError(f"三维标签文件不存在: {segmentation}")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空: {destination}")
        destination.rmdir()

    source_hash = _sha256_file(segmentation)
    if source_hash != EXPECTED_SEGMENTATION_SHA256:
        raise ValueError(
            f"三维标签 SHA-256 不一致: {source_hash} != "
            f"{EXPECTED_SEGMENTATION_SHA256}"
        )
    records_path = source_root / RECORDS_FILENAME
    records = _load_records(records_path)
    _validate_inputs(source_root, records)

    vertices_batch = np.asarray(
        [record["square_vertices_world"] for record in records],
        dtype=np.float64,
    )
    volume = load_label_volume(segmentation, input_coordinate_system="LPS")
    backend = CpuLabelBackend(volume)
    planes = backend.sample_many(vertices_batch, OUTPUT_RESOLUTION)
    backend.close()
    if planes.shape != (
        EXPECTED_POSITION_COUNT,
        OUTPUT_RESOLUTION,
        OUTPUT_RESOLUTION,
    ):
        raise ValueError(f"标签重采样输出形状异常: {planes.shape}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    pixel_totals = {identifier: 0 for identifier in CLASS_SPECS}
    try:
        for identifier in CLASS_SPECS:
            (temporary / identifier).mkdir()
        for index, record in enumerate(records):
            filename = f"{record['slice_id']}.png"
            labels = planes[index]
            for identifier, spec in CLASS_SPECS.items():
                source_mask = np.isin(labels, spec.label_values)
                boundary_path = source_root / identifier / filename
                boundary_mask = _read_boundary_mask(boundary_path, spec)
                validate_boundary_subset(boundary_mask, source_mask, boundary_path)
                rgb = render_filled_rgb(labels, spec)
                output_path = temporary / identifier / filename
                Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
                _validate_written_image(output_path, source_mask, spec)
                pixel_totals[identifier] += int(np.count_nonzero(source_mask))

        shutil.copyfile(records_path, temporary / RECORDS_FILENAME)
        (temporary / "README_填充图说明.txt").write_text(
            _readme_text(segmentation, pixel_totals),
            encoding="utf-8",
        )
        _write_checksums(temporary)

        for identifier in CLASS_SPECS:
            count = len(list((temporary / identifier).glob("*.png")))
            if count != EXPECTED_POSITION_COUNT:
                raise ValueError(f"输出图像数量异常: {identifier}={count}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "positions": len(records),
        "images": len(records) * len(CLASS_SPECS),
        "validation": "passed",
        "output_root": str(destination),
        "segmentation_sha256": source_hash,
        "filled_pixel_totals": pixel_totals,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--segmentation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_filled_images(
            input_root=args.input_root,
            segmentation_path=args.segmentation,
            output_root=args.output_root,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
