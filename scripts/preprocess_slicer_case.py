"""将 DICOM CT 与 3D Slicer分割导出为重采样管线输入。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import SimpleITK as sitk


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_vascular_resampling.preprocessing import write_preprocessed_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 DICOM CT 和 Slicer .seg.nrrd 预处理为重采样模型")
    parser.add_argument("--dicom-dir", required=True, help="包含 DICOM 序列的目录")
    parser.add_argument("--segmentation", required=True, help="3D Slicer .seg.nrrd 标签图")
    parser.add_argument("--output", required=True, help="预处理输出目录")
    parser.add_argument("--series-id", help="明确指定 Series Instance UID")
    parser.add_argument("--series-description", default="2.0 x 2.0_V", help="未指定 UID 时匹配的 SeriesDescription")
    parser.add_argument("--registration-module", help="下游检索使用的 2021.py 路径")
    parser.add_argument("--case-id", default="case_2", help="写入下游 YAML 的病例标识")
    parser.add_argument("--overwrite", action="store_true", help="允许清空已有输出目录后重建")
    return parser


def _series_description(path: str) -> str:
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    return reader.GetMetaData("0008|103e").strip() if reader.HasMetaDataKey("0008|103e") else ""


def read_dicom_series(directory: str | Path, series_id: str | None, description: str) -> tuple[sitk.Image, str]:
    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"DICOM 目录不存在: {source}")
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(source)) or []
    if not series_ids:
        raise ValueError(f"DICOM 目录中没有可读取的序列: {source}")
    if series_id is None:
        candidates = []
        for candidate in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(source), candidate)
            if files and _series_description(files[0]) == description:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError(f"SeriesDescription={description!r} 匹配到 {len(candidates)} 个序列，请使用 --series-id 明确指定")
        selected = candidates[0]
    else:
        if series_id not in series_ids:
            raise ValueError(f"指定序列不在目录中: {series_id}")
        selected = series_id
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(source), selected))
    return reader.Execute(), selected


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.registration_module:
            parser.error("--registration-module 是生成下游病例 YAML 所必需的参数")
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            if not args.overwrite:
                raise FileExistsError(f"输出目录已有内容，请添加 --overwrite: {output}")
            shutil.rmtree(output)
        ct, selected_series_id = read_dicom_series(args.dicom_dir, args.series_id, args.series_description)
        segmentation_path = Path(args.segmentation)
        if not segmentation_path.is_file():
            raise FileNotFoundError(f"分割文件不存在: {segmentation_path}")
        segmentation = sitk.ReadImage(str(segmentation_path))
        summary = write_preprocessed_case(
            ct,
            segmentation,
            output,
            args.registration_module,
            case_id=args.case_id,
        )
        print(json.dumps({"series_id": selected_series_id, **summary}, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
