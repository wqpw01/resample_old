"""生成手工器官网格，并在病例配置中引用既有重建动静脉。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_vascular_resampling.manual_preprocessing import write_manual_segmentation_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="预处理手工 Slicer 器官标签并引用既有重建血管")
    parser.add_argument("--ct", required=True, help="与手工分割对齐的 CT NIfTI/NRRD 或 DICOM 目录")
    parser.add_argument("--dicom-series-uid", help="CT 为 DICOM 目录时选择目标 Series UID")
    parser.add_argument("--segmentation", required=True, help="3D Slicer .seg.nrrd 标签图")
    parser.add_argument("--artery-model", required=True, help="既有重建 artery_tree.ply")
    parser.add_argument("--vein-model", required=True, help="既有重建 vein_tree.ply")
    parser.add_argument("--output", required=True, help="新的手工分割预处理输出目录")
    parser.add_argument("--output-root", required=True, help="新重采样 Gallery 输出根目录")
    parser.add_argument("--case-id", required=True, help="病例标识")
    parser.add_argument("--overwrite", action="store_true", help="允许清空指定输出目录后重建")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser()
    try:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            if not args.overwrite:
                raise FileExistsError(f"输出目录已有内容，请添加 --overwrite: {output}")
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        summary = write_manual_segmentation_case(
            ct_path=args.ct,
            dicom_series_uid=args.dicom_series_uid,
            segmentation_path=args.segmentation,
            artery_model_path=args.artery_model,
            vein_model_path=args.vein_model,
            output_directory=output,
            output_root=args.output_root,
            case_id=args.case_id,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
