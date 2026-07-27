"""从 picked_10cm_cropped 的标签 TAR 批量生成二维检索特征。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_vascular_resampling.cropped_retrieval import process_cropped_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从裁剪标签 TAR 提取 artery/vein 检索特征")
    parser.add_argument("--root", type=Path, required=True, help="包含 frame_* 子目录的根目录")
    parser.add_argument("--width-mm", type=float, default=100.0, help="裁剪图像宽度，默认 100 mm")
    parser.add_argument("--length-mm", type=float, default=100.0, help="裁剪图像高度，默认 100 mm")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = process_cropped_root(args.root, args.width_mm, args.length_mm)
    print(
        json.dumps(
            {
                "root": str(summary.root),
                "folder_count": summary.folder_count,
                "gallery_record_count": summary.gallery_record_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
