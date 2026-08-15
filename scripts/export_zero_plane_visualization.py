#!/usr/bin/env python3
"""导出病例 2 采样点与零度基准面的本地可视化交付包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_vascular_resampling.zero_plane_visualization import export_visualization_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-records-jsonl", type=Path, required=True)
    parser.add_argument("--sample-ply-dir", type=Path, required=True)
    parser.add_argument("--organ-mesh-dir", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = export_visualization_bundle(
            zero_records_jsonl=args.zero_records_jsonl,
            sample_ply_directory=args.sample_ply_dir,
            organ_mesh_directory=args.organ_mesh_dir,
            run_metadata_path=args.run_metadata,
            source_manifest_sha256=args.source_manifest_sha256,
            output_directory=args.output_dir,
        )
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
