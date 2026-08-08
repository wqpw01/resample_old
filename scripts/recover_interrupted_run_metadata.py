#!/usr/bin/env python3
"""为信号中断的病例重建严格校验的运行元数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ct_vascular_resampling.config import load_case_config
from ct_vascular_resampling.pipeline import recover_interrupted_run_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重建 SIGHUP 中断病例的运行元数据，不改写切面结果")
    parser.add_argument("--case-config", required=True, help="已具备器官网格的病例 YAML 配置")
    parser.add_argument("--expected-completed-count", required=True, type=int, help="根 manifest 的预期完成条数")
    parser.add_argument("--reason", required=True, choices=["sighup"], help="中断原因")
    parser.add_argument("--exit-code", required=True, type=int, help="原任务退出码，例如 129")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_case_config(args.case_config)
        metadata = recover_interrupted_run_metadata(
            config,
            expected_completed_count=args.expected_completed_count,
            reason=args.reason,
            exit_code=args.exit_code,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
