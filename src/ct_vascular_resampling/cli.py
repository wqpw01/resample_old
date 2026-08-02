"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, replace

from .auto_preprocessing import describe_auto_case, load_auto_case_config, prepare_auto_case
from .config import load_case_config
from .logging_utils import configure_logging
from .pipeline import run_case
from .rejected_audit import load_rejected_audit_config, run_rejected_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="无 P/N/D 的 CT 血管模型重采样与检索图库构建")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--case-config", help="已具备器官网格的病例 YAML 配置")
    config_group.add_argument("--auto-case-config", help="CT、混合标签体与 TotalSegmentator 自动预处理配置")
    config_group.add_argument("--rejected-audit-config", help="既有 rejected 图库的 CT FOV 审计 YAML 配置")
    parser.add_argument("--steps", nargs="+", choices=["all", "sample", "square", "render", "filter", "index"], default=["all"])
    parser.add_argument("--dry-run", action="store_true", help="仅统计候选，不写入任何文件")
    parser.add_argument("--no-resume", action="store_true", help="输出目录存在时失败，不恢复已有样本")
    parser.add_argument("--workers", type=int, help="覆盖 YAML 的渲染线程数")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], help="覆盖 YAML 的 CT 重采样后端")
    parser.add_argument("--gpu-device", type=int, help="覆盖 YAML 的 CUDA 设备编号")
    parser.add_argument("--gpu-batch-size", type=int, help="覆盖 YAML 的 CT 方形批大小")
    parser.add_argument("--verbose", action="store_true", help="输出调试级运行日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.rejected_audit_config:
            if args.dry_run:
                raise ValueError("rejected 审计不支持 --dry-run")
            audit_config = load_rejected_audit_config(args.rejected_audit_config)
            summary = run_rejected_audit(audit_config)
            print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
            return 0
        if args.auto_case_config:
            auto_config = load_auto_case_config(args.auto_case_config)
            if args.dry_run:
                print(describe_auto_case(auto_config))
                return 0
            generated_config = prepare_auto_case(auto_config)
            config = load_case_config(generated_config)
        else:
            config = load_case_config(args.case_config)
        runtime_overrides = {}
        if args.workers is not None:
            runtime_overrides["workers"] = args.workers
        if args.backend is not None:
            runtime_overrides["backend"] = args.backend
        if args.gpu_device is not None:
            if args.gpu_device < 0:
                raise ValueError("gpu_device 必须不小于零")
            runtime_overrides["gpu_device"] = args.gpu_device
        if args.gpu_batch_size is not None:
            if args.gpu_batch_size < 1:
                raise ValueError("gpu_batch_size 必须大于零")
            runtime_overrides["gpu_batch_size"] = args.gpu_batch_size
        if runtime_overrides:
            config = replace(config, runtime=replace(config.runtime, **runtime_overrides))
        logger = logging.getLogger("ct_vascular_resampling")
        if not args.dry_run:
            logger = configure_logging(config.output_root / config.case_id, args.verbose)
        summary = run_case(config, dry_run=args.dry_run, resume=not args.no_resume, steps=args.steps, workers=args.workers)
        logger.info("病例 %s 完成: %s", config.case_id, summary.status_counts)
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        logging.getLogger("ct_vascular_resampling").error("%s", error)
        print(f"错误: {error}", file=sys.stderr)
        return 1
