#!/usr/bin/env python3
"""只读审计一个已完成的重采样病例输出。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ct_vascular_resampling.output_audit import audit_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-pixels", action="store_true")
    parser.add_argument("--expected-core-design-sha256")
    parser.add_argument("--expected-build-git-commit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_output(
        args.case_directory,
        check_pixels=args.check_pixels,
        expected_core_design_sha256=args.expected_core_design_sha256,
        expected_build_git_commit=args.expected_build_git_commit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.report)
    print(json.dumps({"passed": report["passed"], "errors": report["errors"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
