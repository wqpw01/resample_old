#!/usr/bin/env python3
"""手工分割 CT-EUS 完整重采样兼容入口。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ct_vascular_resampling.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
