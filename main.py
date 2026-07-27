#!/usr/bin/env python3
"""CT 血管重采样项目入口。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ct_vascular_resampling.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
