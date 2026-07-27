"""运行日志配置。"""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(case_directory: Path, verbose: bool = False) -> logging.Logger:
    """同时写入终端与病例日志文件。"""

    log_directory = case_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ct_vascular_resampling")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_directory / "run.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
