"""旧流程兼容的中间 PLY 产物。"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Iterable

from .sampling_pipeline import SquareSample, SurfaceSamples


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_surface_samples_ply(path: str | Path, samples: SurfaceSamples) -> None:
    """写出含顶点法线的 ASCII PLY。"""

    destination = Path(path)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(samples.points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "end_header",
    ]
    lines.extend(
        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}"
        for point, normal in zip(samples.points, samples.normals, strict=True)
    )
    _atomic_write(destination, "\n".join(lines) + "\n")


def write_square_samples_ply(path: str | Path, samples: Iterable[SquareSample]) -> None:
    """写出连续四顶点、无 face 的 ASCII PLY。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = destination.with_name(f".{destination.name}.body.tmp")
    temporary = destination.with_name(f".{destination.name}.tmp")
    sample_count = 0
    try:
        with body.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                for vertex in sample.vertices:
                    handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                sample_count += 1
        with temporary.open("w", encoding="utf-8", newline="\n") as output, body.open(
            "r", encoding="utf-8"
        ) as source:
            output.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {sample_count * 4}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            )
            shutil.copyfileobj(source, output)
        os.replace(temporary, destination)
    finally:
        body.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
