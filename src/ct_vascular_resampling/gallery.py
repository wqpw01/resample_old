"""图库目录、PNG、四点 PLY 与 JSONL 清单持久化。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterable

import numpy as np

from .geometry import SquareFrame
from .quality import QualityResult
from .rendering import RenderedSample


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64)]


def write_rectangles_ply(path: str | Path, frames: Iterable[SquareFrame]) -> None:
    """以无 face 的连续四点 ASCII PLY 原子写出方形。"""

    destination = Path(path)
    all_frames = list(frames)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(all_frames) * 4}\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        for frame in all_frames:
            for vertex in frame.vertices:
                handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
    os.replace(temporary, destination)


class GalleryWriter:
    """将样本稳定写入 gallery、unindexed、rejected 或 FOV 排除清单。"""

    def __init__(self, case_directory: str | Path, case_id: str):
        self.case_directory = Path(case_directory)
        self.case_id = case_id
        self.manifest_path = self.case_directory / "manifest.jsonl"
        self.case_directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.completed_statuses = self._load_completed_statuses()

    def _load_completed_statuses(self) -> dict[str, str]:
        if not self.manifest_path.is_file():
            return {}
        completed: dict[str, str] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    completed[str(record["slice_id"])] = str(record["status"])
                except (json.JSONDecodeError, KeyError) as error:
                    raise ValueError(f"全量清单第 {line_number} 行损坏: {error}") from error
        return completed

    @staticmethod
    def _save_png(image, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        image.save(temporary, format="PNG")
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _status_for(self, rendered: RenderedSample, quality: QualityResult) -> str:
        if not quality.accepted:
            return "rejected"
        return "gallery" if rendered.features else "unindexed"

    def completed_status(self, sample_id: str) -> str | None:
        """返回已持久化样本的状态，供恢复模式在重采样前短路。"""

        with self._lock:
            return self.completed_statuses.get(sample_id)

    def write_sample(
        self,
        sample_id: str,
        organ: str,
        probe_point_world: np.ndarray,
        input_normal_world: np.ndarray,
        frame: SquareFrame,
        rendered: RenderedSample,
        quality: QualityResult,
        resampling_backend: str | None = None,
        fov_diagnostics: dict[str, object] | None = None,
    ) -> str:
        with self._lock:
            return self._write_sample(
                sample_id,
                organ,
                probe_point_world,
                input_normal_world,
                frame,
                rendered,
                quality,
                resampling_backend,
                fov_diagnostics,
            )

    def write_fov_exclusion(
        self,
        sample_id: str,
        organ: str,
        probe_point_world: np.ndarray,
        input_normal_world: np.ndarray,
        frame: SquareFrame,
        fov_diagnostics: dict[str, object],
    ) -> str:
        """记录 CT FOV 外的方形，不生成 CT 或血管 PNG。"""

        with self._lock:
            if sample_id in self.completed_statuses:
                return self.completed_statuses[sample_id]
            record = {
                "frame_id": self.case_id,
                "slice_id": sample_id,
                "status": "excluded_fov",
                "organ": organ,
                "source": "organ_surface",
                "probe_point_world": _vector(probe_point_world),
                "input_normal_world": _vector(input_normal_world),
                "input_direction_world": _vector(frame.v_axis),
                "square_vertices_world": [_vector(vertex) for vertex in frame.vertices],
                "origin_world": _vector(frame.vertices[0]),
                "center_world": _vector(frame.center),
                "u_axis_world": _vector(frame.u_axis),
                "v_axis_world": _vector(frame.v_axis),
                "normal_world": _vector(frame.normal),
                "width_mm": float(frame.width_mm),
                "length_mm": float(frame.length_mm),
                "exclusion_reason": "ct_fov_exceeded",
                "fov_diagnostics": fov_diagnostics,
            }
            self._append_jsonl(self.manifest_path, record)
            self._append_jsonl(self.case_directory / "excluded_fov.jsonl", record)
            self.completed_statuses[sample_id] = "excluded_fov"
            return "excluded_fov"

    def _write_sample(
        self,
        sample_id: str,
        organ: str,
        probe_point_world: np.ndarray,
        input_normal_world: np.ndarray,
        frame: SquareFrame,
        rendered: RenderedSample,
        quality: QualityResult,
        resampling_backend: str | None,
        fov_diagnostics: dict[str, object] | None,
    ) -> str:
        """保存一个样本；已完成的 ID 返回其既有状态。"""

        if sample_id in self.completed_statuses:
            return self.completed_statuses[sample_id]
        status = self._status_for(rendered, quality)
        root = self.case_directory / status
        ct_path = root / "ct" / f"{sample_id}.png"
        boundary_path = root / "boundary_only" / f"{sample_id}.png"
        overlay_path = root / "ct_overlay" / f"{sample_id}.png"
        self._save_png(rendered.ct, ct_path)
        self._save_png(rendered.boundary_only, boundary_path)
        self._save_png(rendered.ct_overlay, overlay_path)
        width_px, height_px = rendered.ct.size
        record = {
            "frame_id": self.case_id,
            "slice_id": sample_id,
            "status": status,
            "organ": organ,
            "source": "organ_surface",
            "probe_point_world": _vector(probe_point_world),
            "input_normal_world": _vector(input_normal_world),
            "input_direction_world": _vector(frame.v_axis),
            "square_vertices_world": [_vector(vertex) for vertex in frame.vertices],
            "origin_world": _vector(frame.vertices[0]),
            "center_world": _vector(frame.center),
            "u_axis_world": _vector(frame.u_axis),
            "v_axis_world": _vector(frame.v_axis),
            "normal_world": _vector(frame.normal),
            "width_mm": float(frame.width_mm),
            "length_mm": float(frame.length_mm),
            "pixel_spacing_mm": [frame.width_mm / (width_px - 1), frame.length_mm / (height_px - 1)],
            "ct_png": str(ct_path.relative_to(root)),
            "boundary_only_png": str(boundary_path.relative_to(root)),
            "ct_overlay_png": str(overlay_path.relative_to(root)),
            "features": rendered.features,
            "quality": {
                "accepted": quality.accepted,
                "reason": quality.reason,
                "black_ratio": quality.black_ratio,
                "black_ratio_exceeded": quality.black_ratio_exceeded,
                "line_length_px": quality.line_length_px,
                "black_side_ratio": quality.black_side_ratio,
                "valid_side_black_ratio": quality.valid_side_black_ratio,
                "line_segment_px": list(quality.line_segment_px) if quality.line_segment_px is not None else None,
            },
        }
        if resampling_backend is not None:
            record["resampling_backend"] = resampling_backend
        if fov_diagnostics is not None:
            record["fov_diagnostics"] = fov_diagnostics
        self._append_jsonl(self.manifest_path, record)
        record_path = root / ({"gallery": "gallery.jsonl", "unindexed": "unindexed.jsonl", "rejected": "rejected.jsonl"}[status])
        self._append_jsonl(record_path, record)
        self.completed_statuses[sample_id] = status
        return status
