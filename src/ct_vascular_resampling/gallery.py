"""图库目录、PNG、四点 PLY 与 JSONL 清单持久化。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import RLock
from typing import Iterable

import numpy as np
from PIL import Image

from .config import ORGAN_BOUNDARY_IDS
from .eus_organs import EUS_ORGAN_METADATA_SCHEMA_VERSION, load_eus_organ_catalog
from .geometry import SquareFrame
from .quality import QualityResult
from .rendering import RenderedSample


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=np.float64)]


def _record_digest(record: dict) -> bytes:
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def write_rectangles_ply(path: str | Path, frames: Iterable[SquareFrame]) -> None:
    """以无 face 的连续四点 ASCII PLY 原子写出方形。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = destination.with_name(f".{destination.name}.body.tmp")
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame_count = 0
    try:
        with body.open("w", encoding="utf-8", newline="\n") as handle:
            for frame in frames:
                for vertex in frame.vertices:
                    handle.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
                frame_count += 1
        with temporary.open("w", encoding="utf-8", newline="\n") as output, body.open(
            "r", encoding="utf-8"
        ) as source:
            output.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {frame_count * 4}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            )
            shutil.copyfileobj(source, output)
        os.replace(temporary, destination)
    finally:
        body.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


class GalleryWriter:
    """将样本稳定写入 gallery、unindexed、rejected 或 FOV 排除清单。"""

    def __init__(
        self,
        case_directory: str | Path,
        case_id: str,
        *,
        required_core_design_sha256: str | None = None,
        repair_missing_state_records: bool = True,
    ):
        self.case_directory = Path(case_directory)
        self.case_id = case_id
        self.required_core_design_sha256 = required_core_design_sha256
        self.repair_missing_state_records = repair_missing_state_records
        self.eus_organ_catalog = load_eus_organ_catalog()
        self.manifest_path = self.case_directory / "manifest.jsonl"
        self.case_directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self.completed_records: dict[str, dict] = {}
        self.completed_statuses = self._load_completed_statuses()

    @staticmethod
    def _is_finite_number(value: object) -> bool:
        return not isinstance(value, (bool, np.bool_)) and isinstance(
            value, (int, float, np.integer, np.floating)
        ) and bool(np.isfinite(value))

    def _validate_pose_record(self, record: dict) -> None:
        if self.required_core_design_sha256 is None:
            return
        if record.get("coordinate_system") != "RAS":
            raise ValueError("位姿 coordinate_system 必须为 RAS")
        if record.get("core_design_sha256") != self.required_core_design_sha256:
            raise ValueError("位姿 core_design_sha256 与当前核心设计不一致")
        commit = record.get("build_git_commit")
        if not isinstance(commit, str) or len(commit) != 40 or any(
            character.lower() not in "0123456789abcdef" for character in commit
        ):
            raise ValueError("位姿 build_git_commit 必须为 40 位十六进制 Git commit")
        if not isinstance(record.get("source_region"), str) or not record["source_region"]:
            raise ValueError("位姿 source_region 必须为非空字符串")
        if record.get("yaw_policy") not in {"standard", "duodenum_bulb", "pancreas_special"}:
            raise ValueError("位姿 yaw_policy 不受支持")
        angles = record.get("angles_degrees")
        if not isinstance(angles, dict) or set(angles) != {"roll", "pitch", "yaw"} or any(
            not self._is_finite_number(angles[axis]) for axis in ("roll", "pitch", "yaw")
        ):
            raise ValueError("位姿 angles_degrees 必须包含有限的 roll、pitch、yaw")
        axes = record.get("local_axes_world")
        if not isinstance(axes, dict) or set(axes) != {"x", "y", "z"}:
            raise ValueError("位姿 local_axes_world 必须包含 x、y、z")
        for axis in ("x", "y", "z"):
            vector = axes[axis]
            if not isinstance(vector, list) or len(vector) != 3 or any(
                not self._is_finite_number(component) for component in vector
            ):
                raise ValueError(f"位姿 local_axes_world.{axis} 必须为三个有限数值")
        for field in ("target_ids", "duplicate_source_regions"):
            values = record.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"位姿 {field} 必须为字符串列表")

    def _validate_gallery_record(self, record: dict) -> None:
        combined_path = record.get("organ_vessel_boundary_png")
        organ_labels = record.get("organ_labels")
        candidate_labels = record.get("eus_candidate_organ_labels")
        if (
            record.get("organ_metadata_schema_version") != EUS_ORGAN_METADATA_SCHEMA_VERSION
            or not isinstance(combined_path, str)
            or not isinstance(organ_labels, list)
            or not isinstance(candidate_labels, list)
        ):
            raise ValueError(
                "检测到旧版 gallery 记录，缺少当前器官元数据 schema、组合图或标签字段；"
                "请使用新的输出目录"
            )
        if (
            any(not isinstance(label, str) or label not in ORGAN_BOUNDARY_IDS for label in organ_labels)
            or organ_labels != sorted(set(organ_labels))
        ):
            raise ValueError("gallery organ_labels 必须是 14 类器官中排序去重后的字符串列表")
        expected_candidates = self.eus_organ_catalog.candidate_labels(organ_labels)
        if (
            any(not isinstance(label, str) for label in candidate_labels)
            or candidate_labels != expected_candidates
        ):
            raise ValueError(
                "gallery eus_candidate_organ_labels 必须严格等于 organ_labels 与 EUS 白名单的交集"
            )
        if not (self.case_directory / "gallery" / combined_path).is_file():
            raise ValueError(f"gallery 组合图不存在: {combined_path}")

    def _state_manifest_paths(self) -> dict[str, Path]:
        return {
            "gallery": self.case_directory / "gallery" / "gallery.jsonl",
            "unindexed": self.case_directory / "unindexed" / "unindexed.jsonl",
            "rejected": self.case_directory / "rejected" / "rejected.jsonl",
            "excluded_fov": self.case_directory / "excluded_fov.jsonl",
        }

    def _load_state_manifest_entries(self, paths: dict[str, Path]) -> dict[str, tuple[str, bytes]]:
        entries: dict[str, tuple[str, bytes]] = {}
        for status, path in paths.items():
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        sample_id = str(record["slice_id"])
                        record_status = str(record["status"])
                        if record_status != status:
                            raise ValueError(f"记录状态应为 {status}，实际为 {record_status}")
                        self._validate_pose_record(record)
                        if status == "gallery":
                            self._validate_gallery_record(record)
                        if sample_id in entries:
                            previous_status = entries[sample_id][0]
                            raise ValueError(
                                f"slice_id 重复或属于多个状态: {sample_id} ({previous_status}, {status})"
                            )
                        entries[sample_id] = (status, _record_digest(record))
                    except (json.JSONDecodeError, KeyError, ValueError) as error:
                        raise ValueError(f"{status} 清单第 {line_number} 行损坏: {error}") from error
        return entries

    def _load_completed_statuses(self) -> dict[str, str]:
        state_manifest_paths = self._state_manifest_paths()
        if not self.manifest_path.is_file():
            if any(path.is_file() and path.stat().st_size > 0 for path in state_manifest_paths.values()):
                raise ValueError("检测到状态清单但缺少根 manifest.jsonl；请使用完整的新输出目录")
            return {}
        state_entries = self._load_state_manifest_entries(state_manifest_paths)
        root_entries: dict[str, tuple[str, bytes]] = {}
        completed: dict[str, str] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    sample_id = str(record["slice_id"])
                    status = str(record["status"])
                    if status not in state_manifest_paths:
                        raise ValueError(f"不支持的样本状态: {status}")
                    self._validate_pose_record(record)
                    if status == "gallery":
                        self._validate_gallery_record(record)
                    if sample_id in root_entries:
                        previous_status = root_entries[sample_id][0]
                        raise ValueError(
                            f"slice_id 重复或属于多个状态: {sample_id} ({previous_status}, {status})"
                        )
                    digest = _record_digest(record)
                    state_entry = state_entries.get(sample_id)
                    if state_entry is None:
                        if not self.repair_missing_state_records:
                            raise ValueError(
                                f"状态清单缺少根 manifest 记录: {status}/{sample_id}"
                            )
                        self._append_jsonl(state_manifest_paths[status], record)
                        state_entries[sample_id] = (status, digest)
                    elif state_entry[0] != status:
                        raise ValueError(
                            f"slice_id 属于多个状态: {sample_id} ({status}, {state_entry[0]})"
                        )
                    elif state_entry[1] != digest:
                        raise ValueError(f"根 manifest 与 {status} 清单的记录内容不一致: {sample_id}")
                    root_entries[sample_id] = (status, digest)
                    completed[sample_id] = status
                    self.completed_records[sample_id] = record
                except (json.JSONDecodeError, KeyError, ValueError) as error:
                    raise ValueError(f"全量清单第 {line_number} 行损坏: {error}") from error
        orphaned = state_entries.keys() - root_entries.keys()
        if orphaned:
            sample_id = min(orphaned)
            status = state_entries[sample_id][0]
            raise ValueError(f"{status} 清单包含根 manifest.jsonl 未记录的样本: {sample_id}")
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

    def completed_record(self, sample_id: str) -> dict | None:
        """返回已持久化记录的副本，供恢复模式核对当前姿态。"""

        with self._lock:
            record = self.completed_records.get(sample_id)
            return None if record is None else dict(record)

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
        pose_metadata: dict[str, object] | None = None,
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
                pose_metadata,
            )

    def write_fov_exclusion(
        self,
        sample_id: str,
        organ: str,
        probe_point_world: np.ndarray,
        input_normal_world: np.ndarray,
        frame: SquareFrame,
        fov_diagnostics: dict[str, object],
        ct_image: Image.Image,
        resampling_backend: str,
        pose_metadata: dict[str, object] | None = None,
    ) -> str:
        """记录 CT FOV 外的方形，只保存黑色填充的灰度 CT PNG。"""

        with self._lock:
            if sample_id in self.completed_statuses:
                return self.completed_statuses[sample_id]
            self._validate_pose_record(pose_metadata or {})
            root = self.case_directory / "excluded_fov"
            ct_path = root / "ct" / f"{sample_id}.png"
            self._save_png(ct_image, ct_path)
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
                "ct_png": str(ct_path.relative_to(root)),
                "resampling_backend": resampling_backend,
            }
            if pose_metadata:
                record.update(pose_metadata)
            self._append_jsonl(self.manifest_path, record)
            self._append_jsonl(self.case_directory / "excluded_fov.jsonl", record)
            self.completed_statuses[sample_id] = "excluded_fov"
            self.completed_records[sample_id] = record
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
        pose_metadata: dict[str, object] | None,
    ) -> str:
        """保存一个样本；已完成的 ID 返回其既有状态。"""

        if sample_id in self.completed_statuses:
            return self.completed_statuses[sample_id]
        self._validate_pose_record(pose_metadata or {})
        status = self._status_for(rendered, quality)
        root = self.case_directory / status
        ct_path = root / "ct" / f"{sample_id}.png"
        boundary_path = root / "boundary_only" / f"{sample_id}.png"
        overlay_path = root / "ct_overlay" / f"{sample_id}.png"
        self._save_png(rendered.ct, ct_path)
        self._save_png(rendered.boundary_only, boundary_path)
        self._save_png(rendered.ct_overlay, overlay_path)
        combined_path = root / "organ_vessel_boundary" / f"{sample_id}.png"
        if status == "gallery":
            self._save_png(rendered.organ_vessel_boundary, combined_path)
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
        if status == "gallery":
            record["organ_metadata_schema_version"] = EUS_ORGAN_METADATA_SCHEMA_VERSION
            record["organ_vessel_boundary_png"] = str(combined_path.relative_to(root))
            record["organ_labels"] = rendered.organ_labels
            record["eus_candidate_organ_labels"] = self.eus_organ_catalog.candidate_labels(
                rendered.organ_labels
            )
        if resampling_backend is not None:
            record["resampling_backend"] = resampling_backend
        if fov_diagnostics is not None:
            record["fov_diagnostics"] = fov_diagnostics
        if pose_metadata:
            record.update(pose_metadata)
        if status == "gallery":
            self._validate_gallery_record(record)
        self._append_jsonl(self.manifest_path, record)
        record_path = root / ({"gallery": "gallery.jsonl", "unindexed": "unindexed.jsonl", "rejected": "rejected.jsonl"}[status])
        self._append_jsonl(record_path, record)
        self.completed_statuses[sample_id] = status
        self.completed_records[sample_id] = record
        return status
