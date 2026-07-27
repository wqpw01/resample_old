"""将 gallery.jsonl 转换为外部 2021.py 检索对象。"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _load_module(path: str | Path) -> ModuleType:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"2021.py 模块不存在: {source}")
    spec = importlib.util.spec_from_file_location("ct_vascular_registration_2021", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 2021.py: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("VesselTriplet", "FeatureVector", "ProbePose", "MultiLabelledCBIR", "HMMPoseEstimator")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(f"2021.py 缺少所需对象: {', '.join(missing)}")
    return module


def _unit(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} 必须是三个数值")
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-8:
        raise ValueError(f"{name} 不能为零向量")
    return vector / magnitude


def _pose_from_record(record: dict[str, Any], module: ModuleType):
    u_axis = _unit(record["u_axis_world"], "u_axis_world")
    v_raw = _unit(record["v_axis_world"], "v_axis_world")
    v_axis = _unit(v_raw - np.dot(v_raw, u_axis) * u_axis, "v_axis_world")
    normal = _unit(np.cross(u_axis, v_axis), "局部坐标轴叉积")
    expected_normal = _unit(record["normal_world"], "normal_world")
    if np.dot(normal, expected_normal) < 0.0:
        v_axis = -v_axis
        normal = -normal
    rotation = Rotation.from_matrix(np.column_stack([u_axis, v_axis, normal]))
    rz, ry, rx = rotation.as_euler("ZYX", degrees=True)
    return module.ProbePose(
        surface_point=np.asarray(record["center_world"], dtype=np.float64),
        rx=float(rx),
        ry=float(ry),
        rz=float(rz),
        depth=0.0,
    )


def _database_key(features: list[Any]) -> str:
    counts: dict[str, int] = {}
    for feature in features:
        counts[str(feature.label)] = counts.get(str(feature.label), 0) + 1
    return "_".join(f"{label}:{counts[label]}" for label in sorted(counts) if label) or "0"


@dataclass(frozen=True)
class GalleryDatabase:
    module: ModuleType
    database: dict[str, list[Any]]
    features: list[Any]

    def create_cbir(self, search_range: int = 2):
        return self.module.MultiLabelledCBIR(database=self.database, search_range=search_range)

    def create_hmm(self, **kwargs):
        return self.module.HMMPoseEstimator(**kwargs)


def load_gallery_database(manifest_path: str | Path, registration_module_path: str | Path) -> GalleryDatabase:
    """读取一个 gallery.jsonl 或包含它的图库目录。"""

    module = _load_module(registration_module_path)
    source = Path(manifest_path)
    if source.is_dir():
        manifests = sorted(source.rglob("gallery.jsonl"))
        if not manifests:
            raise FileNotFoundError(f"图库目录中未找到 gallery.jsonl: {source}")
    elif source.is_file():
        manifests = [source]
    else:
        raise FileNotFoundError(f"图库清单不存在: {source}")
    database: dict[str, list[Any]] = {}
    feature_vectors: list[Any] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    pose = _pose_from_record(record, module)
                    triplets = [
                        module.VesselTriplet(
                            x=float(item["x_mm"]),
                            y=float(item["y_mm"]),
                            area=float(item["area_mm2"]),
                            label=str(item["label"]),
                        )
                        for item in record.get("features", [])
                    ]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{manifest} 第 {line_number} 行无效: {error}") from error
                if not triplets:
                    continue
                vector = module.FeatureVector(triplets=triplets, pose=pose)
                database.setdefault(_database_key(triplets), []).append(vector)
                feature_vectors.append(vector)
    return GalleryDatabase(module=module, database=database, features=feature_vectors)
