"""版本化 EUS 器官白名单及其严格校验。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib
from importlib import resources
import json
from typing import Iterable


EUS_CATALOG_SCHEMA_VERSION = "eus-possible-organs/v1"
EUS_ORGAN_METADATA_SCHEMA_VERSION = "eus-organ-metadata/v1"
EXCLUDED_ORGAN_LABELS = frozenset({"bile_duct", "common_bile_duct"})
EUS_ORGAN_GEOMETRY_SOURCES = {"portal_vein": "portal_vein_and_splenic_vein"}

_SPECIAL_ROLES = {
    "aorta": ("organ_and_vessel", "artery"),
    "inferior_vena_cava": ("organ_and_vessel", "vein"),
    "portal_vein": ("organ_and_vessel", "vein"),
}
_ENTRY_KEYS = {
    "organ_label",
    "eus_label_ids",
    "eus_label_names",
    "role",
    "vessel_type",
    "canonical_vessel_label_id",
}


@dataclass(frozen=True)
class EUSOrganDefinition:
    organ_label: str
    eus_label_ids: tuple[int, ...]
    eus_label_names: tuple[str, ...]
    role: str
    vessel_type: str | None
    canonical_vessel_label_id: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "organ_label": self.organ_label,
            "eus_label_ids": list(self.eus_label_ids),
            "eus_label_names": list(self.eus_label_names),
            "role": self.role,
            "vessel_type": self.vessel_type,
            "canonical_vessel_label_id": self.canonical_vessel_label_id,
        }


@dataclass(frozen=True)
class EUSOrganCatalog:
    schema_version: str
    organs: tuple[EUSOrganDefinition, ...]
    sha256: str

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(item.organ_label for item in self.organs)

    def candidate_labels(self, visible_labels: Iterable[str]) -> list[str]:
        return sorted(set(visible_labels).intersection(self.labels))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sha256": self.sha256,
            "organs": [item.to_record() for item in self.organs],
            "geometry_sources": dict(EUS_ORGAN_GEOMETRY_SOURCES),
        }


def _parse_definition(raw: object) -> EUSOrganDefinition:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
        raise ValueError("EUS 器官条目字段无效")
    label = raw["organ_label"]
    if not isinstance(label, str) or not label:
        raise ValueError("EUS 器官规范名无效")
    if label in EXCLUDED_ORGAN_LABELS:
        raise ValueError(f"胆管不得进入器官白名单: {label}")

    ids = raw["eus_label_ids"]
    if not isinstance(ids, list) or not ids or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError(f"{label} 的 EUS 标签 ID 无效")
    if ids != sorted(set(ids)):
        raise ValueError(f"{label} 的 EUS 标签 ID 必须排序且唯一")

    names = raw["eus_label_names"]
    if (
        not isinstance(names, list)
        or len(names) != len(ids)
        or any(not isinstance(value, str) or not value for value in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(f"{label} 的 EUS 标签名称无效")

    role = raw["role"]
    vessel_type = raw["vessel_type"]
    if role == "organ":
        if vessel_type is not None:
            raise ValueError(f"{label} 的普通器官 vessel_type 必须为 null")
    elif role == "organ_and_vessel":
        if vessel_type not in {"artery", "vein"}:
            raise ValueError(f"{label} 的双身份血管类型无效")
    else:
        raise ValueError(f"{label} 的角色无效")

    canonical_id = raw["canonical_vessel_label_id"]
    if canonical_id is not None and (type(canonical_id) is not int or canonical_id not in ids):
        raise ValueError(f"{label} 的规范血管标签 ID 无效")
    if role == "organ" and canonical_id is not None:
        raise ValueError(f"{label} 的普通器官规范血管标签 ID 必须为 null")

    return EUSOrganDefinition(
        organ_label=label,
        eus_label_ids=tuple(ids),
        eus_label_names=tuple(names),
        role=role,
        vessel_type=vessel_type,
        canonical_vessel_label_id=canonical_id,
    )


def parse_eus_organ_catalog(payload: bytes) -> EUSOrganCatalog:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("EUS 器官白名单不是有效 UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "organs"}:
        raise ValueError("EUS 器官白名单顶层字段无效")
    if raw["schema_version"] != EUS_CATALOG_SCHEMA_VERSION:
        raise ValueError(f"EUS 器官白名单 schema 必须为 {EUS_CATALOG_SCHEMA_VERSION}")
    if not isinstance(raw["organs"], list) or not raw["organs"]:
        raise ValueError("EUS 器官白名单 organs 必须是非空列表")

    definitions = tuple(_parse_definition(item) for item in raw["organs"])
    labels = tuple(item.organ_label for item in definitions)
    if labels != tuple(sorted(set(labels))):
        raise ValueError("EUS 器官规范名必须排序且唯一")
    all_ids = [label_id for item in definitions for label_id in item.eus_label_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("EUS 标签 ID 不能跨器官重复")
    roles = {item.organ_label: (item.role, item.vessel_type) for item in definitions}
    if any(roles.get(label) != expected for label, expected in _SPECIAL_ROLES.items()):
        raise ValueError("三类双身份血管定义无效")

    return EUSOrganCatalog(
        schema_version=raw["schema_version"],
        organs=definitions,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


@cache
def load_eus_organ_catalog() -> EUSOrganCatalog:
    payload = resources.files("ct_vascular_resampling").joinpath("data/eus_possible_organs.json").read_bytes()
    return parse_eus_organ_catalog(payload)
