"""姿态计划的顺序无关流式摘要。"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping


POSE_PLAN_SCHEMA_VERSION = "pose-plan/v1"
_DIGEST_MODULUS = 1 << 256


class OrderIndependentDigest:
    """对 JSON 可序列化记录计算常量内存的顺序无关多重集摘要。"""

    def __init__(self, schema_version: str):
        self.schema_version = schema_version
        self.count = 0
        self.digest_sum = 0
        self.digest_xor = 0

    def update(self, entry: object) -> None:
        canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest(), "big")
        self.digest_sum = (self.digest_sum + digest) % _DIGEST_MODULUS
        self.digest_xor ^= digest
        self.count += 1

    def to_record(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "count": self.count,
            "sha256_sum": f"{self.digest_sum:064x}",
            "sha256_xor": f"{self.digest_xor:064x}",
        }


def summarize_pose_entries(entries: Iterable[Mapping[str, object]]) -> dict[str, int | str]:
    digest = OrderIndependentDigest(POSE_PLAN_SCHEMA_VERSION)
    for entry in entries:
        digest.update(entry)
    return digest.to_record()
