# EUS Gallery Organ Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Gallery construction with a versioned EUS organ catalog, three dual-role vascular structures, and auditable per-slice EUS organ metadata without changing vessel resampling behavior.

**Architecture:** Store the approved JSON as validated package data behind a focused `eus_organs` module. Keep generic visible-organ geometry in the existing organ rendering path, derive EUS candidate labels only when Gallery records are written, and include catalog provenance in run metadata and library summaries. Existing vessel models, features, routing, and the registration adapter remain unchanged.

**Tech Stack:** Python 3.12, NumPy, trimesh, Pillow, SimpleITK, setuptools package data, pytest, mamba.

---

### Task 1: Version and validate the EUS organ catalog

**Files:**
- Create: `src/ct_vascular_resampling/data/eus_possible_organs.json`
- Create: `src/ct_vascular_resampling/eus_organs.py`
- Create: `tests/test_eus_organs.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing catalog contract tests**

Test the approved hash, schema, canonical labels, dual roles, filtering, and explicit bile-duct rejection:

```python
import json
from pathlib import Path

import pytest

from ct_vascular_resampling.eus_organs import (
    EUS_CATALOG_SCHEMA_VERSION,
    EUS_ORGAN_METADATA_SCHEMA_VERSION,
    load_eus_organ_catalog,
    parse_eus_organ_catalog,
)

EXPECTED_SHA256 = "54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d"
EXPECTED_LABELS = (
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "duodenum",
    "inferior_vena_cava", "kidney_left", "kidney_right", "liver",
    "pancreas", "portal_vein", "spleen",
)

def test_packaged_eus_catalog_matches_approved_source():
    catalog = load_eus_organ_catalog()
    assert EUS_CATALOG_SCHEMA_VERSION == "eus-possible-organs/v1"
    assert EUS_ORGAN_METADATA_SCHEMA_VERSION == "eus-organ-metadata/v1"
    assert catalog.sha256 == EXPECTED_SHA256
    assert tuple(item.organ_label for item in catalog.organs) == EXPECTED_LABELS
    assert catalog.candidate_labels(["stomach", "liver", "aorta", "liver"]) == ["aorta", "liver"]

@pytest.mark.parametrize("label", ["bile_duct", "common_bile_duct"])
def test_catalog_rejects_bile_duct_labels(label):
    source = Path(__file__).parents[1] / "src/ct_vascular_resampling/data/eus_possible_organs.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["organs"][0]["organ_label"] = label
    payload = (json.dumps(raw, ensure_ascii=False) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match="胆管"):
        parse_eus_organ_catalog(payload)
```

- [ ] **Step 2: Run the new tests and confirm the expected import failure**

Run: `mamba run -n base python -m pytest tests/test_eus_organs.py -q`

Expected: collection FAIL because `ct_vascular_resampling.eus_organs` does not exist.

- [ ] **Step 3: Add the exact resource and parser**

Add the desktop JSON byte-for-byte with `apply_patch`. Implement these stable interfaces:

```python
from dataclasses import dataclass
from functools import cache
import hashlib
from importlib import resources
import json
from typing import Iterable

EUS_CATALOG_SCHEMA_VERSION = "eus-possible-organs/v1"
EUS_ORGAN_METADATA_SCHEMA_VERSION = "eus-organ-metadata/v1"
EXCLUDED_ORGAN_LABELS = frozenset({"bile_duct", "common_bile_duct"})
_SPECIAL_ROLES = {
    "aorta": ("organ_and_vessel", "artery"),
    "inferior_vena_cava": ("organ_and_vessel", "vein"),
    "portal_vein": ("organ_and_vessel", "vein"),
}
_ENTRY_KEYS = {
    "organ_label", "eus_label_ids", "eus_label_names", "role",
    "vessel_type", "canonical_vessel_label_id",
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

def _parse_definition(raw: object) -> EUSOrganDefinition:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
        raise ValueError("EUS 器官条目字段无效")
    label = raw["organ_label"]
    if not isinstance(label, str) or not label:
        raise ValueError("EUS 器官规范名无效")
    if label in EXCLUDED_ORGAN_LABELS:
        raise ValueError(f"胆管不得进入器官白名单: {label}")
    ids = raw["eus_label_ids"]
    names = raw["eus_label_names"]
    if not isinstance(ids, list) or not ids or any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError(f"{label} 的 EUS 标签 ID 无效")
    if ids != sorted(set(ids)):
        raise ValueError(f"{label} 的 EUS 标签 ID 必须排序且唯一")
    if not isinstance(names, list) or len(names) != len(ids) or any(not isinstance(value, str) or not value for value in names):
        raise ValueError(f"{label} 的 EUS 标签名称无效")
    role, vessel_type = raw["role"], raw["vessel_type"]
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
    return EUSOrganDefinition(label, tuple(ids), tuple(names), role, vessel_type, canonical_id)

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
            "geometry_sources": {"portal_vein": "portal_vein_and_splenic_vein"},
        }

def parse_eus_organ_catalog(payload: bytes) -> EUSOrganCatalog:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("EUS 器官白名单不是有效 UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "organs"}:
        raise ValueError("EUS 器官白名单顶层字段无效")
    if raw["schema_version"] != EUS_CATALOG_SCHEMA_VERSION or not isinstance(raw["organs"], list):
        raise ValueError("EUS 器官白名单 schema 无效")
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
    return EUSOrganCatalog(raw["schema_version"], definitions, hashlib.sha256(payload).hexdigest())

@cache
def load_eus_organ_catalog() -> EUSOrganCatalog:
    payload = resources.files("ct_vascular_resampling").joinpath("data/eus_possible_organs.json").read_bytes()
    return parse_eus_organ_catalog(payload)
```

The parser order is intentional: forbidden bile-duct labels are rejected before catalog ordering checks so the explicit business rule appears in errors.

Load via `importlib.resources.files("ct_vascular_resampling").joinpath("data/eus_possible_organs.json")`, hash raw bytes, and cache. Configure package data:

```toml
[tool.setuptools.package-data]
ct_vascular_resampling = ["data/*.json"]
```

- [ ] **Step 4: Verify catalog tests and byte identity**

Run:

```bash
mamba run -n base python -m pytest tests/test_eus_organs.py -q
sha256sum src/ct_vascular_resampling/data/eus_possible_organs.json \
  /mnt/c/Users/zhangyutang/Desktop/学姐标注EUS_10cm裁剪结果/eus_possible_organs.json
```

Expected: tests PASS and both hashes equal the approved SHA-256.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ct_vascular_resampling/data/eus_possible_organs.json \
  src/ct_vascular_resampling/eus_organs.py tests/test_eus_organs.py
git commit -m "feat: add versioned EUS organ catalog"
```

### Task 2: Add dual-role structures to organ geometry

**Files:**
- Modify: `src/ct_vascular_resampling/config.py`
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_rendering.py`

- [ ] **Step 1: Write failing mapping and vessel-invariance tests**

```python
def test_organ_boundary_models_include_dual_role_vessels():
    from ct_vascular_resampling.config import ORGAN_BOUNDARY_MODEL_IDS
    assert len(ORGAN_BOUNDARY_MODEL_IDS) == 14
    assert ORGAN_BOUNDARY_MODEL_IDS["aorta"] == "aorta"
    assert ORGAN_BOUNDARY_MODEL_IDS["inferior_vena_cava"] == "inferior_vena_cava"
    assert ORGAN_BOUNDARY_MODEL_IDS["portal_vein"] == "portal_vein_and_splenic_vein"
    assert "bile_duct" not in ORGAN_BOUNDARY_MODEL_IDS
```

Render one complete vessel contour twice: once with no organs, once with the same contour in `aorta`, `inferior_vena_cava`, and `portal_vein` organ layers. Assert the second render has all three organ labels while `features`, `boundary_only.tobytes()`, and `ct_overlay.tobytes()` exactly equal the first render.

- [ ] **Step 2: Run focused tests and confirm failures**

Run: `mamba run -n base python -m pytest tests/test_config.py tests/test_rendering.py -q`

Expected: FAIL because the mapping and labels are not implemented.

- [ ] **Step 3: Implement canonical label to source-model mapping**

```python
ORGAN_BOUNDARY_MODEL_IDS = {
    "adrenal_gland_left": "adrenal_gland_left",
    "adrenal_gland_right": "adrenal_gland_right",
    "aorta": "aorta",
    "duodenum": "duodenum",
    "esophagus": "esophagus",
    "gallbladder": "gallbladder",
    "inferior_vena_cava": "inferior_vena_cava",
    "kidney_left": "kidney_left",
    "kidney_right": "kidney_right",
    "liver": "liver",
    "pancreas": "pancreas",
    "portal_vein": "portal_vein_and_splenic_vein",
    "spleen": "spleen",
    "stomach": "stomach",
}
ORGAN_BOUNDARY_IDS = tuple(ORGAN_BOUNDARY_MODEL_IDS)
```

Extend `DEFAULT_ORGAN_COLORS` with aorta using the artery color and both venous structures using the vein color. In the render preparation loop, iterate `(label, model_id)`, load `config.organ_models[model_id]`, then create `PreparedOrgan(model_id, label, color, mesh, bounds)`. Do not change `PreparedVessel` or vessel iteration.

- [ ] **Step 4: Run focused pipeline tests**

Run: `mamba run -n base python -m pytest tests/test_config.py tests/test_rendering.py tests/test_pipeline.py -q`

Expected: PASS including byte-level vessel invariance and the existing clipped-positive-area label test.

- [ ] **Step 5: Commit**

```bash
git add src/ct_vascular_resampling/config.py src/ct_vascular_resampling/pipeline.py \
  tests/test_config.py tests/test_pipeline.py tests/test_rendering.py
git commit -m "feat: add vascular structures to organ geometry"
```

### Task 3: Version Gallery organ metadata and resume validation

**Files:**
- Modify: `src/ct_vascular_resampling/gallery.py`
- Modify: `tests/test_gallery_and_adapter.py`

- [ ] **Step 1: Write failing Gallery schema tests**

Require every Gallery record to contain:

```python
assert record["organ_metadata_schema_version"] == "eus-organ-metadata/v1"
assert record["organ_labels"] == ["liver"]
assert record["eus_candidate_organ_labels"] == ["liver"]
```

Add cases proving `stomach` and `gallbladder` remain valid generic labels with no EUS candidate; all three special labels are valid in both lists; and resume rejects missing schema, unsorted lists, non-visible candidates, non-whitelist candidates, and bile-duct labels.

- [ ] **Step 2: Run Gallery tests and confirm missing-field failures**

Run: `mamba run -n base python -m pytest tests/test_gallery_and_adapter.py -q`

Expected: FAIL because new records lack schema and candidate fields.

- [ ] **Step 3: Persist and validate the exact candidate intersection**

Cache `load_eus_organ_catalog()` in `GalleryWriter.__init__`. Write new fields only for `status == "gallery"`:

```python
record["organ_metadata_schema_version"] = EUS_ORGAN_METADATA_SCHEMA_VERSION
record["organ_vessel_boundary_png"] = str(combined_path.relative_to(root))
record["organ_labels"] = rendered.organ_labels
record["eus_candidate_organ_labels"] = self.eus_organ_catalog.candidate_labels(rendered.organ_labels)
```

Validation requires the exact schema, sorted unique generic labels from `ORGAN_BOUNDARY_IDS`, and exact equality of candidates with `catalog.candidate_labels(organ_labels)`. Missing new fields must produce a direct old-schema/new-output-root error. Do not add fields to other statuses.

- [ ] **Step 4: Verify persistence, recovery, and adapter compatibility**

Run: `mamba run -n base python -m pytest tests/test_gallery_and_adapter.py -q`

Expected: PASS; the registration adapter builds the same vascular objects and ignores added fields.

- [ ] **Step 5: Commit**

```bash
git add src/ct_vascular_resampling/gallery.py tests/test_gallery_and_adapter.py
git commit -m "feat: persist EUS Gallery organ metadata"
```

### Task 4: Add catalog provenance and aggregate counts

**Files:**
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing metadata and summary assertions**

Extend the end-to-end pipeline test:

```python
catalog = metadata["eus_possible_organs"]
assert catalog["schema_version"] == "eus-possible-organs/v1"
assert catalog["sha256"] == "54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d"
assert catalog["organ_labels"] == sorted(catalog["organ_labels"])
assert set(library_summary["organ_boundary_colors"]) == set(ORGAN_BOUNDARY_IDS)
assert set(library_summary["organ_label_counts"]) <= set(ORGAN_BOUNDARY_IDS)
assert set(library_summary["eus_candidate_organ_label_counts"]) <= load_eus_organ_catalog().labels
assert library_summary["eus_possible_organs"]["geometry_sources"] == {
    "portal_vein": "portal_vein_and_splenic_vein"
}
```

Use two Gallery records sharing a candidate label to prove summary values count slices, not contours.

- [ ] **Step 2: Run pipeline tests and confirm absent metadata**

Run: `mamba run -n base python -m pytest tests/test_pipeline.py tests/test_cli.py -q`

Expected: FAIL on missing `eus_possible_organs` or `eus_candidate_organ_label_counts`.

- [ ] **Step 3: Add compact catalog identity to the resume protocol**

Before computing `resume_protocol_sha256`, add:

```python
catalog = load_eus_organ_catalog()
protocol["eus_possible_organs"] = {
    "schema_version": catalog.schema_version,
    "sha256": catalog.sha256,
    "organ_labels": sorted(catalog.labels),
    "excluded_organ_labels": sorted(EXCLUDED_ORGAN_LABELS),
    "geometry_sources": {"portal_vein": "portal_vein_and_splenic_vein"},
}
```

This catalog identity is code protocol, not patient input provenance. Its inclusion must deterministically change the resume hash.

- [ ] **Step 4: Aggregate EUS labels and publish full mapping once**

Maintain a second `Counter` while reading Gallery records:

```python
eus_candidate_counts.update(set(record.get("eus_candidate_organ_labels", [])))
```

Publish:

```python
"eus_candidate_organ_label_counts": dict(sorted(eus_candidate_counts.items())),
"eus_possible_organs": load_eus_organ_catalog().to_record(),
```

Leave blood feature counting and `load_gallery_database` unchanged.

- [ ] **Step 5: Verify protocol and summary output**

Run: `mamba run -n base python -m pytest tests/test_pipeline.py tests/test_cli.py -q`

Expected: PASS, including resume and interrupted-run metadata tests.

- [ ] **Step 6: Commit**

```bash
git add src/ct_vascular_resampling/pipeline.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: audit EUS organ catalog in Gallery runs"
```

### Task 5: Document the build-only contract

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-11-eus-gallery-organ-metadata-design.md` only if tested names differ

- [ ] **Step 1: Update Chinese result documentation**

Replace the active “11 类非血管器官” claim with the 14-label generic set. Document both new record fields, positive-area clipped visibility, all three dual identities, portal/splenic combined geometry, gallbladder versus bile duct, and old-output resume incompatibility.

- [ ] **Step 2: Add concise English compatibility documentation**

State that this is Gallery build metadata only, `registration_adapter` ignores EUS fields, vessel `features` are unchanged, and existing outputs remain readable but require a new output root for any new run.

- [ ] **Step 3: Run documentation checks**

```bash
rg -n "11 类非血管器官|eus_candidate_organ_labels|portal_vein|胆管|bile duct" \
  README.md docs/superpowers/specs/2026-08-11-eus-gallery-organ-metadata-design.md
git diff --check
```

Expected: no stale active 11-class claim, all compatibility terms present, no whitespace errors.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-11-eus-gallery-organ-metadata-design.md
git commit -m "docs: explain EUS Gallery organ metadata"
```

### Task 6: Full verification and independent review

**Files:**
- Review: all files changed after design commit `2eae6a1`

- [ ] **Step 1: Run fresh complete verification**

```bash
mamba run -n base python -m pytest -q
mamba run -n base python -m compileall -q src tests
git diff --check 2eae6a1..HEAD
```

Expected: all tests PASS, compileall exits zero, diff check has no output.

- [ ] **Step 2: Verify the JSON is shipped in a wheel**

```bash
wheel_dir="$(mktemp -d /tmp/ct-eus-wheel-XXXXXX)"
mamba run -n base python -m pip wheel --no-deps --wheel-dir "$wheel_dir" .
unzip -l "$wheel_dir"/*.whl | rg "ct_vascular_resampling/data/eus_possible_organs.json"
```

Expected: wheel build exits zero and lists the JSON exactly once.

- [ ] **Step 3: Run one independent code-review pass**

Use the requesting-code-review workflow with at most one reviewer. Review vessel-feature invariance, portal geometry truthfulness, schema compatibility, deterministic hashing, package data, memory behavior, and accidental status-routing changes. Any high/medium finding first receives a failing regression test, then a scoped fix.

- [ ] **Step 4: Re-run verification after review fixes**

```bash
mamba run -n base python -m pytest -q
mamba run -n base python -m compileall -q src tests
git status --short --branch
```

Expected: all tests PASS; only the pre-existing untracked `.superpowers/` may remain outside committed work.

### Task 7: Push and synchronize server code without rebuilding Gallery

**Files:**
- Remote branch: `origin/feature/eus-gallery-organ-metadata-20260811`
- Server project: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project`
- Server backups: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/backups`

- [ ] **Step 1: Push only the verified feature branch**

```bash
git status --short --branch
git push -u origin feature/eus-gallery-organ-metadata-20260811
git ls-remote origin refs/heads/feature/eus-gallery-organ-metadata-20260811
```

Expected: remote hash equals local `HEAD`; do not merge or force-push `main`.

- [ ] **Step 2: Re-establish the exact server boundary**

Connect only to the supplied server. Under `/root/autodl-tmp/ct_vascular_resampling_case2_20260731`, verify `pwd`, project Git root, branch/HEAD, tracked cleanliness, expected untracked case data/configs, free space, and absence of a running Gallery job. If connection remains refused, paths differ, or tracked files are dirty, stop before backup or sync.

- [ ] **Step 3: Create and validate the required backup**

Estimate the complete `project/` plus `run/` source size and free space. From `/root/autodl-tmp/ct_vascular_resampling_case2_20260731`, create `backups/project_backup_<YYYYMMDD_HHMMSS>.tar.gz` with exactly `project/` and `run/` as archive roots. Validate with `tar -tzf`, `gzip -t`, and `sha256sum`; record its path and hash. Do not read or archive unrelated projects.

- [ ] **Step 4: Fetch and switch non-destructively**

Fetch the verified feature branch inside the exact server project. Confirm its hash against local/GitHub and switch without overwriting server-only untracked case data or configs. Stop if Git reports any collision.

- [ ] **Step 5: Verify synchronized code without resampling**

Run the server mamba environment’s catalog/Gallery focused tests and full pytest suite; verify the packaged JSON hash, `git status`, and `git rev-parse HEAD`. Do not invoke `main.py`, TotalSegmentator, pilot, resume, or any output root.

- [ ] **Step 6: Report delivery evidence**

Report local/GitHub/server hashes, branch names, test totals, source/package JSON hashes, backup path/hash, server worktree status, and explicit confirmation that the existing 134,386-record Gallery was not modified.
