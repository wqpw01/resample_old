# Duodenum Manual Centerline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the user-confirmed E1-to-E2 duodenum centerline from explicit RAS endpoint hints, fail closed on geometry mismatch, and resume the audited case 2 resampling run.

**Architecture:** Extend the strict YAML sampling configuration with optional manual endpoint hints. The centerline module will validate the unpruned skeleton as a connected tree, match both RAS hints to distinct endpoints within tolerance, extract their unique path, and attach structured selection provenance to `CenterlinePath`. The sampling and run-metadata layers will pass through and persist that provenance without changing surface sample points, spacing, pose angles, CT interpolation, or output layout.

**Tech Stack:** Python 3.12, dataclasses, NumPy, SciPy `cKDTree`, scikit-image `skeletonize`, Trimesh voxel grids, PyYAML, pytest, Git, mamba, SSH/screen, NVIDIA/CuPy.

---

## File Map

- `src/ct_vascular_resampling/config.py`: parse and validate optional RAS endpoint hints and matching tolerance.
- `src/ct_vascular_resampling/centerline.py`: select the unique path between confirmed endpoints and describe the selection.
- `src/ct_vascular_resampling/sampling_pipeline.py`: pass configured hints into centerline extraction.
- `src/ct_vascular_resampling/pipeline.py`: include configured and resolved centerline provenance in the resumable run protocol.
- `tests/test_config.py`: configuration acceptance and failure cases.
- `tests/test_spacing_and_centerline.py`: manual path, orientation, tolerance, topology, and automatic-mode regression tests.
- `tests/test_sampling_pipeline.py`: settings-to-extractor propagation.
- `tests/test_pipeline.py`: structured run-metadata serialization.
- `configs/case.example.yaml`: document the optional keys without activating case-specific coordinates.
- `docs/core-design-change-log-20260807.md`: record the user-confirmed E1-to-E2 decision and implementation evidence.
- Desktop Markdown/HTML project documentation: explain the case-specific endpoint evidence and runtime status without treating the old project description as an independent source of truth.

### Task 1: Strict Manual Endpoint Configuration

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/ct_vascular_resampling/config.py`

- [ ] **Step 1: Write the failing valid-configuration test**

Append a test that supplies both endpoint vectors and the 1 mm tolerance:

```python
def test_case_config_loads_manual_duodenum_centerline_endpoints_in_ras(tmp_path):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models)
        + """
sampling:
  duodenum_centerline_endpoint_hints_ras_mm:
    proximal: [19.0, 24.0, 700.0]
    distal: [-33.0, 1.0, 664.0]
  duodenum_centerline_endpoint_match_tolerance_mm: 1.0
""",
        encoding="utf-8",
    )

    config = load_case_config(config_path)

    assert config.sampling.duodenum_centerline_endpoint_hints_ras_mm == (
        (19.0, 24.0, 700.0),
        (-33.0, 1.0, 664.0),
    )
    assert config.sampling.duodenum_centerline_endpoint_match_tolerance_mm == 1.0
```

- [ ] **Step 2: Run the valid test and verify RED**

Run:

```bash
pytest -q tests/test_config.py::test_case_config_loads_manual_duodenum_centerline_endpoints_in_ras
```

Expected: FAIL because `SamplingConfig` has no endpoint-hint field.

- [ ] **Step 3: Write failing invalid-configuration tests**

Add parametrized cases for a missing `distal`, non-three-element vectors, `NaN`/infinite coordinates, and a non-positive tolerance. Assert `ValueError` contains the full `sampling.duodenum_centerline...` field name. Also assert that specifying the tolerance without the endpoint mapping is rejected rather than ignored.

```python
@pytest.mark.parametrize(
    ("sampling_yaml", "message"),
    [
        ("duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [1, 2, 3]", "distal"),
        ("duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [1, 2]\n    distal: [3, 4, 5]", "proximal"),
        ("duodenum_centerline_endpoint_hints_ras_mm:\n    proximal: [.nan, 2, 3]\n    distal: [3, 4, 5]", "有限"),
        ("duodenum_centerline_endpoint_match_tolerance_mm: 0", "endpoint_hints"),
    ],
)
def test_case_config_rejects_invalid_manual_duodenum_centerline_configuration(
    tmp_path, sampling_yaml, message
):
    organ_models = "\n".join(f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS)
    indented = "\n  ".join(sampling_yaml.splitlines())
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + f"\nsampling:\n  {indented}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_case_config(config_path)
```

- [ ] **Step 4: Implement minimal strict parsing**

Add fields after the existing centerline settings:

```python
@dataclass(frozen=True)
class SamplingConfig:
    point_counts: dict[str, int]
    ray_length_mm: float = 100.0
    ray_batch_size: int = 2048
    minimum_spacing_mm: float = 10.0
    centerline_voxel_pitch_mm: float = 1.0
    centerline_tangent_window_mm: float = 10.0
    centerline_max_terminal_spur_mm: float = 5.0
    duodenum_centerline_endpoint_hints_ras_mm: (
        tuple[tuple[float, float, float], tuple[float, float, float]] | None
    ) = None
    duodenum_centerline_endpoint_match_tolerance_mm: float = 1.0
```

Use `math.isfinite` in a focused `_finite_vector3` helper. Add both YAML keys to `supported`. Require the mapping to contain exactly `proximal` and `distal`; reject a standalone tolerance. Preserve `None` when neither key is supplied.

- [ ] **Step 5: Run configuration tests and verify GREEN**

Run:

```bash
pytest -q tests/test_config.py
```

Expected: all configuration tests pass.

- [ ] **Step 6: Commit configuration behavior**

```bash
git add src/ct_vascular_resampling/config.py tests/test_config.py
git commit -m "feat: configure manual duodenum endpoints"
```

### Task 2: Fail-Closed E1-to-E2 Skeleton Path

**Files:**
- Modify: `tests/test_spacing_and_centerline.py`
- Modify: `src/ct_vascular_resampling/centerline.py`

- [ ] **Step 1: Write a failing manual-path test with long side branches**

Construct a connected tree whose proximal-to-distal trunk has side branches longer than 5 mm. Supply exact endpoint world coordinates and assert the result contains only the confirmed trunk, remains proximal-to-distal, and reports no automatic terminal-spur pruning.

```python
def branched_tree_skeleton() -> np.ndarray:
    skeleton = np.zeros((21, 21, 21), dtype=bool)
    center = np.asarray([10, 10, 10])
    skeleton[tuple(center)] = True
    for direction, length in (
        (np.asarray([-1, -1, -1]), 7),
        (np.asarray([1, 1, -1]), 7),
        (np.asarray([1, -1, 1]), 6),
    ):
        for step in range(1, length + 1):
            skeleton[tuple(center + direction * step)] = True
    return skeleton


def test_manual_endpoint_hints_select_unique_path_without_pruning_long_side_branches():
    skeleton = branched_tree_skeleton()
    skeleton_indices = np.argwhere(skeleton)
    world_points = skeleton_indices.astype(np.float64)

    selected, audit = select_skeleton_indices_by_endpoint_hints(
        skeleton,
        world_points,
        proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
        distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
        match_tolerance_mm=1.0,
    )

    assert np.array_equal(selected[0], [3, 3, 3])
    assert np.array_equal(selected[-1], [17, 17, 3])
    assert not np.any(np.all(selected == [16, 4, 16], axis=1))
    assert len(selected) == 15
    assert audit.mode == "manual_endpoint_hints"
    assert audit.automatic_terminal_spur_pruning_applied is False
    assert audit.excluded_endpoints_ras_mm == ((16.0, 4.0, 16.0),)
```

- [ ] **Step 2: Run the manual-path test and verify RED**

Run:

```bash
pytest -q tests/test_spacing_and_centerline.py::test_manual_endpoint_hints_select_unique_path_without_pruning_long_side_branches
```

Expected: collection/import FAIL because the selector does not exist.

- [ ] **Step 3: Write failing safety tests**

Add separate tests that require rejection when an endpoint is outside the tolerance, both hints resolve to the same endpoint, the graph is disconnected, the graph contains a cycle, or two endpoints are equally close to one hint. Keep the existing `test_skeleton_order_rejects_a_branch_when_all_terminal_arms_are_long` unchanged to protect automatic mode.

```python
def test_manual_endpoint_hints_reject_match_outside_tolerance():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)
    with pytest.raises(ValueError, match="proximal.*误差|近端.*误差"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 5.0]),
            distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_same_resolved_endpoint():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)
    with pytest.raises(ValueError, match="同一个端点"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            distal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_disconnected_skeleton():
    skeleton = np.zeros((12, 12, 12), dtype=bool)
    skeleton[1:5, 1, 1] = True
    skeleton[7:11, 10, 10] = True
    points = np.argwhere(skeleton).astype(np.float64)
    with pytest.raises(ValueError, match="单连通"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([1.0, 1.0, 1.0]),
            distal_hint_ras_mm=np.asarray([10.0, 10.0, 10.0]),
            match_tolerance_mm=1.0,
        )


def test_manual_endpoint_hints_reject_cyclic_skeleton():
    skeleton = np.zeros((5, 5, 5), dtype=bool)
    skeleton[2, 2, 2] = True
    skeleton[2, 2, 3] = True
    skeleton[2, 3, 2] = True
    points = np.argwhere(skeleton).astype(np.float64)
    with pytest.raises(ValueError, match="无环树"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([2.0, 2.0, 2.0]),
            distal_hint_ras_mm=np.asarray([2.0, 2.0, 3.0]),
            match_tolerance_mm=2.0,
        )


def test_manual_endpoint_hints_reject_equal_distance_endpoint_match():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)
    midpoint = np.asarray([10.0, 10.0, 3.0])
    with pytest.raises(ValueError, match="无法唯一匹配"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=midpoint,
            distal_hint_ras_mm=np.asarray([16.0, 4.0, 16.0]),
            match_tolerance_mm=20.0,
        )


def test_manual_endpoint_hints_reject_nonfinite_world_coordinates():
    skeleton = branched_tree_skeleton()
    points = np.argwhere(skeleton).astype(np.float64)
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        select_skeleton_indices_by_endpoint_hints(
            skeleton,
            points,
            proximal_hint_ras_mm=np.asarray([3.0, 3.0, 3.0]),
            distal_hint_ras_mm=np.asarray([17.0, 17.0, 3.0]),
            match_tolerance_mm=1.0,
        )
```

- [ ] **Step 4: Implement graph construction and manual selection**

Extract the existing 26-neighbor adjacency construction into a private helper reused by automatic and manual modes. Add an immutable audit object:

```python
@dataclass(frozen=True)
class CenterlineSelectionAudit:
    mode: str
    coordinate_system: str
    configured_proximal_ras_mm: tuple[float, float, float] | None
    configured_distal_ras_mm: tuple[float, float, float] | None
    matched_proximal_ras_mm: tuple[float, float, float] | None
    matched_distal_ras_mm: tuple[float, float, float] | None
    proximal_match_error_mm: float | None
    distal_match_error_mm: float | None
    endpoint_match_tolerance_mm: float | None
    path_point_count: int
    path_length_mm: float
    skeleton_point_count: int
    endpoint_count: int
    branchpoint_count: int
    connected_component_count: int
    excluded_node_count: int
    excluded_endpoints_ras_mm: tuple[tuple[float, float, float], ...]
    automatic_terminal_spur_pruning_applied: bool

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "coordinate_system": self.coordinate_system,
            "configured_proximal_ras_mm": list(self.configured_proximal_ras_mm) if self.configured_proximal_ras_mm else None,
            "configured_distal_ras_mm": list(self.configured_distal_ras_mm) if self.configured_distal_ras_mm else None,
            "matched_proximal_ras_mm": list(self.matched_proximal_ras_mm) if self.matched_proximal_ras_mm else None,
            "matched_distal_ras_mm": list(self.matched_distal_ras_mm) if self.matched_distal_ras_mm else None,
            "proximal_match_error_mm": self.proximal_match_error_mm,
            "distal_match_error_mm": self.distal_match_error_mm,
            "endpoint_match_tolerance_mm": self.endpoint_match_tolerance_mm,
            "path_point_count": self.path_point_count,
            "path_length_mm": self.path_length_mm,
            "skeleton_point_count": self.skeleton_point_count,
            "endpoint_count": self.endpoint_count,
            "branchpoint_count": self.branchpoint_count,
            "connected_component_count": self.connected_component_count,
            "excluded_node_count": self.excluded_node_count,
            "excluded_endpoints_ras_mm": [list(point) for point in self.excluded_endpoints_ras_mm],
            "automatic_terminal_spur_pruning_applied": self.automatic_terminal_spur_pruning_applied,
        }
```

Add `selection_audit: CenterlineSelectionAudit | None = None` as the final `CenterlinePath` field so existing positional constructors remain compatible.

The manual selector must:

```python
indices = np.argwhere(skeleton)
adjacency = _skeleton_adjacency(indices)
components = _connected_components(adjacency)
if len(components) != 1:
    raise ValueError("人工端点中心线骨架必须单连通")
edge_count = sum(len(neighbors) for neighbors in adjacency) // 2
if edge_count != len(indices) - 1:
    raise ValueError("人工端点中心线骨架必须是无环树")
endpoints = np.asarray([i for i, neighbors in enumerate(adjacency) if len(neighbors) == 1])
```

Match in RAS world coordinates, reject equal-distance ambiguity and over-tolerance matches, then follow parent pointers in the tree from proximal to distal. Compute audit counts from the full unpruned graph.

Before matching, require `skeleton_points_world.shape == (len(indices), 3)` and `np.isfinite` for skeleton points, both hints, and the tolerance. The selected node sequence comes from the unique tree parent chain, so it cannot contain repeated nodes. Keep the existing `_cumulative_lengths` and `centerline_tangents` checks as the final duplicate-point and minimum-10-mm-window guards.

- [ ] **Step 5: Integrate manual mode into mesh extraction**

Extend `extract_duodenum_centerline` with optional endpoint hints and tolerance. When hints are present, convert all `np.argwhere(skeleton)` indices through `voxels.indices_to_points`, call the manual selector, and do not call `_order_skeleton_indices`. When hints are absent, preserve existing automatic behavior and its short-spur audit.

- [ ] **Step 6: Run focused and full centerline tests**

Run:

```bash
pytest -q tests/test_spacing_and_centerline.py
```

Expected: manual tests and all pre-existing spacing/centerline tests pass.

- [ ] **Step 7: Commit the path implementation**

```bash
git add src/ct_vascular_resampling/centerline.py tests/test_spacing_and_centerline.py
git commit -m "feat: select audited duodenum centerline path"
```

### Task 3: Sampling Propagation and Run Metadata

**Files:**
- Modify: `tests/test_sampling_pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `src/ct_vascular_resampling/sampling_pipeline.py`
- Modify: `src/ct_vascular_resampling/pipeline.py`

- [ ] **Step 1: Write a failing propagation test**

Replace the existing loose centerline monkeypatch in the full-organ sampling test with a recorder and assert the exact manual arguments:

```python
captured = {}

def record_centerline(*args, **kwargs):
    captured.update(kwargs)
    return centerline

monkeypatch.setattr("ct_vascular_resampling.sampling_pipeline.extract_duodenum_centerline", record_centerline)
settings = replace(
    settings,
    duodenum_centerline_endpoint_hints_ras_mm=((19.0, 24.0, 700.0), (-33.0, 1.0, 664.0)),
    duodenum_centerline_endpoint_match_tolerance_mm=1.0,
)
sample_organs(paths, settings, seed=0, input_coordinate_system="RAS")
assert captured["endpoint_hints_ras_mm"] == settings.duodenum_centerline_endpoint_hints_ras_mm
assert captured["endpoint_match_tolerance_mm"] == 1.0
```

- [ ] **Step 2: Run the propagation test and verify RED**

Run the exact modified test and expect failure because `sample_organs` does not pass the new keywords.

- [ ] **Step 3: Pass endpoint settings to extraction**

Update the call in `sample_organs`:

```python
centerline = extract_duodenum_centerline(
    meshes["duodenum"].mesh,
    meshes["stomach"].vertices,
    voxel_pitch_mm=settings.centerline_voxel_pitch_mm,
    tangent_window_mm=settings.centerline_tangent_window_mm,
    max_terminal_spur_mm=settings.centerline_max_terminal_spur_mm,
    endpoint_hints_ras_mm=settings.duodenum_centerline_endpoint_hints_ras_mm,
    endpoint_match_tolerance_mm=settings.duodenum_centerline_endpoint_match_tolerance_mm,
)
```

- [ ] **Step 4: Write a failing metadata test**

Create a `CenterlineSelectionAudit` on the mocked duodenum `SurfaceSamples`, run `run_case`, and assert:

```python
assert metadata["sampling_configuration"]["duodenum_centerline_endpoint_hints_ras_mm"] == {
    "proximal": [19.0, 24.0, 700.0],
    "distal": [-33.0, 1.0, 664.0],
}
selection = metadata["duodenum_centerline_selection"]
assert selection["mode"] == "manual_endpoint_hints"
assert selection["matched_proximal_ras_mm"] == [19.0, 24.0, 700.0]
assert selection["path_point_count"] == 166
assert selection["automatic_terminal_spur_pruning_applied"] is False
```

- [ ] **Step 5: Persist configuration and resolved selection in the resume protocol**

Serialize the optional hints under `sampling_configuration`. After `sample_organs`, read `surfaces["duodenum"].centerline.selection_audit`; pass its `to_record()` output into `_run_protocol_metadata`. Include the resolved selection before computing `resume_protocol_sha256`, so a changed manual path cannot resume into an existing output directory.

- [ ] **Step 6: Run focused tests and verify GREEN**

```bash
pytest -q tests/test_sampling_pipeline.py tests/test_pipeline.py
```

Expected: all sampling-pipeline and pipeline tests pass.

- [ ] **Step 7: Commit integration and provenance**

```bash
git add src/ct_vascular_resampling/sampling_pipeline.py src/ct_vascular_resampling/pipeline.py tests/test_sampling_pipeline.py tests/test_pipeline.py
git commit -m "feat: audit manual centerline provenance"
```

### Task 4: Configuration Example, Documentation, and Local Verification

**Files:**
- Modify: `configs/case.example.yaml`
- Modify: `docs/core-design-change-log-20260807.md`
- Modify: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.md`
- Regenerate: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.html`

- [ ] **Step 1: Document the optional configuration without inventing defaults**

Add commented keys to `configs/case.example.yaml` and state that they are case-specific RAS coordinates requiring manual anatomical confirmation. Record E1/E2, the 190-node/189-edge tree evidence, and the user confirmation in the change log and desktop documentation. Continue labeling the desktop project description as explanatory material subordinate to the core DOCX and verified code/runtime evidence.

- [ ] **Step 2: Run the complete local test suite**

```bash
pytest -q
```

Expected: all tests pass; warnings are reported separately and no test fails.

- [ ] **Step 3: Run repository integrity checks**

```bash
git diff --check
rg -n 'T[B]D|T[O]DO|F[I]XME' docs configs src tests
rg -n "duodenum_centerline_endpoint_hints_ras_mm|duodenum_centerline_endpoint_match_tolerance_mm" docs configs src tests
```

Expected: no whitespace error, no unresolved placeholder, and all new keys use one consistent spelling.

- [ ] **Step 4: Regenerate and validate desktop HTML**

Use the preserved Pandoc toolchain and template assets that produced the current HTML:

```bash
MD=/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.md
HTML=/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.html
BUILD=/tmp/hmm-document-build

"$BUILD/pandoc/bin/pandoc" "$MD" \
  --from=gfm --to=html5 --standalone \
  --template="$BUILD/template.html" \
  --metadata=lang:zh-CN \
  --metadata=title:"CT 血管重采样图库项目详细说明" \
  --include-in-header="$BUILD/style.html" \
  --include-in-header="$BUILD/style-overrides.html" \
  --lua-filter="$BUILD/mermaid.lua" \
  --include-after-body="$BUILD/after-template.html" \
  --output="$HTML"

test "$(rg -c '<main>' "$HTML")" -eq 1
test "$(rg -c '</main>' "$HTML")" -eq 1
test "$(rg -o '<pre[ >]' "$HTML" | wc -l)" -eq "$(rg -o '</pre>' "$HTML" | wc -l)"
test "$(rg -o '<code[ >]' "$HTML" | wc -l)" -eq "$(rg -o '</code>' "$HTML" | wc -l)"
rg -n "E1|E2|人工端点|项目说明文档.*不.*设计" "$MD" "$HTML"
sha256sum "$MD" "$HTML"
```

Expected: exactly one balanced `main`, balanced `pre`/`code` tags, and the E1/E2 section in both formats.

- [ ] **Step 5: Commit documentation**

```bash
git add configs/case.example.yaml docs/core-design-change-log-20260807.md
git commit -m "docs: record confirmed duodenum main path"
```

- [ ] **Step 6: Push and verify the exact remote branch commit**

```bash
git push origin feature/core-design-alignment-20260806
git ls-remote origin refs/heads/feature/core-design-alignment-20260806
```

Expected: the remote hash exactly equals local `git rev-parse HEAD`; `main` remains untouched.

### Task 5: Server Backup, Deployment, and Real-Case Dry Run

**Files:**
- Server backup: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/backups/project_backup_<timestamp>.tar.gz`
- Server config: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project/configs/case_2_core_design_20260806.yaml`
- Dry-run log: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/run/core_design_dry_run_<date>.log`

- [ ] **Step 1: Reconfirm isolation and current state**

SSH only to `root@connect.westc.seetacloud.com:23078`, then verify `pwd`, the project root, branch, commit, tracked cleanliness, expected untracked case files, and current cgroup/disk/GPU status. Do not read or modify paths outside `/root/autodl-tmp/ct_vascular_resampling_case2_20260731` except standard read-only `/proc`, `/sys/fs/cgroup`, and `nvidia-smi` resource metrics.

- [ ] **Step 2: Create a second pre-update backup**

From the allowed root, archive the current `project` and `run` directories into a new timestamped file under `backups/`. Run `gzip -t`, list the archive roots, verify required code/config/log entries, and record SHA256. Do not overwrite the prior backup `project_backup_20260807_042845.tar.gz`.

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731
timestamp=$(date +%Y%m%d_%H%M%S)
backup="backups/project_backup_${timestamp}.tar.gz"
tar -czf "$backup" project run
gzip -t "$backup"
tar -tzf "$backup" | sed 's#^\./##' | cut -d/ -f1 | sort -u
tar -tzf "$backup" | rg 'project/\.git/HEAD|project/configs/case_2_core_design_20260806.yaml|run/core_design_dry_run_20260807.log'
sha256sum "$backup"
```

Expected archive roots: only `project` and `run`; gzip verification exits zero.

- [ ] **Step 3: Update only the approved feature branch**

Fetch the feature branch from GitHub over HTTPS, fast-forward/switch the server project to the exact remote commit, and verify expected untracked case data/configs remain present. Run:

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
git fetch https://github.com/wqpw01/resample_old.git feature/core-design-alignment-20260806
git switch feature/core-design-alignment-20260806
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
git status --short --branch
mamba run -n ct-vessel-resampling-totalseg-gpu pytest -q
```

Expected: the full suite passes in the existing scientific-computing mamba environment.

- [ ] **Step 4: Add E1/E2 to the independent server configuration**

Update only `configs/case_2_core_design_20260806.yaml` with:

```yaml
  duodenum_centerline_endpoint_hints_ras_mm:
    proximal: [19.0, 24.0, 700.0]
    distal: [-33.0, 1.0, 664.0]
  duodenum_centerline_endpoint_match_tolerance_mm: 1.0
```

Keep `workers: 4`, `backend: gpu`, `gpu_device: 0`, and `gpu_batch_size: 8`. Validate the YAML with `load_case_config` and record its SHA256.

- [ ] **Step 5: Run the real-case dry run with resource observations**

Capture baseline `memory.current`, `memory.events`, `df -B1`, and `nvidia-smi`, then run:

```bash
mamba run -n ct-vessel-resampling-totalseg-gpu \
  python main.py \
  --case-config configs/case_2_core_design_20260806.yaml \
  --dry-run --verbose

PYTHONPATH=src mamba run -n ct-vessel-resampling-totalseg-gpu python -c '
import json
from ct_vascular_resampling.config import load_case_config
from ct_vascular_resampling.sampling_pipeline import sample_organs
c = load_case_config("configs/case_2_core_design_20260806.yaml")
s = sample_organs(c.organ_models, c.sampling, c.runtime.seed, input_coordinate_system=c.geometry.input_coordinate_system)
print(json.dumps(s["duodenum"].centerline.selection_audit.to_record(), ensure_ascii=False, indent=2))
'
```

Expected facts:

- matched path E1 `[19, 24, 700]` to E2 `[-33, 1, 664]`
- selected path approximately 166 points and 224.741 mm
- sampled points: stomach 122, liver 85, pancreas 41, duodenum 51, esophagus 20; total 319
- total poses: 43,695
- no output directory created by dry-run
- `oom` and `oom_kill` remain zero

- [ ] **Step 6: Recalculate the disk gate**

Use the observed historical baseline `13,750,045,990 bytes / 167,724 records`. Require:

```text
estimated_new_bytes = historical_bytes_per_record * 43,695
required_free_bytes = estimated_new_bytes * 1.25 + 5 GiB
```

Proceed only if current free bytes exceed `required_free_bytes` and the new output path is absent.

### Task 6: Full GPU Resampling, Monitoring, and Acceptance

**Files:**
- Output: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_core_design_20260806/case_2`
- Main log: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/run/core_design_20260806.log`
- Resource log: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/run/core_design_20260806_resources.log`

- [ ] **Step 1: Start the resumable job in a named screen session**

Start from the confirmed server project directory with `screen -dmS core_design_20260806`. Write the launched process PID inside the allowed `run` directory and redirect stdout/stderr to the main log. Do not run from an interactive SSH foreground process.

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
screen -dmS core_design_20260806 bash -lc '
  cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
  echo $$ > ../run/core_design_20260806.pid
  exec mamba run -n ct-vessel-resampling-totalseg-gpu \
    python main.py \
    --case-config configs/case_2_core_design_20260806.yaml \
    --verbose >> ../run/core_design_20260806.log 2>&1
'
screen -ls
```

Expected: one detached `core_design_20260806` session and a PID file inside `run`.

- [ ] **Step 2: Monitor memory, GPU, process, and disk every 30 seconds**

Record timestamped `memory.current`, `memory.events`, the job process tree RSS, GPU memory/utilization, output size, and data-disk free bytes. The authoritative memory limit is 90 GiB cgroup, not the host-wide `free` output. If memory rises toward the cgroup limit, GPU allocation fails, OOM counters change, or disk approaches the 5 GiB reserve, stop only this job cleanly and preserve resumable artifacts.

Use a second named screen session for the read-only sampler:

```bash
screen -dmS core_design_20260806_monitor bash -lc '
  root=/root/autodl-tmp/ct_vascular_resampling_case2_20260731
  pid_file="$root/run/core_design_20260806.pid"
  log="$root/run/core_design_20260806_resources.log"
  while test -f "$pid_file" && kill -0 "$(cat "$pid_file")" 2>/dev/null; do
    {
      date --iso-8601=seconds
      printf "memory_current_bytes="; cat /sys/fs/cgroup/memory.current
      cat /sys/fs/cgroup/memory.events
      ps -eo pid,ppid,rss,%mem,etime,args | rg "PID|core_design_20260806.yaml|mamba run"
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
      df -B1 /root/autodl-tmp | tail -1
      du -sb "$root/output_core_design_20260806" 2>/dev/null || true
    } >> "$log" 2>&1
    sleep 30
  done
'
```

Poll the resource log and main log from SSH at intervals no longer than 60 seconds while the job is active.

- [ ] **Step 3: Wait for terminal completion**

Do not report completion while the screen job or its child processes are running. Inspect the final process exit code, tail the complete log for traceback/error markers, and compare final `memory.events` with baseline.

- [ ] **Step 4: Validate structural output invariants**

Require all of the following:

- manifest/state counts sum to exactly 43,695 unique pose IDs
- sampled point counts remain 122/85/41/51/20
- pose policies total to standard 33,345, duodenum bulb 5,328, and pancreas special 5,022
- run metadata contains the core DOCX SHA256, exact build commit, RAS coordinate system, E1/E2 selection audit, 10 mm spacing, cubic B-spline CT interpolation, and GPU calibration result
- all referenced output files exist and the gallery adapter/library summary can load
- no stale pose IDs, traceback, OOM, or incomplete state remains

- [ ] **Step 5: Record final evidence**

Record output byte size, manifest SHA256, metadata SHA256, status counts, peak observed cgroup memory, peak job RSS, peak GPU memory, elapsed time, backup path/hash, build commit, config hash, and output path. Update the audit report/desktop documentation with facts from the completed run, clearly separating core design requirements from explanatory project documentation.
