# Zero-Plane Organ Opacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the user-selected B opacity to make organ surfaces clearer in the zero-plane visualization without changing sampling geometry or resampling outputs.

**Architecture:** Keep opacity as two module-level presentation constants: one for Plotly HTML and one for all Matplotlib static views. Renderers consume those constants while every geometry, color, point, plane, axis, and coordinate-system path remains unchanged.

**Tech Stack:** Python 3.12, Plotly, Matplotlib, pytest, trimesh

---

### Task 1: Lock The B Opacity In Tests

**Files:**
- Modify: `tests/test_zero_plane_visualization.py`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: Add failing assertions for the selected style**

Import the module as `visualization` and extend `test_render_outputs_are_offline_and_nonblank`:

```python
from ct_vascular_resampling import zero_plane_visualization as visualization

assert visualization.INTERACTIVE_ORGAN_MESH_OPACITY == pytest.approx(0.22)
assert visualization.STATIC_ORGAN_MESH_ALPHA == pytest.approx(0.14)
assert '"opacity":0.22' in html
```

The constant assertions fix the approved values; the HTML assertion verifies that the Plotly trace actually consumes the interactive value.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: FAIL because `INTERACTIVE_ORGAN_MESH_OPACITY` and `STATIC_ORGAN_MESH_ALPHA` do not exist yet.

### Task 2: Apply Named Presentation Constants

**Files:**
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: Define the approved constants near the existing visualization constants**

```python
INTERACTIVE_ORGAN_MESH_OPACITY = 0.22
STATIC_ORGAN_MESH_ALPHA = 0.14
```

- [ ] **Step 2: Replace only organ-mesh opacity literals**

In `render_interactive_html`, replace the organ `Mesh3d` argument:

```python
opacity=INTERACTIVE_ORGAN_MESH_OPACITY,
```

In both the isometric `Poly3DCollection` and orthographic `PolyCollection`, replace the organ mesh face alpha:

```python
facecolors=[(*color, STATIC_ORGAN_MESH_ALPHA)],
```

Do not change the zero-plane alpha `0.055/0.022`, edge opacity, local-axis opacity, marker size, colors, camera, or geometry.

- [ ] **Step 3: Run the focused test and verify GREEN**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: `1 passed`.

- [ ] **Step 4: Run the complete visualization test module**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "fix: strengthen zero-plane organ visibility"
```

### Task 3: Regenerate And Replace The Desktop Delivery

**Files:**
- Read: `.work/zero_plane_visualization_20260815/zero_records.jsonl`
- Read: `.work/zero_plane_visualization_20260815/ResampledpointPLY/`
- Read: `.work/zero_plane_visualization_20260815/target_organ_meshes/`
- Read: `.work/zero_plane_visualization_20260815/run_metadata.json`
- Create: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815_透明度B临时\`
- Replace after verification: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815\`

- [ ] **Step 1: Generate a new delivery without overwriting the current one**

```bash
PYTHONPATH=src python scripts/export_zero_plane_visualization.py \
  --zero-records-jsonl .work/zero_plane_visualization_20260815/zero_records.jsonl \
  --sample-ply-dir .work/zero_plane_visualization_20260815/ResampledpointPLY \
  --organ-mesh-dir .work/zero_plane_visualization_20260815/target_organ_meshes \
  --run-metadata .work/zero_plane_visualization_20260815/run_metadata.json \
  --source-manifest-sha256 d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce \
  --output-dir /mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815_透明度B临时
```

Expected: JSON reports `record_count: 400` and counts `118/162/37/53/30`.

- [ ] **Step 2: Verify the generated package**

Run `sha256sum -c SHA256SUMS.txt` inside the temporary directory, parse `sampling_points_zero_planes.json`, and confirm:

```text
coordinate_system=RAS
record_count=400
source_manifest_sha256=d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce
```

Also confirm the HTML contains `"opacity":0.22` and that the five organ sample-point-to-mesh maximum nearest-vertex distances remain `0 mm`.

- [ ] **Step 3: Inspect a fixed-view browser screenshot**

Open the temporary `sampling_points_zero_planes_interactive.html` in a headless browser, save one screenshot, and verify organ surfaces are stronger than the current package while points and zero planes remain visible.

- [ ] **Step 4: Replace the desktop package with rollback preserved**

Rename the current corrected package to:

```text
本次重采样_采样点与零度基准面可视化_20260815_透明度调整前备份_勿用
```

Then rename the verified B temporary directory to the original delivery name. Do not delete either the earlier LPS-error backup or this opacity rollback backup.

### Task 4: Final Verification

**Files:**
- Verify: `src/ct_vascular_resampling/zero_plane_visualization.py`
- Verify: `tests/test_zero_plane_visualization.py`
- Verify: desktop B delivery directory

- [ ] **Step 1: Run repository verification**

```bash
pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compilation exits zero, and `git diff --check` emits no output.

- [ ] **Step 2: Re-run delivery checksums from the final desktop path**

```bash
cd /mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815
sha256sum -c SHA256SUMS.txt
```

Expected: every listed file reports `OK`.

