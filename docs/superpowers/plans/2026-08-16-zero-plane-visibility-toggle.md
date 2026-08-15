# Zero-Plane Visibility Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, globally synchronized checkbox that controls zero-degree plane faces and edges independently from sampling points, organ meshes, and local axes.

**Architecture:** Keep Python responsible for assigning trace roles and serializing exact plane-trace indices. Inject a small native-JavaScript controller through Plotly `post_script`, using a stable plot div id and no external runtime dependency.

**Tech Stack:** Python 3.12, Plotly 6, native JavaScript, pytest, headless Microsoft Edge

---

### Task 1: Lock The HTML Contract With A Failing Test

**Files:**
- Modify: `tests/test_zero_plane_visualization.py`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: Extend the existing render test**

After reading the generated HTML in `test_render_outputs_are_offline_and_nonblank`, add:

```python
assert 'id="zero-plane-visualization"' in html
assert 'id="zero-plane-visibility-toggle"' in html
assert "显示 0° 基准面" in html
assert "const planeTraceIndices = [2,3];" in html
assert "plotly_buttonclicked" in html
assert "indeterminate" in html
```

The one-organ fixture produces traces in the order organ mesh, points, plane face, plane edges, then three axes, so the controlled indices must be exactly `[2,3]`.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: FAIL because the stable div id and visibility controller are not present.

### Task 2: Generate The Native Visibility Controller

**Files:**
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: Add stable identifiers and a script builder**

Add constants near the existing visualization constants:

```python
INTERACTIVE_PLOT_DIV_ID = "zero-plane-visualization"
ZERO_PLANE_TOGGLE_ID = "zero-plane-visibility-toggle"
```

Add `_zero_plane_visibility_script(roles: list[str]) -> str`. It must:

```text
1. Serialize every index whose role equals "planes" into planeTraceIndices.
2. Create a checked checkbox with id zero-plane-visibility-toggle and label 显示 0° 基准面.
3. Append the compact control to the plot div with an accessible aria-label.
4. Restyle only planeTraceIndices when the checkbox changes.
5. Preserve the plane state for All and Hide organ meshes.
6. Set the state false for Points only and true for Points + zero planes.
7. Set checkbox.indeterminate when only part of the plane traces are visible after legend interaction.
```

Use `json.dumps` for the trace-index array and all JavaScript string literals originating in Python. Treat Plotly trace visibility values `false` and `"legendonly"` as hidden.

- [ ] **Step 2: Attach the controller to the offline HTML**

Update `pio.write_html`:

```python
pio.write_html(
    figure,
    file=str(temporary),
    include_plotlyjs=True,
    post_script=_zero_plane_visibility_script(roles),
    full_html=True,
    auto_open=False,
    config={"displaylogo": False, "responsive": True},
    div_id=INTERACTIVE_PLOT_DIV_ID,
)
```

Do not alter trace order, opacity, geometry, colors, camera, or existing preset labels.

- [ ] **Step 3: Run the focused test and verify GREEN**

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: `1 passed`.

- [ ] **Step 4: Run the visualization test module**

```bash
pytest -q tests/test_zero_plane_visualization.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: toggle zero-degree planes in visualization"
```

### Task 3: Regenerate And Verify The Real Desktop Page

**Files:**
- Read: `.work/zero_plane_visualization_20260815/zero_records.jsonl`
- Read: `.work/zero_plane_visualization_20260815/ResampledpointPLY/`
- Read: `.work/zero_plane_visualization_20260815/target_organ_meshes/`
- Read: `.work/zero_plane_visualization_20260815/run_metadata.json`
- Create: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260816_开关临时\`

- [ ] **Step 1: Generate an isolated real delivery**

```bash
PYTHONPATH=src python scripts/export_zero_plane_visualization.py \
  --zero-records-jsonl .work/zero_plane_visualization_20260815/zero_records.jsonl \
  --sample-ply-dir .work/zero_plane_visualization_20260815/ResampledpointPLY \
  --organ-mesh-dir .work/zero_plane_visualization_20260815/target_organ_meshes \
  --run-metadata .work/zero_plane_visualization_20260815/run_metadata.json \
  --source-manifest-sha256 d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce \
  --output-dir /mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_开关临时
```

Expected: JSON reports 400 records and organ counts `118/162/37/53/30`.

- [ ] **Step 2: Verify package invariants**

Run `sha256sum -c SHA256SUMS.txt`, verify the HTML contains the checkbox and `"opacity":0.22`, and verify all five sample-point-to-mesh maximum nearest-vertex distances remain `0 mm`.

- [ ] **Step 3: Verify interaction in a browser**

Open the temporary HTML in Edge and verify:

```text
Default: checkbox checked; all plane faces and edges visible.
Unchecked: all plane faces and edges hidden; points, organ meshes, axes unchanged.
Checked again: all plane faces and edges restored.
Points only: checkbox becomes unchecked.
Points + zero planes: checkbox becomes checked.
Legend partial visibility: checkbox becomes indeterminate.
```

Save screenshots for default and unchecked states and confirm the Plotly canvas remains nonblank.

- [ ] **Step 4: Replace the desktop delivery with rollback preserved**

Rename the current directory to:

```text
本次重采样_采样点与零度基准面可视化_20260815_开关加入前备份_勿用
```

Then rename the verified temporary directory to the original delivery name:

```text
本次重采样_采样点与零度基准面可视化_20260815
```

Do not delete existing rollback directories.

### Task 4: Final Verification

**Files:**
- Verify: `src/ct_vascular_resampling/zero_plane_visualization.py`
- Verify: `tests/test_zero_plane_visualization.py`
- Verify: final desktop delivery

- [ ] **Step 1: Run repository verification**

```bash
pytest -q
python -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compilation exits zero, and no whitespace error is reported.

- [ ] **Step 2: Re-run final delivery checksums**

```bash
cd /mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815
sha256sum -c SHA256SUMS.txt
```

Expected: every listed file reports `OK`.

