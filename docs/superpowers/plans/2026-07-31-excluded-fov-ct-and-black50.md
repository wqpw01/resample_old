# Black 50% And Excluded FOV CT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise every active black-ratio default/configuration to 50% and emit a grayscale CT PNG with pure-black out-of-FOV pixels for every `excluded_fov` sample.

**Architecture:** Keep the geometric `excluded_fov` classification, but stop persisting it before CT interpolation. Send all unfinished squares through the selected batch backend, then route FOV-exceeding squares through a CT-only writer before quality evaluation or vessel intersection. Apply the exact FOV mask after windowing so outside pixels are always zero regardless of HU fill configuration.

**Tech Stack:** Python 3.12, NumPy, Pillow, SimpleITK, SciPy/CuPy resampling backends, JSONL, pytest.

---

## File Structure

- Modify `src/ct_vascular_resampling/config.py`: change the runtime default black-ratio threshold.
- Modify `src/ct_vascular_resampling/preprocessing.py`: change generated case-config defaults.
- Modify `src/ct_vascular_resampling/gallery.py`: persist excluded CT images and their record fields.
- Modify `src/ct_vascular_resampling/pipeline.py`: batch-resample excluded squares and render their CT-only output.
- Modify `tests/test_config.py`, `tests/test_quality.py`: specify the new 50% threshold behavior.
- Modify `tests/test_gallery_and_adapter.py`: specify the excluded writer contract.
- Modify `tests/test_pipeline.py`: specify pixel masking, state isolation, backend use, and metadata.
- Modify `configs/case.example.yaml`, `configs/case_2_autodl.yaml`, `configs/case_2_autodl_pilot.yaml`: align active configuration values.
- Modify `README.md`: document the new threshold and excluded output layout.

### Task 1: Unify The Black-Ratio Threshold At 50%

**Files:**
- Modify: `tests/test_config.py`
- Modify: `tests/test_quality.py`
- Modify: `src/ct_vascular_resampling/config.py`
- Modify: `src/ct_vascular_resampling/preprocessing.py`
- Modify: `configs/case.example.yaml`
- Modify: `configs/case_2_autodl.yaml`
- Modify: `configs/case_2_autodl_pilot.yaml`

- [ ] **Step 1: Write failing default and quality tests**

Change the config assertion to:

```python
assert config.filtering.black_ratio_limit == 0.50
```

Replace the old 31% rejection test with deterministic scattered-pixel tests:

```python
def _pixels_with_black_ratio(ratio: float) -> np.ndarray:
    pixels = np.full((100, 100), 127, dtype=np.uint8)
    rng = np.random.default_rng(0)
    pixels.ravel()[rng.choice(pixels.size, int(pixels.size * ratio), replace=False)] = 0
    return pixels


def test_quality_accepts_black_pixels_below_fifty_percent():
    result = evaluate_ct_quality(_pixels_with_black_ratio(0.40), FilterConfig())
    assert result.accepted is True
    assert result.black_ratio == 0.40


def test_quality_rejects_black_pixels_over_fifty_percent():
    result = evaluate_ct_quality(_pixels_with_black_ratio(0.51), FilterConfig())
    assert result.accepted is False
    assert result.reason == "black_ratio"
    assert result.black_ratio == 0.51
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q tests/test_config.py::test_case_config_resolves_relative_paths_and_uses_confirmed_defaults \
  tests/test_quality.py::test_quality_accepts_black_pixels_below_fifty_percent \
  tests/test_quality.py::test_quality_rejects_black_pixels_over_fifty_percent
```

Expected: the config/default-acceptance assertions fail because the current default remains `0.30`.

- [ ] **Step 3: Implement the 50% defaults**

Set both `FilterConfig.black_ratio_limit` and `_load_filter`'s fallback to `0.50`. Set the generated preprocessing case config and all listed YAML values to `0.50`. Leave the already-versioned `case_2_autodl_black50_line60.yaml` and historical design documents unchanged.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all three tests pass.

- [ ] **Step 5: Commit the threshold change**

```bash
git add tests/test_config.py tests/test_quality.py src/ct_vascular_resampling/config.py \
  src/ct_vascular_resampling/preprocessing.py configs/case.example.yaml \
  configs/case_2_autodl.yaml configs/case_2_autodl_pilot.yaml
git commit -m "feat: raise black ratio limit to fifty percent"
```

### Task 2: Add The Excluded CT Persistence Contract

**Files:**
- Modify: `tests/test_gallery_and_adapter.py`
- Modify: `src/ct_vascular_resampling/gallery.py`

- [ ] **Step 1: Write the failing writer test**

Import `Image` from Pillow, rename the old no-assets test, and pass a known grayscale image plus backend:

```python
ct_image = Image.fromarray(np.full((20, 20), 91, dtype=np.uint8))
status = writer.write_fov_exclusion(
    sample_id="esophagus-000010-x-01",
    organ="esophagus",
    probe_point_world=np.asarray([1.0, 2.0, 3.0]),
    input_normal_world=np.asarray([0.0, 0.0, 1.0]),
    frame=_frame(),
    fov_diagnostics={"contains_ct_fov_exceedance": True, "out_of_bounds_ratio": 0.62},
    ct_image=ct_image,
    resampling_backend="cpu",
)
```

Assert:

```python
ct_path = case_directory / "excluded_fov" / "ct" / "esophagus-000010-x-01.png"
assert Image.open(ct_path).mode == "L"
assert record["ct_png"] == "ct/esophagus-000010-x-01.png"
assert record["resampling_backend"] == "cpu"
assert "boundary_only_png" not in record
assert "ct_overlay_png" not in record
```

- [ ] **Step 2: Run the writer test and verify RED**

Run:

```bash
pytest -q tests/test_gallery_and_adapter.py::test_gallery_writer_persists_fov_exclusion_with_ct_only
```

Expected: fail because `write_fov_exclusion` does not accept `ct_image` or `resampling_backend`.

- [ ] **Step 3: Implement the writer contract**

Extend `GalleryWriter.write_fov_exclusion` with required `ct_image` and `resampling_backend` arguments. Under the existing lock, save the image atomically to:

```python
root = self.case_directory / "excluded_fov"
ct_path = root / "ct" / f"{sample_id}.png"
self._save_png(ct_image, ct_path)
```

Add these fields to the record before appending it:

```python
"ct_png": str(ct_path.relative_to(root)),
"resampling_backend": resampling_backend,
```

Do not add boundary or overlay paths.

- [ ] **Step 4: Run the writer test and verify GREEN**

Run the command from Step 2. Expected: pass.

- [ ] **Step 5: Commit the writer change**

```bash
git add tests/test_gallery_and_adapter.py src/ct_vascular_resampling/gallery.py
git commit -m "feat: persist excluded FOV CT images"
```

### Task 3: Route Excluded Squares Through Batch Resampling

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `src/ct_vascular_resampling/pipeline.py`

- [ ] **Step 1: Write failing CT-mask and routing tests**

Update the single-square FOV test to open `excluded_fov/ct/<id>.png`, compute `diagnose_square_fov`, and assert:

```python
pixels = np.asarray(Image.open(ct_path))
diagnosis = diagnose_square_fov(volume, sample.vertices, resolution=20)
assert pixels.ndim == 2
assert np.all(pixels[diagnosis.out_of_bounds_mask] == 0)
assert np.any(pixels[~diagnosis.out_of_bounds_mask] > 0)
assert not (tmp_path / "case" / "gallery").exists()
assert not (tmp_path / "case" / "unindexed").exists()
assert not (tmp_path / "case" / "rejected").exists()
```

Monkeypatch `evaluate_ct_quality` and `intersect_mesh_with_square` to raise if called, proving excluded samples stop before those stages.

Replace the old run-case “before interpolation” assertion with a wrapper around `CachedCpuBackend.sample_many` that records batch calls and delegates to the original method. Assert the excluded CT exists, `sample_many` was called, the record backend is `cpu`, and both summary and `run_metadata.json` report one excluded sample.

- [ ] **Step 2: Run the two pipeline tests and verify RED**

Run:

```bash
pytest -q tests/test_pipeline.py::test_out_of_fov_square_writes_black_filled_ct_only \
  tests/test_pipeline.py::test_run_case_resamples_fov_square_and_records_exclusion
```

Expected: fail because excluded samples are written before interpolation and no CT exists.

- [ ] **Step 3: Implement the CT-only FOV route**

Replace the pre-interpolation writer helper with a helper accepting `hu` and backend name. Its control flow is:

```python
if square_vertices_inside_ct(volume, sample.vertices):
    return None
frame = frame_from_vertices(sample.vertices)
diagnosis = diagnose_square_fov(volume, frame.vertices, ct_settings.output_resolution,
                                probe_point_world=sample.probe_point_world)
ct_pixels = hu_to_grayscale(hu, ct_settings.window_level, ct_settings.window_width)
ct_pixels = ct_pixels.copy()
ct_pixels[diagnosis.out_of_bounds_mask] = 0
return writer.write_fov_exclusion(
    sample_id=sample.sample_id,
    organ=sample.organ,
    probe_point_world=sample.probe_point_world,
    input_normal_world=sample.input_normal_world,
    frame=frame,
    fov_diagnostics=diagnosis.to_record(),
    ct_image=Image.fromarray(ct_pixels),
    resampling_backend=resampling_backend,
)
```

Call this helper at the start of `render_precomputed_square`, before quality evaluation and vessel intersection. In `render_square_sample`, always sample first and identify the reference path as backend `cpu`.

In `run_case`, remove the loop branch that immediately persists exclusions. Put every unfinished sample into `pending_samples`, allowing the existing CPU/GPU batch loops to sample them. Set `backend_metadata["excluded_fov_count"]` from `statuses["excluded_fov"]` in the finalization block after rendering results have updated the counter.

- [ ] **Step 4: Run the two pipeline tests and verify GREEN**

Run the command from Step 2. Expected: both pass.

- [ ] **Step 5: Run all related tests**

```bash
pytest -q tests/test_ct_resampling.py tests/test_resampling_backend.py \
  tests/test_gallery_and_adapter.py tests/test_pipeline.py
```

Expected: all pass.

- [ ] **Step 6: Commit the pipeline change**

```bash
git add tests/test_pipeline.py src/ct_vascular_resampling/pipeline.py
git commit -m "feat: render black-filled CT for FOV exclusions"
```

### Task 4: Document And Verify The Complete Behavior

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update user-facing documentation**

Change the output description to state that `excluded_fov` writes `excluded_fov/ct/<sample_id>.png`, with outside-FOV pixels black and no boundary/overlay images. Change the rejected-audit section to explain that current runs classify FOV exceedance separately while still emitting the diagnostic CT. State that the default black-ratio rejection threshold is 50%.

- [ ] **Step 2: Scan active code and configs for stale behavior**

Run:

```bash
rg -n 'black_ratio_limit: 0\.(3|30)|black_ratio_limit.*, 0\.30|不生成 CT|without.*PNG' \
  README.md configs src tests
```

Expected: no active 30% default/config or “excluded CT is not generated” statement remains.

- [ ] **Step 3: Run the full verification suite**

```bash
pytest -q
git diff --check
git status --short --branch
```

Expected: all tests pass, no whitespace errors, and only the intended README change remains uncommitted.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: describe excluded FOV CT output"
```

- [ ] **Step 5: Verify final repository state**

```bash
pytest -q
git diff --check
git status --short --branch
git log --oneline -6
```

Expected: all tests pass, worktree is clean, and the design, plan, threshold, writer, pipeline, and documentation commits are present locally.

## Plan Self-Review

- Spec coverage: Tasks 1-4 cover the global 50% threshold, exact post-window FOV masking, CT-only persistence, shared CPU/GPU batching, state isolation, metadata, resume compatibility documentation, and full regression verification.
- Placeholder scan: no deferred implementation or unspecified behavior remains.
- Type consistency: `write_fov_exclusion` receives a Pillow image and backend string from the pipeline helper; record paths are relative to the `excluded_fov` status root, matching the other status writers.
- Scope: no migration tool, blood-vessel processing for exclusions, or FOV-classification change is included.
