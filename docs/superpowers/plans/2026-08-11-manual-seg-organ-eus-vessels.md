# Manual Segmentation Organ and EUS Vessel Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Case 2 from the server CT, the user-supplied Slicer segmentation, and the existing reconstructed artery/vein meshes so every original Gallery record keeps its original vessel behavior while gaining pixel-presence organ metadata and a separate three-class EUS vessel result.

**Architecture:** Keep the existing cubic CT backend and PLY vessel intersection path unchanged. Add a separate nearest-neighbor label-volume backend using the same world-coordinate square grid, then analyze each Gallery label plane into manual organ boundaries and three-class EUS vessel outputs. Persist those outputs only under an explicitly locked manual-segmentation Gallery schema so legacy configurations and state routing remain unchanged.

**Tech Stack:** Python 3.12, NumPy, SciPy `ndimage`, SimpleITK, CuPy/CUDA, Pillow, trimesh, PyYAML, pytest, mamba, Git, SSH/screen.

---

## File Structure

- Create `src/ct_vascular_resampling/label_resampling.py`: load a discrete label volume in canonical RAS, validate CT geometry, and sample square batches with nearest-neighbor CPU/GPU backends.
- Create `src/ct_vascular_resampling/manual_segmentation.py`: own the approved organ/EUS mappings, analyze one raw label plane, filter incomplete features, and render manual-label images.
- Create `src/ct_vascular_resampling/manual_preprocessing.py`: export only the 14 manual organ masks/meshes while preserving exact segmentation and external reconstructed-vessel provenance.
- Create `scripts/preprocess_manual_segmentation_case.py`: non-destructive CLI for the new preprocessing mode.
- Modify `src/ct_vascular_resampling/config.py`: parse an optional, strict `manual_segmentation` contract without changing legacy cases.
- Modify `src/ct_vascular_resampling/rendering.py`: carry optional manual-label render products without changing original vessel images or `features`.
- Modify `src/ct_vascular_resampling/gallery.py`: lock and validate the manual Gallery schema, write two new image families, and restore safely.
- Modify `src/ct_vascular_resampling/pipeline.py`: load/sample the label volume, attach manual outputs only to original Gallery slices, and report provenance/counts.
- Modify `configs/case.example.yaml`, `README.md`, and the Chinese project documentation: document the manual mode, exact mappings, 60% threshold, and incomplete-vessel rule.
- Create focused tests in `tests/test_label_resampling.py` and `tests/test_manual_segmentation.py`; extend `tests/test_config.py`, `tests/test_preprocessing.py`, `tests/test_rendering.py`, `tests/test_gallery_and_adapter.py`, `tests/test_pipeline.py`, and `tests/test_quality.py`.

### Task 1: Strict Manual-Segmentation Configuration

**Files:**
- Modify: `src/ct_vascular_resampling/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Add a helper YAML block and tests that require the exact approved mappings:

```python
MANUAL_SEGMENTATION_YAML = """
manual_segmentation:
  path: labels/EUS-main-organ.seg.nrrd
  organ_label_values:
    spleen: [1]
    kidney_right: [2]
    kidney_left: [3]
    gallbladder: [4]
    esophagus: [5]
    liver: [6]
    stomach: [7]
    aorta: [8]
    inferior_vena_cava: [9]
    pancreas: [11]
    adrenal_gland_right: [12]
    adrenal_gland_left: [13]
    duodenum: [14]
    portal_vein: [23, 26, 33, 34, 35, 36, 37]
  eus_vessel_label_values:
    aorta: [8]
    inferior_vena_cava: [9]
    portal_vein: [26, 33, 34, 35, 36, 37]
  eus_vessel_colors:
    aorta: [255, 0, 0]
    inferior_vena_cava: [0, 0, 255]
    portal_vein: [170, 85, 255]
"""

def test_case_config_loads_strict_manual_segmentation_mode(tmp_path):
    config = load_case_config(_write_case(tmp_path, MANUAL_SEGMENTATION_YAML))
    manual = config.manual_segmentation
    assert manual is not None
    assert manual.path == tmp_path / "labels/EUS-main-organ.seg.nrrd"
    assert manual.organ_label_values["portal_vein"] == (23, 26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_label_values["portal_vein"] == (26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_colors["portal_vein"] == (170, 85, 255)
```

Parametrize rejection of a missing canonical key, unknown key, empty list, duplicate value inside one mapping, `True` as a label, non-integer label, and a color other than three `0..255` integers. Add a legacy assertion that omitting `manual_segmentation` yields `None`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py -k manual_segmentation
```

Expected: FAIL because `CaseConfig` has no `manual_segmentation` field.

- [ ] **Step 3: Implement the strict dataclass and parser**

Add these public constants and type:

```python
EUS_VESSEL_IDS = ("aorta", "inferior_vena_cava", "portal_vein")

@dataclass(frozen=True)
class ManualSegmentationConfig:
    path: Path
    organ_label_values: dict[str, tuple[int, ...]]
    eus_vessel_label_values: dict[str, tuple[int, ...]]
    eus_vessel_colors: dict[str, tuple[int, int, int]]
```

Add `manual_segmentation: ManualSegmentationConfig | None = None` to `CaseConfig`. Parse only `path`, `organ_label_values`, `eus_vessel_label_values`, and `eus_vessel_colors`; require exactly `ORGAN_BOUNDARY_IDS` and `EUS_VESSEL_IDS`; reject booleans before integer conversion; reject duplicates within each mapping while allowing the approved cross-mapping reuse.

- [ ] **Step 4: Run configuration tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py
git add src/ct_vascular_resampling/config.py tests/test_config.py
git commit -m "feat: add strict manual segmentation configuration"
```

Expected: all `tests/test_config.py` tests PASS.

### Task 2: Nearest-Neighbor Label Volume and Backends

**Files:**
- Create: `src/ct_vascular_resampling/label_resampling.py`
- Create: `tests/test_label_resampling.py`

- [ ] **Step 1: Write failing geometry and CPU sampling tests**

Create tests for LPS-to-RAS conversion, CT/label Size/Spacing/Origin/Direction equality, nearest-neighbor output, and outside-FOV fill `0`:

```python
def test_cpu_label_backend_samples_nearest_labels_and_fills_zero():
    labels = np.zeros((3, 4, 5), dtype=np.uint8)
    labels[1, 1, 1] = 8
    label_volume = LabelVolume.from_sitk(_image(labels), input_coordinate_system="LPS")
    vertices = _square_through_voxels(_image(labels), z=1)
    sampled = CpuLabelBackend(label_volume).sample_many(vertices[None], resolution=5)
    assert sampled.dtype == np.uint8
    assert sampled.shape == (1, 5, 5)
    assert 8 in sampled[0]
    assert np.all(CpuLabelBackend(label_volume).sample_many(_outside()[None], 5) == 0)
```

Geometry tests must accept `Origin` delta `5e-7 mm` and reject `2e-6 mm`, altered spacing, direction, or size.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_label_resampling.py
```

Expected: collection FAIL because `label_resampling.py` does not exist.

- [ ] **Step 3: Implement the reference CPU path**

Implement:

```python
@dataclass(frozen=True)
class LabelVolume:
    data_zyx: np.ndarray
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction_xyz: np.ndarray

    @property
    def physical_to_index_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.direction_xyz @ np.diag(self.spacing_xyz))

    def world_to_continuous_indices(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        flat = points.reshape(-1, 3)
        indices = (flat - self.origin_xyz) @ self.physical_to_index_matrix.T
        return indices.reshape(points.shape)

def validate_label_geometry(ct: CTVolume, labels: LabelVolume, *, atol_mm: float = 1e-6) -> None: ...

class CpuLabelBackend:
    name = "cpu"
    def sample_many(self, vertices_batch: np.ndarray, resolution: int) -> np.ndarray:
        # square_coordinates_zyx uses the same world grid as CT.
        values = map_coordinates(
            self.volume.data_zyx,
            square_coordinates_zyx(self.volume, square, resolution),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
```

Require a scalar three-dimensional integer segmentation whose values fit `uint8`, store one `uint8` volume, and return one `uint8` batch. Never allocate one 3-D volume per organ. `square_coordinates_zyx` can then use `LabelVolume.world_to_continuous_indices` on exactly the same RAS world grid as CT.

- [ ] **Step 4: Add RED tests for GPU equality and fallback**

Using a fake CuPy device and fake `map_coordinates`, assert exact pixel equality with CPU. Test `auto` fallback on initialization failure, forced `gpu` rejection, and validation rejection for one altered label pixel.

- [ ] **Step 5: Implement GPU backend and exact calibration**

Implement `CuPyLabelBackend`, `create_label_sampling_backend`, and `validate_label_backend_against_cpu`. Use `order=0`, `prefilter=False`, `cval=0`, and compare with `np.array_equal`; do not use a numeric tolerance for labels.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_label_resampling.py tests/test_coordinates.py tests/test_resampling_backend.py
git add src/ct_vascular_resampling/label_resampling.py tests/test_label_resampling.py
git commit -m "feat: sample aligned segmentation label planes"
```

Expected: all focused tests PASS.

### Task 3: Organ Presence and Three-Class EUS Vessel Analysis

**Files:**
- Create: `src/ct_vascular_resampling/manual_segmentation.py`
- Create: `tests/test_manual_segmentation.py`

- [ ] **Step 1: Write failing organ-presence and boundary tests**

Test these facts independently:

```python
def test_one_pixel_and_full_frame_organs_are_labels_without_artificial_frame():
    labels = np.full((9, 9), 6, dtype=np.uint8)
    labels[4, 4] = 11
    result = analyze_manual_label_plane(labels, 100.0, 100.0, CONFIG)
    assert result.organ_labels == ["liver", "pancreas"]
    assert np.all(result.organ_boundary_rgb[0, :, :] == 255)
    assert np.all(result.organ_boundary_rgb[-1, :, :] == 255)
```

Also assert a source label absent from the sampled pixels is absent even if continuous geometry could be tangent. Assert labels `23,26,33,34,35,36,37` all map to organ `portal_vein`, while EUS `portal_vein` excludes `23` and includes `26,33..37`.

- [ ] **Step 2: Write the decisive incomplete-vessel RED test**

Use one closed component and one component touching row `0` in the same raw plane:

```python
def test_incomplete_eus_vessel_is_drawn_but_not_featured():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[4:7, 4:7] = 8       # closed Ao component
    labels[0:3, 8:11] = 9      # IVC clipped by top edge
    result = analyze_manual_label_plane(labels, 110.0, 110.0, CONFIG)
    assert [feature["label"] for feature in result.eus_vessel_features] == ["aorta"]
    assert result.eus_vessel_labels == ["aorta", "inferior_vena_cava"]
    pixels = set(map(tuple, result.eus_vessel_boundary_rgb.reshape(-1, 3)))
    assert (255, 0, 0) in pixels
    assert (0, 0, 255) in pixels
```

Add a merge test where touching SMV `26` and SV `33` form one component and no internal source-label boundary appears. Assert 8-connectivity joins diagonal pixels. Assert the complete component feature uses `x_mm = mean(column) * width_mm/(width-1)`, `y_mm = mean(row) * length_mm/(height-1)`, and area `pixel_count * x_spacing * y_spacing`.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_manual_segmentation.py
```

Expected: collection FAIL because `manual_segmentation.py` does not exist.

- [ ] **Step 4: Implement one-pass analysis**

Define:

```python
EUS_VESSEL_METADATA_SCHEMA_VERSION = "eus-vessel-metadata/v1"

@dataclass(frozen=True)
class ManualLabelPlaneAnalysis:
    organ_labels: list[str]
    organ_boundary_rgb: np.ndarray
    eus_vessel_labels: list[str]
    eus_vessel_features: list[dict[str, float | str]]
    eus_vessel_boundary_rgb: np.ndarray
```

Map each canonical class from the raw plane with `np.isin`. Compute boundaries from in-image neighbor transitions only, treating outside-image neighbors as foreground so a full-frame class does not acquire an artificial rectangular border. Merge `26,33,34,35,36,37` before boundary/component work. Use `ndimage.label(..., structure=np.ones((3,3)))`; classify a component as incomplete when any pixel touches row `0`, row `H-1`, column `0`, or column `W-1`. Generate the visible boundary before feature filtering so incomplete components remain colored in the image.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_manual_segmentation.py tests/test_cropped_retrieval.py
git add src/ct_vascular_resampling/manual_segmentation.py tests/test_manual_segmentation.py
git commit -m "feat: analyze manual organ and EUS vessel planes"
```

Expected: all focused tests PASS, including the open-boundary regression.

### Task 4: Render Manual Organ and EUS Vessel Images Without Changing Original Outputs

**Files:**
- Modify: `src/ct_vascular_resampling/rendering.py`
- Modify: `src/ct_vascular_resampling/manual_segmentation.py`
- Modify: `tests/test_rendering.py`
- Modify: `tests/test_manual_segmentation.py`

- [ ] **Step 1: Write failing rendering-isolation tests**

For identical CT pixels and PLY vessel contours, render once in legacy mode and once with a manual label analysis. Assert:

```python
assert manual.features == legacy.features
assert manual.boundary_only.tobytes() == legacy.boundary_only.tobytes()
assert manual.ct_overlay.tobytes() == legacy.ct_overlay.tobytes()
assert manual.organ_labels == analysis.organ_labels
assert manual.eus_vessel_features == analysis.eus_vessel_features
```

For the clipped IVC component, assert blue pixels exist in both `eus_vessel_boundary` and `ct_eus_vessel_overlay`, while no IVC feature exists. Assert the white-background EUS image remains white inside vessel regions and outside boundaries.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_rendering.py tests/test_manual_segmentation.py
```

Expected: FAIL because `RenderedSample` has no EUS fields and no manual compositing function exists.

- [ ] **Step 3: Implement optional render products**

Extend `RenderedSample` with default-`None` manual fields:

```python
eus_vessel_boundary: Image.Image | None = None
ct_eus_vessel_overlay: Image.Image | None = None
eus_vessel_features: list[dict[str, float | str]] | None = None
eus_vessel_labels: list[str] | None = None
```

Implement `apply_manual_label_analysis(rendered, analysis)` with `dataclasses.replace`. Build `organ_vessel_boundary` from `analysis.organ_boundary_rgb`, then overwrite only non-white pixels from the original `rendered.boundary_only` so reconstructed-vessel colors remain on top. Build the new CT overlay by replacing CT RGB pixels only where the unfiltered EUS boundary is non-white.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_rendering.py tests/test_manual_segmentation.py
git add src/ct_vascular_resampling/rendering.py src/ct_vascular_resampling/manual_segmentation.py tests/test_rendering.py tests/test_manual_segmentation.py
git commit -m "feat: render separate EUS vessel boundary outputs"
```

Expected: original output byte-equality assertions and new image assertions PASS.

### Task 5: Gallery Persistence and Schema Lock

**Files:**
- Modify: `src/ct_vascular_resampling/gallery.py`
- Modify: `tests/test_gallery_and_adapter.py`

- [ ] **Step 1: Write failing manual Gallery persistence tests**

Construct a `RenderedSample` with original features plus manual fields. Instantiate `GalleryWriter(..., manual_segmentation_enabled=True)` and assert it writes:

```python
assert record["eus_vessel_metadata_schema_version"] == "eus-vessel-metadata/v1"
assert record["eus_vessel_labels"] == ["aorta", "inferior_vena_cava"]
assert record["eus_vessel_features"] == [{"label": "aorta", "x_mm": 50.0, "y_mm": 50.0, "area_mm2": 100.0}]
assert record["eus_vessel_boundary_png"].startswith("eus_vessel_boundary/")
assert record["ct_eus_vessel_overlay_png"].startswith("ct_eus_vessel_overlay/")
```

Assert both PNGs exist. Assert `unindexed`, `rejected`, and `excluded_fov` records do not receive these fields or files. Assert status is still determined only by original `rendered.features`.

- [ ] **Step 2: Add failing restoration/schema tests**

Assert manual mode refuses existing Gallery records with a missing schema, missing image, unknown/duplicate label, feature label outside the three classes, non-finite feature values, or a feature label not present in `eus_vessel_labels`. Assert legacy mode continues loading valid legacy records and emits no `eus_vessel_*` fields.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_gallery_and_adapter.py -k 'manual or eus_vessel or legacy'
```

Expected: FAIL because `GalleryWriter` has no schema-mode flag or new paths.

- [ ] **Step 4: Implement persistence and validation**

Add `manual_segmentation_enabled: bool = False` to `GalleryWriter`. In manual mode require all manual render fields for Gallery status, save `eus_vessel_boundary/<slice_id>.png` and `ct_eus_vessel_overlay/<slice_id>.png` atomically, attach the versioned fields, and validate them during both write and resume. Never consult `eus_vessel_features` in `_status_for`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_gallery_and_adapter.py tests/test_rendering.py
git add src/ct_vascular_resampling/gallery.py tests/test_gallery_and_adapter.py
git commit -m "feat: persist versioned EUS vessel Gallery metadata"
```

Expected: all focused tests PASS.

### Task 6: Pipeline Integration and Original-Vessel Invariance

**Files:**
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing single-square integration tests**

Create an accepted square with one complete reconstructed PLY vessel and a raw label plane containing a complete Ao plus clipped IVC. Assert the output is Gallery, original `features` come only from PLY, EUS features contain only Ao, and both EUS images contain IVC blue.

Render the same square without manual mode and compare status, original `features`, `boundary_only.png`, and `ct_overlay.png` byte-for-byte.

- [ ] **Step 2: Write failing full-pipeline backend tests**

Patch the pose stream to a small deterministic set. Assert manual mode:

- loads the label volume once;
- validates it against CT before any PNG/JSONL write;
- batches CT and label sampling using the same square vertices and resolution;
- calibrates GPU labels by exact equality;
- falls back to CPU in `auto` after label calibration/runtime failure;
- rejects forced `gpu` instead of silently mixing an unverified label backend;
- does not analyze labels for rejected, excluded-FOV, or original-unindexed slices.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_pipeline.py -k 'manual or label_plane or eus_vessel or invariant'
```

Expected: FAIL because `run_case` never loads or passes label planes.

- [ ] **Step 4: Integrate the independent label path**

Update `_preflight` to require the configured segmentation path. In render mode load one `LabelVolume`, validate it against CT, create its backend, and calibrate it on the same pending squares. Each batch becomes `(sample, hu, raw_label_plane, backend_names)`. In `render_precomputed_square`, preserve the existing quality and PLY intersection order; only when original complete PLY features exist, call `analyze_manual_label_plane` and `apply_manual_label_analysis`. Legacy mode continues using mesh-derived organ layers.

If either GPU backend fails in `auto`, close it, select its CPU reference, record the fallback, and continue without changing completed records. Forced `gpu` raises before writing the affected batch.

- [ ] **Step 5: Run pipeline and invariance tests**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_pipeline.py tests/test_resampling_backend.py tests/test_label_resampling.py
```

Expected: all tests PASS and legacy assertions remain unchanged.

- [ ] **Step 6: Commit integration**

```bash
git add src/ct_vascular_resampling/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate manual label planes into Gallery builds"
```

### Task 7: Manual Preprocessing With External Reconstructed Vessels

**Files:**
- Create: `src/ct_vascular_resampling/manual_preprocessing.py`
- Create: `scripts/preprocess_manual_segmentation_case.py`
- Modify: `tests/test_preprocessing.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing preprocessing tests**

Use a small CT/segmentation pair with all approved organ labels and two external mesh files. Assert preprocessing:

- writes exactly 14 organ masks and 14 organ PLYs;
- makes organ `portal_vein_and_splenic_vein` from `23,26,33,34,35,36,37`;
- copies the source segmentation byte-for-byte and records its SHA-256;
- records external `artery_tree.ply` and `vein_tree.ply` hashes without regenerating or overwriting them;
- emits manual configuration with black ratio `0.60` and the exact two mappings/colors;
- fails before output on geometry mismatch, missing source labels, or missing external vessel files.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_preprocessing.py -k manual
```

Expected: FAIL because the manual preprocessing module does not exist.

- [ ] **Step 3: Implement non-destructive preprocessing**

Implement `write_manual_segmentation_case(...)` using the approved mappings. Reuse `validate_geometry`, `build_binary_masks`, and `mask_to_mesh` only for the 14 organ outputs. Copy the segmentation to `segmentation/EUS-main-organ.seg.nrrd` with `shutil.copyfile`; never derive `artery_tree` or `vein_tree` from segmentation. Generate a manifest containing source paths, SHA-256 hashes, geometry, label mappings, voxel counts, mesh counts/watertight diagnostics, and external-vessel provenance.

- [ ] **Step 4: Implement the CLI without destructive overwrite defaults**

The script accepts `--ct`, `--segmentation`, `--artery-model`, `--vein-model`, `--output`, `--registration-module`, `--output-root`, and `--case-id`. A nonempty output fails unless `--overwrite` is explicitly supplied; the production server procedure will always use a new output directory and will not pass `--overwrite`.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_preprocessing.py tests/test_cli.py
git add src/ct_vascular_resampling/manual_preprocessing.py scripts/preprocess_manual_segmentation_case.py tests/test_preprocessing.py tests/test_cli.py
git commit -m "feat: preprocess manual organs with external vessels"
```

Expected: all focused tests PASS.

### Task 8: Run Protocol, Summary Counts, Resume Rejection, and 60% Quality Rule

**Files:**
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_quality.py`

- [ ] **Step 1: Write failing protocol and summary tests**

Require `run_metadata.json` to contain segmentation path/hash/geometry, nearest interpolation, fill label `0`, mappings, EUS colors, 8-connectivity and touch-edge completeness rule, organ meshes, external vessel hashes, and `black_ratio_limit: 0.60` inside the resume-protocol digest.

Require `library_summary.json` to contain:

```python
assert summary["eus_vessel_label_counts"] == {
    "aorta": expected_aorta_slices,
    "inferior_vena_cava": expected_ivc_slices,
}
assert summary["eus_vessel_feature_counts"] == {"aorta": expected_complete_aorta_components}
assert summary["eus_vessel_colors"] == {
    "aorta": [255, 0, 0],
    "inferior_vena_cava": [0, 0, 255],
    "portal_vein": [170, 85, 255],
}
```

Assert resume rejects changed segmentation bytes, mappings, colors, threshold, geometry, or missing EUS images before appending records.

- [ ] **Step 2: Add exact 60% threshold tests**

```python
def test_quality_accepts_exactly_sixty_percent_black():
    result = evaluate_ct_quality(
        _pixels_with_black_ratio(0.60),
        FilterConfig(black_ratio_limit=0.60, line_min_diagonal_fraction=1.0),
    )
    assert result.black_ratio == 0.60
    assert result.black_ratio_exceeded is False

def test_quality_rejects_more_than_sixty_percent_black():
    result = evaluate_ct_quality(
        _pixels_with_black_ratio(0.61),
        FilterConfig(black_ratio_limit=0.60, line_min_diagonal_fraction=1.0),
    )
    assert result.black_ratio_exceeded is True
```

- [ ] **Step 3: Run and verify RED**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_pipeline.py tests/test_quality.py -k 'protocol or summary or sixty or resume'
```

Expected: new metadata/summary tests FAIL; quality comparison already uses strict `>` and should confirm the intended boundary once configured.

- [ ] **Step 4: Implement metadata/counting and strict restoration**

Extend `_input_provenance`, `_run_protocol_metadata`, and Gallery scanning to include the manual contract and stream counts without loading all records into memory. Make the manual schema part of `resume_protocol_sha256`. Keep legacy summary shape unchanged when manual mode is disabled.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_pipeline.py tests/test_quality.py tests/test_gallery_and_adapter.py
git add src/ct_vascular_resampling/pipeline.py tests/test_pipeline.py tests/test_quality.py
git commit -m "feat: audit manual EUS vessel Gallery runs"
```

Expected: all focused tests PASS.

### Task 9: Configuration Examples and Chinese/English Documentation

**Files:**
- Modify: `configs/case.example.yaml`
- Modify: `README.md`
- Modify: `C:\Users\zhangyutang\Desktop\CT血管重采样项目说明文档` files that describe Gallery construction
- Test: `tests/test_config.py`

- [ ] **Step 1: Update the configuration example**

Add a commented `manual_segmentation` block with the exact approved mappings and colors. State that Main Portal Vein `23` belongs to the organ union but not the new EUS vessel portal group; SMV `26` belongs to both. Show `black_ratio_limit: 0.60` for this mode while retaining the legacy default explanation.

- [ ] **Step 2: Update Chinese and English behavior documentation**

Document these non-negotiable facts:

- organ labels require at least one nearest-neighbor label pixel, not a closed mesh contour;
- a full-frame organ receives a label but no artificial square border;
- original PLY `features`, `boundary_only`, `ct_overlay`, and state routing are unchanged;
- the new three-class outputs are separate;
- incomplete/touch-edge EUS vessel components do not enter `eus_vessel_features`;
- every visible EUS boundary, including open/clipped/touch-edge boundaries, is drawn in both new images;
- the formal Case 2 run uses `black_ratio_limit: 0.60` with strict `>` rejection.

- [ ] **Step 3: Verify documentation facts and commit**

Run:

```bash
rg -n "23|26|33|34|35|36|37|0\.60|eus_vessel_features|触边|incomplete|touch-edge" README.md configs/case.example.yaml docs
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py
git add README.md configs/case.example.yaml
git commit -m "docs: describe manual EUS vessel Gallery outputs"
```

The desktop documentation is outside this Git repository. Update it in place, do not attempt to stage it in Git, and record its changed-file list plus before/after SHA-256 values in the final audit rather than copying it into the repository.

### Task 10: Full Local Verification and Independent Quality Review

**Files:**
- Review all files changed since `7c5e1cb`

- [ ] **Step 1: Run formatting/static sanity checks**

```bash
git diff --check 7c5e1cb..HEAD
python -m compileall -q src scripts tests
rg -n 'T(BD)|TO(DO)|implement[ ]later|fill[ ]in[ ]details' src tests scripts configs README.md
```

Expected: no whitespace errors, compile errors, or implementation placeholders.

- [ ] **Step 2: Run the complete local test suite**

```bash
mamba run -n ct-vessel-resampling pytest -q
```

Expected: all tests PASS with no warnings introduced by this branch.

- [ ] **Step 3: Review high-risk invariants against the design**

Inspect the diff and explicitly verify:

- raw label sampling is nearest-neighbor and canonical RAS geometry matches CT;
- no per-organ 3-D label copies are retained;
- label batches are bounded by `gpu_batch_size`/CPU batch size;
- label `23` is absent from EUS vessel portal features but present in organ portal mapping;
- incomplete EUS components are excluded only from features, never from either new image;
- new EUS features do not affect Gallery routing;
- legacy mode and original vessel output bytes remain protected by tests;
- no unrelated files or `.superpowers/` are staged.

- [ ] **Step 4: Request one code review and address findings with TDD**

Use the `requesting-code-review` skill with one reviewer, limited to the branch diff and design document. For each valid finding, first add/reproduce a failing test, then implement the correction and rerun the focused/full suite.

- [ ] **Step 5: Commit any review fixes**

```bash
git status --short
git log --oneline 7c5e1cb..HEAD
```

Expected: only intentional changes; `.superpowers/` remains untracked and unstaged.

### Task 11: GitHub Push, Server Backup, Input Staging, Pilot, and Full Run

**Files/locations:**
- Push branch: `feature/manual-seg-eus-vessels-20260811`
- Server project: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project`
- Server backup directory: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/backups`
- New preprocessing root: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/case_data_manual_eus_20260811`
- New output root: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_manual_seg_eus_vessels_20260811`

- [ ] **Step 1: Push the reviewed branch**

```bash
git push -u origin feature/manual-seg-eus-vessels-20260811
git rev-parse HEAD
git ls-remote origin refs/heads/feature/manual-seg-eus-vessels-20260811
```

Expected: local and remote hashes are identical.

- [ ] **Step 2: Reconfirm server isolation and resources read-only**

Connect only to `ssh -p 35258 root@connect.westd.seetacloud.com`, then run:

```bash
pwd
readlink -f /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
git -C /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project remote -v
git -C /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project status --short --branch
df -h /root/autodl-tmp
free -h
nvidia-smi
```

Expected: exact project root, expected Git remote, at least the previously observed 50 GB free before new outputs, and no unrelated project path selected.

- [ ] **Step 3: Create and verify the required server backup before code sync**

```bash
stamp=$(date +%Y%m%d_%H%M%S)
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731
tar -czf "backups/project_backup_${stamp}.tar.gz" project run
sha256sum "backups/project_backup_${stamp}.tar.gz" | tee "backups/project_backup_${stamp}.tar.gz.sha256"
tar -tzf "backups/project_backup_${stamp}.tar.gz" >/dev/null
```

Expected: archive verification exits `0`; no existing output or data directory is modified.

- [ ] **Step 4: Sync only the approved branch and stage the segmentation**

Fetch/fast-forward the server project to the pushed branch. Copy the local source segmentation to the new case-data directory, then verify:

```bash
sha256sum '/root/autodl-tmp/ct_vascular_resampling_case2_20260731/case_data_manual_eus_20260811/source/EUS main organ---.seg(1).nrrd'
```

Expected SHA-256: `0b56268488411925d96bb070e25e72a0105a8502e87ffd349a9ba01cd32dc124`.

- [ ] **Step 5: Run server tests before preprocessing**

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
/root/miniconda3/bin/mamba run -n ct-vessel-resampling-totalseg-gpu pytest -q
```

Expected: complete suite PASS.

- [ ] **Step 6: Generate manual organ inputs and audited production config**

Run the new preprocessing CLI with:

- CT: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project/case_data/ct/ct_venous.nrrd`
- segmentation: the staged file above;
- external artery: existing `project/case_data/models/artery_tree.ply`;
- external vein: existing `project/case_data/models/vein_tree.ply`;
- registration: existing `project/registration/2021.py`;
- output root: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_manual_seg_eus_vessels_20260811`.

Copy the audited partial-rotation Case 2 sampling/endpoints into a new server-only configuration, changing only manual organ model paths, adding `manual_segmentation`, setting the new output root, and setting `black_ratio_limit: 0.60`. Diff it against the previous audited config and record the diff under `run/`.

- [ ] **Step 7: Dry-run and pilot under bounded resources**

Run dry-run first and verify the sampled point/pose counts match the currently approved partial-rotation configuration. Create a separate pilot case/output, cap each point count to `2`, use `workers: 4` and `gpu_batch_size: 8`, and run all steps. Validate every pilot Gallery record, both new PNG paths, exact colors, feature completeness, metadata, summary, and original-output invariance.

- [ ] **Step 8: Launch the full run in an isolated screen session**

Use `workers: 8` initially and `gpu_batch_size: 8` to control RAM/VRAM. Launch under `setsid screen`, write the main log, PID, exit code, and resource CSV under this project's `run/` only. Do not touch the old 18 GB Gallery or 7.8 GB temporary Gallery.

- [ ] **Step 9: Monitor to terminal state**

At regular intervals record `nvidia-smi`, `free -h`, `df -h`, process RSS, manifest line count, Gallery line count, and errors from the new run log. If memory pressure, disk exhaustion, geometry mismatch, schema failure, or nonzero exit appears, stop only this new run and report the exact evidence before changing parameters.

- [ ] **Step 10: Stream final acceptance checks**

After exit code `0`, verify without loading all JSONL records into RAM:

- manifest count equals the pose count and equals the sum of four states;
- Gallery JSONL count equals each of `ct`, `boundary_only`, `ct_overlay`, `organ_vessel_boundary`, `eus_vessel_boundary`, and `ct_eus_vessel_overlay` PNG counts;
- all Gallery records have both metadata schemas and valid paths;
- organ/EUS labels use only approved names and exact mappings;
- EUS features use only complete non-touching components and labels present on the slice;
- open/touch-edge components sampled during pilot/audit remain colored in both images despite feature exclusion;
- original reconstructed-vessel fields and three original image families have their historical semantics;
- `run_metadata.json` is `complete`, hashes match inputs/commit, threshold is `0.60`;
- `library_summary.json` counts exactly match a fresh streaming recount;
- the old and temporary Gallery roots retain their pre-run counts and mtimes.

- [ ] **Step 11: Record final evidence**

Record local/GitHub/server commit hashes, backup path/hash, segmentation hash, CT/external-vessel hashes, dry-run/pilot/full commands, point/pose/state counts, output sizes, resource peaks, test totals, and final output paths in the delivery report. Do not merge to `main` without a separate user decision.
