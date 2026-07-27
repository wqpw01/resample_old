# Case 2 Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert case 2 venous-phase DICOM CT and Slicer segmentation into geometry-aligned NRRD CT, binary masks, PLY meshes, and a downstream case YAML.

**Architecture:** Add a focused preprocessing module that owns label taxonomy, DICOM/segmentation validation, mask construction, physical-space mesh extraction, and artifact writing. A thin script supplies the known case 2 paths and calls the module; existing resampling CLI behavior stays unchanged.

**Tech Stack:** Python 3.12, SimpleITK, NumPy, scikit-image Marching Cubes, trimesh, PyYAML, pytest.

---

### Task 1: Add Reproducible Mesh Dependency And Label Taxonomy

**Files:**
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Create: `src/ct_vascular_resampling/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing taxonomy test**

```python
from ct_vascular_resampling.preprocessing import ARTERY_LABEL_VALUES, VEIN_LABEL_VALUES, PORTAL_AUXILIARY_LABEL_VALUES

def test_case2_vessel_taxonomy_matches_confirmed_slicer_colours():
    assert ARTERY_LABEL_VALUES == (8, 20, 22, 24, 25, 39, 40)
    assert VEIN_LABEL_VALUES == (9, 23, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 41, 42)
    assert PORTAL_AUXILIARY_LABEL_VALUES == (23, 26, 33, 34, 35, 36, 37)
```

- [ ] **Step 2: Run the taxonomy test and verify it fails**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_case2_vessel_taxonomy_matches_confirmed_slicer_colours -q`

Expected: FAIL because `ct_vascular_resampling.preprocessing` does not exist.

- [ ] **Step 3: Add dependencies and the immutable taxonomy constants**

```python
ORGAN_LABEL_VALUES = {
    "spleen": (1,), "kidney_right": (2,), "kidney_left": (3,),
    "gallbladder": (4,), "esophagus": (5,), "liver": (6,),
    "stomach": (7,), "aorta": (8,), "inferior_vena_cava": (9,),
    "pancreas": (11,), "adrenal_gland_right": (12,),
    "adrenal_gland_left": (13,), "duodenum": (14,),
}
ARTERY_LABEL_VALUES = (8, 20, 22, 24, 25, 39, 40)
VEIN_LABEL_VALUES = (9, 23, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 41, 42)
PORTAL_AUXILIARY_LABEL_VALUES = (23, 26, 33, 34, 35, 36, 37)
```

Add `scikit-image>=0.25` to `requirements.txt` and `scikit-image` to `environment.yml`.

- [ ] **Step 4: Run the taxonomy test and verify it passes**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_case2_vessel_taxonomy_matches_confirmed_slicer_colours -q`

Expected: PASS.

### Task 2: Validate Geometry And Generate Binary Masks

**Files:**
- Modify: `src/ct_vascular_resampling/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing geometry and union-mask tests**

```python
def test_validate_geometry_accepts_sub_micrometre_origin_difference(ct_image, segmentation_image):
    validate_geometry(ct_image, segmentation_image)

def test_build_binary_masks_unions_requested_label_values(segmentation_image):
    masks = build_binary_masks(segmentation_image, {"artery_tree": (8, 20)})
    assert np.array_equal(sitk.GetArrayFromImage(masks["artery_tree"]), expected_union)
    assert masks["artery_tree"].GetOrigin() == segmentation_image.GetOrigin()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py -k 'geometry or binary_masks' -q`

Expected: FAIL because the functions do not exist.

- [ ] **Step 3: Implement validation and mask construction**

```python
def validate_geometry(ct: sitk.Image, segmentation: sitk.Image, atol_mm: float = 1e-6) -> None:
    if ct.GetSize() != segmentation.GetSize() or not np.allclose(ct.GetSpacing(), segmentation.GetSpacing(), atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与分割的 Size 或 Spacing 不一致")
    if not np.allclose(ct.GetOrigin(), segmentation.GetOrigin(), atol=atol_mm, rtol=0.0) or not np.allclose(ct.GetDirection(), segmentation.GetDirection(), atol=1e-8, rtol=0.0):
        raise ValueError("CT 与分割不在同一物理空间")

def build_binary_masks(segmentation: sitk.Image, mappings: Mapping[str, tuple[int, ...]]) -> dict[str, sitk.Image]:
    labels = sitk.GetArrayViewFromImage(segmentation)
    masks = {}
    for name, values in mappings.items():
        binary = np.isin(labels, values).astype(np.uint8)
        if not binary.any():
            raise ValueError(f"{name} 没有分割体素")
        image = sitk.GetImageFromArray(binary)
        image.CopyInformation(segmentation)
        masks[name] = image
    return masks
```

- [ ] **Step 4: Run the focused test group and verify it passes**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py -k 'geometry or binary_masks' -q`

Expected: PASS.

### Task 3: Extract Physical-Space PLY Meshes

**Files:**
- Modify: `src/ct_vascular_resampling/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write a failing mesh-coordinate test**

```python
def test_mask_to_mesh_uses_origin_spacing_and_direction():
    mask = rotated_single_voxel_mask()
    mesh = mask_to_mesh(mask)
    expected = np.asarray(mask.TransformContinuousIndexToPhysicalPoint((1.5, 1.0, 1.0)))
    assert np.any(np.linalg.norm(mesh.vertices - expected, axis=1) < 1e-6)
    assert len(mesh.faces) > 0
```

- [ ] **Step 2: Run the mesh test and verify it fails**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_mask_to_mesh_uses_origin_spacing_and_direction -q`

Expected: FAIL because `mask_to_mesh` does not exist.

- [ ] **Step 3: Implement Marching Cubes and PLY writer**

```python
def mask_to_mesh(mask: sitk.Image) -> trimesh.Trimesh:
    array = sitk.GetArrayViewFromImage(mask)
    vertices_zyx, faces, _, _ = marching_cubes(array.astype(np.float32), level=0.5)
    indices_xyz = vertices_zyx[:, [2, 1, 0]]
    scaled = indices_xyz * np.asarray(mask.GetSpacing())
    vertices_xyz = scaled @ np.asarray(mask.GetDirection()).reshape(3, 3).T + np.asarray(mask.GetOrigin())
    mesh = trimesh.Trimesh(vertices=vertices_xyz, faces=faces, process=False)
    mesh.fix_normals(multibody=True)
    return mesh
```

Write meshes with `mesh.export(path, file_type="ply")`, rejecting empty arrays before export.

- [ ] **Step 4: Run the mesh test and verify it passes**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_mask_to_mesh_uses_origin_spacing_and_direction -q`

Expected: PASS.

### Task 4: Write Case Artifacts And Downstream YAML

**Files:**
- Modify: `src/ct_vascular_resampling/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write a failing artifact-layout test**

```python
def test_write_preprocessed_case_writes_ct_masks_meshes_manifest_and_case_yaml(tmp_path, ct_image, segmentation_image):
    result = write_preprocessed_case(ct_image, segmentation_image, tmp_path, Path('/tmp/2021.py'))
    assert (tmp_path / 'ct' / 'ct_venous.nrrd').is_file()
    assert (tmp_path / 'masks' / 'artery_tree.nrrd').is_file()
    assert (tmp_path / 'models' / 'vein_tree.ply').is_file()
    assert (tmp_path / 'manifest.json').is_file()
    assert yaml.safe_load((tmp_path / 'case_preprocessed.yaml').read_text())['vessel_models'][0]['label'] == 'artery'
```

- [ ] **Step 2: Run the artifact test and verify it fails**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_write_preprocessed_case_writes_ct_masks_meshes_manifest_and_case_yaml -q`

Expected: FAIL because `write_preprocessed_case` does not exist.

- [ ] **Step 3: Implement atomic artifact writing and manifest generation**

`write_preprocessed_case` must create `ct/`, `masks/`, and `models/`; write all organ, arterial, venous and portal-auxiliary masks and PLY files; write counts and watertight state to `manifest.json`; and emit a config whose `organ_models.portal_vein_and_splenic_vein` points at the auxiliary mesh and whose vessel labels are `artery` and `vein`.

- [ ] **Step 4: Run the artifact test and verify it passes**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_write_preprocessed_case_writes_ct_masks_meshes_manifest_and_case_yaml -q`

Expected: PASS.

### Task 5: Add The Case 2 Runner And Produce Real Artifacts

**Files:**
- Create: `scripts/preprocess_slicer_case.py`
- Modify: `README.md`

- [ ] **Step 1: Write the runner contract test**

```python
def test_case2_runner_parser_requires_dicom_segmentation_and_output():
    parser = build_parser()
    args = parser.parse_args(['--dicom-dir', 'dicom', '--segmentation', 'seg.nrrd', '--output', 'out'])
    assert args.series_id is None
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py::test_case2_runner_parser_requires_dicom_segmentation_and_output -q`

Expected: FAIL because `scripts.preprocess_slicer_case` does not exist.

- [ ] **Step 3: Implement the runner and case selection**

The script must use `ImageSeriesReader.GetGDCMSeriesIDs`, select a supplied `--series-id` or exactly one series whose description matches `2.0 x 2.0_V`, read the segmentation, call `write_preprocessed_case`, and print a JSON summary. Add the documented invocation to `README.md`.

- [ ] **Step 4: Run unit tests and generate the real case output**

Run: `mamba run -n base python -m pytest tests/test_preprocessing.py -q`

Expected: PASS.

Run:

```bash
mamba run -n base python scripts/preprocess_slicer_case.py \
  --dicom-dir '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例2/1.2.840.78.85.7.5.1809089.1755084265' \
  --segmentation '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例2/EUS main organ---.seg(1).nrrd' \
  --output '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例2/预处理后' \
  --series-id '1.2.156.112605.189250946070685.250813112346.3.8268.135835' \
  --registration-module '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/2021.py'
```

- [ ] **Step 5: Verify real artifacts**

Run: `mamba run -n base python -m pytest -q` and inspect `预处理后/manifest.json` plus every emitted PLY with `trimesh.load`.

Expected: all project tests pass; 13 organ meshes, one portal auxiliary mesh, one arterial mesh and one venous mesh load as triangular meshes; CT and all masks retain the validated CT geometry.

## Plan Self-Review

- Spec coverage: Tasks 1-5 cover exact taxonomy, DICOM CT conversion, binary masks, physical-space PLY extraction, manifests, downstream YAML, tests and real output generation.
- Placeholder scan: no deferred implementation steps or unspecified label mappings remain.
- Type consistency: `validate_geometry`, `build_binary_masks`, `mask_to_mesh` and `write_preprocessed_case` are introduced before the orchestration task that consumes them.
- Repository note: the project is not a Git repository, so the normal per-task commit steps cannot be performed.
