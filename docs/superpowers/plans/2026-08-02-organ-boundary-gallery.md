# TotalSegmentator Organ Boundary Gallery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use test-driven-development for every behavior change and subagent-driven-development reviews at each checkpoint.

**Goal:** Extend the automatic CT plus mixed-NRRD workflow so gallery frames contain TotalSegmentator organ labels and a white-background organ-plus-vessel boundary image, then rebuild and verify the remote case in a versioned output directory.

**Architecture:** Keep the existing CT, sampling, quality, vessel-feature, and retrieval contracts. Make automatic preprocessing accept `artery`/`vein` mappings with optional legacy `portal`, reuse validated TotalSegmentator output, and generate the same internal case config. Add a separate organ rendering layer that runs only after a frame qualifies for `gallery`, using bounds rejection before exact mesh sections. Persist organ artifacts separately from vessel features.

**Tech Stack:** Python 3.12, NumPy, SimpleITK, trimesh, Pillow, PyYAML, pytest, mamba, TotalSegmentator, CUDA/CuPy.

---

### Task 1: Automatic input and TotalSegmentator cache

**Files:** `src/ct_vascular_resampling/auto_preprocessing.py`, `configs/auto_case.example.yaml`, `tests/test_auto_preprocessing.py`

1. Add failing tests proving `vessel_label_values` accepts exactly `artery` and `vein`, permits optional legacy `portal`, rejects overlap/unknown keys, and unions `portal` into the generated venous tree.
2. Add failing tests proving a configured/default TotalSegmentator directory is reused only when all 14 masks are non-empty and CT-aligned, while missing or invalid output invokes TotalSegmentator and records whether cache was reused.
3. Implement the minimal parsing, validation, cache selection, subprocess orchestration, and provenance changes. Keep mixed NRRD organ labels ignored by constructing only configured vessel masks.
4. Update the auto YAML example and README input contract. Run focused tests after each red/green cycle.

### Task 2: Organ mesh section and renderer

**Files:** `src/ct_vascular_resampling/geometry.py`, `src/ct_vascular_resampling/rendering.py`, `src/ct_vascular_resampling/config.py`, matching focused tests

1. Add failing tests for a conservative mesh-bounds-versus-square predicate, including disjoint plane, disjoint square projection, and possible intersection.
2. Add failing renderer tests proving the vessel-only image is unchanged, the combined image is white-backed, organs are drawn before vessels, labels are sorted/deduplicated, and clipped positive-area contours count as present.
3. Implement `OrganLayer`, the fixed 11-organ ID/color maps, bounds rejection, and combined rendering without adding organs to vascular `features`.

### Task 3: Gallery-only pipeline and persistence

**Files:** `src/ct_vascular_resampling/pipeline.py`, `src/ct_vascular_resampling/gallery.py`, `tests/test_pipeline.py`, `tests/test_gallery_and_adapter.py`

1. Add failing tests proving organ mesh intersection occurs only for quality-accepted frames with complete vessel features.
2. Add failing persistence tests proving gallery receives `organ_vessel_boundary/<id>.png`, `organ_vessel_boundary_png`, and `organ_labels`, while unindexed/rejected/excluded outputs remain unchanged.
3. Add a failing resume test proving an old gallery manifest without the new artifact fails with an actionable error.
4. Implement prepared organ meshes with cached bounds, conditional intersections, atomic image/record persistence, and stale-output validation. Keep adapter behavior unchanged and prove it ignores organ fields.
5. Extend `library_summary.json` with organ label counts and the fixed color legend.

### Task 4: Documentation, review, and local verification

1. Update README output layout, auto-input behavior, cache rules, status scope, JSONL fields, and migration note.
2. Run focused tests, then the complete project suite in the project mamba environment.
3. Review the implementation against the approved design, then perform a separate code-quality review; fix every critical or important finding and rerun verification.
4. Merge the feature branch into local `main`, rerun the full suite on the merge result, and push `origin/main` without force.

### Task 5: Remote deployment and full build

1. Record old output manifest hashes and disk/GPU state. Fast-forward the remote project to the pushed `main` while preserving untracked `case_data/`, `registration/`, and local configs.
2. Create a remote auto config that uses the existing CT, `vascular_labels.nrrd`, `artery: [1]`, `vein: [2, 3]`, the validated TotalSegmentator cache, and the new `output_organs_v1` root.
3. Run a small real-data pilot and verify image dimensions/modes/colors, allowed labels, unchanged vascular features, and expected state isolation. Use the pilot to estimate full runtime.
4. Start the full run in a persistent session, monitor logs, process/GPU/disk growth, and status counts until completion. Resume only within the new output if interrupted.
5. Verify the expected 167,724 total samples and prior state counts, all referenced files, one-to-one filename sets for four gallery image directories, 112,749 adapter-loadable gallery records, allowed organ labels, and unchanged old output hashes. Save the effective config, run log, validation JSON, and representative samples; do not create a full ZIP.
