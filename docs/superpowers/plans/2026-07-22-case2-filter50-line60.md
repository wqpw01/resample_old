# Case 2 Filter 50 Line 60 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Case 2 resampling gallery with a black-area rejection limit of 50% and a long black-boundary-line minimum length of 60% of the image diagonal.

**Architecture:** Create a versioned case configuration rather than altering the validated 30%/70% run. The remote pipeline resamples every square from the original NRRD and meshes, then writes gallery, unindexed, and rejected artifacts plus the rebuilt retrieval features. Validation compares manifests with artifacts and loads the resulting gallery database.

**Tech Stack:** Python 3.12, SimpleITK, NumPy/SciPy, OpenCV, trimesh, CUDA/CuPy fallback-aware pipeline, JSONL, SSH/SCP.

---

## File Structure

- Create: `configs/case_2_autodl_black50_line60.yaml` — immutable Case 2 configuration for the revised filter run.
- Create: `docs/superpowers/specs/2026-07-22-case2-filter50-line60-design.md` — approved behavioral specification.
- Create: `docs/superpowers/plans/2026-07-22-case2-filter50-line60.md` — this execution plan.
- Remote output: `/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/` — new gallery version, never overwriting `/output/case_2/`.

### Task 1: Derive and Validate the Versioned Configuration

**Files:**
- Create: `configs/case_2_autodl_black50_line60.yaml`
- Test: `src/ct_vascular_resampling/config.py` via the CLI dry run

- [ ] **Step 1: Create the derived YAML config**

  Copy all source paths, sampling values, CT values, vessels and runtime settings from `configs/case_2_autodl.yaml`. Set exactly:

  ```yaml
  output_root: /root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60
  filtering:
    black_threshold: 50
    black_ratio_limit: 0.5
    line_min_diagonal_fraction: 0.6
    black_side_min_ratio: 0.9
    valid_side_max_black_ratio: 0.1
  ```

- [ ] **Step 2: Copy the config into the remote project**

  Run:

  ```bash
  ssh ... 'test -d /root/autodl-tmp/ct_vascular_resampling_case2/project/configs'
  scp ... configs/case_2_autodl_black50_line60.yaml root@...:/root/autodl-tmp/ct_vascular_resampling_case2/project/configs/
  ```

  Expected: remote config file exists with the same SHA-256 as the local file.

- [ ] **Step 3: Validate configuration and inputs without writing samples**

  Run:

  ```bash
  /root/miniconda3/envs/ct-vessel-resampling-gpu/bin/python main.py \
    --case-config configs/case_2_autodl_black50_line60.yaml --dry-run
  ```

  Expected: configuration, NRRD CT, all organ meshes, both vessel meshes and registration module load successfully; the existing 30%/70% output remains unchanged.

### Task 2: Rebuild the Revised Gallery and Retrieval Database

**Files:**
- Read: `configs/case_2_autodl_black50_line60.yaml`
- Create: `/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/*`

- [ ] **Step 1: Start a fresh remote run and persist its log**

  Run from `/root/autodl-tmp/ct_vascular_resampling_case2/project`:

  ```bash
  /root/miniconda3/envs/ct-vessel-resampling-gpu/bin/python main.py \
    --case-config configs/case_2_autodl_black50_line60.yaml \
    2>&1 | tee /root/autodl-tmp/ct_vascular_resampling_case2/full-run-black50-line60.log
  ```

  Expected: all 74,187 deterministic square IDs are processed. Each former rejected sample is recomputed before status assignment, so it may produce blood-vessel features and enter `gallery`.

- [ ] **Step 2: Check runtime backend fidelity metadata**

  Inspect `run_metadata.json`:

  ```bash
  jq '.requested_backend, .selected_backend, .fallback_reason, .calibration' \
    /root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/run_metadata.json
  ```

  Expected: requested and selected backends plus the CPU/GPU calibration decision are documented; output must be from the fidelity-validated path.

### Task 3: Validate Artifacts and Retrieval Features

**Files:**
- Read: `/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/manifest.jsonl`
- Read: `/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/{gallery,unindexed,rejected}/*.jsonl`
- Read: `/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/library_summary.json`

- [ ] **Step 1: Validate status count and PNG pairing integrity**

  Run a read-only Python verification that counts manifest rows, groups statuses, and confirms every listed `ct`, `boundary_only`, and `ct_overlay` path exists. For every status, compare sorted `ct/*.png` names with `ct_overlay/*.png` names.

  Expected: manifest row count equals the sum of the three state counts; every artifact exists; each CT image has a same-name overlay.

- [ ] **Step 2: Load the rebuilt gallery database**

  Run with the project source path set:

  ```bash
  PYTHONPATH=src /root/miniconda3/envs/ct-vessel-resampling-gpu/bin/python - <<'PY'
  from ct_vascular_resampling.registration_adapter import load_gallery_database
  db = load_gallery_database('/root/autodl-tmp/ct_vascular_resampling_case2/output_black50_line60/case_2/gallery/gallery.jsonl')
  print(len(db))
  PY
  ```

  Expected: all gallery records load without parse errors and the count matches `library_summary.json`.

### Task 4: Deliver Random Local Samples

**Files:**
- Create: `C:\\Users\\zhangyutang\\Desktop\\CT血管重采样随机样本_20260722_145836_black50_line60/`

- [ ] **Step 1: Select and package 10 random samples per final state**

  On the remote server, use each final-state `ct/*.png` directory as the population. For `gallery`, `rejected`, and `unindexed`, select 10 filenames with `shuf -n 10`; copy each matching CT and `ct_overlay` PNG into a temporary transfer directory and write `selection.tsv`.

- [ ] **Step 2: Transfer and validate locally**

  Transfer the package to the stated desktop directory. Verify the directory contains 10 CT/overlay pairs per state, same-name lists match, CT PNGs are 300 x 300 grayscale, and overlay PNGs are 300 x 300 RGB.

- [ ] **Step 3: Remove only remote transfer staging artifacts**

  Remove the temporary preview archive and staging directory after local checks pass. Retain the versioned remote output, log and all JSONL files.

## Plan Self-Review

- Spec coverage: Tasks 1-2 apply the two confirmed threshold changes without changing any other sampling or feature definition; Task 3 rebuilds and proves retrieval-library integrity; Task 4 delivers all three requested sample classes.
- Placeholders: none. Exact paths, values, commands, and expected outputs are included.
- Consistency: the versioned config, output root, case ID and validation paths all consistently resolve to `output_black50_line60/case_2`.
- Repository state: the project directory has no `.git` metadata, so no commit is possible or required for this run-specific configuration and documentation.
