# Core Design Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` and `test-driven-development` to implement this plan task-by-task.

**Goal:** Align the CT-EUS sampling pipeline with the 2026-08-06 core design and the user's approved clarifications, then deploy and run the complete case on the isolated server project.

**Architecture:** Convert all input geometry to canonical RAS at the load boundary, build design-specific candidate regions, apply deterministic 10 mm constrained FPS, and generate semantic local-frame poses as a stream. Preserve the approved CT rendering and quality policies while adding enough provenance to audit every output pose.

**Tech Stack:** Python 3.12, NumPy, SciPy, SimpleITK, trimesh, scikit-image, pytest, mamba, YAML/JSONL/PLY.

---

### Task 1: Baseline and rollback

- [x] Verify `main@b1898a0`, core DOCX SHA-256 and DOCX archive integrity.
- [x] Create `feature/core-design-alignment-20260806`.
- [x] Archive and hash the desktop Markdown/HTML documentation.
- [x] Commit the approved design specification and traceability matrix.

### Task 2: Canonical RAS and design filters

- [ ] Write failing coordinate conversion and filter boundary tests.
- [ ] Implement explicit LPS-to-RAS CT and mesh boundaries.
- [ ] Implement the 100 mm target-organ ray filter and R04-R06/R08-R09 region formulas.
- [ ] Run focused and regression tests, update traceability evidence, and commit.

### Task 3: Spacing, esophagus extension and centerline

- [ ] Write failing tests for 10 mm constrained FPS, full-span esophagus extension and centerline topology/tangent rules.
- [ ] Implement deterministic region-capped FPS and sampling statistics.
- [ ] Implement the validated 1 mm duodenum skeleton path and 10 mm chord tangent.
- [ ] Run focused and regression tests, update traceability evidence, and commit.

### Task 4: Local frames and semantic poses

- [ ] Write failing tests for ordinary/duodenum zero planes, right-handed axes, Z-Y-X rotations and pose counts.
- [ ] Implement ordinary, supplemental and duodenum local-frame construction.
- [ ] Implement yaw-region classification, stable pose IDs and strict duplicate suppression.
- [ ] Run focused and regression tests, update traceability evidence, and commit.

### Task 5: Streaming outputs and documentation

- [ ] Write failing integration tests for streaming pose output, resume behavior and RAS metadata.
- [ ] Extend configuration and JSONL/run metadata without breaking retrieval fields.
- [ ] Update repository and desktop Markdown documentation; regenerate desktop HTML from the Markdown source.
- [ ] Run the full local suite, local-model sampling acceptance and code review; commit each coherent module.

### Task 6: GitHub and server deployment

- [ ] Push only the feature branch to `origin`; do not merge `main`.
- [ ] Verify the exact allowed server root and current worktree state.
- [ ] Archive `project/` and `run/`, hash the archive and verify its member list before changing server code.
- [ ] Fetch and check out the feature branch without cleaning or overwriting unrelated/untracked content.

### Task 7: Production resampling and acceptance

- [ ] Generate the complete sampling/pose count before rendering.
- [ ] Enforce the approved disk gate: observed estimate times 1.25 plus 5 GiB must fit.
- [ ] Run the complete case in the new output directory while monitoring logs, memory, GPU and disk.
- [ ] Verify manifest totals, spacing, RAS metadata, output hashes and absence of runtime errors.
- [ ] Produce the final compliance and change report with rollback instructions.
