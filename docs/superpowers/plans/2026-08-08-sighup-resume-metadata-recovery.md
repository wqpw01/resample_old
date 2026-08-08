# SIGHUP Resume Metadata Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a verifiable resume protocol before the first rendered slice, reconstruct the missing protocol for the audited 88,280-record SIGHUP interruption, and resume the server run in an SSH-independent session.

**Architecture:** Keep the existing `GalleryWriter` manifests and strict pose validation as the source of completed-work truth. Add lifecycle fields to atomically written `run_metadata.json`, plus an explicit recovery function and CLI script that recompute the current protocol and validate every persisted pose before reconstructing missing metadata. Run the recovered job under detached `screen`, with a separate detached resource monitor and immutable attempt-specific evidence files.

**Tech Stack:** Python 3.10, pytest, NumPy, SimpleITK, trimesh, YAML case configuration, mamba, GNU screen, Git.

---

### Task 1: Reproduce the missing startup checkpoint

**Files:**
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing first-render checkpoint test**

Wrap the real renderer in the existing one-sample CPU pipeline fixture and assert that metadata exists before the first record is written:

```python
metadata_seen_before_render: dict[str, object] = {}
original_render = pipeline_module.render_precomputed_square

def observe_checkpoint(*args, **kwargs):
    metadata_path = config.output_root / config.case_id / "run_metadata.json"
    metadata_seen_before_render.update(json.loads(metadata_path.read_text(encoding="utf-8")))
    return original_render(*args, **kwargs)

monkeypatch.setattr(pipeline_module, "render_precomputed_square", observe_checkpoint)
summary = run_case(config, steps=["render"], workers=1)
assert metadata_seen_before_render["run_state"] == "running"
assert metadata_seen_before_render["completed_pose_count"] == 0
assert metadata_seen_before_render["status_counts"] == {}
assert summary.total_squares == 1
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
mamba run -n base python -m pytest tests/test_pipeline.py::test_run_case_writes_metadata_before_first_render -q
```

Expected: FAIL because `run_metadata.json` does not exist when the first renderer is called.

- [ ] **Step 3: Write the failing interrupted-state test**

Use the same small fixture, make `render_precomputed_square` raise `RuntimeError("simulated render failure")`, and assert after the exception:

```python
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert metadata["run_state"] == "interrupted"
assert metadata["completed_pose_count"] == 0
assert metadata["status_counts"] == {}
```

- [ ] **Step 4: Run the second focused test and verify RED**

Expected: FAIL because existing metadata has no `run_state` lifecycle field.

- [ ] **Step 5: Commit the reproducing tests**

```bash
git add tests/test_pipeline.py
git commit -m "test: reproduce missing resume checkpoint"
```

### Task 2: Persist lifecycle metadata before rendering

**Files:**
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add one metadata-state helper**

Add a small helper next to `_write_run_metadata`:

```python
def _metadata_with_state(
    metadata: dict[str, object],
    *,
    state: str,
    statuses: Counter[str],
    total_squares: int,
) -> dict[str, object]:
    completed = sum(statuses.values())
    return {
        **metadata,
        "run_state": state,
        "total_squares": total_squares,
        "completed_pose_count": completed,
        "status_counts": dict(sorted(statuses.items())),
        "excluded_fov_count": statuses["excluded_fov"],
    }
```

Reject any state outside `{"running", "interrupted", "complete"}`.

- [ ] **Step 2: Write `running` before the first pending sample**

After backend selection and GPU calibration, carry forward an existing `recovery_history` list if present and atomically write a `running` checkpoint before entering either CPU or GPU batch loop.

- [ ] **Step 3: Write terminal lifecycle states**

In the rendering `finally`, write `complete` only when the status total equals `len(samples)`; otherwise write `interrupted`. If all poses were already complete on entry, update reconstructed metadata to `complete` before skipping rendering.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
mamba run -n base python -m pytest \
  tests/test_pipeline.py::test_run_case_writes_metadata_before_first_render \
  tests/test_pipeline.py::test_run_case_marks_metadata_interrupted_after_render_error -q
```

Expected: 2 passed.

- [ ] **Step 5: Run all pipeline tests**

```bash
mamba run -n base python -m pytest tests/test_pipeline.py -q
```

Expected: all pipeline tests pass.

- [ ] **Step 6: Commit lifecycle implementation**

```bash
git add src/ct_vascular_resampling/pipeline.py tests/test_pipeline.py
git commit -m "fix: persist resumable run metadata before rendering"
```

### Task 3: Add explicit interrupted-metadata reconstruction

**Files:**
- Modify: `src/ct_vascular_resampling/pipeline.py`
- Create: `scripts/recover_interrupted_run_metadata.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing recovery tests**

Create a one-sample output in `tmp_path`, remove only its temporary-test `run_metadata.json`, and call the wished-for API:

```python
metadata = recover_interrupted_run_metadata(
    config,
    expected_completed_count=1,
    reason="sighup",
    exit_code=129,
    recovered_at_utc="2026-08-08T10:43:29Z",
)
assert metadata["run_state"] == "interrupted"
assert metadata["completed_pose_count"] == 1
assert metadata["status_counts"] == {"excluded_fov": 1}
assert metadata["recovery_history"] == [{
    "reason": "sighup",
    "exit_code": 129,
    "recovered_at_utc": "2026-08-08T10:43:29Z",
    "completed_pose_count": 1,
    "status_counts": {"excluded_fov": 1},
}]
assert len(metadata["resume_protocol_sha256"]) == 64
```

Add separate tests proving the function rejects an existing metadata file, an expected count mismatch, and a stale or geometrically changed pose.

The partial-resume test must use different 40-character commits before and after interruption. It must prove that old records retain the interrupted commit, new records use the recovery implementation commit, and final metadata preserves the explicitly compatible completed-commit list and recovery history.

- [ ] **Step 2: Run recovery tests and verify RED**

Expected: collection or import failure because `recover_interrupted_run_metadata` does not exist.

- [ ] **Step 3: Extract a shared pose-plan preparation helper**

Move the existing preflight, organ sampling, centerline-selection audit, square generation, sampled counts and protocol calculation into one private `_prepare_pose_plan(config)` helper used unchanged by both `run_case` and recovery. Preserve the existing operation order and returned `RunSummary` behavior.

- [ ] **Step 4: Implement strict metadata reconstruction**

Implement:

```python
def recover_interrupted_run_metadata(
    config: CaseConfig,
    *,
    expected_completed_count: int,
    reason: str,
    exit_code: int,
    recovered_at_utc: str | None = None,
) -> dict[str, object]:
    ...
```

It must refuse an existing metadata file, load all manifests through `GalleryWriter`, require the exact expected count, validate every completed pose with `_validate_completed_pose`, reject stale IDs, compute current input hashes and protocol, atomically write marked `interrupted` metadata, and return it. It must not load CT voxel data, create a resampling backend, or modify PNG, PLY, or JSONL files.

- [ ] **Step 5: Add the recovery CLI script**

The script accepts exactly:

```text
--case-config PATH
--expected-completed-count INTEGER
--reason sighup
--exit-code 129
```

It loads the case YAML, calls the production recovery function, prints the resulting JSON, and returns nonzero on validation failure.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run recovery tests plus `tests/test_cli.py`; expected: all pass.

- [ ] **Step 7: Commit recovery implementation**

```bash
git add src/ct_vascular_resampling/pipeline.py scripts/recover_interrupted_run_metadata.py tests/test_pipeline.py tests/test_cli.py
git commit -m "feat: reconstruct audited metadata for interrupted runs"
```

### Task 4: Document and verify the recovery contract

**Files:**
- Modify: `README.md`
- Modify: `docs/core-design-change-log-20260807.md`

- [ ] **Step 1: Update Chinese and English documentation**

Document the three `run_state` values, startup checkpoint timing, strict recovery command, refusal conditions, and `screen` deployment rule. State explicitly that recovery does not change sampling geometry or medical image resampling parameters.

- [ ] **Step 2: Run documentation and full local checks**

```bash
git diff --check
rg -n "TBD|PLACEHOLDER" README.md docs/core-design-change-log-20260807.md || true
mamba run -n base python -m pytest -q
```

Expected: no diff errors or placeholders; the full suite passes.

- [ ] **Step 3: Commit documentation**

```bash
git add README.md docs/core-design-change-log-20260807.md
git commit -m "docs: explain interrupted run recovery"
```

### Task 5: Push and deploy the verified fix

**Files:**
- Remote Git branch: `feature/core-design-alignment-20260806`
- Remote server root: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731`

- [ ] **Step 1: Verify and push the feature branch**

Confirm tracked status is clean, record the new HEAD, and push only the feature branch. Verify remote `main` remains `b1898a0e6e09e369d39fe6c6136313f84e514ed5`.

- [ ] **Step 2: Back up server code and run evidence**

Create a new timestamped TAR under the allowed `backups/` directory containing only `project/` and `run/`; verify gzip integrity, member prefixes and SHA-256.

- [ ] **Step 3: Update server code without touching outputs**

Transfer the verified Git bundle, fast-forward the server feature branch, and confirm the server HEAD and tracked tree match local/GitHub. Preserve the untracked runtime config and all partial output files.

- [ ] **Step 4: Run server tests**

Run the full server suite in `ct-vessel-resampling-totalseg-gpu`; expected: all tests pass before metadata reconstruction.

### Task 6: Reconstruct metadata and resume under detached screen

**Files:**
- Runtime config: `project/configs/case_2_partial_rotation_20260808.yaml`
- Existing output: `output_partial_rotation_20260808/case_2/`
- New evidence: attempt-specific files under `run/`

- [ ] **Step 1: Revalidate the interrupted output**

Require no active resampling process, old exit code 129, root count 88,280, exact state-count sum 88,280, OOM zero, expected Git/config hashes and sufficient disk.

- [ ] **Step 2: Test SSH-independent detachment**

Start a harmless five-second `setsid screen -DmS` probe from a separate SSH connection, let that connection end, and require its evidence file to appear afterward.

- [ ] **Step 3: Reconstruct metadata**

Run:

```bash
mamba run -n ct-vessel-resampling-totalseg-gpu python scripts/recover_interrupted_run_metadata.py \
  --case-config configs/case_2_partial_rotation_20260808.yaml \
  --expected-completed-count 88280 \
  --reason sighup \
  --exit-code 129
```

Verify `run_state`, recovery history, protocol hash, exact counts and atomic-file cleanup before resuming.

- [ ] **Step 4: Start the resume and monitor sessions**

Use unique screen names and new logs. The run session writes a new exit-code file; the monitor session appends 30-second CPU RSS, cgroup memory, OOM, GPU, disk and manifest count samples to a new CSV. Preserve all first-attempt evidence files.

- [ ] **Step 5: Confirm skip-and-resume behavior**

After planning and manifest validation, require the count to advance beyond 88,280 without duplicate IDs and without rewriting the first-attempt files. Verify OOM remains zero.

### Task 7: Complete and validate the full result

**Files:**
- Final output: `output_partial_rotation_20260808/case_2/`
- Final evidence: `run/partial_rotation_resume_*`

- [ ] **Step 1: Monitor to terminal exit**

Continue until the detached run writes exit code 0. Stop and request direction on any nonzero exit, OOM event, protocol mismatch or unsafe disk threshold.

- [ ] **Step 2: Run structured final validation**

Require exactly 208,250 unique root records; exact equality with the sum and ID union of gallery, unindexed, rejected and excluded-FOV manifests; all referenced files present; final `run_state: complete`; `completed_pose_count: 208250`; unchanged angle arrays, point spacing, design hash, build commit and input hashes.

- [ ] **Step 3: Record hashes and resource results**

Record SHA-256 for runtime config, metadata, summary, all JSONL manifests, logs and resource CSV. Report peak sampled memory, final disk usage, OOM counters, status counts and elapsed time.

- [ ] **Step 4: Complete the plan only after verification**

Run the `verification-before-completion` checklist, update the task plan, and report both the recovered interruption evidence and final successful result without describing the output as clinically validated.
