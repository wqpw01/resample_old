# Partial Rotation Range Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand fixed roll to `[-15, +15]` degrees and pitch to `[-10, +10]` degrees at 5-degree steps while preserving the 400 case-2 sampling points, all yaw policies, local-frame geometry, and the isolated production workflow.

**Architecture:** Keep angle policy in the existing pose module, derive allocation from the actual roll/pitch arrays, and let existing metadata consume those arrays without a new YAML schema. Prove the Cartesian product with unit and end-to-end tests, update tracked and desktop documentation, then deploy the exact feature commit through a verified Git bundle. A server dry-run and resource gate must pass before the complete GPU run starts in a new output root.

**Tech Stack:** Python 3.12, NumPy, SciPy, trimesh, SimpleITK, pytest, Mamba, Markdown/HTML, Git bundle, Bash, CUDA GPU backend.

---

## File Map

- `src/ct_vascular_resampling/squares.py`: owns fixed roll, pitch, yaw arrays and Cartesian-product generation.
- `src/ct_vascular_resampling/sampling_pipeline.py`: classifies yaw policy and preallocates the pose stream.
- `src/ct_vascular_resampling/pipeline.py`: already serializes angle arrays into run metadata; no production change is expected.
- `tests/test_pose_generation.py`: exact angle sets, per-policy counts, zero pose, and `Rz @ Ry @ Rx` geometry.
- `tests/test_sampling_pipeline.py`: stream counts, yaw classification, IDs, and exact-pose deduplication.
- `tests/test_pipeline.py`: dry-run totals, batches, resume protocol, manifests, and angle metadata.
- `README.md`, `configs/case.example.yaml`, `docs/core-design-traceability-20260806.md`, `docs/core-design-change-log-20260807.md`: tracked implementation and audit documentation.
- Desktop `CT血管重采样项目详细说明.md/.html`: detailed guide source and generated delivery.
- Server-only `configs/case_2_partial_rotation_20260808.yaml`: audited case config copied to a new output root; untracked because it contains server paths.

### Task 1: Test-drive angle arrays and dynamic pose allocation

**Files:**
- Modify: `tests/test_pose_generation.py:46-64`
- Modify: `tests/test_sampling_pipeline.py:1-145`
- Modify: `tests/test_pipeline.py:409-475`
- Modify: `src/ct_vascular_resampling/squares.py:12-21`
- Modify: `src/ct_vascular_resampling/sampling_pipeline.py:548-570`

- [ ] **Step 1: Verify the focused baseline**

```bash
mamba run -n base python -m pytest \
  tests/test_pose_generation.py \
  tests/test_sampling_pipeline.py \
  tests/test_pipeline.py -q
```

Expected: 22 tests pass with the old 117/333/279 contracts.

- [ ] **Step 2: Change pose-generation contract tests first**

Replace the old assertions in `test_pose_variants_use_confirmed_roll_pitch_and_region_yaw_ranges`:

```python
assert len(standard) == 455
assert len(bulb) == 1295
assert len(special) == 1085
assert {value.roll_degrees for value in standard} == {
    -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0
}
assert {value.pitch_degrees for value in standard} == {-10.0, -5.0, 0.0, 5.0, 10.0}
assert {value.yaw_degrees for value in standard} == set(np.arange(-30.0, 31.0, 5.0))
assert {value.yaw_degrees for value in bulb} == set(np.arange(-90.0, 91.0, 5.0))
assert {value.yaw_degrees for value in special} == set(np.arange(-120.0, 31.0, 5.0))
```

Keep zero-pose and `Rz @ Ry @ Rx` tests unchanged because 0 and 5 degrees remain valid.

- [ ] **Step 3: Add a regression test for derived allocation**

Add imports in `tests/test_sampling_pipeline.py`:

```python
import ct_vascular_resampling.sampling_pipeline as sampling_pipeline_module
from ct_vascular_resampling.sampling_pipeline import _candidate_pose_count
```

Add:

```python
def test_candidate_pose_count_is_derived_from_roll_pitch_and_yaw_arrays(monkeypatch):
    surfaces = {
        "stomach": SurfaceSamples(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0]]),
            region_ids=("stomach",),
            target_ids=(("liver",),),
            zero_plane_anchor_world=np.asarray([0.0, -1.0, 0.0]),
            pancreas_special_x_limit=10.0,
        )
    }
    monkeypatch.setattr(sampling_pipeline_module, "ROLL_ANGLES_DEGREES", (-10.0, 0.0, 10.0))
    monkeypatch.setattr(sampling_pipeline_module, "PITCH_ANGLES_DEGREES", (-5.0, 5.0))

    assert _candidate_pose_count(surfaces) == 3 * 2 * 13
```

It must fail as `117 != 78` while allocation still contains literal `9`.

- [ ] **Step 4: Update dependent contract expectations**

In `tests/test_sampling_pipeline.py`, replace ordinary stream counts `117 -> 455`, bulb `333 -> 1295`, special `279 -> 1085`, including duplicate-stream and unique-ID assertions.

In `tests/test_pipeline.py`, update the one-point integration case:

```python
assert dry.total_squares == 455
assert completed.total_squares == 455
assert metadata["total_squares"] == 455
assert metadata["pose_angles_degrees"]["roll"] == [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
assert metadata["pose_angles_degrees"]["pitch"] == [-10.0, -5.0, 0.0, 5.0, 10.0]
assert metadata["completed_pose_count"] == 455
assert cpu_batch_sizes == [8] * 56 + [7]
assert sum(resumed.status_counts.values()) == 455
assert len((tmp_path / "output" / "case_001" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 455
```

- [ ] **Step 5: Observe intended failures**

```bash
mamba run -n base python -m pytest \
  tests/test_pose_generation.py::test_pose_variants_use_confirmed_roll_pitch_and_region_yaw_ranges \
  tests/test_sampling_pipeline.py::test_candidate_pose_count_is_derived_from_roll_pitch_and_yaw_arrays -q
```

Expected: both fail against old arrays/allocation.

- [ ] **Step 6: Implement the minimum change**

In `squares.py`:

```python
ROLL_ANGLES_DEGREES = tuple(float(value) for value in range(-15, 16, 5))
PITCH_ANGLES_DEGREES = tuple(float(value) for value in range(-10, 11, 5))
```

Keep all yaw sequences unchanged. In `_candidate_pose_count`:

```python
axis_pose_count = len(ROLL_ANGLES_DEGREES) * len(PITCH_ANGLES_DEGREES)
count += len(YAW_ANGLES_DEGREES[yaw_policy]) * axis_pose_count
```

Do not change rotation matrices, local frames, point selection, or YAML parsing.

- [ ] **Step 7: Run impacted tests**

```bash
mamba run -n base python -m pytest \
  tests/test_pose_generation.py \
  tests/test_sampling_pipeline.py \
  tests/test_pipeline.py -q
```

Expected: 23 tests pass; integration writes/resumes exactly 455 poses in batches `[8] * 56 + [7]`.

- [ ] **Step 8: Commit code and tests**

```bash
git add src/ct_vascular_resampling/squares.py \
  src/ct_vascular_resampling/sampling_pipeline.py \
  tests/test_pose_generation.py \
  tests/test_sampling_pipeline.py \
  tests/test_pipeline.py
git commit -m "feat: partially restore roll and pitch ranges"
```

### Task 2: Update tracked documentation and configuration guidance

**Files:**
- Modify: `README.md:81-90`
- Modify: `configs/case.example.yaml:54-56`
- Modify: `docs/core-design-traceability-20260806.md:14-45`
- Modify: `docs/core-design-change-log-20260807.md:18-49`

- [ ] **Step 1: Update README without rewriting historical evidence**

Use this current contract:

```markdown
- 2026-08-08 用户批准部分恢复旋转范围：滚动为 `-15..+15` 度、俯仰为 `-10..+10` 度，步长均为 5 度；普通区、十二指肠球部和胰腺特殊区偏航仍分别为 `-30..+30`、`-90..+90`、`-120..+30` 度。因此每点分别生成 455、1295、1085 个姿态。该范围是对核心 DOCX 完整滚转/俯仰范围的受控折中，不改写核心文件原文。
```

Retain the 2026-08-07 43,695-pose result only as a labeled historical old-angle acceptance result.

Add a compact English subsection in the same maintained README instead of creating an otherwise absent `README.en.md`:

```markdown
### Controlled rotation range (English)

As approved on 2026-08-08, roll uses `-15..+15 degrees` and pitch uses `-10..+10 degrees`, both at 5-degree steps. The standard, duodenal-bulb, and pancreas-special yaw policies remain unchanged, producing 455, 1295, and 1085 poses per point. This is a controlled partial restoration; it does not rewrite the complete ranges in the core DOCX and does not change sampling points or the 10 mm minimum spacing.
```

- [ ] **Step 2: Document fixed angles in the generic YAML example**

```yaml
square:
  side_length_mm: 100.0
  # 三轴角度是项目级固定算法，不是病例参数：
  # roll -15..+15 / 5 deg；pitch -10..+10 / 5 deg；yaw 按三类区域策略。
```

Run `mamba run -n base python -m pytest tests/test_config.py -q`. Expected: all config tests pass; comments do not add schema.

- [ ] **Step 3: Update traceability and change log**

Change current R08-R12 pose evidence from 333/117/279 to 1295/455/1085. R13 must state that P049-P051 is the original full design while effective values are the user's controlled partial range. Add:

```markdown
| 有效滚转/俯仰范围 | roll `[-15,+15]/5 deg`；pitch `[-10,+10]/5 deg`；偏航不变 | 用户在获知 208,250 姿态和资源估算后批准的受控折中；不是核心 DOCX 完整原范围 |
```

Add change-log rows for `squares.py` angle arrays and dynamic `_candidate_pose_count`. Do not claim server dry-run/full run completion yet.

- [ ] **Step 4: Scan tracked current-state text**

```bash
rg -n "滚动和俯仰均为|-5/0/\+5|117、333、279|117/333/279|\* 9" \
  README.md configs/case.example.yaml \
  docs/core-design-traceability-20260806.md \
  docs/core-design-change-log-20260807.md
```

Expected: no stale current contract. Explicitly dated historical totals may remain.

### Task 3: Update and regenerate the desktop Markdown/HTML guide

**Files:**
- Modify: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.md`
- Regenerate: `/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.html`

- [ ] **Step 1: Verify pre-edit rollback evidence**

```bash
sha256sum \
  '/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.md' \
  '/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档/CT血管重采样项目详细说明.html' \
  '/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档_backup_20260807_000933.tar.gz'
```

Expected: Markdown `d09e4744641fb531089571e8688ec6e68579d3b7533169e942d29e6ea080b97e`, HTML `ec10854f1018ad44fe95f7c9f5aabf3c3c02e815c4ec835bbec0f8f390e75bd8`, archive `354c72a890185afbb2692f75877d1e6f124be1a5e48ca39f9ec176d461b33eb7`. Stop on mismatch.

- [ ] **Step 2: Patch every current angle/count statement in Markdown**

Update revision date to 2026-08-08 and use:

```markdown
滚动 roll：-15°, -10°, -5°, 0°, +5°, +10°, +15°
俯仰 pitch：-10°, -5°, 0°, +5°, +10°
偏航 yaw：普通 -30..+30；十二指肠球部 -90..+90；胰腺特殊 -120..+30，均为 5° 步长
```

Update current formulas:

```text
普通点数 x 455
+ 十二指肠球部点数 x 1295
+ 胰腺特殊区点数 x 1085
```

Update test/checklist references to 455/1295/1085. Keep `roll=-5` where it is a valid single-record example. Label the partial range as user-approved controlled behavior, not rewritten DOCX text.

- [ ] **Step 3: Regenerate HTML from Markdown while preserving the shell**

```bash
DOC_DIR='/mnt/c/Users/zhangyutang/Desktop/CT血管重采样项目说明文档'
BUILD_DIR="$(mktemp -d)"
PANDOC='/tmp/hmm-document-build/pandoc/bin/pandoc'
sed -n '1,/<main>/p' "$DOC_DIR/CT血管重采样项目详细说明.html" | sed '$d' > "$BUILD_DIR/header.html"
sed -n '/<\/main>/,$p' "$DOC_DIR/CT血管重采样项目详细说明.html" | sed '1d' > "$BUILD_DIR/footer.html"
"$PANDOC" "$DOC_DIR/CT血管重采样项目详细说明.md" \
  --from=gfm --to=html5 --wrap=none --output="$BUILD_DIR/body.html"
```

Assemble header, `<main>`, body, `</main>`, and footer into a temporary sibling file, then atomically replace the HTML. Remove the build directory after validation.

- [ ] **Step 4: Validate Markdown/HTML structure and content**

Use a read-only validator requiring these strings in both sources:

```python
required = ("-15..+15", "-10..+10", "455", "1295", "1085", "208,250", "受控")
obsolete = ("滚动/俯仰均为 `-5/0/+5`", "117/333/279")
```

Also assert balanced `main`, `pre`, `code`, `h1`, `h2`, `h3`, plus the Mermaid loader and return-to-top button. Expected: `desktop-angle-docs-ok`.

- [ ] **Step 5: Record post-edit hashes**

Run the Step 1 `sha256sum` command. Add observed Markdown/HTML hashes to traceability; the original archive hash must remain unchanged.

### Task 4: Complete local verification, review, and documentation commit

**Files:**
- Modify with observed evidence: `docs/core-design-traceability-20260806.md`
- Modify with observed evidence: `docs/core-design-change-log-20260807.md`

- [ ] **Step 1: Run the complete local suite**

```bash
mamba run -n base python -m pytest -q
```

Expected: 145 tests pass. Only the existing three SimpleITK/SWIG deprecation warnings are accepted; any failure or new warning category stops deployment.

- [ ] **Step 2: Run syntax and diff checks**

```bash
mamba run -n base python -m compileall -q src tests
git diff --check
git status --short --branch
```

Expected: compile/diff checks are silent. Status contains intended tracked docs plus pre-existing `.superpowers/`; never stage `.superpowers/`.

- [ ] **Step 3: Verify the exact angle contract directly**

```bash
mamba run -n base python - <<'PY'
from ct_vascular_resampling.squares import PITCH_ANGLES_DEGREES, ROLL_ANGLES_DEGREES, YAW_ANGLES_DEGREES

assert ROLL_ANGLES_DEGREES == (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)
assert PITCH_ANGLES_DEGREES == (-10.0, -5.0, 0.0, 5.0, 10.0)
assert {name: len(values) for name, values in YAW_ANGLES_DEGREES.items()} == {
    "standard": 13,
    "duodenum_bulb": 37,
    "pancreas_special": 31,
}
print("angle-contract-ok")
PY
```

Expected: `angle-contract-ok`.

- [ ] **Step 4: Perform code review**

Invoke `requesting-code-review` and inspect the diff from commit `5426e5f` through the worktree. Resolve every correctness, resource, test, and documentation finding.

- [ ] **Step 5: Commit tracked documentation only**

```bash
git add README.md configs/case.example.yaml \
  docs/core-design-traceability-20260806.md \
  docs/core-design-change-log-20260807.md
git commit -m "docs: record partial rotation range"
```

Desktop files remain outside Git; their SHA-256 values are the audit link.

- [ ] **Step 6: Verify committed HEAD**

```bash
mamba run -n base python -m pytest -q
git diff --check
git status --short --branch
git log -4 --oneline --decorate
```

Expected: 145 tests pass; only `.superpowers/` is unrelated/untracked.

### Task 5: Push, back up, deploy, and execute server dry-run

**Files:**
- Push: `feature/core-design-alignment-20260806`
- Create locally: `/tmp/ct_vascular_resampling_feature_${short_commit}.bundle`
- Create remotely: `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/backups/project_backup_${timestamp}.tar.gz`
- Create remotely: `project/configs/case_2_partial_rotation_20260808.yaml`
- Create remotely: `run/partial_rotation_dry_run_20260808.log`

- [ ] **Step 1: Push only the feature branch and verify GitHub**

```bash
git push origin feature/core-design-alignment-20260806
local_head="$(git rev-parse HEAD)"
remote_head="$(git ls-remote origin refs/heads/feature/core-design-alignment-20260806 | awk '{print $1}')"
test "$local_head" = "$remote_head"
```

Expected: hashes match. Do not merge or push `main`.

- [ ] **Step 2: Build and verify the transfer bundle**

```bash
short_commit="$(git rev-parse --short=12 HEAD)"
bundle="/tmp/ct_vascular_resampling_feature_${short_commit}.bundle"
git bundle create "$bundle" feature/core-design-alignment-20260806
git bundle verify "$bundle"
sha256sum "$bundle"
```

Expected: the bundle advertises the feature branch at GitHub's full commit.

- [ ] **Step 3: Reconfirm server isolation boundary**

Connect only with `ssh -p 23078 root@connect.westc.seetacloud.com`, then:

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
pwd
git rev-parse HEAD
git status --short --branch
sha256sum configs/case_2_core_design_20260806.yaml
```

Expected: exact allowed path, HEAD `b4148817263271c12f53b91a9a2bcf32d3b1eb97`, config hash `66e7c811cdb82d0823edfe3c27f3bd7d2143e97f8fe7f206b2de82de1ede2ae6`. Stop on unexpected tracked changes/path/HEAD/hash; preserve known untracked case data, configs, and registration files.

- [ ] **Step 4: Back up current project and run state before update**

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731
timestamp="$(date +%Y%m%d_%H%M%S)"
backup="backups/project_backup_${timestamp}.tar.gz"
tar -czf "$backup" project run
gzip -t "$backup"
tar -tzf "$backup" | awk -F/ '$1 != "project" && $1 != "run" {bad=1} END {exit bad}'
sha256sum "$backup"
```

Expected: valid gzip, members only under `project/` or `run/`, recorded SHA-256.

- [ ] **Step 5: Upload and fast-forward from the bundle**

Use `scp -P 23078 "$bundle" root@connect.westc.seetacloud.com:/root/autodl-tmp/ct_vascular_resampling_case2_20260731/run/$(basename "$bundle")` to place the bundle in the allowed `run/` directory. Verify remote/local bundle hashes, then interpolate the exact verified basename from the controlling shell:

```bash
bundle_name="$(basename "$bundle")"
ssh -p 23078 root@connect.westc.seetacloud.com \
  "cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project && \
   git fetch ../run/$bundle_name refs/heads/feature/core-design-alignment-20260806:refs/remotes/bundle/feature/core-design-alignment-20260806 && \
   git merge --ff-only refs/remotes/bundle/feature/core-design-alignment-20260806 && \
   git rev-parse HEAD && git status --short --branch"
```

Expected: server/local/GitHub HEAD match; tracked files clean. Never clean untracked data/config/registration paths.

- [ ] **Step 6: Run server tests**

```bash
mamba run -n ct-vessel-resampling-totalseg-gpu python -m pytest -q
```

Expected: 145 tests pass. Do not use the broken direct `pytest` launcher.

- [ ] **Step 7: Create an isolated runtime config**

Copy the audited config to `configs/case_2_partial_rotation_20260808.yaml` and change only:

```yaml
output_root: "/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_partial_rotation_20260808"
```

Use structured YAML comparison to prove `output_root` is the only difference. Confirm the output root and `case_2/` child do not exist. Preserve `workers: 4`, GPU backend/device/batch 8, E1/E2 hints, caps, and 10 mm spacing.

- [ ] **Step 8: Run and monitor dry-run**

```bash
mamba run -n ct-vessel-resampling-totalseg-gpu python main.py \
  --case-config configs/case_2_partial_rotation_20260808.yaml \
  --dry-run --verbose 2>&1 | tee ../run/partial_rotation_dry_run_20260808.log
```

Monitor RSS, cgroup memory/peak/events, GPU memory, and disk from a second connection at no more than 60-second intervals. Required values:

```text
stomach 118; liver 158; pancreas 40; duodenum 53 (17+36); esophagus 31
total points 400
yaw policies standard 364; duodenum_bulb 17; pancreas_special 19
total squares 208250
every reported minimum spacing >= 10.0 mm
output directory remains absent
OOM/oom_kill increments 0
```

Stop on any mismatch or unexpected output write.

- [ ] **Step 9: Apply the disk safety gate**

Historical average: `13,750,045,990 / 167,724 = 81,980.193591853284 bytes/record`.

```bash
estimated_bytes=17072375316
required_bytes=26709178264
free_bytes="$(df -PB1 /root/autodl-tmp/ct_vascular_resampling_case2_20260731 | awk 'NR==2 {print $4}')"
test "$free_bytes" -ge "$required_bytes"
```

Expected: free space at least 26,709,178,264 bytes (24.875 GiB), including 25% margin plus 5 GiB. Failure stops production; do not delete unrelated data or reduce points/angles.

### Task 6: Run full GPU workflow and audit outputs

**Files:**
- Create remotely: `run/partial_rotation_full_run_20260808.log`
- Create remotely: `run/partial_rotation_full_run_20260808.exit`
- Create remotely: `run/partial_rotation_resources_20260808.csv`
- Create remotely: `output_partial_rotation_20260808/case_2/`

- [ ] **Step 1: Recheck production preconditions**

Require matching GitHub/server HEAD, absent output path, dry-run 208,250, unchanged OOM counters, available GPU 0, and `free_bytes >= 26709178264`. Record `nvidia-smi`, `memory.events`, disk, config hash, Git commit, and timestamps in `run/`.

- [ ] **Step 2: Launch a disconnect-tolerant run**

```bash
nohup bash -lc '
  set -o pipefail
  cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
  mamba run -n ct-vessel-resampling-totalseg-gpu python main.py \
    --case-config configs/case_2_partial_rotation_20260808.yaml \
    --verbose 2>&1 | tee ../run/partial_rotation_full_run_20260808.log
  printf "%s\n" "${PIPESTATUS[0]}" > ../run/partial_rotation_full_run_20260808.exit
' > ../run/partial_rotation_full_run_20260808.launcher.log 2>&1 &
printf '%s\n' "$!" > ../run/partial_rotation_full_run_20260808.pid
```

Expected: one writer process and no competing process for the output root.

- [ ] **Step 3: Monitor resources and progress**

At intervals no longer than 60 seconds append:

```text
timestamp,pid_rss_bytes,cgroup_current_bytes,cgroup_peak_bytes,oom,oom_kill,gpu_used_mib,gpu_total_mib,disk_free_bytes,manifest_lines
```

Inspect logs for traceback, CUDA OOM, allocation/I/O/worker failure. If free disk approaches 5 GiB, OOM increments, or exit is nonzero, preserve state and diagnose; do not delete/relaunch blindly.

- [ ] **Step 4: Verify termination**

```bash
test "$(cat ../run/partial_rotation_full_run_20260808.exit)" = "0"
```

Confirm no writer remains and the log has a normal completion summary.

- [ ] **Step 5: Audit structured invariants**

Require:

```text
run_metadata.total_squares == 208250
run_metadata.completed_pose_count == 208250
roll == [-15,-10,-5,0,5,10,15]
pitch == [-10,-5,0,5,10]
all yaw arrays unchanged
sum(status_counts.values()) == 208250
manifest line count == 208250
sample_id values unique and each in one status only
library_summary indexed count == gallery line count
coordinate_system == RAS; minimum_point_spacing_mm == 10.0
core_design_sha256 == 4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b
build_git_commit == deployed feature commit
```

- [ ] **Step 6: Audit files, sizes, and key hashes**

Require five sampling-point PLYs, square/rectangle PLYs, four status locations, manifest, metadata, summary, and logs. Hash runtime config, dry/full logs, run metadata, library summary, and JSONL manifests. Record output bytes/final free space. Validate PNG counts against owning JSONL rather than hashing millions of PNGs individually.

- [ ] **Step 7: Produce final compliance and rollback handoff**

Report local/GitHub/server commit; server backup path/size/hash; bundle/config hashes; dry-run counts and resource peak; full exit/elapsed/resource/disk data; four status counts; manifest invariants; desktop hashes; and rollback commands constrained to the allowed root.

Do not describe technical completion as medical validation; separate clinical acceptance remains required.
