# Sampling Points And Zero Plane Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从病例 2 正式输出中只读提取全部 400 个采样点及对应零度基准面，并在 Windows 桌面生成可交互 HTML、四视图 PNG、PLY、CSV、JSON、器官网格和中文说明。

**Architecture:** 在 `zero_plane_visualization.py` 中实现纯数据解析、几何验证和本地导出；薄 CLI 负责读取已经通过 SSH 流式筛选到本地的零度 JSONL、采样点 PLY 与器官网格。远程提取由一次只读 SSH 命令完成，凭证不进入仓库或交付物。渲染使用 Plotly 生成离线 HTML、Matplotlib 生成固定视角 PNG，PLY/CSV/JSON 使用原子写入。

**Tech Stack:** Python 3.12、NumPy、Matplotlib、Plotly、Trimesh、Pillow、pytest、mamba。

---

## 文件结构

- Create: `src/ct_vascular_resampling/zero_plane_visualization.py`：记录模型、零度筛选、几何验证、PLY/CSV/JSON、Plotly 和 Matplotlib 渲染。
- Create: `scripts/export_zero_plane_visualization.py`：本地输入目录到最终交付目录的命令行入口。
- Create: `tests/test_zero_plane_visualization.py`：零度筛选、几何拒绝、导出格式和最小渲染测试。
- Modify: `README.md`：增加可视化导出命令和输入文件说明。

### Task 1: 零度记录解析与几何验证

**Files:**
- Create: `tests/test_zero_plane_visualization.py`
- Create: `src/ct_vascular_resampling/zero_plane_visualization.py`

- [ ] **Step 1: 写零度筛选和几何验证失败测试**

测试构造一个满足以下几何的真实语义记录：探头在底边 `v0-v1` 中点，局部 `x` 指向方形深度，`y` 沿底边，`z=x×y`，边长 100 mm。断言非零角记录被忽略，缺失、重复、非正交轴和探头不在底边中点均抛出 `ValueError`。

```python
def _record(index: int = 0) -> dict:
    return {
        "slice_id": f"stomach-{index:06d}-rp000-pp000-yp000",
        "organ": "stomach",
        "probe_point_world": [0.0, 0.0, 0.0],
        "angles_degrees": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "square_vertices_world": [[0.0, -50.0, 0.0], [0.0, 50.0, 0.0], [100.0, 50.0, 0.0], [100.0, -50.0, 0.0]],
        "local_axes_world": {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]},
        "input_normal_world": [1.0, 0.0, 0.0],
        "coordinate_system": "RAS",
    }

def test_select_zero_planes_rejects_duplicate_sample():
    with pytest.raises(ValueError, match="重复"):
        select_zero_planes([_record(), _record()], {"stomach": 1})
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py -q`

Expected: FAIL，原因是模块或 `select_zero_planes` 尚不存在。

- [ ] **Step 3: 实现最小数据模型和验证**

```python
@dataclass(frozen=True)
class ZeroPlaneRecord:
    slice_id: str
    organ: str
    point_index: int
    probe: np.ndarray
    vertices: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    input_normal: np.ndarray

def select_zero_planes(records: Iterable[dict], expected_counts: Mapping[str, int]) -> list[ZeroPlaneRecord]:
    # 只接受三角均为 0、RAS、有限 3D 数值的记录；验证唯一键、计数、100 mm 正方形、
    # probe==(v0+v1)/2、局部轴单位正交及 cross(x,y)==z，最终按器官和 point_index 排序。
```

容差固定为：点和边长 `atol=1e-5 mm`，轴与正交关系 `atol=1e-8`，不使用相对容差。

- [ ] **Step 4: 运行聚焦测试并确认 GREEN**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py -q`

Expected: PASS。

- [ ] **Step 5: 提交解析与验证**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: validate zero-plane visualization records"
```

### Task 2: PLY、CSV 与 JSON 结构化导出

**Files:**
- Modify: `tests/test_zero_plane_visualization.py`
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py`

- [ ] **Step 1: 写结构化导出失败测试**

测试调用 `write_structured_exports(records, output, provenance)`，断言：

```python
assert "element vertex 1" in (output / "sampling_points.ply").read_text()
assert "element edge 4" in (output / "zero_planes_edges.ply").read_text()
assert "element face 1" in (output / "zero_planes_faces.ply").read_text()
assert json.loads((output / "sampling_points_zero_planes.json").read_text())["record_count"] == 1
assert len(list(csv.DictReader((output / "sampling_points_zero_planes.csv").open()))) == 1
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py::test_write_structured_exports -q`

Expected: FAIL，原因是 `write_structured_exports` 尚不存在。

- [ ] **Step 3: 实现原子导出**

实现 `_atomic_text`、`write_points_ply`、`write_edges_ply`、`write_faces_ply`、`write_csv` 和 `write_json`。点 PLY 写 `x/y/z/nx/ny/nz/red/green/blue/organ_id`；边和面 PLY 写逐器官 RGB。JSON 顶层固定包含：

```python
{
    "schema_version": "zero-plane-visualization/v1",
    "coordinate_system": "RAS",
    "unit": "mm",
    "record_count": len(records),
    "organ_counts": dict(Counter(record.organ for record in records)),
    "provenance": provenance,
    "records": [...],
}
```

- [ ] **Step 4: 运行测试并确认 GREEN**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py -q`

Expected: PASS。

- [ ] **Step 5: 提交结构化导出**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: export zero-plane geometry artifacts"
```

### Task 3: 离线 HTML 和四视图 PNG

**Files:**
- Modify: `tests/test_zero_plane_visualization.py`
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py`

- [ ] **Step 1: 写渲染失败测试**

使用一个零度面和一个最小三角网格，调用 `render_interactive_html` 与 `render_static_views`，断言 HTML 内嵌 Plotly、不引用 CDN，四张 PNG 为非空 RGB/RGBA 图片且像素标准差大于 0。

```python
html = (output / "sampling_points_zero_planes_interactive.html").read_text()
assert "cdn.plot.ly" not in html
assert "Plotly.newPlot" in html
for name in ("isometric", "axial", "coronal", "sagittal"):
    pixels = np.asarray(Image.open(output / f"sampling_points_zero_planes_{name}.png"))
    assert pixels.std() > 0.0
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank -q`

Expected: FAIL，原因是渲染函数尚不存在。

- [ ] **Step 3: 实现 Plotly 交互图**

每个器官建立网格、点、面边框和局部轴 trace，并使用 `legendgroup` 绑定器官显隐；添加“全部、仅点、点和零度面、隐藏器官”按钮。使用 `plotly.offline.plot(..., include_plotlyjs=True, auto_open=False)` 输出自包含 HTML，场景使用 `aspectmode="data"` 和明确 R/A/S 轴标题。

- [ ] **Step 4: 实现 Matplotlib 四视图**

器官网格按固定随机种子最多抽取 5,000 个面；零度面填充透明度不高于 0.035，边框不高于 0.20，点始终位于上层。四个相机为：等距 `(elev=25, azim=-55)`、轴位沿 `+S`、冠状位沿 `-A`、矢状位沿 `+R`。所有图共享世界坐标范围和等比例盒体。

- [ ] **Step 5: 运行测试并确认 GREEN**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py -q`

Expected: PASS。

- [ ] **Step 6: 提交渲染实现**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: render sampling points and zero planes"
```

### Task 4: CLI、真实数据提取与桌面交付

**Files:**
- Create: `scripts/export_zero_plane_visualization.py`
- Modify: `tests/test_zero_plane_visualization.py`
- Modify: `README.md`

- [ ] **Step 1: 写 CLI 失败测试**

测试最小本地输入包执行后退出 0，缺少器官网格或零度记录时退出非零，且最终目录不存在半成品 `.tmp`。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `mamba run -n base python -m pytest tests/test_zero_plane_visualization.py::test_cli_exports_complete_bundle -q`

Expected: FAIL，原因是 CLI 尚不存在。

- [ ] **Step 3: 实现 CLI 和中文说明**

CLI 参数固定为：

```text
--zero-records-jsonl PATH
--sample-ply-dir PATH
--organ-mesh-dir PATH
--run-metadata PATH
--source-manifest-sha256 HEX
--output-dir PATH
```

CLI 先写同级临时目录，验证全部预期文件后以 `os.replace` 发布。`README_中文.txt` 解释五种颜色、RAS、毫米单位、探头位于底边中点、零度筛选和 3D Slicer/MeshLab 打开方法；最后写 `SHA256SUMS.txt`。

- [ ] **Step 4: 运行聚焦和完整测试**

Run:

```bash
mamba run -n base python -m pytest tests/test_zero_plane_visualization.py -q
mamba run -n base python -m pytest -q
mamba run -n base python -m compileall -q src scripts tests
git diff --check
```

Expected: 新测试全部通过；完整测试 0 failed；compileall 和 diff check 退出 0。

- [ ] **Step 5: 提交 CLI 与文档**

```bash
git add scripts/export_zero_plane_visualization.py tests/test_zero_plane_visualization.py README.md
git commit -m "feat: add zero-plane visualization export command"
```

- [ ] **Step 6: 从服务器只读提取真实输入**

使用新端口 `45590`，在服务器端以 Python 流式扫描根 `manifest.jsonl`，只输出三角均为 0 的 400 条 JSONL。下载五个 `ResampledpointPLY/FPS-*.ply`、`run_metadata.json` 和手工预处理的五个目标器官网格；计算并记录根 manifest SHA-256。密码只通过 `sshpass -e` 的当前进程环境传递，不写文件。

- [ ] **Step 7: 生成桌面交付并验收**

Run:

```bash
mamba run -n base python scripts/export_zero_plane_visualization.py \
  --zero-records-jsonl /home/zyt/ct_vascular_resampling/.work/zero_plane_visualization_20260815/zero_records.jsonl \
  --sample-ply-dir /home/zyt/ct_vascular_resampling/.work/zero_plane_visualization_20260815/ResampledpointPLY \
  --organ-mesh-dir /home/zyt/ct_vascular_resampling/.work/zero_plane_visualization_20260815/target_organ_meshes \
  --run-metadata /home/zyt/ct_vascular_resampling/.work/zero_plane_visualization_20260815/run_metadata.json \
  --source-manifest-sha256 "$MANIFEST_SHA256" \
  --output-dir /mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815
```

Expected: 400 条、逐器官计数精确、全部交付文件存在。随后用 Pillow 检查四张 PNG 非空，用浏览器截图检查 HTML 首屏非空、器官图例与控件不遮挡，并用 `trimesh.load` 验证三个 PLY 可解析。

- [ ] **Step 8: 记录最终状态**

确认远程 Gallery 的修改时间和哈希未因只读提取改变；记录本地交付目录、总大小、SHA-256 清单、源 manifest 哈希和生成提交。
