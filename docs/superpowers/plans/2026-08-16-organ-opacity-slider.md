# 器官网格显示控制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为离线交互网页增加默认 `0.70` 的器官网格不透明度滑块、可切换的 `1.00` 不透明连续表面，并放大默认三维视角，同时保证现有可见性预设和所有重采样数据不变。

**Architecture:** 沿用现有 35 条 Plotly trace，不增加第二套器官网格。生成 HTML 时内嵌五类原始网格的完整 `i/j/k` 面索引；顶部工具栏用滑块控制性能表面 opacity，用复选框在保存的性能面索引与完整面索引之间切换。所有操作只 restyle 五条 `organ_mesh`，不改变 `visible`；默认相机 eye 调整为 `(1.05, -1.12, 0.76)`。

**Tech Stack:** Python 3.12、NumPy、Trimesh、Plotly、pytest、离线 HTML/JavaScript、Microsoft Edge Chrome DevTools Protocol。

---

## 文件结构

- 修改 `src/ct_vascular_resampling/zero_plane_visualization.py`：定义显示控件常量、生成工具栏、内嵌完整面索引、绑定浏览器事件、调整默认相机并更新交付说明。
- 修改 `tests/test_zero_plane_visualization.py`：固定默认值、滑块与连续表面 HTML 契约、完整面载荷、默认相机和器官网格 trace 索引。
- 生成新的桌面候选目录：不纳入 Git，只用于真实 400 点网页和输出哈希验证。
- 不修改重采样管线、采样点算法、Gallery、服务器结果和静态 PNG alpha。

### Task 1: 固定滑块契约并验证失败

**Files:**
- Modify: `tests/test_zero_plane_visualization.py:175-225`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: 修改现有离线渲染测试，写入滑块契约**

在 `test_render_outputs_are_offline_and_nonblank` 中把现有交互不透明度断言改为 `0.70`，保留静态 alpha `0.14`，并增加以下断言：

```python
assert visualization.INTERACTIVE_ORGAN_MESH_OPACITY == pytest.approx(0.70)
assert visualization.STATIC_ORGAN_MESH_ALPHA == pytest.approx(0.14)
assert visualization.ORGAN_OPACITY_MIN == pytest.approx(0.10)
assert visualization.ORGAN_OPACITY_MAX == pytest.approx(1.00)
assert visualization.ORGAN_OPACITY_STEP == pytest.approx(0.05)
assert visualization.DEFAULT_CAMERA_EYE == {"x": 1.05, "y": -1.12, "z": 0.76}
assert '"opacity":0.7' in html
assert 'id="organ-mesh-opacity-slider"' in html
assert 'id="organ-mesh-opacity-value"' in html
assert 'id="continuous-organ-surface-toggle"' in html
assert 'min="0.1" max="1.0" step="0.05" value="0.7"' in html
assert "器官网格不透明度" in html
assert "使用不透明连续表面" in html
assert "70%" in html
assert "const organMeshTraceIndices = [0];" in html
assert "const continuousSurfaceFaces" in html
assert "Plotly.restyle" in html
assert '"camera":{"eye":{"x":1.05,"y":-1.12,"z":0.76}}' in html
```

增加一个独立测试，以 12,001 个三角面构造网格并调用 `_mesh_arrays`，断言性能面为 12,000、完整面为 12,001，且完整面数组逐项保持原始顺序。该测试避免只用四面体时无法证明完整面载荷与性能面确实不同。

- [ ] **Step 2: 运行聚焦测试并确认因功能尚未实现而失败**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: FAIL，报告连续表面控件、完整面载荷、默认相机或 `_mesh_arrays` 三返回值尚不存在；不得是测试语法错误。

- [ ] **Step 3: 检查失败输出和当前差异**

Run:

```bash
git diff --check
git diff -- tests/test_zero_plane_visualization.py
```

Expected: `git diff --check` 退出码为 `0`；差异只包含新的预期契约。

### Task 2: 实现器官网格不透明度控制器

**Files:**
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py:24-32`
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py:484-607`
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py:616-780`
- Modify: `src/ct_vascular_resampling/zero_plane_visualization.py:1150-1185`
- Test: `tests/test_zero_plane_visualization.py`

- [ ] **Step 1: 定义默认值、范围和稳定 DOM ID**

在模块常量区定义：

```python
INTERACTIVE_ORGAN_MESH_OPACITY = 0.70
STATIC_ORGAN_MESH_ALPHA = 0.14
ORGAN_OPACITY_MIN = 0.10
ORGAN_OPACITY_MAX = 1.00
ORGAN_OPACITY_STEP = 0.05
ORGAN_OPACITY_SLIDER_ID = "organ-mesh-opacity-slider"
ORGAN_OPACITY_VALUE_ID = "organ-mesh-opacity-value"
CONTINUOUS_SURFACE_TOGGLE_ID = "continuous-organ-surface-toggle"
DEFAULT_CAMERA_EYE = {"x": 1.05, "y": -1.12, "z": 0.76}
```

不得改变 `STATIC_ORGAN_MESH_ALPHA`。

- [ ] **Step 2: 同时保留性能面和完整面**

把 `_mesh_arrays` 返回值扩展为 `(vertices, performance_faces, continuous_faces)`。`continuous_faces` 必须是输入网格全部三角面的副本；`performance_faces` 在面数超过上限时继续使用现有固定 seed 随机抽样，否则与完整面相同。不得简化、重排或删除完整面中的连通分量。

在 `render_interactive_html` 为每个 `organ_mesh` trace 记录其完整面索引，键使用该 trace 的实际整数索引。Plotly trace 仍只创建一条，并以性能面作为初始 `i/j/k`。

- [ ] **Step 3: 扩展现有浏览器控制脚本**

使用 `_interactive_controls_script(roles, continuous_faces_by_trace)`。除保留现有零度面与滑块逻辑外，从同一个 `roles` 列表生成器官网格索引，并把完整面转换成紧凑 JSON：

```python
organ_mesh_indices = [index for index, role in enumerate(roles) if role == "organ_mesh"]
if not organ_mesh_indices:
    raise ValueError("交互可视化缺少器官网格 trace")
organ_indices_json = json.dumps(organ_mesh_indices, separators=(",", ":"))
continuous_faces_json = json.dumps(
    {
        str(index): {
            "i": faces[:, 0].tolist(),
            "j": faces[:, 1].tolist(),
            "k": faces[:, 2].tolist(),
        }
        for index, faces in continuous_faces_by_trace.items()
    },
    separators=(",", ":"),
)
```

嵌入脚本使用稳定 DOM ID，并只更新 opacity：

```javascript
const organMeshTraceIndices = ORGAN_INDICES_JSON;
const opacitySlider = document.getElementById(OPACITY_SLIDER_ID_JSON);
const opacityValue = document.getElementById(OPACITY_VALUE_ID_JSON);

const applyOrganOpacity = () => {
  const opacity = Number(opacitySlider.value);
  opacityValue.value = `${Math.round(opacity * 100)}%`;
  opacityValue.textContent = opacityValue.value;
  return Plotly.restyle(graph, {opacity}, organMeshTraceIndices);
};

opacitySlider.addEventListener("input", applyOrganOpacity);
```

初次加载保存每条器官网格的性能面数组。连续表面复选框变化时逐条更新同一 trace；数组属性必须按 Plotly 单 trace restyle 语义使用外层数组包装：

```javascript
const performanceSurfaceFaces = Object.fromEntries(
  organMeshTraceIndices.map(index => [index, {
    i: Array.from(graph.data[index].i),
    j: Array.from(graph.data[index].j),
    k: Array.from(graph.data[index].k)
  }])
);

const applyContinuousSurfaceState = async () => {
  const continuous = continuousSurfaceToggle.checked;
  opacitySlider.disabled = continuous;
  opacityValue.value = continuous ? "100%" : `${Math.round(Number(opacitySlider.value) * 100)}%`;
  opacityValue.textContent = opacityValue.value;
  for (const index of organMeshTraceIndices) {
    const faces = continuous ? continuousSurfaceFaces[index] : performanceSurfaceFaces[index];
    await Plotly.restyle(graph, {
      i: [faces.i], j: [faces.j], k: [faces.k],
      opacity: continuous ? 1.0 : Number(opacitySlider.value)
    }, [index]);
  }
};
```

脚本不得修改 `visible`，不得在 `plotly_buttonclicked` 中重置滑块或连续表面状态。现有零度面复选框逻辑保持原行为。

- [ ] **Step 4: 在顶部工具栏增加连续表面控件并支持窄屏换行**

在 `_interactive_html_document` 的 `.zero-plane-toolbar` 中保留复选框，并加入：

```html
<div class="organ-opacity-control">
  <label for="organ-mesh-opacity-slider">
    器官网格不透明度
    <output id="organ-mesh-opacity-value" for="organ-mesh-opacity-slider">70%</output>
  </label>
  <input id="organ-mesh-opacity-slider"
         type="range"
         min="0.1" max="1.0" step="0.05" value="0.7"
         aria-label="器官网格不透明度">
</div>
<label class="continuous-surface-control" for="continuous-organ-surface-toggle">
  <input id="continuous-organ-surface-toggle" type="checkbox" aria-label="使用不透明连续表面">
  <span>使用不透明连续表面</span>
</label>
```

工具栏样式改为自动高度并允许换行：

```css
.zero-plane-toolbar {
  flex: 0 0 auto;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  min-height: 38px;
}
.organ-opacity-control {
  display: flex;
  align-items: center;
  gap: 8px;
}
.organ-opacity-control input[type="range"] { width: 160px; }
.organ-opacity-control output { min-width: 3ch; }
.continuous-surface-control {
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}
```

不得使用固定定位，不得把控件放入 Plotly 图表内部。

- [ ] **Step 5: 应用放大的默认相机并更新交付说明**

`figure.update_layout` 保持 `aspectmode="data"`，仅将 camera 改为：

```python
"camera": {"eye": dict(DEFAULT_CAMERA_EYE)},
```

在 `pio.to_html(..., post_script=...)` 中传入 `roles` 和完整面映射。在 `_readme_text` 的打开方式中明确：

```text
- HTML 可直接用浏览器离线打开；顶部工具栏可控制零度基准面、器官网格不透明度和不透明连续表面，右侧图例可按器官显隐。
```

- [ ] **Step 6: 运行聚焦测试和可视化模块测试**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
pytest -q tests/test_zero_plane_visualization.py
git diff --check
```

Expected: 聚焦测试 `1 passed`，模块测试 `16 passed`，`git diff --check` 退出码为 `0`。

- [ ] **Step 7: 提交实现与测试**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: toggle continuous organ surfaces"
```

Expected: 提交只包含上述两个文件。

### Task 3: 浏览器验证滑块和预设联动

**Files:**
- Read: `src/ct_vascular_resampling/zero_plane_visualization.py`
- Generate only: `C:\Users\zhangyutang\AppData\Local\Temp\ct-organ-opacity-slider-browser-test\toggle_test.html`

- [ ] **Step 1: 生成一器官最小浏览器测试页**

用 `select_zero_planes` 和 `render_interactive_html` 生成含一条器官网格、一组采样点、一组零度面和三条局部轴的临时 HTML。输入记录使用测试中的 100 mm 方形和原点探头，器官网格使用四面体；不得写入仓库受跟踪文件。

- [ ] **Step 2: 使用 Edge CDP 验证初始状态**

以唯一 `--user-data-dir` 和调试端口启动 Edge，读取页面状态并断言：

```javascript
slider.value === "0.7"
output.textContent === "70%"
graph.data[0].opacity === 0.7
continuousSurfaceToggle.checked === false
continuousSurfaceToggle.disabled === false
graph.layout.scene.camera.eye.x === 1.05
graph.layout.scene.camera.eye.y === -1.12
graph.layout.scene.camera.eye.z === 0.76
```

Expected: 各项均为真，零度面复选框、连续表面复选框和四个预设均存在。

- [ ] **Step 3: 验证滑块只改变器官网格 opacity**

保存全部 trace 的 `visible` 和非器官网格 trace 的 `opacity`，执行：

```javascript
slider.value = "0.35";
slider.dispatchEvent(new Event("input", {bubbles: true}));
```

等待 Plotly 重绘后断言：

```javascript
graph.data[0].opacity === 0.35
output.textContent === "35%"
visibleAfter === visibleBefore
nonOrganOpacityAfter === nonOrganOpacityBefore
```

- [ ] **Step 4: 验证连续表面切换不改变可见性**

记录所有 trace 的 visible 和器官网格初始面数，勾选 `使用不透明连续表面` 后断言：滑块 disabled，百分比为 `100%`，器官网格 opacity 为 `1.0`，完整面数与输入四面体一致，全部 visible 保持不变。取消勾选后断言滑块重新启用，百分比与 opacity 恢复 `35%` 和 `0.35`，性能面数组逐项恢复。

- [ ] **Step 5: 验证四个预设不重置两个器官控制**

先勾选连续表面，再依次实际点击 `Points + zero planes`、`Points only`、`Hide organ meshes`、`All`。每次均断言连续表面仍勾选、滑块值仍为 `0.35` 且保持 disabled；显示器官的 `Points only` 和 `All` 中 opacity 为 `1.0`。取消连续表面后，显示器官的预设必须恢复 opacity `0.35`。同时复核 `Points only` 继续显示器官网格和采样点。

- [ ] **Step 6: 验证桌面和窄屏布局**

在 `1280×800` 和 `390×844` 两个视口读取工具栏、标题、预设按钮的 `getBoundingClientRect()`，断言任意两个不发生矩形相交；截图确认两个复选框、百分比和滑块均未溢出容器，放大后的模型未被场景边界裁切。

### Task 4: 生成并审计真实 400 点候选包

**Files:**
- Read: `.work/zero_plane_visualization_20260815/zero_records.jsonl`
- Read: `.work/zero_plane_visualization_20260815/ResampledpointPLY/`
- Read: `.work/zero_plane_visualization_20260815/target_organ_meshes/`
- Read: `.work/zero_plane_visualization_20260815/run_metadata.json`
- Generate only: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260816_器官显示控制临时\`

- [ ] **Step 1: 用正式输入生成全新候选目录**

Run:

```bash
PYTHONPATH=src python scripts/export_zero_plane_visualization.py \
  --zero-records-jsonl .work/zero_plane_visualization_20260815/zero_records.jsonl \
  --sample-ply-dir .work/zero_plane_visualization_20260815/ResampledpointPLY \
  --organ-mesh-dir .work/zero_plane_visualization_20260815/target_organ_meshes \
  --run-metadata .work/zero_plane_visualization_20260815/run_metadata.json \
  --source-manifest-sha256 d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce \
  --output-dir '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_器官显示控制临时'
```

Expected: `record_count` 为 `400`；器官计数为胃 `118`、肝 `162`、胰腺 `37`、十二指肠 `53`、食管 `30`。

- [ ] **Step 2: 校验候选目录完整性和数据不变性**

Run:

```bash
cd '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_器官显示控制临时'
sha256sum -c SHA256SUMS.txt
```

解析 `sampling_points_zero_planes.json` 并断言：坐标系为 `RAS`、记录数为 `400`、器官计数符合上一步、源 manifest SHA-256 为 `d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce`。

将候选目录与当前正式目录逐文件比较。除 `sampling_points_zero_planes_interactive.html`、`README_中文.txt` 和 `SHA256SUMS.txt` 外，其余文件 SHA-256 必须完全一致。

- [ ] **Step 3: 在真实页面验证五条器官网格轨迹**

通过 Edge CDP 加载候选 HTML，断言器官网格索引为 `[0, 7, 14, 21, 28]`，默认相机 eye 为 `(1.05, -1.12, 0.76)`，五条轨迹初始 opacity 均为 `0.70`。性能模式面数必须为 `[12000, 12000, 12000, 12000, 11522]`。最后一项是食管有效段复制并沿 S/I 轴平移 42 mm 后形成的实际可视化模型面数，不是原始输入 PLY 的 `9834` 面。把滑块改为 `0.40` 后，五条轨迹均变为 `0.40`，其余 30 条轨迹的 opacity 和全部 35 条轨迹的 visible 值保持不变。

勾选连续表面后，五条器官网格面数必须变为 `[61644, 370432, 37132, 30120, 11522]`，opacity 均为 `1.00`，滑块 disabled、百分比为 `100%`，全部 visible 保持不变。取消后必须逐项恢复性能面数和 `0.40`。

- [ ] **Step 4: 在真实页面验证预设和布局**

在连续表面开启时点击 `Points only`，断言五条器官网格和五组采样点可见，零度面与局部轴隐藏，完整面数和 `1.00` opacity 不变。随后关闭连续表面、恢复滑块 `0.70` 和默认 `All` 状态，截图检查工具栏、标题、预设按钮和图例无重叠，模型无裁切。

### Task 5: 可回退发布与最终验证

**Files:**
- Preserve: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815\`
- Backup to: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815_器官显示控制加入前备份_勿用\`
- Promote: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260816_器官显示控制临时\`

- [ ] **Step 1: 预检三个目录，禁止覆盖既有备份**

确认正式目录和候选目录均存在，且目标备份目录不存在。任一条件不满足立即停止，不执行移动。

- [ ] **Step 2: 先备份正式目录，再提升候选目录**

依次执行两个独立的 `mv`：

```bash
mv '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815' \
   '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815_器官显示控制加入前备份_勿用'
```

```bash
mv '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_器官显示控制临时' \
   '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815'
```

不得删除任何此前备份或临时候选目录。

- [ ] **Step 3: 运行全量验证**

Run:

```bash
pytest -q
python -m compileall -q src scripts tests
git diff --check
git status --short --branch
```

Expected: 加入完整面保留测试后共 `257` 项测试全部通过；`compileall` 和 `git diff --check` 退出码为 `0`；Git 只保留既有未跟踪 `.superpowers/` 和 `.work/`。

- [ ] **Step 4: 重新校验正式目录**

Run:

```bash
cd '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815'
sha256sum -c SHA256SUMS.txt
```

Expected: 清单中的全部文件均为 `OK`。再次在正式 HTML 中确认滑块默认 `70%`、连续表面默认关闭、默认相机值正确；浏览器重载后的五条器官网格 opacity 均为 `0.70`，切换完整面后面数与五个原始 PLY 一致。

- [ ] **Step 5: 保留功能分支**

保持 `feature/core-design-20260813-full-rotation` 分支和当前工作区，不推送、不合并、不删除分支；报告实现提交、正式网页路径、回退目录和全部验证证据。
