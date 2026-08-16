# 器官网格不透明度滑块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为采样点与零度基准面离线交互网页增加全局器官网格不透明度滑块，范围 `0.10–1.00`、步长 `0.05`、默认 `0.70`，并保证现有可见性预设和所有重采样数据不变。

**Architecture:** 沿用 `zero_plane_visualization.py` 的 trace 角色列表，在生成 HTML 时同时嵌入器官网格 trace 索引和零度面 trace 索引。顶部工具栏增加原生 `range` 控件；浏览器端仅通过 `Plotly.restyle(..., {opacity: value}, organMeshTraceIndices)` 更新五条器官网格，不介入预设的 `visible` 状态。

**Tech Stack:** Python 3.12、NumPy、Trimesh、Plotly、pytest、离线 HTML/JavaScript、Microsoft Edge Chrome DevTools Protocol。

---

## 文件结构

- 修改 `src/ct_vascular_resampling/zero_plane_visualization.py`：定义滑块常量、生成工具栏、绑定浏览器事件、将交互器官网格默认不透明度改为 `0.70`、更新交付说明。
- 修改 `tests/test_zero_plane_visualization.py`：固定默认值、滑块 HTML 契约和器官网格 trace 索引。
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
assert '"opacity":0.7' in html
assert 'id="organ-mesh-opacity-slider"' in html
assert 'id="organ-mesh-opacity-value"' in html
assert 'min="0.1" max="1.0" step="0.05" value="0.7"' in html
assert "器官网格不透明度" in html
assert "70%" in html
assert "const organMeshTraceIndices = [0];" in html
assert "Plotly.restyle" in html
```

- [ ] **Step 2: 运行聚焦测试并确认因功能尚未实现而失败**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
```

Expected: FAIL，首先报告 `INTERACTIVE_ORGAN_MESH_OPACITY` 仍为 `0.22`，或滑块常量/HTML 尚不存在；不得是测试语法错误。

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
```

不得改变 `STATIC_ORGAN_MESH_ALPHA`。

- [ ] **Step 2: 扩展现有浏览器控制脚本**

将 `_zero_plane_visibility_script(roles)` 重命名为 `_interactive_controls_script(roles)`。除保留现有零度面逻辑外，从同一个 `roles` 列表生成器官网格索引：

```python
organ_mesh_indices = [index for index, role in enumerate(roles) if role == "organ_mesh"]
if not organ_mesh_indices:
    raise ValueError("交互可视化缺少器官网格 trace")
organ_indices_json = json.dumps(organ_mesh_indices, separators=(",", ":"))
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

脚本不得调用 `Plotly.restyle` 修改 `visible`，不得在 `plotly_buttonclicked` 中重置滑块值。现有零度面复选框逻辑保持原行为。

- [ ] **Step 3: 在顶部工具栏增加滑块并支持窄屏换行**

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
```

不得使用固定定位，不得把控件放入 Plotly 图表内部。

- [ ] **Step 4: 使用新的控制脚本并更新交付说明**

在 `pio.to_html(..., post_script=...)` 中调用 `_interactive_controls_script(roles)`。在 `_readme_text` 的打开方式中明确：

```text
- HTML 可直接用浏览器离线打开；顶部工具栏可控制零度基准面显示和器官网格不透明度，右侧图例可按器官显隐。
```

- [ ] **Step 5: 运行聚焦测试和可视化模块测试**

Run:

```bash
pytest -q tests/test_zero_plane_visualization.py::test_render_outputs_are_offline_and_nonblank
pytest -q tests/test_zero_plane_visualization.py
git diff --check
```

Expected: 聚焦测试 `1 passed`，模块测试 `16 passed`，`git diff --check` 退出码为 `0`。

- [ ] **Step 6: 提交实现与测试**

```bash
git add src/ct_vascular_resampling/zero_plane_visualization.py tests/test_zero_plane_visualization.py
git commit -m "feat: control organ opacity in visualization"
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
```

Expected: 三项均为真，复选框和四个预设仍存在。

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

- [ ] **Step 4: 验证四个预设不重置滑块**

依次实际点击 `Points + zero planes`、`Points only`、`Hide organ meshes`、`All`。每次均断言滑块仍为 `0.35`；显示器官的 `Points only` 和 `All` 中 `graph.data[0].opacity` 仍为 `0.35`。同时复核 `Points only` 继续显示器官网格和采样点。

- [ ] **Step 5: 验证桌面和窄屏布局**

在 `1280×800` 和 `390×844` 两个视口读取工具栏、标题、预设按钮的 `getBoundingClientRect()`，断言任意两个不发生矩形相交；截图确认标签、百分比和滑块均未溢出容器。

### Task 4: 生成并审计真实 400 点候选包

**Files:**
- Read: `.work/zero_plane_visualization_20260815/zero_records.jsonl`
- Read: `.work/zero_plane_visualization_20260815/ResampledpointPLY/`
- Read: `.work/zero_plane_visualization_20260815/target_organ_meshes/`
- Read: `.work/zero_plane_visualization_20260815/run_metadata.json`
- Generate only: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260816_不透明度滑块临时\`

- [ ] **Step 1: 用正式输入生成全新候选目录**

Run:

```bash
PYTHONPATH=src python scripts/export_zero_plane_visualization.py \
  --zero-records-jsonl .work/zero_plane_visualization_20260815/zero_records.jsonl \
  --sample-ply-dir .work/zero_plane_visualization_20260815/ResampledpointPLY \
  --organ-mesh-dir .work/zero_plane_visualization_20260815/target_organ_meshes \
  --run-metadata .work/zero_plane_visualization_20260815/run_metadata.json \
  --source-manifest-sha256 d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce \
  --output-dir '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_不透明度滑块临时'
```

Expected: `record_count` 为 `400`；器官计数为胃 `118`、肝 `162`、胰腺 `37`、十二指肠 `53`、食管 `30`。

- [ ] **Step 2: 校验候选目录完整性和数据不变性**

Run:

```bash
cd '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_不透明度滑块临时'
sha256sum -c SHA256SUMS.txt
```

解析 `sampling_points_zero_planes.json` 并断言：坐标系为 `RAS`、记录数为 `400`、器官计数符合上一步、源 manifest SHA-256 为 `d8dd401455968e31dceb31f96e4ef7aeeb9e78297f7738e733b309c44caa70ce`。

将候选目录与当前正式目录逐文件比较。除 `sampling_points_zero_planes_interactive.html`、`README_中文.txt` 和 `SHA256SUMS.txt` 外，其余文件 SHA-256 必须完全一致。

- [ ] **Step 3: 在真实页面验证五条器官网格轨迹**

通过 Edge CDP 加载候选 HTML，断言器官网格索引为 `[0, 7, 14, 21, 28]`，五条轨迹初始 opacity 均为 `0.70`。把滑块改为 `0.40` 后，五条轨迹均变为 `0.40`，其余 30 条轨迹的 opacity 和全部 35 条轨迹的 visible 值保持不变。

- [ ] **Step 4: 在真实页面验证预设和布局**

点击 `Points only` 后断言五条器官网格和五组采样点可见，零度面与局部轴隐藏，滑块仍为上一步的 `0.40`。随后恢复滑块 `0.70` 和默认 `All` 状态，截图检查工具栏、标题、预设按钮和图例无重叠。

### Task 5: 可回退发布与最终验证

**Files:**
- Preserve: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815\`
- Backup to: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815_不透明度滑块加入前备份_勿用\`
- Promote: `C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260816_不透明度滑块临时\`

- [ ] **Step 1: 预检三个目录，禁止覆盖既有备份**

确认正式目录和候选目录均存在，且目标备份目录不存在。任一条件不满足立即停止，不执行移动。

- [ ] **Step 2: 先备份正式目录，再提升候选目录**

依次执行两个独立的 `mv`：

```bash
mv '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815' \
   '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815_不透明度滑块加入前备份_勿用'
```

```bash
mv '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260816_不透明度滑块临时' \
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

Expected: 当前基线 `256` 项测试全部通过；`compileall` 和 `git diff --check` 退出码为 `0`；Git 只保留既有未跟踪 `.superpowers/` 和 `.work/`。

- [ ] **Step 4: 重新校验正式目录**

Run:

```bash
cd '/mnt/c/Users/zhangyutang/Desktop/本次重采样_采样点与零度基准面可视化_20260815'
sha256sum -c SHA256SUMS.txt
```

Expected: 清单中的全部文件均为 `OK`。再次在正式 HTML 中确认滑块默认 `70%`，浏览器重载后的五条器官网格 opacity 均为 `0.70`。

- [ ] **Step 5: 保留功能分支**

保持 `feature/core-design-20260813-full-rotation` 分支和当前工作区，不推送、不合并、不删除分支；报告实现提交、正式网页路径、回退目录和全部验证证据。
