# Filled EUS Vessel Boundary Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从原始手工 `.seg.nrrd` 精确重采样100个切面位置，生成主动脉、下腔静脉和门静脉系各100张白底实心填充图，同时保留原边界图不变。

**Architecture:** 新增一个独立导出脚本，复用项目现有 `LabelVolume` 和 `CpuLabelBackend` 完成 LPS 到 RAS 坐标转换及最近邻标签采样。脚本先严格验证输入包、源标签哈希和三类文件名集合，在同级临时目录中生成、逐像素验证、写出说明与哈希清单，全部成功后再原子发布目标目录。

**Tech Stack:** Python 3.12、NumPy、Pillow、SimpleITK、SciPy `map_coordinates`、pytest、项目现有 `ct_vascular_resampling.label_resampling`。

---

## 文件结构

- Create: `scripts/export_filled_eus_vessel_boundaries.py`：命令行入口、输入验证、标签重采样、填充图生成、事务式发布和校验清单。
- Create: `tests/test_export_filled_eus_vessel_boundaries.py`：类别映射、触边掩膜填充、输入集合校验、边界包含校验和输出颜色测试。
- Create at runtime: `C:\Users\zhangyutang\Desktop\重新随机100位置_三类血管独立边界填充图_20260817`：最终300张图及审计文件。

### Task 1: 用测试锁定填充与输入校验语义

**Files:**
- Create: `tests/test_export_filled_eus_vessel_boundaries.py`
- Test: `tests/test_export_filled_eus_vessel_boundaries.py`

- [ ] **Step 1: 写入失败测试**

测试必须覆盖以下具体行为：

```python
from pathlib import Path

import numpy as np
import pytest

from scripts.export_filled_eus_vessel_boundaries import (
    CLASS_SPECS,
    render_filled_rgb,
    validate_boundary_subset,
    validate_filename_sets,
)


def test_render_filled_rgb_fills_edge_touching_pixels() -> None:
    labels = np.zeros((4, 5), dtype=np.uint8)
    labels[:, :2] = 8

    rendered = render_filled_rgb(labels, CLASS_SPECS["aorta"])

    assert np.all(rendered[:, :2] == (255, 0, 0))
    assert np.all(rendered[:, 2:] == 255)


def test_render_filled_rgb_merges_only_eus_portal_labels() -> None:
    labels = np.asarray([[23, 26, 33, 34, 35, 36, 37]], dtype=np.uint8)

    rendered = render_filled_rgb(labels, CLASS_SPECS["portal_vein"])

    assert np.all(rendered[0, 0] == 255)
    assert np.all(rendered[0, 1:] == (170, 85, 255))


def test_validate_boundary_subset_rejects_pixels_outside_source_mask() -> None:
    source_mask = np.asarray([[False, True], [False, False]])
    boundary_mask = np.asarray([[True, True], [False, False]])

    with pytest.raises(ValueError, match="边界像素不属于重采样标签"):
        validate_boundary_subset(boundary_mask, source_mask, Path("sample.png"))


def test_validate_filename_sets_requires_exact_100_file_match() -> None:
    names = {f"sample-{index:03d}.png" for index in range(100)}
    validate_filename_sets({key: names for key in CLASS_SPECS}, names)

    with pytest.raises(ValueError, match="文件名集合不一致"):
        validate_filename_sets(
            {**{key: names for key in CLASS_SPECS}, "aorta": names - {"sample-000.png"}},
            names,
        )
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```bash
mamba run -n base pytest tests/test_export_filled_eus_vessel_boundaries.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `scripts.export_filled_eus_vessel_boundaries`。

### Task 2: 实现可审计、事务式导出脚本

**Files:**
- Create: `scripts/export_filled_eus_vessel_boundaries.py`
- Test: `tests/test_export_filled_eus_vessel_boundaries.py`

- [ ] **Step 1: 定义不可变类别映射与纯函数**

实现以下公开接口：

```python
@dataclass(frozen=True)
class VesselClassSpec:
    label_values: tuple[int, ...]
    color_rgb: tuple[int, int, int]


CLASS_SPECS = {
    "aorta": VesselClassSpec((8,), (255, 0, 0)),
    "inferior_vena_cava": VesselClassSpec((9,), (0, 0, 255)),
    "portal_vein": VesselClassSpec((26, 33, 34, 35, 36, 37), (170, 85, 255)),
}


def render_filled_rgb(labels: np.ndarray, spec: VesselClassSpec) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("标签平面必须是二维数组")
    mask = np.isin(values, spec.label_values)
    rgb = np.full((*values.shape, 3), 255, dtype=np.uint8)
    rgb[mask] = spec.color_rgb
    return rgb


def validate_boundary_subset(
    boundary_mask: np.ndarray,
    source_mask: np.ndarray,
    path: Path,
) -> None:
    if boundary_mask.shape != source_mask.shape:
        raise ValueError(f"边界图尺寸不一致: {path}")
    outside_count = int(np.count_nonzero(boundary_mask & ~source_mask))
    if outside_count:
        raise ValueError(f"边界像素不属于重采样标签: {path}, count={outside_count}")
```

- [ ] **Step 2: 实现严格输入预检**

脚本必须：

- 验证输入根、三个类别目录、JSONL 和标签文件存在。
- 验证标签 SHA-256 等于 `0b56268488411925d96bb070e25e72a0105a8502e87ffd349a9ba01cd32dc124`。
- 验证 JSONL 恰好100条、`slice_id` 唯一、每条 `square_vertices_world` 为 `4x3` 有限数值。
- 验证三个输入目录各有100张 RGB 300x300 PNG，且文件名集合与100个 `slice_id + ".png"` 完全一致。
- 验证目标目录不存在；若存在则退出，不覆盖。

- [ ] **Step 3: 实现批量最近邻重采样与填充**

核心调用固定为：

```python
volume = load_label_volume(segmentation_path, input_coordinate_system="LPS")
backend = CpuLabelBackend(volume)
planes = backend.sample_many(vertices_batch, resolution=300)
```

逐位置、逐类别生成白底 RGB 图。原边界图的类别颜色像素必须是对应源掩膜的子集；任何不一致立即失败。禁止 flood fill、形态学闭合和平滑。

- [ ] **Step 4: 实现事务式发布与审计文件**

在目标目录同级创建隐藏临时目录，完成以下工作后使用 `os.replace` 发布：

- 三个类别子目录各写100张 PNG。
- 复制 `gallery_sample_100_unique_positions.jsonl`。
- 写入 `README_填充图说明.txt`，记录源路径、源 SHA-256、LPS/RAS、300x300、最近邻、标签映射、颜色和触边填充规则。
- 写入 `SHA256SUMS.txt`，按相对路径排序记录300张 PNG、JSONL 和 README 的 SHA-256。
- 重新读取每张 PNG，验证其彩色像素掩膜与源标签掩膜逐像素相等，并验证只包含白色和该类颜色。
- 发生异常时删除临时目录，原始边界图和正式目标路径均保持不变。

- [ ] **Step 5: 运行测试并确认通过**

Run:

```bash
mamba run -n base pytest tests/test_export_filled_eus_vessel_boundaries.py -q
```

Expected: all tests pass。

- [ ] **Step 6: 运行相关回归测试**

Run:

```bash
mamba run -n base pytest tests/test_label_resampling.py tests/test_manual_segmentation.py -q
```

Expected: all tests pass，确认未改变既有标签采样和三类血管语义。

- [ ] **Step 7: 提交脚本与测试**

```bash
git add scripts/export_filled_eus_vessel_boundaries.py tests/test_export_filled_eus_vessel_boundaries.py
git commit -m "feat: export filled EUS vessel masks"
```

### Task 3: 对真实100位置执行导出

**Files:**
- Read: `C:\Users\zhangyutang\Desktop\重新随机100位置_三类血管独立边界图_20260817\gallery_sample_100_unique_positions.jsonl`
- Read: `C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\血管重建病例2\EUS main organ---.seg(1).nrrd`
- Create: `C:\Users\zhangyutang\Desktop\重新随机100位置_三类血管独立边界填充图_20260817`

- [ ] **Step 1: 再次确认正式输出路径为空闲**

Run:

```bash
test ! -e '/mnt/c/Users/zhangyutang/Desktop/重新随机100位置_三类血管独立边界填充图_20260817'
```

Expected: exit code 0。

- [ ] **Step 2: 运行真实导出**

Run:

```bash
mamba run -n base python scripts/export_filled_eus_vessel_boundaries.py \
  --input-root '/mnt/c/Users/zhangyutang/Desktop/重新随机100位置_三类血管独立边界图_20260817' \
  --segmentation '/mnt/c/Users/zhangyutang/Desktop/CT-EUS定位项目/数据/血管重建病例2/EUS main organ---.seg(1).nrrd' \
  --output-root '/mnt/c/Users/zhangyutang/Desktop/重新随机100位置_三类血管独立边界填充图_20260817'
```

Expected: 输出 `positions=100 images=300 validation=passed`，exit code 0。

### Task 4: 独立验证与视觉抽查

**Files:**
- Read: `C:\Users\zhangyutang\Desktop\重新随机100位置_三类血管独立边界填充图_20260817`

- [ ] **Step 1: 独立统计文件、尺寸和颜色**

使用独立验证命令重新读取全部300张图并断言：每类100张、尺寸300x300、RGB模式、文件名集合相等，且每张仅含白色和对应类别色。

Expected: `aorta=100 inferior_vena_cava=100 portal_vein=100 invalid=0`。

- [ ] **Step 2: 独立逐像素重采样比对**

再次从 `.seg.nrrd` 和 JSONL 构建100个标签平面，不复用导出阶段缓存，比较300张图的彩色掩膜与源标签掩膜。

Expected: `mismatched_images=0 mismatched_pixels=0`。

- [ ] **Step 3: 校验原图未被修改**

Run:

```bash
cd '/mnt/c/Users/zhangyutang/Desktop/重新随机100位置_三类血管独立边界图_20260817'
sha256sum -c PACKAGE_SHA256SUMS.txt
```

Expected: 所有原包校验项目均为 `OK`。

- [ ] **Step 4: 生成并检查10位置联系表**

以固定随机种子 `20260817` 抽取10个位置，将每个位置的原三类边界图与新三类填充图并列组成联系表，写入临时工作目录并用图像查看工具检查。抽样必须至少包含一个触边分量和一个门静脉多标签合并结果。

Expected: 白底未被误填，所有可见源标签区域实心着色，触边区域保留，三类颜色无串色。

---

## 执行方式

本任务在当前会话中使用 `superpowers:executing-plans` 内联执行。根据当前协作约束不派生子代理，每个任务完成后检查测试和产物再进入下一项。
