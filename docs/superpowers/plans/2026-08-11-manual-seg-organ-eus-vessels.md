# 手工分割器官与独立 EUS 三类血管 Gallery 实施计划

> **供执行代理使用：** 批准后优先使用 `superpowers:executing-plans` 在当前会话逐项执行；只有用户另行批准并行代理时才使用 `superpowers:subagent-driven-development`。所有步骤使用复选框（`- [ ]`）跟踪状态。

**目标：** 使用服务器 CT、用户指定的 3D Slicer 分割以及既有重建动静脉网格，从头重建病例 2 Gallery；完整保留原血管建库行为，同时新增基于标签像素的器官信息和相互独立的三类 EUS 血管结果。

**架构：** 现有 CT 三次插值后端和 PLY 血管求交链路保持不变。新增一条使用相同世界坐标方形网格的最近邻标签体采样链路，由同一个原始标签平面派生器官信息、器官边界和三类 EUS 血管结果。只有显式启用手工分割模式的 Gallery 才写入新字段和两类新图像，旧配置、原血管特征和四状态路由不改变。

**技术栈：** Python 3.12、NumPy、SciPy `ndimage`、SimpleITK、CuPy/CUDA、Pillow、trimesh、PyYAML、pytest、mamba、Git、SSH、screen。

---

## 一、文件职责与修改边界

- 新建 `src/ct_vascular_resampling/label_resampling.py`：读取离散标签体、转换到规范 RAS、校验 CT/标签几何，并提供最近邻 CPU/GPU 方形批采样。
- 新建 `src/ct_vascular_resampling/manual_segmentation.py`：保存已批准的器官与 EUS 血管映射；分析单张原始标签平面；区分可见边界和完整特征；生成手工标签图像。
- 新建 `src/ct_vascular_resampling/manual_preprocessing.py`：仅导出 14 类手工器官掩膜/网格，记录原始标签体与外部重建血管来源，不从标签体重建正式动静脉。
- 新建 `scripts/preprocess_manual_segmentation_case.py`：提供非破坏性的手工分割预处理命令行入口。
- 修改 `src/ct_vascular_resampling/config.py`：新增可选且严格校验的 `manual_segmentation` 配置，不改变旧病例配置。
- 修改 `src/ct_vascular_resampling/rendering.py`：携带可选的手工标签渲染结果，不改变原 `features`、`boundary_only` 和 `ct_overlay`。
- 修改 `src/ct_vascular_resampling/gallery.py`：锁定和校验手工分割 Gallery schema，写入两类新图像，并保证断点恢复安全。
- 修改 `src/ct_vascular_resampling/pipeline.py`：加载并采样标签体，仅为按原规则进入 Gallery 的切面附加新结果，并记录运行溯源与汇总计数。
- 修改 `configs/case.example.yaml`、`README.md` 和桌面项目说明文档：说明手工分割模式、精确标签映射、60% 黑色阈值和不完整血管规则。
- 新建 `tests/test_label_resampling.py`、`tests/test_manual_segmentation.py`；扩展 `tests/test_config.py`、`tests/test_preprocessing.py`、`tests/test_rendering.py`、`tests/test_gallery_and_adapter.py`、`tests/test_pipeline.py`、`tests/test_quality.py`。

## 二、必须保持的业务不变量

1. 原 `artery_tree.ply`、`vein_tree.ply` 仍按原算法求交；原 `features`、`boundary_only_png`、`ct_overlay_png` 和 Gallery/unindexed/rejected/excluded_fov 路由不受新逻辑影响。
2. 器官标签来自切面最近邻标签像素。只要当前方形内至少存在 1 个该器官像素就加入 `organ_labels`，不要求轮廓闭合；整张切面位于器官内部也必须加入。
3. EUS 三类血管固定为腹主动脉、下腔静脉和门静脉系。门静脉系血管只合并 `26,33,34,35,36,37`；Main Portal Vein `23` 只属于器官门静脉联合，不属于新 EUS 门静脉血管。
4. 新 `eus_vessel_features` 只收录二维标签平面中完整、不触及图像四边的 8 邻域连通分量。
5. 开放、截断或触及图像边缘的血管分量不得进入 `eus_vessel_features`，但必须进入 `eus_vessel_labels`，并且必须绘制在 `eus_vessel_boundary_png` 与 `ct_eus_vessel_overlay_png` 中。
6. 正式病例使用 `black_ratio_limit: 0.60`；只有黑色比例严格大于 60% 才因比例规则拒绝，恰好 60% 保留。

### 任务 1：增加严格的手工分割配置合同

**涉及文件：**

- 修改：`src/ct_vascular_resampling/config.py`
- 测试：`tests/test_config.py`

- [ ] **步骤 1：先写失败的配置测试**

增加以下配置样例和成功加载测试：

```python
MANUAL_SEGMENTATION_YAML = """
manual_segmentation:
  path: labels/EUS-main-organ.seg.nrrd
  organ_label_values:
    spleen: [1]
    kidney_right: [2]
    kidney_left: [3]
    gallbladder: [4]
    esophagus: [5]
    liver: [6]
    stomach: [7]
    aorta: [8]
    inferior_vena_cava: [9]
    pancreas: [11]
    adrenal_gland_right: [12]
    adrenal_gland_left: [13]
    duodenum: [14]
    portal_vein: [23, 26, 33, 34, 35, 36, 37]
  eus_vessel_label_values:
    aorta: [8]
    inferior_vena_cava: [9]
    portal_vein: [26, 33, 34, 35, 36, 37]
  eus_vessel_colors:
    aorta: [255, 0, 0]
    inferior_vena_cava: [0, 0, 255]
    portal_vein: [170, 85, 255]
"""

def test_case_config_loads_strict_manual_segmentation_mode(tmp_path):
    organ_models = "\n".join(
        f"  {name}: models/{name}.obj" for name in REQUIRED_ORGAN_IDS
    )
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        _case_yaml(organ_models) + MANUAL_SEGMENTATION_YAML,
        encoding="utf-8",
    )
    config = load_case_config(config_path)
    manual = config.manual_segmentation
    assert manual is not None
    assert manual.path == tmp_path / "labels/EUS-main-organ.seg.nrrd"
    assert manual.organ_label_values["portal_vein"] == (23, 26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_label_values["portal_vein"] == (26, 33, 34, 35, 36, 37)
    assert manual.eus_vessel_colors["portal_vein"] == (170, 85, 255)
```

参数化测试以下错误：缺少规范键、出现未知键、标签列表为空、同一映射内标签值重复、把 `True` 当作整数、标签值不是整数、颜色不是三个 `0..255` 整数。另加旧配置测试，确认未配置 `manual_segmentation` 时结果为 `None`。

- [ ] **步骤 2：运行测试并确认红灯原因正确**

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py -k manual_segmentation
```

预期：失败原因是 `CaseConfig` 尚无 `manual_segmentation` 字段，而不是测试语法或夹具错误。

- [ ] **步骤 3：实现最小配置类型与解析器**

新增：

```python
EUS_VESSEL_IDS = ("aorta", "inferior_vena_cava", "portal_vein")

@dataclass(frozen=True)
class ManualSegmentationConfig:
    path: Path
    organ_label_values: dict[str, tuple[int, ...]]
    eus_vessel_label_values: dict[str, tuple[int, ...]]
    eus_vessel_colors: dict[str, tuple[int, int, int]]
```

为 `CaseConfig` 增加 `manual_segmentation: ManualSegmentationConfig | None = None`。解析器只允许 `path`、`organ_label_values`、`eus_vessel_label_values`、`eus_vessel_colors` 四个键；分别要求键集合严格等于 `ORGAN_BOUNDARY_IDS` 与 `EUS_VESSEL_IDS`。整数转换前先拒绝布尔值；禁止同一映射内部的标签重叠；允许器官映射与 EUS 血管映射按已批准规则共享数值标签。

- [ ] **步骤 4：验证绿灯并提交本模块**

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py
git add src/ct_vascular_resampling/config.py tests/test_config.py
git commit -m "feat: add strict manual segmentation configuration"
```

预期：`tests/test_config.py` 全部通过。

### 任务 2：建立最近邻标签体及 CPU/GPU 后端

**涉及文件：**

- 新建：`src/ct_vascular_resampling/label_resampling.py`
- 新建：`tests/test_label_resampling.py`

- [ ] **步骤 1：先写几何和 CPU 采样失败测试**

覆盖 LPS 到 RAS 转换、CT/标签 Size/Spacing/Origin/Direction 一致性、最近邻输出和 FOV 外填 0：

```python
def test_cpu_label_backend_samples_nearest_labels_and_fills_zero():
    labels = np.zeros((3, 4, 5), dtype=np.uint8)
    labels[1, 1, 1] = 8
    image = sitk.GetImageFromArray(labels)
    volume = LabelVolume.from_sitk(image, input_coordinate_system="LPS")
    vertices = np.asarray(
        [[0.0, 0.0, 1.0], [-4.0, 0.0, 1.0], [-4.0, -3.0, 1.0], [0.0, -3.0, 1.0]],
        dtype=np.float64,
    )
    sampled = CpuLabelBackend(volume).sample_many(vertices[None], resolution=5)
    assert sampled.dtype == np.uint8
    assert sampled.shape == (1, 5, 5)
    assert 8 in sampled[0]
    outside = vertices + np.asarray([100.0, 0.0, 0.0])
    assert np.all(CpuLabelBackend(volume).sample_many(outside[None], 5) == 0)
```

几何测试必须接受 `5e-7 mm` 的 Origin 文本精度差，并拒绝 `2e-6 mm` Origin 差异以及 Size、Spacing、Direction 任一不一致。

- [ ] **步骤 2：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_label_resampling.py
```

预期：因 `label_resampling.py` 尚不存在而失败。

- [ ] **步骤 3：实现参考 CPU 路径**

实现以下接口：

```python
@dataclass(frozen=True)
class LabelVolume:
    data_zyx: np.ndarray
    spacing_xyz: np.ndarray
    origin_xyz: np.ndarray
    direction_xyz: np.ndarray

    @classmethod
    def from_sitk(
        cls,
        image: sitk.Image,
        *,
        input_coordinate_system: str,
    ) -> "LabelVolume":
        data = sitk.GetArrayFromImage(image)
        if image.GetDimension() != 3 or data.ndim != 3:
            raise ValueError("标签图必须是三维单标量图像")
        if not np.issubdtype(data.dtype, np.integer) or np.min(data) < 0 or np.max(data) > 255:
            raise ValueError("标签值必须是可表示为 uint8 的非负整数")
        return cls(
            data_zyx=data.astype(np.uint8, copy=False),
            spacing_xyz=np.asarray(image.GetSpacing(), dtype=np.float64),
            origin_xyz=to_ras_points(
                np.asarray(image.GetOrigin(), dtype=np.float64),
                input_coordinate_system,
            ),
            direction_xyz=to_ras_direction(
                np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3),
                input_coordinate_system,
            ),
        )

    @property
    def physical_to_index_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.direction_xyz @ np.diag(self.spacing_xyz))

    def world_to_continuous_indices(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        flat = points.reshape(-1, 3)
        indices = (flat - self.origin_xyz) @ self.physical_to_index_matrix.T
        return indices.reshape(points.shape)

def validate_label_geometry(
    ct: CTVolume,
    labels: LabelVolume,
    *,
    atol_mm: float = 1e-6,
) -> None:
    if ct.data_zyx.shape != labels.data_zyx.shape:
        raise ValueError("CT 与标签图的 Size 不一致")
    if not np.allclose(ct.spacing_xyz, labels.spacing_xyz, atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与标签图的 Spacing 不一致")
    if not np.allclose(ct.origin_xyz, labels.origin_xyz, atol=atol_mm, rtol=0.0):
        raise ValueError("CT 与标签图的 Origin 不一致")
    if not np.allclose(ct.direction_xyz, labels.direction_xyz, atol=1e-8, rtol=0.0):
        raise ValueError("CT 与标签图的 Direction 不一致")

class CpuLabelBackend:
    name = "cpu"

    def __init__(self, volume: LabelVolume) -> None:
        self.volume = volume

    def sample_many(self, vertices_batch: np.ndarray, resolution: int) -> np.ndarray:
        vertices = np.asarray(vertices_batch, dtype=np.float64)
        if vertices.ndim != 3 or vertices.shape[1:] != (4, 3):
            raise ValueError("vertices_batch 必须是 N×4×3 数组")
        sampled = []
        for square in vertices:
            values = map_coordinates(
                self.volume.data_zyx,
                square_coordinates_zyx(self.volume, square, resolution),
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            )
            sampled.append(values.reshape(resolution, resolution).astype(np.uint8, copy=False))
        return (
            np.stack(sampled, axis=0)
            if sampled
            else np.empty((0, resolution, resolution), dtype=np.uint8)
        )
```

输入必须是三维单标量整数标签体，且数值可安全转换为 `uint8`。内存中只保存一个 `uint8` 三维标签体，批输出也是 `uint8`；不得为每个器官复制一份三维体。`square_coordinates_zyx` 必须通过 `LabelVolume.world_to_continuous_indices` 使用与 CT 相同的 RAS 世界坐标网格。

- [ ] **步骤 4：增加 GPU 精确一致性与回退失败测试**

使用假的 CuPy device 和假的 `map_coordinates`，要求 CPU/GPU 输出逐像素完全相同。覆盖：`auto` 初始化失败回退 CPU、强制 `gpu` 初始化失败直接报错、GPU 结果任一像素变化时校验失败。

- [ ] **步骤 5：实现 GPU 后端与精确校准**

实现 `CuPyLabelBackend`、`create_label_sampling_backend`、`validate_label_backend_against_cpu`。GPU 路径固定使用 `order=0`、`prefilter=False`、`cval=0`；校准使用 `np.array_equal`，标签结果不得使用浮点容差。

- [ ] **步骤 6：运行聚焦测试并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_label_resampling.py \
  tests/test_coordinates.py \
  tests/test_resampling_backend.py
git add src/ct_vascular_resampling/label_resampling.py tests/test_label_resampling.py
git commit -m "feat: sample aligned segmentation label planes"
```

预期：全部通过，既有 CT 后端测试不发生变化。

### 任务 3：实现器官出现判定与三类 EUS 血管分析

**涉及文件：**

- 新建：`src/ct_vascular_resampling/manual_segmentation.py`
- 新建：`tests/test_manual_segmentation.py`

- [ ] **步骤 1：先写器官像素语义失败测试**

```python
CONFIG = ManualSegmentationConfig(
    path=Path("unused.seg.nrrd"),
    organ_label_values={
        "spleen": (1,),
        "kidney_right": (2,),
        "kidney_left": (3,),
        "gallbladder": (4,),
        "esophagus": (5,),
        "liver": (6,),
        "stomach": (7,),
        "aorta": (8,),
        "inferior_vena_cava": (9,),
        "pancreas": (11,),
        "adrenal_gland_right": (12,),
        "adrenal_gland_left": (13,),
        "duodenum": (14,),
        "portal_vein": (23, 26, 33, 34, 35, 36, 37),
    },
    eus_vessel_label_values={
        "aorta": (8,),
        "inferior_vena_cava": (9,),
        "portal_vein": (26, 33, 34, 35, 36, 37),
    },
    eus_vessel_colors={
        "aorta": (255, 0, 0),
        "inferior_vena_cava": (0, 0, 255),
        "portal_vein": (170, 85, 255),
    },
)

def test_one_pixel_and_full_frame_organs_are_labels_without_artificial_frame():
    labels = np.full((9, 9), 6, dtype=np.uint8)
    labels[4, 4] = 11
    result = analyze_manual_label_plane(labels, 100.0, 100.0, CONFIG)
    assert result.organ_labels == ["liver", "pancreas"]
    assert np.all(result.organ_boundary_rgb[0, :, :] == 255)
    assert np.all(result.organ_boundary_rgb[-1, :, :] == 255)
```

再分别断言：最近邻平面中没有像素时，即使连续几何可能相切也不加标签；`23,26,33,34,35,36,37` 均映射到器官 `portal_vein`；EUS 血管 `portal_vein` 排除 `23`，只包含 `26,33,34,35,36,37`。

- [ ] **步骤 2：写入本次最关键的不完整血管失败测试**

同一标签平面同时构造一个闭合 Ao 分量和一个触及顶边的 IVC 分量：

```python
def test_incomplete_eus_vessel_is_drawn_but_not_featured():
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[4:7, 4:7] = 8       # 完整闭合 Ao
    labels[0:3, 8:11] = 9      # 被顶边截断的 IVC
    result = analyze_manual_label_plane(labels, 110.0, 110.0, CONFIG)
    assert [item["label"] for item in result.eus_vessel_features] == ["aorta"]
    assert result.eus_vessel_labels == ["aorta", "inferior_vena_cava"]
    colors = set(map(tuple, result.eus_vessel_boundary_rgb.reshape(-1, 3)))
    assert (255, 0, 0) in colors
    assert (0, 0, 255) in colors
```

另加合并测试：相邻的 SMV `26` 和 SV `33` 必须先合并为一个门静脉二值类，只形成一个连通分量，二者原始标签交界处不得产生内部边界。增加对角像素测试，确认使用 8 邻域。完整分量特征必须满足：

```python
x_mm = mean_column * width_mm / (width_px - 1)
y_mm = mean_row * length_mm / (height_px - 1)
area_mm2 = pixel_count * x_spacing_mm * y_spacing_mm
```

- [ ] **步骤 3：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_manual_segmentation.py
```

预期：因目标模块尚不存在而失败。

- [ ] **步骤 4：实现一次标签平面分析**

定义：

```python
EUS_VESSEL_METADATA_SCHEMA_VERSION = "eus-vessel-metadata/v1"

@dataclass(frozen=True)
class ManualLabelPlaneAnalysis:
    organ_labels: list[str]
    organ_boundary_rgb: np.ndarray
    eus_vessel_labels: list[str]
    eus_vessel_features: list[dict[str, float | str]]
    eus_vessel_boundary_rgb: np.ndarray
```

每个规范类通过 `np.isin` 从同一原始标签平面派生。边界只根据图像内部相邻像素变化计算；图像外侧不得被当作背景，因此整面器官不会产生人为方框。门静脉标签 `26,33,34,35,36,37` 先合并为二值类，再执行边界与连通域分析。连通域使用：

```python
components, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
```

任一分量只要触及第 `0` 行、第 `H-1` 行、第 `0` 列或第 `W-1` 列，就按二维有限视野中的开放/截断轮廓处理。必须先从未过滤掩膜生成可见边界，再过滤特征，保证不完整分量仍被绘制。

- [ ] **步骤 5：验证绿灯并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_manual_segmentation.py \
  tests/test_cropped_retrieval.py
git add src/ct_vascular_resampling/manual_segmentation.py tests/test_manual_segmentation.py
git commit -m "feat: analyze manual organ and EUS vessel planes"
```

预期：全部通过，尤其是不完整血管“图像保留、特征排除”回归测试通过。

### 任务 4：渲染手工器官与 EUS 血管图，同时保护原输出

**涉及文件：**

- 修改：`src/ct_vascular_resampling/rendering.py`
- 修改：`src/ct_vascular_resampling/manual_segmentation.py`
- 测试：`tests/test_rendering.py`
- 测试：`tests/test_manual_segmentation.py`

- [ ] **步骤 1：先写渲染隔离失败测试**

对同一组 CT 像素和 PLY 血管轮廓分别执行旧模式和手工标签模式，断言：

```python
assert manual.features == legacy.features
assert manual.boundary_only.tobytes() == legacy.boundary_only.tobytes()
assert manual.ct_overlay.tobytes() == legacy.ct_overlay.tobytes()
assert manual.organ_labels == analysis.organ_labels
assert manual.eus_vessel_features == analysis.eus_vessel_features
```

对触边 IVC 断言：蓝色同时出现在 `eus_vessel_boundary` 与 `ct_eus_vessel_overlay`，但不存在 IVC 特征。白底图的血管内部和无边界背景必须保持白色，不能改成填充色块。

- [ ] **步骤 2：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_rendering.py \
  tests/test_manual_segmentation.py
```

预期：失败原因是 `RenderedSample` 尚无 EUS 字段，且尚无手工结果合成函数。

- [ ] **步骤 3：实现可选渲染产物**

为 `RenderedSample` 增加默认值为 `None` 的字段：

```python
eus_vessel_boundary: Image.Image | None = None
ct_eus_vessel_overlay: Image.Image | None = None
eus_vessel_features: list[dict[str, float | str]] | None = None
eus_vessel_labels: list[str] | None = None
```

实现 `apply_manual_label_analysis(rendered, analysis)`，使用 `dataclasses.replace` 返回新对象。`organ_vessel_boundary` 先采用 `analysis.organ_boundary_rgb`，再只用原 `rendered.boundary_only` 中的非白像素覆盖，使既有重建血管颜色位于器官层之上。新的 CT 叠加图只在未过滤 EUS 边界的非白像素处覆盖 CT RGB，开放或触边轮廓不得被特征过滤影响。

- [ ] **步骤 4：验证绿灯并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_rendering.py \
  tests/test_manual_segmentation.py
git add \
  src/ct_vascular_resampling/rendering.py \
  src/ct_vascular_resampling/manual_segmentation.py \
  tests/test_rendering.py \
  tests/test_manual_segmentation.py
git commit -m "feat: render separate EUS vessel boundary outputs"
```

预期：原输出逐字节相等测试和两张新图测试全部通过。

### 任务 5：持久化 Gallery 新字段并锁定 schema

**涉及文件：**

- 修改：`src/ct_vascular_resampling/gallery.py`
- 测试：`tests/test_gallery_and_adapter.py`

- [ ] **步骤 1：先写手工 Gallery 持久化失败测试**

构造同时具有原血管特征和手工字段的 `RenderedSample`，用以下 writer 写入：

```python
writer = GalleryWriter(
    tmp_path / "case",
    "case",
    manual_segmentation_enabled=True,
)
status = writer.write_sample(
    sample_id="stomach-000000-x-00",
    organ="stomach",
    probe_point_world=probe_point,
    input_normal_world=normal,
    frame=frame,
    rendered=rendered,
    quality=accepted_quality,
)
record = json.loads(
    (tmp_path / "case/gallery/gallery.jsonl").read_text(encoding="utf-8")
)
assert status == "gallery"
assert record["eus_vessel_metadata_schema_version"] == "eus-vessel-metadata/v1"
assert record["eus_vessel_labels"] == ["aorta", "inferior_vena_cava"]
assert record["eus_vessel_features"] == [
    {"label": "aorta", "x_mm": 50.0, "y_mm": 50.0, "area_mm2": 100.0}
]
assert record["eus_vessel_boundary_png"].startswith("eus_vessel_boundary/")
assert record["ct_eus_vessel_overlay_png"].startswith("ct_eus_vessel_overlay/")
```

同时确认两张 PNG 实际存在。`unindexed`、`rejected`、`excluded_fov` 不得获得新字段或新图像。状态必须继续只由原 `rendered.features` 决定。

- [ ] **步骤 2：增加恢复与 schema 失败测试**

手工模式必须拒绝以下既有记录：缺少 schema、缺少图片、标签未知、标签重复、特征标签不属于三类、特征数值非有限、特征标签未出现在该切面的 `eus_vessel_labels` 中。旧模式必须继续读取合法旧记录，并且不产生任何 `eus_vessel_*` 字段。

- [ ] **步骤 3：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_gallery_and_adapter.py \
  -k 'manual or eus_vessel or legacy'
```

预期：因 `GalleryWriter` 尚无 schema 模式参数和新路径而失败。

- [ ] **步骤 4：实现原子写入与严格校验**

为 `GalleryWriter` 增加 `manual_segmentation_enabled: bool = False`。手工模式下，Gallery 状态必须同时具备所有手工渲染字段；原子写入：

```text
gallery/eus_vessel_boundary/<slice_id>.png
gallery/ct_eus_vessel_overlay/<slice_id>.png
```

写入 JSONL 前附加版本化字段并执行校验；初始化和断点恢复读取旧记录时执行同样校验。`_status_for` 严禁读取 `eus_vessel_features`。

- [ ] **步骤 5：验证绿灯并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_gallery_and_adapter.py \
  tests/test_rendering.py
git add src/ct_vascular_resampling/gallery.py tests/test_gallery_and_adapter.py
git commit -m "feat: persist versioned EUS vessel Gallery metadata"
```

预期：全部通过。

### 任务 6：接入主流程并证明原血管结果不变

**涉及文件：**

- 修改：`src/ct_vascular_resampling/pipeline.py`
- 测试：`tests/test_pipeline.py`

- [ ] **步骤 1：先写单切面集成失败测试**

构造一张质量合格切面：原重建 PLY 中有一个完整血管；标签平面中有一个完整 Ao 和一个触边 IVC。断言最终状态为 Gallery，原 `features` 只来自 PLY，新 EUS 特征只包含 Ao，两张 EUS 图均包含 IVC 蓝色。

再用同一 CT、PLY 和位姿关闭手工模式运行，逐项比较：状态、原 `features`、`boundary_only.png`、`ct_overlay.png` 必须完全一致。

- [ ] **步骤 2：先写小型全流程与后端失败测试**

将姿态流替换为少量确定性样本，验证手工模式：

1. 标签体只加载一次；
2. 任何 PNG/JSONL 写入前完成 CT/标签几何校验；
3. CT 与标签使用同一批方形顶点和相同分辨率；
4. GPU 标签后端按逐像素完全相等进行校准；
5. `auto` 模式下标签校准或运行失败时回退 CPU；
6. 强制 `gpu` 时不得静默使用未经验证的标签后端；
7. rejected、excluded_fov 和按原规则 unindexed 的切面不执行手工标签分析。

- [ ] **步骤 3：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_pipeline.py \
  -k 'manual or label_plane or eus_vessel or invariant'
```

预期：因 `run_case` 尚未加载或传递标签平面而失败。

- [ ] **步骤 4：接入独立标签链路**

修改 `_preflight`，手工模式下要求分割路径存在。render 阶段只加载一个 `LabelVolume`，先与 CT 校验，再创建标签后端，并在与 CT 相同的待处理方形上校准。批数据接口固定为：

```python
(sample, hu_square, raw_label_plane, backend_names)
```

`render_precomputed_square` 保持现有顺序：FOV 判定、质量筛选、原 PLY 求交和原状态判定均不改变。只有原 PLY 存在完整特征、切面将进入 Gallery 时，才调用 `analyze_manual_label_plane` 和 `apply_manual_label_analysis`。旧模式继续使用原网格器官层。

任一 GPU 后端在 `auto` 模式失败时，只关闭失败后端并切换到对应 CPU 参考后端，同时记录回退原因；已完成记录不得改写。强制 `gpu` 时必须在写入当前批次前报错。

- [ ] **步骤 5：验证主流程和不变量**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_pipeline.py \
  tests/test_resampling_backend.py \
  tests/test_label_resampling.py
```

预期：全部通过，所有旧模式断言保持不变。

- [ ] **步骤 6：提交主流程集成**

```bash
git add src/ct_vascular_resampling/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate manual label planes into Gallery builds"
```

### 任务 7：仅预处理手工器官，并引用既有重建血管

**涉及文件：**

- 新建：`src/ct_vascular_resampling/manual_preprocessing.py`
- 新建：`scripts/preprocess_manual_segmentation_case.py`
- 测试：`tests/test_preprocessing.py`
- 测试：`tests/test_cli.py`

- [ ] **步骤 1：先写预处理失败测试**

使用包含全部批准器官标签的小型 CT/分割和两个外部网格文件，断言：

1. 只写出 14 个器官 mask 和 14 个器官 PLY；
2. `portal_vein_and_splenic_vein` 网格来自 `23,26,33,34,35,36,37` 联合；
3. 原始 segmentation 被逐字节复制，并记录 SHA-256；
4. 只记录外部 `artery_tree.ply`、`vein_tree.ply` 的路径与 SHA-256，不重建也不覆盖它们；
5. 生成的手工病例配置使用 `black_ratio_limit: 0.60` 和精确映射/颜色；
6. 几何不一致、必要标签缺失或外部血管文件缺失时，在写入输出前失败。

- [ ] **步骤 2：运行测试并确认红灯**

```bash
mamba run -n ct-vessel-resampling pytest -q tests/test_preprocessing.py -k manual
```

预期：因手工预处理模块尚不存在而失败。

- [ ] **步骤 3：实现非破坏性预处理**

实现以下明确接口：

```text
write_manual_segmentation_case(
    *,
    ct_path: str | Path,
    segmentation_path: str | Path,
    artery_model_path: str | Path,
    vein_model_path: str | Path,
    output_directory: str | Path,
    registration_module_path: str | Path,
    output_root: str | Path,
    case_id: str,
) -> dict[str, object]
```

复用 `validate_geometry`、`build_binary_masks`、`mask_to_mesh`，但只处理 14 类器官。器官配置键 `portal_vein` 在预处理输出中对应模型文件名 `portal_vein_and_splenic_vein`。使用 `shutil.copyfile` 将源分割复制到：

```text
segmentation/EUS-main-organ.seg.nrrd
```

不得从分割生成 `artery_tree` 或 `vein_tree`。manifest 必须记录源绝对路径、SHA-256、几何、标签映射、体素数、网格顶点/面数、watertight 诊断和两个外部重建血管的溯源。

- [ ] **步骤 4：实现不默认覆盖的 CLI**

脚本参数固定为：

```text
--ct
--segmentation
--artery-model
--vein-model
--output
--registration-module
--output-root
--case-id
--overwrite
```

输出目录非空时默认失败，只有显式 `--overwrite` 才允许清理；正式服务器流程始终使用新目录，不传 `--overwrite`。

- [ ] **步骤 5：验证绿灯并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_preprocessing.py \
  tests/test_cli.py
git add \
  src/ct_vascular_resampling/manual_preprocessing.py \
  scripts/preprocess_manual_segmentation_case.py \
  tests/test_preprocessing.py \
  tests/test_cli.py
git commit -m "feat: preprocess manual organs with external vessels"
```

预期：全部通过。

### 任务 8：完善运行协议、汇总、恢复校验与 60% 质量规则

**涉及文件：**

- 修改：`src/ct_vascular_resampling/pipeline.py`
- 测试：`tests/test_pipeline.py`
- 测试：`tests/test_quality.py`

- [ ] **步骤 1：先写运行协议与汇总失败测试**

`run_metadata.json` 必须包含：segmentation 路径/SHA-256/几何、最近邻插值、FOV 外标签值 0、两套标签映射、三类颜色、8 邻域、触边即不完整的二维判定、手工器官网格来源、外部重建血管 SHA-256，以及 `black_ratio_limit: 0.60`。这些内容必须进入 `resume_protocol_sha256`。

`library_summary.json` 必须包含：

```python
assert summary["eus_vessel_label_counts"] == {
    "aorta": 2,
    "inferior_vena_cava": 1,
}
assert summary["eus_vessel_feature_counts"] == {
    "aorta": 1,
}
assert summary["eus_vessel_colors"] == {
    "aorta": [255, 0, 0],
    "inferior_vena_cava": [0, 0, 255],
    "portal_vein": [170, 85, 255],
}
```

断点恢复必须在追加记录前拒绝：segmentation 内容变化、映射变化、颜色变化、阈值变化、几何变化或任一 EUS 图片缺失。

- [ ] **步骤 2：写入 60% 精确边界测试**

```python
def test_quality_accepts_exactly_sixty_percent_black():
    result = evaluate_ct_quality(
        _pixels_with_black_ratio(0.60),
        FilterConfig(black_ratio_limit=0.60, line_min_diagonal_fraction=1.0),
    )
    assert result.black_ratio == 0.60
    assert result.black_ratio_exceeded is False

def test_quality_rejects_more_than_sixty_percent_black():
    result = evaluate_ct_quality(
        _pixels_with_black_ratio(0.61),
        FilterConfig(black_ratio_limit=0.60, line_min_diagonal_fraction=1.0),
    )
    assert result.black_ratio_exceeded is True
```

- [ ] **步骤 3：运行测试并确认红灯/既有边界行为**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_pipeline.py \
  tests/test_quality.py \
  -k 'protocol or summary or sixty or resume'
```

预期：新元数据和汇总测试失败；质量模块现有严格 `>` 比较应使两条 60% 边界测试在指定配置下通过。

- [ ] **步骤 4：实现协议、流式计数与恢复保护**

扩展 `_input_provenance`、`_run_protocol_metadata` 和 Gallery 流式扫描。手工分割合同必须参与 `resume_protocol_sha256`；汇总计数逐行处理 JSONL，不得将完整 Gallery 读入内存。未启用手工模式时保留旧版 summary 结构。

- [ ] **步骤 5：验证绿灯并提交**

```bash
mamba run -n ct-vessel-resampling pytest -q \
  tests/test_pipeline.py \
  tests/test_quality.py \
  tests/test_gallery_and_adapter.py
git add \
  src/ct_vascular_resampling/pipeline.py \
  tests/test_pipeline.py \
  tests/test_quality.py
git commit -m "feat: audit manual EUS vessel Gallery runs"
```

预期：全部通过。

### 任务 9：更新配置示例和中英文文档

**涉及文件：**

- 修改：`configs/case.example.yaml`
- 修改：`README.md`
- 修改：`C:\Users\zhangyutang\Desktop\CT血管重采样项目说明文档` 中描述重采样建库的文件
- 测试：`tests/test_config.py`

- [ ] **步骤 1：更新配置示例**

增加注释形式的完整 `manual_segmentation` 配置，使用已批准的精确映射和颜色。明确 Main Portal Vein `23` 属于器官门静脉联合、不属于新 EUS 血管门静脉；SMV `26` 同时属于二者。展示本模式正式配置使用 `black_ratio_limit: 0.60`，并保留旧模式默认阈值说明。

- [ ] **步骤 2：更新中文和英文行为说明**

文档必须明确：

1. 器官标签以最近邻采样后至少 1 个像素为准，不以网格闭合截面为准；
2. 整面位于器官内部时有标签，但不沿图像四边画人为边框；
3. 原 PLY `features`、`boundary_only`、`ct_overlay` 和状态路由不变；
4. 新三类 EUS 结果与原血管结果分开；
5. 不完整/触边 EUS 血管不进入 `eus_vessel_features`；
6. 开放、截断和触边边界仍必须出现在两张新图中；
7. 正式病例黑色比例阈值为 60%，比较符为严格大于。

- [ ] **步骤 3：校验文档事实并分别记录**

```bash
rg -n \
  "23|26|33|34|35|36|37|0\.60|eus_vessel_features|触边|不完整|开放" \
  README.md configs/case.example.yaml docs
mamba run -n ct-vessel-resampling pytest -q tests/test_config.py
git add README.md configs/case.example.yaml
git commit -m "docs: describe manual EUS vessel Gallery outputs"
```

桌面说明文档位于 Git 仓库外，只在原位置更新，不尝试 Git 暂存，也不复制进仓库。最终审计记录该目录内实际修改文件及修改前/后的 SHA-256。

### 任务 10：完成本地全量验证和独立质量审查

**审查范围：** `7c5e1cb` 之后的全部实现改动。

- [ ] **步骤 1：执行格式、编译和占位符检查**

```bash
git diff --check 7c5e1cb..HEAD
python -m compileall -q src scripts tests
rg -n 'T(BD)|TO(DO)|implement[ ]later|fill[ ]in[ ]details' \
  src tests scripts configs README.md
```

预期：无空白错误、编译错误或实现占位符。

- [ ] **步骤 2：运行完整本地测试**

```bash
mamba run -n ct-vessel-resampling pytest -q
```

预期：全部测试通过，无本分支新引入的 warning。

- [ ] **步骤 3：按高风险不变量人工复核 diff**

逐项确认：

1. 标签图使用最近邻，且规范 RAS 几何与 CT 相同；
2. 内存中没有按器官复制 14 份三维标签体；
3. 标签批大小受 `gpu_batch_size` 或 CPU 批大小约束；
4. 标签 `23` 不进入 EUS 血管门静脉特征，但进入器官门静脉映射；
5. 不完整 EUS 分量只从特征排除，两张图均保留；
6. 新 EUS 特征不影响 Gallery 路由；
7. 旧模式和原血管图像逐字节不变受到自动测试保护；
8. 未暂存无关文件和既有未跟踪 `.superpowers/`。

- [ ] **步骤 4：只使用一个审查代理进行代码审查**

使用 `requesting-code-review` skill，只启用一个审查者，范围限定为本分支 diff 和已批准设计文档。每个有效问题先补充能复现的失败测试，再修改实现，并重新运行聚焦测试和全量测试。

- [ ] **步骤 5：核对提交边界**

```bash
git status --short
git log --oneline 7c5e1cb..HEAD
```

预期：只有任务相关改动；`.superpowers/` 保持未跟踪且未暂存。

### 任务 11：推送 GitHub、备份服务器、同步、pilot 与正式重建

**固定路径与边界：**

- 分支：`feature/manual-seg-eus-vessels-20260811`
- 服务器项目：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project`
- 服务器备份目录：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/backups`
- 新预处理目录：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/case_data_manual_eus_20260811`
- 新正式输出：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_manual_seg_eus_vessels_20260811`
- 服务器连接：`ssh -p 35258 root@connect.westd.seetacloud.com`

- [ ] **步骤 1：推送已审查分支并核对哈希**

```bash
git push -u origin feature/manual-seg-eus-vessels-20260811
git rev-parse HEAD
git ls-remote origin refs/heads/feature/manual-seg-eus-vessels-20260811
```

预期：本地与 GitHub 分支哈希完全相同。

- [ ] **步骤 2：只读确认服务器项目边界与资源**

登录指定服务器后执行：

```bash
pwd
readlink -f /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
git -C /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project remote -v
git -C /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project status --short --branch
df -h /root/autodl-tmp
free -h
nvidia-smi
```

预期：路径严格指向本项目，Git 远程正确；同步前可用磁盘不少于先前观测的约 50 GB；不得进入或读取无关项目目录。

- [ ] **步骤 3：同步代码前完成服务器压缩备份**

```bash
stamp=$(date +%Y%m%d_%H%M%S)
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731
tar -czf "backups/project_backup_${stamp}.tar.gz" project run
sha256sum "backups/project_backup_${stamp}.tar.gz" \
  | tee "backups/project_backup_${stamp}.tar.gz.sha256"
tar -tzf "backups/project_backup_${stamp}.tar.gz" >/dev/null
```

预期：压缩包校验退出码为 `0`，同时获得 SHA-256；不得修改任何既有数据或输出目录。

- [ ] **步骤 4：只同步批准分支并上传 segmentation**

服务器项目 fetch 后只快进到已推送分支。将本地源分割复制到新病例数据目录，并验证：

```bash
sha256sum \
  '/root/autodl-tmp/ct_vascular_resampling_case2_20260731/case_data_manual_eus_20260811/source/EUS main organ---.seg(1).nrrd'
```

预期 SHA-256：

```text
0b56268488411925d96bb070e25e72a0105a8502e87ffd349a9ba01cd32dc124
```

- [ ] **步骤 5：预处理前运行服务器完整测试**

```bash
cd /root/autodl-tmp/ct_vascular_resampling_case2_20260731/project
/root/miniconda3/bin/mamba run \
  -n ct-vessel-resampling-totalseg-gpu \
  pytest -q
```

预期：完整测试通过。

- [ ] **步骤 6：生成手工器官输入与正式病例配置**

使用新预处理 CLI，输入固定为：

- CT：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project/case_data/ct/ct_venous.nrrd`
- segmentation：步骤 4 上传并校验的源文件
- 外部 artery：`project/case_data/models/artery_tree.ply`
- 外部 vein：`project/case_data/models/vein_tree.ply`
- 检索模块：`project/registration/2021.py`
- 输出根：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_manual_seg_eus_vessels_20260811`

以服务器现有、已审计的部分旋转病例配置为基准创建一个新的服务器专用 YAML，只允许以下变化：器官网格路径改为新手工器官、增加 `manual_segmentation`、改为新输出根、`black_ratio_limit` 改为 `0.60`。采样点上限、10 mm 间距、E1/E2 十二指肠端点、三轴旋转和原血管路径不得改变。把新旧配置 diff 保存到本项目 `run/`。

- [ ] **步骤 7：先执行 dry-run，再执行小规模 pilot**

dry-run 必须确认实际采样点数和姿态数与当前已批准的部分旋转配置一致。pilot 使用独立 case ID 和独立输出目录，每个区域点数上限设为 `2`，`workers: 4`、`gpu_batch_size: 8`，运行完整流程。

逐条检查 pilot Gallery：两张新图片存在、颜色准确、开放/触边血管无特征但两图均显示、器官像素语义正确、元数据与汇总一致、原血管字段和原三类图像不变。

- [ ] **步骤 8：以内存受控参数启动正式任务**

正式任务初始使用 `workers: 8`、`gpu_batch_size: 8`，通过独立 `setsid screen` 会话运行。主日志、PID、退出码和资源 CSV 只写入本项目 `run/`。严禁修改原 18 GB Gallery 和约 7.8 GB 临时 Gallery。

- [ ] **步骤 9：持续监控直到明确终态**

定期记录：`nvidia-smi`、`free -h`、`df -h`、进程 RSS、manifest 行数、Gallery 行数和新日志错误。出现内存压力、磁盘不足、几何不一致、schema 错误或非零退出时，只停止本次新任务，并先报告精确证据，不擅自更改业务参数。

- [ ] **步骤 10：对正式结果执行流式验收**

退出码为 `0` 后，不把完整 JSONL 一次性载入内存，逐项验证：

1. manifest 数等于总姿态数，也等于四状态数量之和；
2. Gallery JSONL 行数等于 `ct`、`boundary_only`、`ct_overlay`、`organ_vessel_boundary`、`eus_vessel_boundary`、`ct_eus_vessel_overlay` 各自 PNG 数量；
3. 每条 Gallery 同时具有器官 schema、EUS 血管 schema 和有效文件路径；
4. 器官/EUS 标签只能使用批准的规范名和映射；
5. EUS 特征只来自完整、不触边分量，且其标签必须出现在当前切面；
6. pilot/正式抽检发现的开放或触边分量虽然无特征，但两图中仍有相应颜色边界；
7. 原血管字段与原三类图像继续保持历史语义；
8. `run_metadata.json` 为 `complete`，输入哈希、构建提交和阈值 `0.60` 正确；
9. `library_summary.json` 与重新流式统计完全一致；
10. 原 Gallery 和临时 Gallery 的计数、mtime 未改变。

- [ ] **步骤 11：整理最终可审计证据**

最终报告必须记录：本地/GitHub/服务器 commit、服务器备份路径与 SHA-256、segmentation/CT/外部血管 SHA-256、dry-run/pilot/正式命令、采样点/姿态/四状态数量、输出大小、资源峰值、测试总数和最终输出路径。未经用户单独批准，不合并到 `main`。

## 三、执行方式与审核门槛

本计划提交用户审核前，不修改生产代码、不推送远程、不登录服务器执行变更、不启动重采样。用户批准后优先采用当前会话内联执行，按任务逐项 TDD，并在模块提交点报告结果；只在最终质量审查阶段使用一个审查代理，符合“不要开过多子代理”的要求。
