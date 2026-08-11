# 手工器官标签图与独立 EUS 血管重采样设计

## 目标与设计边界

病例 2 从 CT、用户指定的 3D Slicer segmentation 和既有重建血管模型完整重建一个新 Gallery。器官模型与切面器官信息只来自 `EUS main organ---.seg(1).nrrd`，不得调用或复用 TotalSegmentator 器官分割。原有重建血管切割继续使用 `artery_tree.ply` 和 `vein_tree.ply`，其状态路由、特征、颜色与图片保持原语义；新增一套独立的三类 EUS 血管结果，不与原 `features` 混合。

本次不改变核心采样点间距、采样点上限、零度面、三轴旋转、方形尺寸、CT 插值、坐标系或血管完整截面定义。黑色像素比例拒绝阈值由 50% 调整为 60%，其他质量和 FOV 规则不变。

## 已确认输入

- 服务器 CT：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/project/case_data/ct/ct_venous.nrrd`。
- 本地手工 segmentation：`C:\Users\zhangyutang\Desktop\CT-EUS定位项目\数据\血管重建病例2\EUS main organ---.seg(1).nrrd`。
- Segmentation SHA-256：`0b56268488411925d96bb070e25e72a0105a8502e87ffd349a9ba01cd32dc124`。
- 原重建血管：服务器现有 `artery_tree.ply` 和 `vein_tree.ply`。
- 检索模块：服务器本项目现有 `registration/2021.py`。
- 新正式输出根：`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_manual_seg_eus_vessels_20260811`。

CT 与 segmentation 均为 `512 x 512 x 132`，Spacing 均为 `[0.822265625, 0.822265625, 2.0]` mm，Direction 相同。Origin 仅有 NRRD 文本精度造成的亚微米差异，必须继续通过现有 `atol=1e-6 mm, rtol=0` 几何校验。运行时仍将 LPS 输入统一转换为 RAS 毫米物理坐标。

## 手工器官预处理

预处理只从 segmentation 导出 14 类器官掩膜和 PLY，不从 segmentation 生成正式 artery/vein 模型：

| 规范器官名 | Slicer 标签值 |
|---|---:|
| `spleen` | 1 |
| `kidney_right` | 2 |
| `kidney_left` | 3 |
| `gallbladder` | 4 |
| `esophagus` | 5 |
| `liver` | 6 |
| `stomach` | 7 |
| `aorta` | 8 |
| `inferior_vena_cava` | 9 |
| `pancreas` | 11 |
| `adrenal_gland_right` | 12 |
| `adrenal_gland_left` | 13 |
| `duodenum` | 14 |
| `portal_vein_and_splenic_vein` | 23, 26, 33, 34, 35, 36, 37 |

器官门静脉明确包含 Main Portal Vein 23、SMV 26、SV 33、Left/Right PV 34/35、PV confluence 36 和 PV branch 37。它可包含多个连通分量且不要求整体 watertight，因为器官是否出现在切面中不再依赖 PLY 截面闭合；PLY 仍用于目标器官射线与现有采样几何。

预处理配置引用既有重建 `artery_tree.ply` 和 `vein_tree.ply`，不得把 segmentation 中的动静脉标签写成正式血管模型。预处理清单记录 CT、segmentation、两个外部血管文件的路径、SHA-256、标签映射、体素数和器官网格统计。

## 单次标签图最近邻采样

每个切面使用与 CT 完全相同的四顶点、世界坐标网格和 `300 x 300` 分辨率：

- CT 使用现有三次 B-spline 插值。
- Segmentation 使用最近邻插值，FOV 外填 0，不执行预滤波。
- 一个原始 `uint8` 标签平面同时派生器官信息、器官边界和三类 EUS 血管结果；不得为每个器官复制一份三维标签体。
- CPU 与 GPU 最近邻结果必须逐像素相同。GPU 校验不通过时，`auto` 后端回退 CPU；强制 `gpu` 时拒绝运行。

标签体约 35 MB。GPU 每批新增输出约 `batch_size x 300 x 300` 字节，不长期保留历史标签平面。运行协议记录标签图 SHA-256、插值方法、原始标签映射和输出规范名。

## 器官出现与边界语义

先把原始标签平面映射到 14 个规范器官类，再执行以下规则：

- 当前有限方形内至少有 1 个该器官像素时，加入排序去重的 `organ_labels`。
- 是否形成闭合轮廓、是否与器官表面相交、截面面积大小均不作为标签前提。
- 整个方形位于器官内部时仍加入标签。
- 连续几何只相切、最近邻结果中没有该器官像素时不加入。
- `eus_candidate_organ_labels` 仍严格等于 `organ_labels` 与版本化 EUS 器官白名单的交集。

`organ_vessel_boundary` 的器官层从规范器官标签平面提取实际像素边界。只检查图像内部相邻像素的类别变化，不假定方形外为背景，因此器官填满整个方形时不沿四边人为画框。原重建血管边界随后按原颜色覆盖器官层。胃、食管和胆囊可进入 `organ_labels`，但不进入 EUS 候选；胆管标签不参与器官映射。

## 独立三类 EUS 血管

新 EUS 血管只使用用户截图确认的八个 Slicer Segment，并归并为三类：

| EUS 血管规范名 | Slicer Segment | 标签值 | RGB |
|---|---|---:|---:|
| `aorta` | Ao | 8 | `[255, 0, 0]` |
| `inferior_vena_cava` | IVC | 9 | `[0, 0, 255]` |
| `portal_vein` | SMV, SV, Left PV, Right PV, PV confluence, PV branch | 26, 33, 34, 35, 36, 37 | `[170, 85, 255]` |

EUS 血管门静脉系不包含 Main Portal Vein 23。26、33–37 在二维标签平面先合并为一个二值类，再以与现有二维 EUS 特征脚本相同的 8 邻域执行连通域与边界提取，内部源标签交界处不绘制边界。

每张原规则 Gallery 切面新增：

- `eus_vessel_metadata_schema_version`: 固定为 `eus-vessel-metadata/v1`。
- `eus_vessel_labels`: 至少出现 1 个像素的三类规范名，排序去重。
- `eus_vessel_features`: 只包含在有限二维标签图中轮廓闭合、不接触首末行或首末列的完整连通域。
- `eus_vessel_boundary_png`: 三类边界共同叠加在一张白底 RGB 图，区域内部保持白色。这张图绘制所有可见边界，不得因轮廓开放、被方形视野截断或触及图像边缘而丢弃。
- `ct_eus_vessel_overlay_png`: 同一组未过滤边界叠加在 CT 灰度图；开放、截断和触边轮廓也必须绘制。

完整连通域特征继续使用方形局部毫米坐标：列方向为 `x_mm`，行方向为 `y_mm`，原点与现有特征相同。质心由连通域像素中心计算；面积为像素数乘以两个方向的像素间距。与现有二维 EUS 特征脚本保持一致，在这种填充标签像素语义下，连通域触及图像四边即表示其轮廓被有限方形视野截断、不闭合；该分量仍进入 `eus_vessel_labels`，也必须出现在两张新图中，但不进入 `eus_vessel_features`。不额外使用三维 PLY 的 watertight 状态替代这个二维完整性判定。

## 原血管结果与状态路由不变

以下字段和文件只由既有重建血管 PLY 产生：

- `features`
- `boundary_only_png`
- `ct_overlay_png`
- Gallery/unindexed/rejected 状态

新 `eus_vessel_features` 不与 `features` 合并，也不把原本的 unindexed 切面提升为 Gallery。新增 EUS 血管字段和图片只附加到按原规则进入 Gallery 的记录。相同输入切面在修改前后的原 `features`、`boundary_only` 和 `ct_overlay` 必须字节或结构完全一致。

## 质量规则

正式病例配置使用：

```yaml
filtering:
  black_threshold: 50
  black_ratio_limit: 0.60
  line_min_diagonal_fraction: 0.70
  black_side_min_ratio: 0.90
  valid_side_max_black_ratio: 0.10
```

黑色像素比例严格大于 60% 时因比例拒绝，恰好 60% 保留。直线黑边、FOV 排除、黑色像素定义和组合原因优先级不变。阈值进入运行协议哈希，不能向 50% 输出续写。

## 配置、摘要与恢复协议

普通病例 YAML 新增显式的“手工 segmentation 标签采样模式”，包含 segmentation 路径和两套标签映射。本次正式病例配置必须启用该模式；启用后所有字段均为必需，加载时拒绝缺失、重复、布尔值、非整数、空集合或不支持的规范名。器官门静脉与 EUS 血管门静脉允许按本设计共享 26、33–37，但必须分别记录。

未启用手工 segmentation 模式的既有配置保持原网格器官行为，不生成 `eus_vessel_*` 字段，也不能被报告为本设计的合规产物。GalleryWriter 根据运行配置选择并锁定记录 schema；同一输出根不得混用两种模式。TotalSegmentator 自动流程不包含本设计要求的 Slicer 血管标签，因此不得伪造三类 EUS 血管结果。

`run_metadata.json` 增加：

- segmentation 路径、SHA-256 与几何；
- 最近邻插值和 FOV 外标签值 0；
- 14 类器官映射；
- 三类 EUS 血管映射、颜色、完整连通域规则；
- 手工器官网格及外部重建血管来源；
- 60% 黑色阈值。

`library_summary.json` 增加 `eus_vessel_label_counts`、`eus_vessel_feature_counts`、三类颜色和源标签映射。Gallery 恢复校验要求新增 schema、字段、图片存在且标签/特征规范有效。任何 segmentation 哈希、映射、颜色、阈值、构建提交或几何变化均拒绝续写，要求新输出根。

## 测试与验收

TDD 覆盖：

- 手工器官预处理只生成器官并引用外部 artery/vein PLY；
- CT 与标签图几何不一致、标签缺失或外部血管缺失时拒绝；
- 器官门静脉包含 23、26、33–37；EUS 血管门静脉只包含 26、33–37；
- CPU/GPU 最近邻逐像素一致、FOV 外为 0；
- 一个像素即产生器官标签；零像素相切不产生；整面器官产生标签但不画方框；
- 三类 EUS 血管颜色、合并边界、开放/触边分量排除特征但保留在两张图中、毫米质心与面积；
- 原血管特征、两张原图和状态路由完全不变；
- 60% 恰好保留、超过 60% 拒绝；
- 新 Gallery schema、恢复拒绝、运行元数据和摘要计数。

本地完整测试与独立质量审查通过后推送新功能分支。服务器再次备份 `project/` 和 `run/`，同步后只在新目录生成手工器官预处理产物、pilot 和正式输出。正式任务在独立 `screen` 会话中运行并监控 GPU、RAM、磁盘和日志。最终验收流式核对所有状态清单、图片数量和路径、标签/特征映射、原血管不变性、运行协议及摘要；原有 18 GB 输出和 7.8 GB 临时 Gallery 不得修改。
