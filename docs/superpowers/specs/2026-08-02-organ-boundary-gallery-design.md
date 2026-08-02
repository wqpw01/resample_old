# TotalSegmentator 器官边界图库扩展设计

## 目标与输入

病例原始输入保持为 CT 与一个可同时包含器官、血管标签的 NRRD 分割体。配置只声明动脉和静脉标签；其他上传标签全部忽略。器官网格始终由输入 CT 经 TotalSegmentator 生成，不使用上传分割中的器官标签。

`vessel_label_values` 新格式只要求 `artery` 和 `vein`，后续病例默认示例为 `artery: [1]`、`vein: [2, 3]`。旧格式可继续提供 `portal`，其标签仍并入静脉。

TotalSegmentator 继续生成采样算法必需的 14 类结构。自动入口增加可选缓存目录；14 个掩膜均存在、非空且与 CT 的 Size、Spacing、Origin、Direction 一致时复用，否则重新分割。当前远端病例使用既有有效缓存。

## 截面与渲染

采样姿态、CT 重采样、FOV 排除、50% 黑色阈值、血管截面和图库状态判定不变。只有质量合格且具有完整血管特征、最终进入 `gallery` 的帧才计算器官截面。

器官边界只包含 11 类非血管结构：左右肾上腺、十二指肠、食管、胆囊、左右肾、肝脏、胰腺、脾脏和胃。主动脉、下腔静脉、门静脉及脾静脉不进入器官层。每个网格先通过平面和方形投影包围盒测试，再调用现有精确网格截面算法。

现有 `boundary_only/<sample_id>.png` 继续表示白底血管边界图。新增 `organ_vessel_boundary/<sample_id>.png`，使用纯白 RGB 背景，先画器官、后画血管，默认 300 px 输出仍使用 2 px 线宽。血管沿用配置颜色；器官使用固定调色板：

| 器官 ID | RGB |
|---|---|
| `adrenal_gland_left` | `[31, 119, 180]` |
| `adrenal_gland_right` | `[174, 199, 232]` |
| `duodenum` | `[44, 160, 44]` |
| `esophagus` | `[152, 223, 138]` |
| `gallbladder` | `[188, 189, 34]` |
| `kidney_left` | `[214, 39, 40]` |
| `kidney_right` | `[255, 152, 150]` |
| `liver` | `[140, 86, 75]` |
| `pancreas` | `[148, 103, 189]` |
| `spleen` | `[227, 119, 194]` |
| `stomach` | `[127, 127, 127]` |

## 输出协议与兼容性

每条 gallery 记录在原有位姿和 `features` 外增加：

- `organ_vessel_boundary_png`: 相对于 `gallery/` 的组合图路径。
- `organ_labels`: 当前方形内具有有效可见截面的器官 ID，排序并去重。被方形边缘裁剪但仍有正面积的轮廓也算出现。

不保存器官轮廓点、中心或面积。器官标签不混入血管 `features`，不改变 `gallery/unindexed` 路由、`2021.py` 适配器或检索规则。`unindexed`、`rejected` 和 `excluded_fov` 不新增图片或字段；`excluded_fov` 继续只保存黑色填充的灰度 CT。

`library_summary.json` 增加 gallery 帧级器官标签计数和调色板。恢复旧图库时若 gallery 记录缺少新增字段或组合图，明确报错并要求使用新输出目录，防止静默跳过。

## 远端交付

代码在功能分支以测试驱动实现并审查，随后本地合并到 `main`、推送 `origin/main`，服务器项目使用快进同步且保留现有未跟踪病例数据。

远端先做真实数据小规模试跑，再使用现有 TotalSegmentator 缓存和 `vascular_labels.nrrd` 在 `/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_organs_v1/case_2/` 全量重建。旧输出与旧 ZIP 均保留；受剩余磁盘限制，不生成新的全量 ZIP。任务持续监控到结束，并校验四类状态计数、JSONL、四套 gallery PNG、器官标签范围、旧检索加载和抽样图像。
