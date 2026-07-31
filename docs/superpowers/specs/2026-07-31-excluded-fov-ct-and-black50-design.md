# 黑色比例 50% 与 excluded_fov CT 输出设计

## 目标

本次修改包含两个行为变更：

1. 将黑色像素比例拒绝阈值从 30% 统一提高到 50%。
2. 保留 `excluded_fov` 独立状态，同时为每个 FOV 越界方形生成一张灰度 CT PNG；超出 CT 原始 FOV 的像素必须为纯黑。

本次修改不改变器官表面采样、方形姿态生成、血管特征定义、质量检测中的直线黑边规则，也不改变 `gallery`、`unindexed` 和 `rejected` 的职责。

## 配置行为

`FilterConfig.black_ratio_limit` 的默认值改为 `0.50`。通用示例配置和仓库内仍使用 `0.30` 的 Case 2 配置同步改为 `0.50`，保证未显式填写配置和复制现有配置时的行为一致。

黑色比例的比较语义保持不变：只有 `black_ratio > 0.50` 时拒绝，恰好 50% 不因比例规则拒绝。直线黑边仍独立检测；一张图同时超过黑色比例阈值并命中直线黑边时，拒绝原因仍优先记录为 `black_boundary_line`。

## excluded_fov 数据流

`excluded_fov` 的几何判定保持不变：将方形四个世界坐标顶点转换成 CT 连续体素索引，只要任一顶点超出合法索引范围，该样本就固定归入 `excluded_fov`。

与现有“判定后直接写清单并跳过插值”不同，未完成的越界样本将进入现有 CT 重采样后端：

1. CPU 或 GPU 后端按当前配置生成完整二维 HU 方形。
2. 使用 `diagnose_square_fov` 生成与输出分辨率一致的逐像素越界掩码。
3. 正常执行窗位窗宽映射，得到二维 `uint8` 灰度图。
4. 将越界掩码对应的像素强制设置为 `0`。
5. 将图像写入 `excluded_fov/ct/<sample_id>.png`。
6. 写入 `manifest.jsonl` 和 `excluded_fov.jsonl`，然后返回 `excluded_fov` 状态。

强制置零发生在窗位窗宽映射之后，因此越界区域始终为纯黑，不依赖 `fill_hu_value`、窗位或窗宽。CT FOV 内的像素保留正常插值和窗口化结果。

越界样本不进入质量筛选，不与血管网格求交，不提取检索特征，也不生成 `boundary_only` 或 `ct_overlay`。它不会进入 `gallery`、`unindexed` 或 `rejected`。

## 输出契约

病例目录新增以下实际文件目录：

```text
<output_root>/<case_id>/
└── excluded_fov/
    └── ct/
        └── <sample_id>.png
```

每条 `excluded_fov` 记录继续包含原有世界坐标、局部坐标轴、四顶点和 FOV 诊断信息，并新增：

```json
{
  "status": "excluded_fov",
  "ct_png": "ct/<sample_id>.png",
  "resampling_backend": "cpu 或 gpu"
}
```

`ct_png` 相对于 `excluded_fov/` 目录。记录中不出现 `boundary_only_png` 或 `ct_overlay_png`。

## 后端与运行元数据

越界样本使用与其他样本相同的批量重采样后端，避免为每个样本重复进行 CPU 三次样条预滤波，也避免 GPU 任务因越界样本退化成逐张 CPU 处理。

GPU 后端仍需通过现有 CPU 对照校准。`auto` 模式的回退规则和显式 `gpu` 模式的失败规则保持不变。`run_metadata.json` 中的 `excluded_fov_count` 在渲染完成后按实际状态计数写入，不能依赖插值前提前落盘的旧行为。

## 恢复与兼容性

断点恢复继续以 `manifest.jsonl` 中的 `slice_id` 为准。已经记录为完成的样本仍会跳过。

因此，旧版输出中已经存在 `excluded_fov` 清单但没有 CT PNG 的病例不会在恢复运行时自动补图。需要获得完整新输出时，必须使用新的 `output_root`，或在明确备份后重新创建病例输出目录。本次修改不加入自动迁移或补图命令。

## 测试与验收

实施采用测试先行，至少覆盖以下行为：

1. 未显式设置 `black_ratio_limit` 时默认值为 `0.50`。
2. 黑色比例为 40% 且未命中直线黑边的图像通过质量筛选。
3. 黑色比例大于 50% 的图像仍被拒绝。
4. 越界方形返回 `excluded_fov`，并生成 `excluded_fov/ct/<sample_id>.png`。
5. 输出 PNG 为二维灰度图，越界掩码内所有像素均为 `0`，FOV 内测试像素保留非零值。
6. `excluded_fov` 记录包含 `ct_png` 和实际重采样后端，不包含边界图或叠加图路径。
7. 越界样本不调用质量筛选和血管求交，不出现在其他三个状态目录中。
8. 完整测试套件通过，README 的输入输出和状态表与新行为一致。

## 非目标

本次不改变直线黑边阈值，不将 `excluded_fov` 合并进 `rejected`，不为越界样本生成血管特征，不增加历史输出迁移工具，也不修改 CT FOV 的几何判定方式。
