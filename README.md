# CT 血管重采样图库

输入同一物理坐标系内的 CT、完整器官网格以及 portal/hepatic 或 artery/vein 血管网格，生成无 P/N/D 的重采样 CT 图库和 `2021.py` 兼容检索特征。

采样几何的唯一设计基准是 `基于目标器官的采样方法-20260806.docx`，SHA-256 为 `4b27aee1a6db1680e501f17bd3492a571bd169c0bf7004d79b4a512d929cc53b`。项目说明文档仅描述实现，不替代核心设计。生产输入显式声明为 LPS，内部计算、PLY、JSONL 和检索位姿统一为 RAS 毫米物理坐标。

## 交付包结构与职责

交付包包含可运行的项目代码、通用配置模板、环境定义和兼容的 `2021.py`；不包含病例 CT、器官或血管模型、重采样结果、TotalSegmentator 权重和测试目录。

```text
ct_vascular_resampling/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── environment*.yml
├── configs/
├── registration/
├── scripts/
└── src/ct_vascular_resampling/
```

### 顶层文件

- `main.py`：命令行入口。`--case-config` 使用已有器官与血管网格重采样；`--auto-case-config` 先调用 TotalSegmentator 自动生成器官模型，再进入同一重采样和建库流程。
- `README.md`：输入要求、环境安装、运行命令、输出结构和检索接入说明。
- `requirements.txt`：pip 依赖，包含 SimpleITK、网格处理、PyTorch 和 TotalSegmentator。
- `pyproject.toml`：Python 包元数据和测试发现配置。
- `environment.yml`、`environment.gpu.yml`、`environment.autodl.gpu.yml`：Conda/Mamba 环境方案。
- `environment.totalseg.gpu.yml`：自动器官分割的推荐 GPU 环境，包含 CUDA PyTorch、CuPy 和 TotalSegmentator。

### 配置、兼容引擎与脚本

- `configs/case.example.yaml`：已有 CT、器官网格、血管网格时的标准病例配置模板。
- `configs/auto_case.example.yaml`：只有 CT 与一份混合 NRRD/NIfTI 标签体时的模板。只需填写 artery、vein 的数值标签；旧配置可保留 portal，它会自动合并至 `vein_tree`。
- `registration/2021.py`：兼容检索引擎，提供 `FeatureVector`、`VesselTriplet`、`ProbePose`、`MultiLabelledCBIR` 和 `HMMPoseEstimator`。
- `scripts/preprocess_slicer_case.py`：从 DICOM CT 与 3D Slicer `.seg.nrrd` 提取对齐掩膜、导出 NRRD 与物理空间 PLY 网格，并生成可运行的内部病例 YAML。
- `scripts/preprocess_manual_segmentation_case.py`：只从手工 `.seg.nrrd` 导出 14 类器官掩膜/网格，在病例配置中引用既有重建 `artery_tree.ply` 和 `vein_tree.ply`；不会从标签图重建或覆盖正式血管模型。

### 主程序模块

- `auto_preprocessing.py`：读取 CT 与混合标签体，只提取配置的动静脉标签，并调用或复用 TotalSegmentator 生成器官掩膜和 PLY；旧配置中的 portal 会并入 `vein_tree`。
- `preprocessing.py`：校验 Size、Spacing、Origin、Direction；提取二值掩膜，使用 Marching Cubes 导出物理空间网格，并写出预处理清单和内部病例配置。
- `sampling.py`、`sampling_pipeline.py`、`centerline.py`、`squares.py`：目标器官射线筛选、区域内 10 mm 约束 FPS、十二指肠中心线、局部坐标和语义三轴姿态。
- `ct_resampling.py`、`resampling_backend.py`：按世界坐标对 CT 插值，提供参考 CPU 与经校验的可选 GPU 后端。
- `label_resampling.py`、`manual_segmentation.py`：使用与 CT 完全相同的世界坐标方形做最近邻标签采样，并从一个原始标签平面派生器官上下文和独立三类 EUS 血管结果。
- `rendering.py`：生成 CT 图、仅血管边界图、CT 血管叠加图和器官/血管组合边界图。
- `quality.py`：按黑色区域占比和直线黑边筛选不合格图像。
- `gallery.py`、`artifacts.py`：将样本写入 `gallery/`、`rejected/`、`unindexed/`，生成 JSONL 清单，并为 Gallery 记录版本化的 EUS 候选器官元数据。
- `eus_organs.py`、`data/eus_possible_organs.json`：校验随包发布的 EUS 器官白名单、双身份血管映射和白名单来源哈希。
- `registration_adapter.py`：读取 `gallery.jsonl`，恢复 `2021.py` 可直接使用的检索对象。
- `scripts/recover_interrupted_run_metadata.py`：在 SIGHUP 等信号中断且 `run_metadata.json` 缺失时，严格校验已有清单并重建恢复协议；不改写 PNG、PLY 或 JSONL。
- `config.py`、`cli.py`、`pipeline.py`：配置解析、参数覆盖、日志和全流程调度。
- `geometry.py`、`mesh_io.py`、`logging_utils.py`：几何计算、网格读取和日志基础设施。

### 重采样结果与检索入口

一个完整病例输出为 `<output_root>/<case_id>/`：`gallery/` 保存可检索图像与 `gallery.jsonl`，`unindexed/` 保存无血管截面但质量合格的图像，`rejected/` 保存黑色区域或直线黑边不合格图像。旧网格模式默认仅在黑色像素比例超过 50% 时按比例拒绝；本次正式手工标签病例配置固定为 `black_ratio_limit: 0.60`，只有严格大于 60% 才按比例拒绝，恰好 60% 保留。若同一张图同时满足黑色占比阈值与直线边界规则，`quality.reason` 固定为 `black_boundary_line`，并用 `quality.black_ratio_exceeded=true` 保留比例超限证据。`excluded_fov.jsonl` 单独记录任一方形顶点超出 CT 原始物理 FOV 的样本，包含方形世界坐标和连续体素索引诊断；对应灰度图写入 `excluded_fov/ct/<sample_id>.png`，超出 FOV 的像素强制为纯黑，不生成血管边界图或叠加图。`manifest.jsonl` 汇总上述四种状态。`library_summary.json` 记录血管特征统计、通用器官与 EUS 候选器官的逐切面计数、完整 EUS 标签映射和器官颜色图例。`run_metadata.json` 还记录 EUS 白名单 schema、SHA-256、规范标签、胆管排除项和门静脉几何来源，连同 RAS、核心设计哈希、构建 Git commit、输入文件 SHA-256、点距、中心线参数与人工端点选择证据、三轴角度、方形尺寸、输出分辨率、三次插值、WL/WW、FOV 填充值、质量/FOV 策略、实际后端和四状态计数共同形成可恢复运行协议。

每个 gallery 帧继续生成 `ct/`、白底仅血管 `boundary_only/` 和 CT 血管叠加 `ct_overlay/`，并生成白底 `organ_vessel_boundary/`。组合图先绘制 14 类通用器官结构，再由原血管层按原配置颜色覆盖；其中腹主动脉、下腔静脉和门静脉具有“器官+血管”双重身份，但原血管求交、颜色、`features`、`boundary_only` 和 `ct_overlay` 均不改变。门静脉的器官轮廓复用现有 `portal_vein_and_splenic_vein` 组合网格，因此该规范标签表示该组合几何在切面内形成的正面积轮廓。

Gallery JSONL 中，旧网格模式的 `organ_labels` 继续按网格正面积截面产生。启用 `manual_segmentation` 后，器官语义改为标签图最近邻采样结果：有限方形内只要至少有 1 个该器官像素就加入排序去重的 `organ_labels`，不要求闭合轮廓；整张方形位于器官内部时仍有标签，但不会假定方形外为背景，也不会沿图像四边画人为边框。`eus_candidate_organ_labels` 是 `organ_labels` 与随包白名单的严格交集。胃、食管和胆囊可保留在 `organ_labels`，但不进入 EUS 候选字段；`bile_duct` 与 `common_bile_duct` 显式禁止进入白名单，胆囊不等于胆管。该字段只提供建库元数据，不在本流程中执行检索初筛。器官标签始终不写入原血管 `features`，因此不改变图库状态或 `2021.py` 的原血管检索语义。`unindexed`、`rejected` 和 `excluded_fov` 不生成组合图，也不写上述器官字段。

### 手工标签与独立 EUS 三类血管

手工模式固定将 Ao `8` 归为 `aorta`（红色 `[255,0,0]`）、IVC `9` 归为 `inferior_vena_cava`（蓝色 `[0,0,255]`），并将 SMV/SV/Left PV/Right PV/PV confluence/PV branch `26,33,34,35,36,37` 合并为 `portal_vein`（紫色 `[170,85,255]`）。Main Portal Vein `23` 只进入器官门静脉联合，不进入新 EUS 血管门静脉；SMV `26` 同时属于二者。

这条新链路不改变原重建 PLY 的 `features`、`boundary_only_png`、`ct_overlay_png` 或 gallery/unindexed/rejected 状态路由。每条原规则 Gallery 记录另外写入 `eus_vessel_labels`、`eus_vessel_features`、白底 `eus_vessel_boundary_png` 和 `ct_eus_vessel_overlay_png`。三类血管按 8 邻域分析；开放、截断或触及方形四边的不完整分量不进入 `eus_vessel_features`，但只要有可见标签像素，仍进入 `eus_vessel_labels`，并保留在两张新边界图中。新 EUS 特征不会把原本的 `unindexed` 切面提升为 Gallery。

断点恢复会检查四种状态的每条已完成记录、当前完整姿态 ID 集合、四顶点和位姿字段，并比较配置、输入 SHA-256、核心设计哈希、EUS 白名单协议和构建 Git commit。`run_metadata.json` 在首个待处理切面前原子写入 `run_state: running`，正常完成为 `complete`，普通异常为 `interrupted`。恢复仍会读取器官网格以重建姿态计划，并流式哈希输入；没有待处理姿态时不会把 CT 解码为体数据、不会加载血管渲染网格或初始化插值后端。缺少新器官字段的旧 Gallery 不兼容本 schema，必须使用新的输出根目录重新建库；旧图库或任何协议/几何不一致都会明确报错，拒绝前不会改写既有 PLY。

若进程因 `SIGHUP` 直接终止，且已有 JSONL 清单完整但缺少运行元数据，必须先执行受控重建：

```bash
python scripts/recover_interrupted_run_metadata.py \
  --case-config configs/case.yaml \
  --expected-completed-count 88280 \
  --reason sighup \
  --exit-code 129
```

该入口拒绝覆盖已有元数据，要求完成数、状态清单、姿态 ID、四顶点和位姿字段全部一致，并只原子写入带 `recovery_history` 的 `interrupted` 元数据。它不会修复或追加任何 JSONL；状态清单缺失、计数不符、陈旧姿态或几何不符都会失败。若恢复实现提交不同于已完成记录，元数据同时记录 `compatible_completed_build_git_commits`、`completed_build_git_commits` 和 `recovery_build_git_commit`；旧记录保留原提交，新记录写入真实恢复实现提交，不以环境变量伪装版本。恢复任务必须在独立 `setsid screen` 会话中运行。

### Interrupted-run recovery

The pipeline writes an atomic `run_metadata.json` checkpoint before the first pending slice. A normal run ends with `run_state: complete`; an ordinary exception leaves `interrupted`. After a `SIGHUP` exit such as code 129, run `scripts/recover_interrupted_run_metadata.py` only after verifying the expected manifest count. The recovery command validates the complete pose plan, geometry, input hashes and design protocol, writes no PNG/PLY/JSONL, and records both the old completed build commits and the recovery implementation commit. Resume the job inside a detached `screen` session so an SSH disconnect cannot terminate the worker.

### EUS organ metadata (English)

In legacy mesh mode, Gallery organ labels continue to require positive-area mesh intersections. With `manual_segmentation` enabled, a nearest-neighbor label plane is sampled on exactly the same world-coordinate square as CT, and one sampled organ pixel is sufficient for `organ_labels`; contour closure is not required. A square fully inside an organ is labelled without drawing an artificial border around the image. The original reconstructed-PLY `features`, `boundary_only_png`, `ct_overlay_png`, and state routing remain unchanged.

Manual mode also writes separate three-class EUS vessel metadata and images. Components that are open, clipped, or touch an image edge are excluded from `eus_vessel_features`, but their visible boundaries remain in both `eus_vessel_boundary_png` and `ct_eus_vessel_overlay_png`. Main Portal Vein label 23 belongs only to the organ union; SMV 26 and labels 33-37 form the EUS portal-vein class. The formal manual case rejects by black ratio only when it is strictly greater than 0.60. Pre-schema Gallery outputs must be rebuilt under a new output root.

供 `2021.py` 检索时只加载 `gallery/gallery.jsonl`；`registration_adapter.load_gallery_database()` 会将其中的特征和方位转换为 `FeatureVector`、`VesselTriplet` 与 `ProbePose`，无需将 JSONL 转换为另一种文件格式。

## 环境

```bash
mamba env create -f environment.yml
mamba activate ct-vessel-resampling
```

## 运行

先基于 `configs/case.example.yaml` 填写病例路径与模型映射。所有模型必须是带三角面的 OBJ、STL 或 PLY，并与 NIfTI/NRRD CT 的原生 LPS 物理坐标一致。若输入已经是 RAS，必须在 `geometry.input_coordinate_system` 中显式声明。

`ct_path` 也可指向 DICOM 目录；目录有多个序列时，必须在病例 YAML 顶层填写 `dicom_series_uid`，以避免误选不同重建层厚或 Scout。

```bash
python main.py --case-config configs/case.yaml --dry-run
python main.py --case-config configs/case.yaml --workers 8
```

## 核心采样与三轴姿态

- 胃、食管和十二指肠沿表面外法线发出最长 100 mm 射线，只保留命中 11 个目标结构之一的候选；射线按默认 2048 条分块，分块只降低内存峰值，不改变几何结果。
- 配置点数是区域上限，不补点。每个区域独立执行确定性 FPS，任意两个实际采样点的 RAS 欧氏距离至少 10 mm；“采样点间隔 1 cm”不表示切面内部像素间距。
- 食管有效段从原始食管最小 z 到肝脏最大 z，并按完整有效段 z 跨度向下复制。肝脏一、二区合并采样时逐点保留 `liver_region_one`、`liver_region_two` 或重叠组合来源。十二指肠以 1 mm 体素骨架得到近端到远端中心线，中心线只用于定义十二指肠 0 度面和 10 mm 弦切向，不改变 CT 或表面采样点。若自动短毛刺规则无法得到无分叉路径，必须经人工解剖确认后以病例专属 RAS 端点锚点选取未剪枝树中的唯一路径；不得通过任意放大毛刺阈值继续运行。
- 局部坐标为右手系，采样点位于 100 mm 方形底边中心。旋转采用局部内禀 `Z(偏航) -> Y(俯仰) -> X(滚动)`，矩阵为 `B' = B Rz Ry Rx`。
- 2026-08-08 用户批准受控部分恢复旋转范围：滚动为 `-15..+15` 度、俯仰为 `-10..+10` 度，步长均为 5 度；普通区、十二指肠球部和胰腺特殊区偏航仍分别为 `-30..+30`、`-90..+90`、`-120..+30` 度，步长均为 5 度。因此每点分别生成 455、1295、1085 个姿态。该范围是对核心 DOCX 完整滚转/俯仰范围的受控折中，不改写核心文件中的完整原范围。
- 四顶点在 `rtol=0, atol=1e-9 mm` 下完全相同时强制去重，基础区域优先；该行为不可通过配置关闭。样本 ID 编码来源点和 roll/pitch/yaw，支持稳定断点恢复。

### Controlled rotation range (English)

As approved on 2026-08-08, roll uses `-15..+15 degrees` and pitch uses `-10..+10 degrees`, both at 5-degree steps. The standard, duodenal-bulb, and pancreas-special yaw policies remain unchanged, producing 455, 1295, and 1085 poses per point. This is a controlled partial restoration; it does not rewrite the complete ranges in the core DOCX and does not change sampling points or the 10 mm minimum spacing.

历史旧角度证据：2026-08-07 在当时滚动/俯仰均为 `-5/0/+5` 度的合同下，使用导出的 case 2 真实器官网格只读验收得到 319 个采样点和 43,695 个姿态，其中普通 33,345、胰腺特殊 5,022、十二指肠球部 5,328；85 个肝点的来源为一区 33、一区/二区重叠 12、二区 40。病例 2 的完整十二指肠骨架为 190 节点/189 边的单连通树；用户按三平面可视化确认 E1 `[19, 24, 700]` 到 E2 `[-33, 1, 664]` 为主路径，选中 166 点、长度 224.7411 mm。以上仅是 2026-08-07 旧角度合同下本次输入与实现的历史验收结果，不是当前姿态数量、其他病例的固定数量或核心 DOCX 原文参数。

`gallery/` 仅包含具备完整血管截面特征的可检索样本，支持 portal/hepatic 或 artery/vein 标签对；`unindexed/` 保留无血管截面的合格图像；`rejected/` 保留黑色区域或直线黑边不合格样本。使用 `ct_vascular_resampling.registration_adapter.load_gallery_database()` 可将 `gallery.jsonl` 载入外部 `2021.py` CBIR 实现。

## Rejected FOV 审计

当前管线会先按方形顶点判断 CT FOV 越界，再使用所选 CPU/GPU 后端生成诊断 CT，并把逐像素越界区域强制填黑。此类样本固定写入 `excluded_fov.jsonl` 和 `excluded_fov/ct/`，不会进入质量筛选、血管求交或 `rejected/`。对于历史上已经生成的 `rejected/rejected.jsonl`，仍可使用原始 NRRD/NIfTI 或指定 DICOM Series 复算每个方形的连续体素坐标，区分 CT FOV 外常量填充与 CT 范围内低 HU/空气。基于 `configs/rejected_audit.example.yaml` 填写路径后运行：

```bash
python main.py --rejected-audit-config configs/rejected_audit.yaml
```

审计在 `output_directory` 写入全量 `rejected_fov_audit.jsonl`、`summary.json`、`summary.csv`，并按原因保存有限数量的代表性越界掩码与叠加图。它不会移动或改写原 `rejected`、`gallery`、`unindexed` 内容。运行时新产生的 rejected 记录也会在其 JSONL 中附加 `fov_diagnostics`。

### 提取历史 FOV 边界样本

若需要单独复核历史结果中“原先为 `black_ratio`、审计后为 `fov_boundary_aligned`”的样本，可运行：

```bash
python scripts/extract_fov_boundary_samples.py \
  --library-root /path/to/case_2 \
  --audit-jsonl /path/to/case_2/rejected/diagnostics/rejected_fov_audit.jsonl \
  --destination /path/to/black_ratio_fov_boundary_aligned_5 \
  --limit 5
```

每个样本子目录包含 `ct.png`、`ct_overlay.png`、`boundary_only.png`、可用时的 FOV 越界掩码/叠加图以及 `metadata.json`。输出根目录的 `locations.csv` 保存探头点、方形四顶点、连续体素索引范围和越界比例。

## 自动器官预处理

自动病例入口只接收 CT 和一份可同时包含器官、血管标签的 NRRD/NIfTI 标签体。`vessel_label_values` 只配置 `artery` 和 `vein`，例如 `artery: [1]`、`vein: [2, 3]`；未配置的器官或其他数值标签全部忽略。旧三键配置仍可额外提供 `portal`，其标签会先并入 `vein`。器官始终来自 TotalSegmentator 的 `total` 任务，不从上传标签体提取。病例配置可从 `configs/auto_case.example.yaml` 开始：

```bash
python main.py --auto-case-config configs/auto_case.yaml --backend auto
```

`totalsegmentator.cache_directory` 可指向已有分割目录，相对路径按自动病例 YAML 所在目录解析。程序仅在 14 个必需掩膜全部存在、非空，且 Size、Spacing、Origin、Direction 均与 CT 一致时跳过 TotalSegmentator；缓存缺失或无效时会向该目录重新生成并再次严格校验。未配置时仍使用 `<output_root>/<case_id>/preprocessing/totalsegmentator/`。预处理 `manifest.json` 的 provenance 会记录实际 `cache_directory`、`cache_reused`、计划命令及 `command_executed`。

自动入口生成的内部病例配置固定使用 100 mm 方形和强制严格去重；旧 `square.deduplicate_degenerate_edge_angles` 已失效并会被明确拒绝，不能与新输出协议混用。

`registration_module_path` 是既有 `2021.py` 检索引擎代码路径，不属于病例影像输入。首次运行会下载 TotalSegmentator 权重。建议在运行前设置 `TOTALSEG_HOME_DIR` 到持久磁盘路径；这与上述分割掩膜缓存相互独立。GPU 环境使用 `environment.totalseg.gpu.yml` 创建；自动输出位于 `<output_root>/<case_id>/preprocessing/`，原有 `gallery/`、`rejected/`、`unindexed/` 目录骨架保持兼容，gallery JSONL 按上文新增器官字段。

## 二维裁剪标签检索特征

对于 `picked_10cm_cropped` 这类每帧含 `<frame>_cropped_jpg_Label.tar` 的二维 100 mm 裁剪标签，可批量提取 artery/vein 连通域特征：

```bash
PYTHONPATH=src python scripts/extract_cropped_retrieval_features.py \
  --root /path/to/picked_10cm_cropped
```

脚本优先读取 TAR 中的单层 NIfTI 标签图，将 ID 26-32 归为 `vein`、ID 33-40 归为 `artery`，仅保留不触及图像边界的完整连通域。仅有 JSON 而没有 NIfTI 的空标签 TAR 会生成 `unindexed` 空特征记录。每个帧目录只会得到 `<frame>_cropped_retrieval_features.json` 和 `<frame>_cropped_gallery.jsonl`，不会在根目录生成汇总结果文件。

任一具有特征的 `<frame>_cropped_gallery.jsonl` 可由 `registration_adapter.load_gallery_database()` 加载到 `2021.py` 的 `MultiLabelledCBIR`。它使用 100 mm x 100 mm 二维裁剪平面生成合成姿态，适用于血管特征检索；不含患者三维世界位姿，不能作为 `HMMPoseEstimator` 的三维导航数据。

## Slicer 分割预处理

对于 DICOM CT 和 3D Slicer `.seg.nrrd`，先导出物理空间一致的 NRRD CT、二值掩膜和 PLY 网格，再使用生成的 `case_preprocessed.yaml` 运行主流程：

```bash
python scripts/preprocess_slicer_case.py \
  --dicom-dir /path/to/dicom \
  --segmentation /path/to/segmentation.seg.nrrd \
  --output /path/to/preprocessed \
  --series-id <venous-series-uid> \
  --registration-module /path/to/2021.py
```

病例 2 的动脉和静脉标签合并规则固定在 `ct_vascular_resampling.preprocessing` 中；未指定 `--series-id` 时，脚本只会接受唯一匹配 `2.0 x 2.0_V` 的序列。
