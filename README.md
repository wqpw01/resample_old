# CT 血管重采样图库

输入同一物理坐标系内的 CT、完整器官网格以及 portal/hepatic 或 artery/vein 血管网格，生成无 P/N/D 的重采样 CT 图库和 `2021.py` 兼容检索特征。

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
- `configs/auto_case.example.yaml`：只有 CT 与一份三类血管标签图时的模板。需填写 artery、vein、portal 的数值标签；portal 会自动合并至 `vein_tree`。
- `registration/2021.py`：兼容检索引擎，提供 `FeatureVector`、`VesselTriplet`、`ProbePose`、`MultiLabelledCBIR` 和 `HMMPoseEstimator`。
- `scripts/preprocess_slicer_case.py`：从 DICOM CT 与 3D Slicer `.seg.nrrd` 提取对齐掩膜、导出 NRRD 与物理空间 PLY 网格，并生成可运行的内部病例 YAML。

### 主程序模块

- `auto_preprocessing.py`：读取 CT 与三类血管标签图，调用 TotalSegmentator 生成器官掩膜和 PLY；输出 `artery_tree`，并将 vein 与 portal 合并为 `vein_tree`。
- `preprocessing.py`：校验 Size、Spacing、Origin、Direction；提取二值掩膜，使用 Marching Cubes 导出物理空间网格，并写出预处理清单和内部病例配置。
- `sampling.py`、`sampling_pipeline.py`、`squares.py`：器官表面候选筛选、固定随机种子的 FPS、100 mm x 100 mm 方形采样面及 27 个局部姿态。
- `ct_resampling.py`、`resampling_backend.py`：按世界坐标对 CT 插值，提供参考 CPU 与经校验的可选 GPU 后端。
- `rendering.py`：生成 CT 图、仅血管边界图和叠加图。
- `quality.py`：按黑色区域占比和直线黑边筛选不合格图像。
- `gallery.py`、`artifacts.py`：将样本写入 `gallery/`、`rejected/`、`unindexed/`，并生成 JSONL 清单和检索库摘要。
- `registration_adapter.py`：读取 `gallery.jsonl`，恢复 `2021.py` 可直接使用的检索对象。
- `config.py`、`cli.py`、`pipeline.py`：配置解析、参数覆盖、日志和全流程调度。
- `geometry.py`、`mesh_io.py`、`logging_utils.py`：几何计算、网格读取和日志基础设施。

### 重采样结果与检索入口

一个完整病例输出为 `<output_root>/<case_id>/`：`gallery/` 保存可检索图像与 `gallery.jsonl`，`unindexed/` 保存无血管截面但质量合格的图像，`rejected/` 保存黑色区域或直线黑边不合格图像；`manifest.jsonl` 汇总三类全部样本，`library_summary.json` 记录检索特征统计，`run_metadata.json` 记录后端与运行信息。

供 `2021.py` 检索时只加载 `gallery/gallery.jsonl`；`registration_adapter.load_gallery_database()` 会将其中的特征和方位转换为 `FeatureVector`、`VesselTriplet` 与 `ProbePose`，无需将 JSONL 转换为另一种文件格式。

## 环境

```bash
mamba env create -f environment.yml
mamba activate ct-vessel-resampling
```

## 运行

先基于 `configs/case.example.yaml` 填写病例路径与模型映射。所有模型必须是带三角面的 OBJ、STL 或 PLY，并与 NIfTI/NRRD CT 的原生物理坐标一致。

`ct_path` 也可指向 DICOM 目录；目录有多个序列时，必须在病例 YAML 顶层填写 `dicom_series_uid`，以避免误选不同重建层厚或 Scout。

```bash
python main.py --case-config configs/case.yaml --dry-run
python main.py --case-config configs/case.yaml --workers 8
```

`gallery/` 仅包含具备完整血管截面特征的可检索样本，支持 portal/hepatic 或 artery/vein 标签对；`unindexed/` 保留无血管截面的合格图像；`rejected/` 保留黑色区域或直线黑边不合格样本。使用 `ct_vascular_resampling.registration_adapter.load_gallery_database()` 可将 `gallery.jsonl` 载入外部 `2021.py` CBIR 实现。

## Rejected FOV 审计

对已生成的 `rejected/rejected.jsonl`，可使用原始 NRRD/NIfTI 或指定 DICOM Series 复算每个方形的连续体素坐标，区分 CT FOV 外常量填充与 CT 范围内低 HU/空气。基于 `configs/rejected_audit.example.yaml` 填写路径后运行：

```bash
python main.py --rejected-audit-config configs/rejected_audit.yaml
```

审计在 `output_directory` 写入全量 `rejected_fov_audit.jsonl`、`summary.json`、`summary.csv`，并按原因保存有限数量的代表性越界掩码与叠加图。它不会移动或改写原 `rejected`、`gallery`、`unindexed` 内容。运行时新产生的 rejected 记录也会在其 JSONL 中附加 `fov_diagnostics`。

## 自动器官预处理

自动病例入口只接收 CT 和一份含动脉、静脉、门静脉的 NRRD/NIfTI 血管标签图。它调用 TotalSegmentator 的 `total` 任务生成源采样规则所需器官掩膜与物理空间 PLY 网格；门静脉会与静脉合并为 `vein_tree`。病例配置可从 `configs/auto_case.example.yaml` 开始：

```bash
python main.py --auto-case-config configs/auto_case.yaml --backend auto
```

`registration_module_path` 是既有 `2021.py` 检索引擎代码路径，不属于病例影像输入。首次运行会下载 TotalSegmentator 权重。建议在运行前设置 `TOTALSEG_HOME_DIR` 到持久磁盘路径；已有权重缓存可直接复用。GPU 环境使用 `environment.totalseg.gpu.yml` 创建；自动输出位于 `<output_root>/<case_id>/preprocessing/`，原有 `gallery/`、`rejected/`、`unindexed/` 及 JSONL 输出不变。

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
