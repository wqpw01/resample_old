# 病例 2 DICOM 与 Slicer 分割预处理设计

## 目标

将 `血管重建病例2` 中的静脉期 DICOM CT 与 `.seg.nrrd` 分割，转换为 CT 血管重采样管线可直接使用的 NRRD CT、二值掩膜和 PLY 三角网格。结果写入病例目录下的 `预处理后`。

本预处理不执行方形采样、CT 切片渲染、质量筛选或检索建库。

## 已确认输入

- DICOM 目录：`1.2.840.78.85.7.5.1809089.1755084265`
- 选定序列：`2.0 x 2.0_V`，Series Instance UID 为 `1.2.156.112605.189250946070685.250813112346.3.8268.135835`
- 分割：`EUS main organ---.seg(1).nrrd`
- CT 与分割的 Size、Spacing、Direction 相同；Origin 在 `1e-6 mm` 容差内一致。

## 标签映射

### 器官模型

| 输出名 | 标签名 | 标签值 |
|---|---|---:|
| `spleen` | `spleen` | 1 |
| `kidney_right` | `right kidney` | 2 |
| `kidney_left` | `left kidney` | 3 |
| `gallbladder` | `GB` | 4 |
| `esophagus` | `esophagus` | 5 |
| `liver` | `liver` | 6 |
| `stomach` | `stomach` | 7 |
| `aorta` | `Ao` | 8 |
| `inferior_vena_cava` | `IVC` | 9 |
| `pancreas` | `pancreas` | 11 |
| `adrenal_gland_right` | `right adr` | 12 |
| `adrenal_gland_left` | `left adr` | 13 |
| `duodenum` | `duodenum` | 14 |

### 血管模型

`artery_tree` 合并：`Ao`(8)、`SA`(25)、`celiac`(20)、`SMA`(22)、`PHA`(24)、`LRA`(39)、`RRA`(40)。

`vein_tree` 合并：`IVC`(9)、`Main Portal Vein`(23)、`SMV`(26)、`SV`(33)、`Left PV`(34)、`Right PV`(35)、`PV confluence`(36)、`PV branch`(37)、`LHV`(27)、`MHV`(28)、`RHV`(29)、`LGV`(32)、`LRV`(41)、`RAV`(42)。

为保持当前胃采样规则的输入契约，另输出辅助 `portal_vein_and_splenic_vein`，合并 Main Portal Vein、SMV、SV、左右 PV、PV confluence 与 PV branch。该辅助模型不作为检索特征类别。

## 实现与物理空间

实现为可复用的项目模块和独立脚本，不修改已有 `main.py` 的病例重采样入口。

1. 读取指定 DICOM 序列，写出 `ct/ct_venous.nrrd`。
2. 从 Slicer 标签图按上述映射生成 `uint8` 二值掩膜；每个掩膜保留 CT 的 Origin、Spacing、Direction。
3. 使用 `skimage.measure.marching_cubes(level=0.5)` 从掩膜提取三角面。
4. 将 Marching Cubes 返回的 z/y/x 连续索引改为 x/y/z，并计算：

   `world_xyz = origin + direction @ (index_xyz * spacing)`。

5. 将世界坐标顶点和三角面写为 PLY；写入前修复闭合网格的法线方向，记录网格是否 watertight。

## 输出结构

```text
预处理后/
  ct/ct_venous.nrrd
  masks/<每个器官或血管树>.nrrd
  models/<每个器官或血管树>.ply
  manifest.json
  case_preprocessed.yaml
```

`case_preprocessed.yaml` 填入全部器官模型，并将检索血管设为 `artery_tree` 与 `vein_tree`。`portal_vein_and_splenic_vein` 仍作为器官模型项提供给胃候选点筛选。

## 校验与失败条件

- 选定 DICOM 序列不存在、标签缺失、标签体素数为零、CT/分割几何不匹配或提取网格为空时失败。
- `manifest.json` 记录源标签、体素数、顶点数、三角面数、网格闭合状态、CT 与分割的最大几何差异及全部输出路径。
- 单元测试使用带非零 Origin、非单位 Spacing、非恒等 Direction 的合成标签图，验证掩膜、合并标签和网格物理顶点变换。

## 依赖

为在项目 mamba 环境中可复现地提取网格，增加 `scikit-image` 依赖。
