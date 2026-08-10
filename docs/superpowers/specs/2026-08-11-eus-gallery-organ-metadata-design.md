# EUS Gallery 器官元数据扩展设计

## 目标与边界

重采样 Gallery 后续用于 EUS 图像检索。本次只调整建库产物中的器官交面、器官标签和审计元数据，不实现 EUS 查询初筛、CBIR 候选过滤或其他检索规则。

现有采样点、零度面、三轴旋转、CT 三次插值、质量过滤、图库状态路由和血管截面特征保持不变。下腔静脉、腹主动脉和门静脉增加器官身份时，仍继续按原有动脉/静脉模型生成 `features` 和血管图层。

白名单基准为用户提供的 `eus_possible_organs.json`，原文件以 Python 包资源纳入仓库，保持字节内容不变。基准 schema 为 `eus-possible-organs/v1`，SHA-256 为 `54b8bf06fc48d1733e98b32a01dc10e056f5db3b4cddb34e18905dd8d97bf63d`。

## 白名单模型与校验

新增独立的 EUS 器官目录加载模块，负责读取包资源并校验：

- 顶层 schema、`organs` 列表和每个条目的字段类型完整。
- `organ_label`、EUS 标签 ID 和名称在各自约束内唯一、非空且确定排序。
- `aorta`、`inferior_vena_cava`、`portal_vein` 必须以 `organ_and_vessel` 存在，血管类型分别为 `artery`、`vein`、`vein`。
- `bile_duct`、`common_bile_duct` 等胆管规范标签显式禁止；`gallbladder` 是不同解剖结构，仍可保留在通用可见器官字段。

资源作为 setuptools package data 安装，避免依赖 Windows 桌面绝对路径。运行协议和图库摘要记录资源 schema 与 SHA-256，使本地、GitHub 和服务器使用同一规则。

## 器官交面与双身份结构

通用器官交面集合由现有 11 类扩展为 14 类：

- 保留左右肾上腺、十二指肠、食管、胆囊、左右肾、肝脏、胰腺、脾脏和胃。
- 增加 `aorta` 和 `inferior_vena_cava`，直接使用同名 TotalSegmentator 网格。
- 增加规范输出标签 `portal_vein`，几何复用现有 `portal_vein_and_splenic_vein` 合并网格；因此该候选在几何上包含脾静脉，必须在文档和摘要中明确记录。

器官交面继续只在 CT 质量合格且至少有一个完整血管截面的 Gallery 候选上执行。网格与方形切面相交后，只要方形视野内保留正面积区域就视为可见；轮廓被方形边缘裁剪仍计入，严格零面积相切不计入。

组合图继续先绘制器官层、后绘制血管层。腹主动脉使用现有动脉橙红色，下腔静脉和门静脉使用现有静脉青色；重合位置最终仍由原血管图层覆盖。血管模型加载、血管交面、`features`、颜色和状态判定不得因双身份器官而改变。

## Gallery 输出协议

每条新 `gallery.jsonl` 记录增加：

- `organ_metadata_schema_version`: 固定为 `eus-organ-metadata/v1`。
- `eus_candidate_organ_labels`: 当前 `organ_labels` 与 EUS 白名单规范名的排序去重交集。

`organ_labels` 继续表示所有 14 类通用结构中当前切面可见的结构。胃、食管和胆囊可以出现在 `organ_labels`，但因不在白名单而不进入 `eus_candidate_organ_labels`。三类双身份结构在可见时同时进入两个字段。候选字段只是建库元数据，不在本次实现中驱动检索。

Gallery 恢复校验要求：

- 两个标签列表均为字符串、排序且无重复。
- 通用标签只来自 14 类集合，EUS 候选只来自版本化白名单。
- 候选列表必须严格等于 `organ_labels` 与白名单的交集。
- 胆管标签不得出现。
- 新 schema 下缺少任一字段的旧 Gallery 明确拒绝续写，要求使用新输出根。

`library_summary.json` 增加按 Gallery 切面计数的 `eus_candidate_organ_label_counts`，同一标签在一张切面最多计一次；同时集中保存白名单 schema、SHA-256、规范标签到 EUS ID/中文名/角色的完整映射和门静脉合并几何说明。`run_metadata.json` 保存白名单 schema、SHA-256 和规范标签集合，并将其纳入恢复协议哈希。

现有 `registration_adapter` 继续只读取血管 `features` 和位姿，忽略新增字段。已有 134,386 条服务器 Gallery 保持原地不动，仍可作为旧库读取，但新代码不得向其追加记录。

## 测试与交付

测试覆盖资源 schema、重复项和胆管拒绝；14 类器官映射；门静脉合并网格到规范标签映射；边缘裁剪正面积轮廓；双身份标签；EUS 候选严格交集；恢复不兼容提示；图库摘要与运行协议；以及增加器官身份前后血管 `features` 完全相同。

完成聚焦测试后运行全量 pytest、compileall 和源码/文档一致性检查，并进行独立代码质量审查。代码在 `feature/eus-gallery-organ-metadata-20260811` 分模块提交并推送 GitHub。

本次服务器范围仅同步代码，不试跑或全量重建。同步前必须在 `/root/autodl-tmp/ct_vascular_resampling_case2_20260731` 内对项目代码和运行配置建立带时间戳的完整压缩备份；只操作该项目根。服务器连接、磁盘、分支和工作区状态异常时停止同步。旧输出目录不得修改、删除或续写。
