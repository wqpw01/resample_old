# 采样点与零度基准面可视化设计

## 目标与边界

为病例 2 本次正式全量重采样生成一套本地可视化交付物，展示全部采样点及每个采样点对应的零度基准面。可视化必须直接读取正式输出记录，不重新执行采样、不重新推导零度面，也不修改远程 Gallery 或任何运行结果。

正式病例目录为：

`/root/autodl-tmp/ct_vascular_resampling_case2_20260731/output_core_design_20260813_full_rotation/case_2`

本地交付目录为：

`C:\Users\zhangyutang\Desktop\本次重采样_采样点与零度基准面可视化_20260815`

## 数据来源与筛选

- 采样点来自 `ResampledpointPLY/FPS-*.ply`，并与根 `manifest.jsonl` 中的 `probe_point_world` 逐器官、逐索引核对。
- 零度基准面只从根 `manifest.jsonl` 中筛选 `roll=0`、`pitch=0`、`yaw=0` 的记录。
- 每个采样点必须且只能匹配一条零度记录；重复、缺失或总数不是 400 时停止生成并报告错误。
- 零度面的四顶点直接使用记录中的 `square_vertices_world`；采样点使用 `probe_point_world`，局部轴使用 `local_axes_world`，不从法向量重新计算。
- 全部坐标保持正式记录的 RAS 世界坐标，单位为毫米。
- `squarePLY/*.ply` 和 `rectangles.ply` 包含全角度切面，不作为零度面数据源。

预期采样点总数为 400：胃 118、肝 162、胰腺 37、十二指肠 53、食管 30。

## 视觉编码

五个目标器官使用稳定且互相区分的颜色。相同器官的采样点、零度面边框和表格记录使用同一主色；半透明器官网格使用较浅的同色系。采样点使用实心球形标记，零度面使用低透明度填充和清晰边框。

每个零度面额外显示：

- 从采样点出发的局部 `+x`、`+y`、`+z` 短轴；
- 采样点位于方形底边中点的几何关系；
- 可选的点编号，默认关闭以避免 400 个标签遮挡。

交互视图支持按器官分别显示或隐藏器官网格、采样点、零度面和局部轴。静态视图使用相同相机范围和等比例坐标，避免透视或轴缩放造成方向误判。

## 交付物

交付目录包含：

- `sampling_points_zero_planes_interactive.html`：自包含 Plotly JavaScript 的离线交互三维页面；
- `sampling_points_zero_planes_isometric.png`：等距三维视图；
- `sampling_points_zero_planes_axial.png`：轴位投影视图；
- `sampling_points_zero_planes_coronal.png`：冠状位投影视图；
- `sampling_points_zero_planes_sagittal.png`：矢状位投影视图；
- `sampling_points.ply`：带器官 RGB 和法向量的 400 个采样点；
- `zero_planes_edges.ply`：400 个零度面边框线段；
- `zero_planes_faces.ply`：400 个零度面四边形面；
- `sampling_points_zero_planes.csv`：每行一个采样点，含器官、索引、采样点、局部轴和四顶点；
- `sampling_points_zero_planes.json`：与 CSV 等价的结构化完整记录及来源哈希；
- `README_中文.txt`：颜色、坐标、零度筛选、软件打开方式和文件含义。

目标器官网格作为独立 PLY 文件放入 `target_organ_meshes/`，用于 3D Slicer、MeshLab 或 CloudCompare 中提供解剖空间参照。

## 实现方式

新增一个独立导出脚本。脚本通过 SSH 在服务器端只读、流式扫描根清单，只把筛选出的 400 条零度记录传回本地；不得下载约数 GB 的完整根清单。采样点 PLY 和五个目标器官网格下载到临时工作目录，完成一致性检查后生成本地交付物。临时文件与最终交付目录分离；只有所有检查和渲染成功后，才原子性替换最终目录。

交互 HTML 内嵌 Plotly 运行代码，不依赖互联网。静态 PNG 使用 Matplotlib 三维渲染，固定画布尺寸、相机方位、等比例世界坐标范围和图例。PLY 使用 ASCII 或二进制标准格式，并在说明中记录面顶点顺序。

## 验证与验收

- 零度记录总数为 400，且逐器官计数与采样点计数一致；
- `slice_id`、采样点和四顶点均唯一且为有限数值；
- 每个零度面的边长为 100 mm，四边形近似正方形；
- 采样点到零度面底边中点的距离在浮点容差内为 0；
- 局部三轴单位化、两两正交并满足右手系；
- HTML、四张 PNG、三个主 PLY、CSV、JSON、说明和五个器官网格均存在且非空；
- HTML 包含全部 400 个采样点和 400 个零度面；
- PNG 通过像素非空检查，并人工检查文字、图例、器官、采样点和零度面没有明显遮挡；
- 在桌面最终目录生成 SHA-256 清单，便于后续确认文件未变。

远程服务器和 Gallery 全程只读；本任务不改变代码的重采样逻辑、配置、模型、运行日志或现有输出。
