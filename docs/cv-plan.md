# CV 路线图:广播视频 → moments schema

管线第一段(视频 → 轨迹)的技术方案。结论先行:这不是"简单的一段",
它是全链条最难的工程——但任务形状已经被足球侧定义清楚了,叫
**Game State Reconstruction(GSR)**:广播视频 → 追踪 + 识别 + 投到
球场 2D 坐标系。我们照搬 GSR 的模块化架构,把足球组件换成篮球组件。

## 参照物

- **BCT**(`~/coding/BCT-main`):大学生原型,YOLO11n 检测 + 分割 +
  自制启发式单应性,notebook、非实时。**只作 pipeline 形状参考,不基于它开发。**
- **SoccerNet GSR baseline**([sn-gamestate](https://github.com/SoccerNet/sn-gamestate),
  CVPR'24W):检测 → 球场定位/标定 → PRTreID(身份+队伍+角色 embedding)→
  追踪 → 号码 OCR → 2D minimap。构建在 [TrackLab](https://github.com/TrackingLaboratory/tracklab)
  框架上,模块可插拔。**这是我们要抄的架构。**

## 模块选型(每个都开源、可微调)

| 模块 | 选型 | 依据 |
|------|------|------|
| 球员/裁判检测 | YOLO11m 或 RT-DETR 微调 | Roboflow universe 有现成篮球广播数据集(球员检测 ~1.4k 图起步) |
| 多目标追踪 | [Deep-EIoU](https://arxiv.org/abs/2306.13074)(运动场景 SOTA 系)或 [CAMELTrack](https://github.com/TrackingLaboratory/CAMELTrack)(2025,SportsMOT HOTA 80+,关联模块可训练) | 2025 上限参考:SAM-Deep-EIoU + 球衣/队伍线索全局关联,SportsMOT 86.8 HOTA,篮球提升最大 |
| ReID / 队伍 / 号码 | PRTreID 风格 embedding + 球衣号 OCR(roboflow/sports 有现成) | 队伍分配 + 断裂 tracklet 缝合都靠它 |
| 球场标定 | court keypoint 模型:[KaliCalib](https://arxiv.org/abs/2209.07795)(篮球专用,MMSports'22)或 keypoint-YOLO 在 Roboflow 篮球场关键点数据集上微调 | 逐帧单应性 + 时间平滑 |
| 球检测 | [WASB](https://github.com/nttcom/WASB-SBDT)(BMVC'23,篮球验证过,带权重) | 小而快的目标要高分辨率 heatmap 模型,YOLO 框不行 |

## 微调数据

- **检测/追踪**:[SportsMOT](https://github.com/MCG-NJU/SportsMOT) 篮球子集
  (带框带 ID)、Roboflow universe 篮球数据集。
- **球场关键点**:Roboflow universe 篮球 court keypoint 数据集(NBA 广播视角)。
- **独有资产(别人没有的)**:632 场 SportVU 轨迹 × nba.com 逐回合广播录像链接。
  时间对齐(比分牌 OCR / game clock)后 = 广播视频→2D 坐标的成对 ground
  truth,标注成本≈0。用作 CV 前端的**验收集**(也可做训练数据)。

## 广播视角的先天残缺(诚实面对)

单机位广播只拍半场局部:镜头外球员不存在 → offball_screen / cut /
弱侧对位在纯广播输入下**不可检测**。CV 前端输出的 moments 必然是
部分可见 + 有噪声的。两个对策:

1. 输出契约不变:CV 适配器写同一 moments schema,缺失球员就缺行,
   下游按"完整帧"过滤的逻辑(events.py `complete_frames`)天然兼容——
   但代价是可检测事件变少,这要在 C1 量化。
2. 检测阈值是在干净 SportVU 上调的,对噪声的容忍度必须先量出来 → Gate 0。

## 里程碑

- **C0 · Gate 0(✅ 已建)**:`jdub robustness <game>` —— 给 SportVU 轨迹注入
  高斯噪声,量出各检测器的 F1-vs-σ 衰减曲线。结论:速度类检测器在 σ=0.25ft
  就崩,视频轨迹必须先平滑;B 线模型对视频输入近乎必需。
- **C1 · baseline 跑通(✅ 已建,cv/)**:YOLO11+BoT-SORT + WASB 球 + 经典
  禁区贴合标定,端到端出 moments;lal-hou 片段首次从视频检出战术事件
  (265 对位 + 1 drive)。经典标定在摇镜头下不稳定 → 直接推进 C2。
- **C2 · 球场关键点模型(🔄 训练中)**:Roboflow reloc2 数据集(1.4k 广播帧,
  18 地标)微调 YOLO11s-pose,每帧独立回归绝对 H。`KeypointCalibrator`
  已接入,以 `cv/src/jdub_cv/stability.py` 全片段指标验收。
  后续(按 QP 指定的栈,暂 hold):SigLIP 分队 → SAM2 跟踪 → RF-DETR
  检测微调 → SmolVLM2/ResNet 球衣号码 → 真实球员身份。
- **C3 · 对齐验收**:SportVU×广播对齐做验收集;迭代到过 Gate 0 门槛 →
  接 moments schema → studio 回放肉眼质检(评估工具=demo,铁律不变)。

## 与轨迹 Transformer 的关系(两条训练线)

- **A 线(本文档)**:CV 前端,2-3 个小模型微调(检测器、court keypoint,
  可选 ReID/关联)。
- **B 线(training-plan.md)**:轨迹 Transformer,自监督预训练 + 弱标签微调,
  替换规则检测器的置信度。
- 汇合点:两条线共享 moments schema。B 线模型天然比几何规则更耐噪声
  (训练时可做噪声增强),所以 B 线成熟后 Gate 0 的门槛会放宽——
  但顺序上先 C0/C1 摸清现实,再决定 B 线要不要提前做噪声增强。

## 依赖纪律

CV 侧依赖(torch/ultralytics/tracklab 等)不进 jdub 主包——C1 起以独立
子包/环境开发(`cv/` 或独立 repo),只通过 moments Parquet 与主管线通信。
主包保持今天这样能秒装秒跑。
