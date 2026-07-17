# AI/训练路线图:从规则弱标签到学习模型

现状定位:events.py 的几何规则不是终点,是**弱标注器**。这是文献验证过的路径——
NETS(Hauri & Vucetic, ECAI 2023)用同款规则在同一份公开 SportVU 数据上打弱标签,
训练 Transformer 后 F1 从规则的 0.869 提到 **0.951**。我们照抄这条已被验证的爬升路线。

## 阶段 T1:数据扩容(纯工程,无训练)

- 把公开 dump 的 **632 场全部**下载解析(现在只用了 7 场)。
  ~45M moments、每场 ~5s 解析,存储 ~40GB Parquet,一晚跑完。
- `jdub detect` 全量跑一遍 → 弱标签库:每场 ~90 screen / ~86 coverage / 全部原子动作。
  规模大约 5.7 万个挡拆样本(NETS 是 45,802,同量级)。

## 阶段 T2:自监督预训练(核心 AI 里程碑)

- 输入张量:每回合重采样到 T=50 帧 × 11 实体 × 特征(x, y, 队伍 one-hot, 球 flag)。
- 架构 v1 保持小:per-entity 时间编码器 + 实体间 cross-attention(set-transformer 风格,
  几百万参数);HoopTransformer 式 axial attention 是 v2 升级项,不是 v1。
- 目标:**masked trajectory modeling**——随机遮 entity-timestep 片段,回归坐标,Huber loss。
  无需任何标签,632 场全用上。
- 硬性 sanity gate(过不了不许进 T3):冻结 encoder,linear probe 预测弱标签
  (PnR 有/无),probe 打不过 base rate 有意义的幅度 = embedding 是垃圾,回去修。

## 阶段 T3:弱监督微调(替换规则置信度)

- 用 T1 的弱标签微调分类头:screen / handoff / drive / cut / post_up / offball_screen
  / iso / transition 多标签逐帧(或逐窗口)预测。
- 对标:规则 F1 ~0.87(NETS 弱标签自测)→ 模型目标 ≥0.93(NETS 微调后 0.951)。
- 产品接入点:actions 表的 `confidence` 字段从规则启发式换成模型概率,
  **schema 不变,下游(studio/事实编译器/解说)零改动**。这是当初把置信度做成
  一等公民的原因。
- coverage 分类头同理(switch/blitz/drop/over/under 五类):McIntyre 逻辑回归基线
  0.69 准确率,序列模型应显著超过;人工标注 20-50 个回合做验收集(M3 的人工抽检
  顺手就是标注)。

## 阶段 T4(v2,PRD 后续):检索与相似回合

- 池化出 possession embedding → FAISS 索引 → "以回合搜回合"。
- 解说引擎引用历史相似回合("这和第二节那次如出一辙")。

## 算力与依赖

- PyTorch + 单卡消费级 GPU 足够 v1(几 M 参数、45M 帧采样训练);无 GPU 时
  MPS(Apple Silicon)可跑小规模验证。
- 新依赖只在 T2 动手时加(torch),现在不进 pyproject——避免无用重依赖。

## 铁律不变

学习模型只替换"检测与置信度",不进解说层:LLM 依旧只见事实流。
模型的每个输出都必须能回放到 studio 里被肉眼质检——评估工具和 demo 是同一个东西。
