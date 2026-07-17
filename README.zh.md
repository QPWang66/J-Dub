# jdub

[English](README.md) | 中文

**J**ustified **Dub**bing:篮球比赛视频 → 有依据的战术解说。

四段管线:**视频 → 轨迹 → 战术 → 解说**。第一段(视频 → 轨迹的 CV 前端)
尚未做,当前直接以公开 SportVU 轨迹为输入,后接:
对位分配 → 原子动作检测(9 种)→ 挡拆 coverage 分类(5 种)
→ 事实编译器 → salience 规划 → 有依据的解说(中/英)。铁律:语言层只见事实流,
不见坐标;低置信事实用模糊措辞或沉默。

当前状态:轨迹→解说三段全通(M1–M4),人工抽检验收(M2/M3/M4 的 DoD)待做。
视频→轨迹进入规划——见 [docs/cv-plan.md](docs/cv-plan.md);其 Gate 0
(规则检测器的噪声鲁棒性,`jdub robustness`)已建成。

接下来两条 AI 训练线:**A)** CV 前端(检测器微调、球场关键点、追踪——
docs/cv-plan.md);**B)** 轨迹 Transformer,替换规则置信度
(docs/training-plan.md)。两条线在共享的 moments schema 汇合。

## Quickstart

```bash
uv sync
uv run jdub download 01.04.2016.SAC.at.OKC   # 下载一场 SportVU 数据到 data/raw/
uv run jdub parse data/raw/0021500517.json   # 解析为 data/parquet/ 下的 Parquet
uv run jdub detect 0021500517                # 对位 + 原子动作 + coverage
uv run jdub pbp 0021500517                   # 官方 play-by-play(抽检地面真值)
uv run jdub studio                           # 本地回合播放器 http://127.0.0.1:8000
uv run jdub commentary 0021500517 217        # 该回合的有依据解说(--lang zh|en)
uv run jdub viz 0021500517 217               # 单回合渲染 mp4 到 out/
uv run pytest
```

mp4 导出依赖系统 ffmpeg(`brew install ffmpeg`);gif(`--out x.gif`)不需要。

## 识别能力

- 原子动作(`data/parquet/actions/`):screen、offball_screen、handoff、pass、
  drive、cut、post_up、iso、transition,均带时间戳与置信度。
- 挡拆 coverage(`data/parquet/coverages/`):switch、blitz、drop、over、under。
- 每回合官方 PBP 文字与 nba.com 逐回合录像链接作为对照。

## 结构

```
src/jdub/
  data.py        SportVU JSON -> Parquet(moments/games/players/pbp),去重、方向归一
  events.py      对位分配(Franks 质心 + 最优指派)、原子动作、coverage 分类
  commentary.py  事实编译器 -> salience -> 中/英解说(模板零幻觉;--llm 走本地模型,
                 默认 ollama qwen3:8b,JDUB_LLM_URL/JDUB_LLM_MODEL 可换任何
                 OpenAI 兼容端点,失败回落模板)
  studio.py      FastAPI 后端(4 个 JSON 接口 + 静态页)
  static/        jdub studio 前端(原生 Canvas 单页,零依赖)
  viz.py         matplotlib 渲染 mp4/gif
  robustness.py  CV 前端的 Gate 0:检测器 F1 随注入定位噪声的衰减曲线
  cli.py         Typer 入口:download / parse / detect / pbp / robustness / studio / commentary / viz
docs/
  detection-research.md   检测阈值的文献依据与 citation(deep-research 产物)
  cv-plan.md              CV 路线图:广播视频 -> moments schema(GSR 式模块化管线)
  training-plan.md        AI 路线图:规则弱标签 -> 自监督轨迹 Transformer
tests/           合成轨迹单测 + 截断的真实数据夹具
```

## 数据边界

公开 SportVU 数据只覆盖 2015-16 常规赛(2015-10-27 至 2016-01-23,632 场);
季后赛轨迹不存在公开渠道。本地样本为 5 场 OKC + 2 场 GSW。

`data/` 整体不进 git,唯一例外是**骑勇圣诞大战(2015-12-25 CLE @ GSW)**作为
sample data 随仓库分发:全套 Parquet(克隆后 `uv sync && uv run jdub studio`
即可直接播放,无需联网)+ 原始 `.7z` 归档(想复现完整管线时
`py7zr` 解压后 `jdub parse` 即可,原始 JSON 108MB 超 GitHub 单文件上限故不入库)。
