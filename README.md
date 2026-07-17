# jdub

**J**ustified **Dub**bing:篮球轨迹数据 → 有依据的战术解说。

管线:SportVU 轨迹 → 对位分配 → 原子动作检测(9 种)→ 挡拆 coverage 分类(5 种)
→ 事实编译器 → salience 规划 → 有依据的中文解说。铁律:语言层只见事实流,
不见坐标;低置信事实用模糊措辞或沉默。

当前状态:M1–M4 管线全通,人工抽检验收(M2/M3/M4 的 DoD)待做。

## Quickstart

```bash
uv sync
uv run jdub download 01.04.2016.SAC.at.OKC   # 下载一场 SportVU 数据到 data/raw/
uv run jdub parse data/raw/0021500517.json   # 解析为 data/parquet/ 下的 Parquet
uv run jdub detect 0021500517                # 对位 + 原子动作 + coverage
uv run jdub pbp 0021500517                   # 官方 play-by-play(抽检地面真值)
uv run jdub studio                           # 本地回合播放器 http://127.0.0.1:8000
uv run jdub commentary 0021500517 217        # 该回合的有依据中文解说
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
  commentary.py  事实编译器 -> salience -> 中文解说(模板零幻觉;--llm 走 claude CLI)
  studio.py      FastAPI 后端(4 个 JSON 接口 + 静态页)
  static/        jdub studio 前端(原生 Canvas 单页,零依赖)
  viz.py         matplotlib 渲染 mp4/gif
  cli.py         Typer 入口:download / parse / detect / pbp / studio / commentary / viz
docs/
  detection-research.md   检测阈值的文献依据与 citation(deep-research 产物)
  training-plan.md        AI 路线图:规则弱标签 -> 自监督轨迹 Transformer
tests/           合成轨迹单测 + 截断的真实数据夹具
```

## 数据边界

公开 SportVU 数据只覆盖 2015-16 常规赛(2015-10-27 至 2016-01-23,632 场);
季后赛轨迹不存在公开渠道。样本为 5 场 OKC + 2 场 GSW。`data/` 不进 git。
