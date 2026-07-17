# jdub

**J**ustified **Dub**bing:篮球轨迹数据 → 有依据的战术解说。当前进度:M4 管线全通(检测阈值来自文献、coverage 分类、事实编译器、有依据中文解说),M2-M4 的人工抽检验收待做。

## Quickstart

```bash
uv sync
uv run jdub download 01.01.2016.CHA.at.TOR   # 下载一场 SportVU 数据到 data/raw/
uv run jdub parse data/raw/0021500492.json   # 解析为 data/parquet/ 下的 Parquet
uv run jdub detect 0021500492                # M2:对位分配 + 原子动作检测
uv run jdub studio                           # 本地回合播放器 http://127.0.0.1:8000
uv run jdub commentary 0021500517 217        # M4:该回合的有依据中文解说
uv run jdub viz 0021500517 217               # 单回合渲染 mp4 到 out/
uv run pytest
```

检测阈值的文献依据与 citation:`docs/detection-research.md`。

mp4 导出依赖系统 ffmpeg(`brew install ffmpeg`);gif(`--out x.gif`)不需要。
