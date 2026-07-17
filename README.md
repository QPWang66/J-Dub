# jdub

**J**ustified **Dub**bing:篮球轨迹数据 → 有依据的战术解说。当前进度:M1(数据管线 + 回合可视化)。

## Quickstart

```bash
uv sync
uv run jdub download 01.01.2016.CHA.at.TOR   # 下载一场 SportVU 数据到 data/raw/
uv run jdub parse data/raw/0021500492.json   # 解析为 data/parquet/ 下的 Parquet
uv run jdub studio                           # 本地回合播放器 http://127.0.0.1:8000
uv run jdub viz 0021500492 6                 # 单回合渲染 mp4 到 out/
uv run pytest
```

mp4 导出依赖系统 ffmpeg(`brew install ffmpeg`);gif(`--out x.gif`)不需要。
