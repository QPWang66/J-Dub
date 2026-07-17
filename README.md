# jdub

English | [中文](README.zh.md)

**J**ustified **Dub**bing: basketball game video → grounded tactical commentary.

Four-stage pipeline: **video → trajectories → tactics → commentary**. Stage 1
(the CV front-end, video → trajectories) is not built yet; the current input is
public SportVU tracking data, feeding:
matchup assignment → atomic action detection (9 types) → pick-and-roll coverage
classification (5 types) → fact compiler → salience planning → grounded
commentary (Chinese or English). Iron rule: the language layer only ever sees
the fact stream, never coordinates; low-confidence facts get hedged wording or
silence.

Current status: trajectories → commentary (M1–M4) works end to end; human
spot-check acceptance (the M2/M3/M4 DoD) is pending. Video → trajectories is
in planning — see [docs/cv-plan.md](docs/cv-plan.md); its Gate 0 (noise
robustness of the rule detectors, `jdub robustness`) is built.

Two AI training tracks ahead: **A)** the CV front-end (fine-tuned detector,
court keypoints, tracking — docs/cv-plan.md) and **B)** the trajectory
Transformer that replaces rule confidences (docs/training-plan.md). They meet
at the shared moments schema.

## Quickstart

```bash
uv sync
uv run jdub download 01.04.2016.SAC.at.OKC   # download one SportVU game into data/raw/
uv run jdub parse data/raw/0021500517.json   # parse into Parquet under data/parquet/
uv run jdub detect 0021500517                # matchups + atomic actions + coverages
uv run jdub pbp 0021500517                   # official play-by-play (spot-check ground truth)
uv run jdub studio                           # local possession player, http://127.0.0.1:8000
uv run jdub commentary 0021500517 217        # grounded commentary for that possession (--lang zh|en)
uv run jdub viz 0021500517 217               # render one possession to mp4 in out/
uv run pytest
```

mp4 export needs system ffmpeg (`brew install ffmpeg`); gif (`--out x.gif`) does not.

## What it detects

- Atomic actions (`data/parquet/actions/`): screen, offball_screen, handoff,
  pass, drive, cut, post_up, iso, transition — each with timestamps and a
  confidence score.
- Pick-and-roll coverages (`data/parquet/coverages/`): switch, blitz, drop,
  over, under.
- Official PBP text and nba.com per-possession clip links for cross-checking.

## Layout

```
src/jdub/
  data.py        SportVU JSON -> Parquet (moments/games/players/pbp), dedupe, direction normalization
  events.py      matchup assignment (Franks centroid + optimal assignment), atomic actions, coverage classification
  commentary.py  fact compiler -> salience -> commentary in zh/en (templates are hallucination-free
                 by construction; --llm uses a local model, default ollama qwen3:8b,
                 JDUB_LLM_URL/JDUB_LLM_MODEL point at any OpenAI-compatible endpoint,
                 falls back to templates on failure)
  studio.py      FastAPI backend (4 JSON endpoints + static page)
  static/        jdub studio front-end (vanilla Canvas single page, zero dependencies)
  viz.py         matplotlib mp4/gif rendering
  robustness.py  Gate 0 for the CV front-end: detector F1 vs injected positional noise
  cli.py         Typer entry points: download / parse / detect / pbp / robustness / studio / commentary / viz
docs/
  detection-research.md   literature basis + citations for detection thresholds (deep-research output)
  cv-plan.md              CV roadmap: broadcast video -> moments schema (GSR-style modular pipeline)
  training-plan.md        AI roadmap: rule-based weak labels -> self-supervised trajectory Transformer
tests/           synthetic-trajectory unit tests + truncated real-data fixtures
```

## Data boundary

The public SportVU dump only covers the 2015-16 regular season (2015-10-27
through 2016-01-23, 632 games); playoff tracking data has no public source.
Local samples: 5 OKC games + 2 GSW games.

`data/` as a whole is git-ignored, with one exception: the **Christmas game
(2015-12-25 CLE @ GSW)** ships with the repo as sample data — the full Parquet
set (after cloning, `uv sync && uv run jdub studio` plays it with no network)
plus the raw `.7z` archive (to reproduce the full pipeline, extract with
`py7zr` and run `jdub parse`; the raw JSON is 108MB, over GitHub's single-file
limit, so it is not committed).
