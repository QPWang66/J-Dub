"""jdub studio: local web viewer (FastAPI + single-page canvas)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from jdub.data import PARQUET_DIR, event_frames, load_games, load_moments

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="jdub studio")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/games")
def games() -> list[dict]:
    return load_games().to_dicts()


@app.get("/api/games/{game_id}/events")
def events(game_id: str) -> list[dict]:
    try:
        m = load_moments(game_id)
    except FileNotFoundError:
        raise HTTPException(404, f"game {game_id} not parsed") from None
    return (
        m.group_by("event_id", maintain_order=True)
        .agg(
            pl.col("quarter").first(),
            pl.col("game_clock").max().alias("start_clock"),
            pl.col("game_clock").min().alias("end_clock"),
            pl.col("moment_idx").n_unique().alias("n_moments"),
        )
        .sort("event_id")
        .to_dicts()
    )


@app.get("/api/games/{game_id}/events/{event_id}")
def event(game_id: str, event_id: int) -> dict:
    m = load_moments(game_id)
    frames = event_frames(m, event_id)
    if not frames:
        raise HTTPException(404, f"event {event_id} has no moments")
    game = load_games().filter(pl.col("game_id") == game_id).to_dicts()[0]
    return {"game": game, "frames": frames}


def run(port: int = 8000, parquet_dir: Path = PARQUET_DIR) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port)
