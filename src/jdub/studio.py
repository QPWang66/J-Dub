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
    ev = m.group_by("event_id", maintain_order=True).agg(
        pl.col("quarter").first(),
        pl.col("game_clock").max().alias("start_clock"),
        pl.col("game_clock").min().alias("end_clock"),
        pl.col("moment_idx").n_unique().alias("n_moments"),
    )
    pbp_path = PARQUET_DIR / "pbp" / f"{game_id}.parquet"
    if pbp_path.exists():
        pbp = pl.read_parquet(pbp_path).select("event_id", "desc")
        ev = ev.join(pbp, on="event_id", how="left")
    else:
        ev = ev.with_columns(pl.lit(None, dtype=pl.String).alias("desc"))
    actions_path = PARQUET_DIR / "actions" / f"{game_id}.parquet"
    if actions_path.exists():
        summary = (
            pl.read_parquet(actions_path)
            .group_by("event_id")
            .agg(
                pl.len().alias("n_actions"),
                pl.col("confidence").max().alias("max_conf"),
                pl.col("type").unique().sort().alias("action_types"),
            )
        )
        ev = ev.join(summary, on="event_id", how="left")
    else:
        ev = ev.with_columns(
            pl.lit(0).alias("n_actions"),
            pl.lit(None, dtype=pl.Float64).alias("max_conf"),
            pl.lit([], dtype=pl.List(pl.String)).alias("action_types"),
        )
    return ev.with_columns(pl.col("n_actions").fill_null(0)).sort("event_id").to_dicts()


@app.get("/api/games/{game_id}/events/{event_id}")
def event(game_id: str, event_id: int) -> dict:
    m = load_moments(game_id)
    frames = event_frames(m, event_id)
    if not frames:
        raise HTTPException(404, f"event {event_id} has no moments")
    game = load_games().filter(pl.col("game_id") == game_id).to_dicts()[0]
    return {
        "game": game,
        "frames": frames,
        "matchups": _detection(game_id, "matchups", event_id),
        "actions": _detection(game_id, "actions", event_id),
        "coverages": _detection(game_id, "coverages", event_id),
        "players": _rosters(game_id),
    }


@app.get("/api/games/{game_id}/events/{event_id}/commentary")
def commentary(game_id: str, event_id: int, llm: bool = False) -> dict:
    from jdub.commentary import generate

    return generate(game_id, event_id, llm=llm)


def _detection(game_id: str, table: str, event_id: int) -> list[dict]:
    """M2 detection rows for one event; [] until `jdub detect` has been run."""
    path = PARQUET_DIR / table / f"{game_id}.parquet"
    if not path.exists():
        return []
    return pl.read_parquet(path).filter(pl.col("event_id") == event_id).to_dicts()


def _rosters(game_id: str) -> dict[str, dict]:
    path = PARQUET_DIR / "players" / f"{game_id}.parquet"
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    return {
        str(r["player_id"]): {"name": r["lastname"], "jersey": r["jersey"]}
        for r in df.iter_rows(named=True)
    }


def run(port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port)
