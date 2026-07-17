"""SportVU raw JSON -> Parquet tables (moments, games, players).

Raw moment layout (verified against 0021500492.json):
    [quarter, wall_clock_ms, game_clock, shot_clock|None, None,
     [[team_id, player_id, x, y, z], ...]]        # ball is team_id=-1, player_id=-1
Court is 94 x 50 ft, origin at one corner. ~25 Hz. Adjacent events overlap in time.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import py7zr
import requests

RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/parquet")
MIRROR = (
    "https://raw.githubusercontent.com/linouk23/NBA-Player-Movements"
    "/master/data/2016.NBA.Raw.SportVU.Game.Logs"
)
COURT_LENGTH = 94.0
COURT_WIDTH = 50.0


def download_game(name: str, raw_dir: Path = RAW_DIR) -> Path:
    """Download one game archive (e.g. '01.01.2016.CHA.at.TOR') and extract its JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / f"{name}.7z"
    if not archive.exists():
        resp = requests.get(f"{MIRROR}/{name}.7z", timeout=300)
        resp.raise_for_status()
        archive.write_bytes(resp.content)
    with py7zr.SevenZipFile(archive) as z:
        json_name = z.getnames()[0]
        if not (raw_dir / json_name).exists():
            z.extractall(raw_dir)
    return raw_dir / json_name


def parse_game(json_path: Path | str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse raw SportVU JSON into (moments, games, players) DataFrames.

    moments is long format (one row per entity per moment), deduped via dedupe_moments.
    """
    raw = json.loads(Path(json_path).read_bytes())
    game_id = raw["gameid"]

    cols: dict[str, list] = {
        "event_id": [],
        "moment_idx": [],
        "quarter": [],
        "game_clock": [],
        "shot_clock": [],
        "entity": [],
        "team_id": [],
        "player_id": [],
        "x": [],
        "y": [],
        "z": [],
    }
    for event in raw["events"]:
        event_id = int(event["eventId"])
        for idx, m in enumerate(event["moments"]):
            quarter, _wall, game_clock, shot_clock, _, positions = m
            for team_id, player_id, x, y, z in positions:
                cols["event_id"].append(event_id)
                cols["moment_idx"].append(idx)
                cols["quarter"].append(quarter)
                cols["game_clock"].append(game_clock)
                cols["shot_clock"].append(shot_clock)
                cols["entity"].append("ball" if player_id == -1 else "player")
                cols["team_id"].append(team_id)
                cols["player_id"].append(player_id)
                cols["x"].append(x)
                cols["y"].append(y)
                cols["z"].append(z)

    moments = (
        pl.DataFrame(cols)
        .with_columns(
            pl.lit(game_id).alias("game_id"),
            pl.col("quarter").cast(pl.Int8),
            pl.col("event_id").cast(pl.Int32),
            pl.col("moment_idx").cast(pl.Int32),
            pl.col("team_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64),
        )
        .select(
            "game_id",
            "event_id",
            "moment_idx",
            "quarter",
            "game_clock",
            "shot_clock",
            "entity",
            "team_id",
            "player_id",
            "x",
            "y",
            "z",
        )
    )
    moments = dedupe_moments(moments)

    home, visitor = raw["events"][0]["home"], raw["events"][0]["visitor"]
    games = pl.DataFrame(
        {
            "game_id": [game_id],
            "date": [raw["gamedate"]],
            "home_team_id": [home["teamid"]],
            "home_team": [home["name"]],
            "home_abbr": [home["abbreviation"]],
            "visitor_team_id": [visitor["teamid"]],
            "visitor_team": [visitor["name"]],
            "visitor_abbr": [visitor["abbreviation"]],
        }
    )
    players = pl.DataFrame(
        [
            {
                "game_id": game_id,
                "team_id": side["teamid"],
                "player_id": p["playerid"],
                "firstname": p["firstname"],
                "lastname": p["lastname"],
                "jersey": p["jersey"],
                "position": p["position"],
            }
            for side in (home, visitor)
            for p in side["players"]
        ]
    )
    return moments, games, players


def dedupe_moments(df: pl.DataFrame) -> pl.DataFrame:
    """Drop moments whose (game_id, quarter, game_clock) already appeared earlier.

    SportVU events overlap: adjacent events repeat the same frames. Keep first
    occurrence. Dedupe at moment granularity (all 11 entity rows kept or dropped
    together), preserving input order.
    """
    kept = (
        df.select("game_id", "event_id", "moment_idx", "quarter", "game_clock")
        .unique(subset=["game_id", "event_id", "moment_idx"], keep="first", maintain_order=True)
        .unique(subset=["game_id", "quarter", "game_clock"], keep="first", maintain_order=True)
        .select("game_id", "event_id", "moment_idx")
    )
    return df.join(
        kept, on=["game_id", "event_id", "moment_idx"], how="semi", maintain_order="left"
    )


def normalize_offense_direction(df: pl.DataFrame) -> pl.DataFrame:
    """Flip coordinates per (game_id, event_id) so offense always attacks the left basket.

    Offense side inferred from the ball's mean x over the event (half-court bias):
    if the ball lives in the right half, mirror x and y.
    """
    # ponytail: ball-mean-x heuristic; upgrade to possession-aware flip in M2
    flip = (
        df.filter(pl.col("entity") == "ball")
        .group_by("game_id", "event_id")
        .agg((pl.col("x").mean() > COURT_LENGTH / 2).alias("flip"))
    )
    return (
        df.join(flip, on=["game_id", "event_id"], how="left", maintain_order="left")
        .with_columns(pl.col("flip").fill_null(False))
        .with_columns(
            pl.when("flip").then(COURT_LENGTH - pl.col("x")).otherwise("x").alias("x"),
            pl.when("flip").then(COURT_WIDTH - pl.col("y")).otherwise("y").alias("y"),
        )
        .drop("flip")
    )


def parse_to_parquet(json_path: Path | str, out_dir: Path = PARQUET_DIR) -> dict[str, int]:
    """Parse one game and write per-game Parquet files. Returns row counts."""
    moments, games, players = parse_game(json_path)
    game_id = games["game_id"][0]
    for name, df in (("moments", moments), ("games", games), ("players", players)):
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        df.write_parquet(d / f"{game_id}.parquet")
    return {"moments": len(moments), "games": len(games), "players": len(players)}


def load_moments(game_id: str, parquet_dir: Path = PARQUET_DIR) -> pl.DataFrame:
    return pl.read_parquet(parquet_dir / "moments" / f"{game_id}.parquet")


def load_games(parquet_dir: Path = PARQUET_DIR) -> pl.DataFrame:
    return pl.read_parquet(parquet_dir / "games" / "*.parquet")


def event_frames(moments: pl.DataFrame, event_id: int) -> list[dict]:
    """Shape one event's moments into render-ready frames for studio/viz.

    Each frame: {quarter, game_clock, shot_clock, ball: [x, y, z] | None,
                 players: [[team_id, player_id, x, y], ...]}
    """
    ev = moments.filter(pl.col("event_id") == event_id).sort("moment_idx")
    frames = []
    for _, group in ev.group_by("moment_idx", maintain_order=True):
        first = group.row(0, named=True)
        frame = {
            "quarter": first["quarter"],
            "game_clock": first["game_clock"],
            "shot_clock": first["shot_clock"],
            "ball": None,
            "players": [],
        }
        for r in group.iter_rows(named=True):
            if r["entity"] == "ball":
                frame["ball"] = [r["x"], r["y"], r["z"]]
            else:
                frame["players"].append([r["team_id"], r["player_id"], r["x"], r["y"]])
        frames.append(frame)
    return frames
