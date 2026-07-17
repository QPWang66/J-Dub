from pathlib import Path

import polars as pl
import pytest

from jdub.data import (
    dedupe_moments,
    event_frames,
    normalize_offense_direction,
    parse_game,
    parse_to_parquet,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fixture_game.json"


@pytest.fixture(scope="module")
def parsed():
    return parse_game(FIXTURE)


def test_parse_schema_and_counts(parsed):
    moments, games, players = parsed
    assert moments.columns == [
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
    ]
    assert games.height == 1
    assert games["game_id"][0] == "0021500492"
    assert players.height == 26
    # every moment has exactly one ball row and <= 10 player rows
    per_moment = moments.group_by("event_id", "moment_idx").agg(
        (pl.col("entity") == "ball").sum().alias("balls"),
        (pl.col("entity") == "player").sum().alias("players"),
    )
    assert per_moment["balls"].max() == 1
    assert per_moment["players"].max() == 10


def test_dedupe_keeps_first_occurrence():
    df = pl.DataFrame(
        {
            "game_id": ["g"] * 4,
            "event_id": [1, 1, 2, 2],
            "moment_idx": [0, 1, 0, 1],
            "quarter": [1, 1, 1, 1],
            "game_clock": [700.0, 699.96, 699.96, 699.92],  # event 2 overlaps event 1's tail
            "entity": ["ball"] * 4,
            "x": [1.0, 2.0, 99.0, 3.0],
        }
    )
    out = dedupe_moments(df)
    assert out.select("event_id", "moment_idx").rows() == [(1, 0), (1, 1), (2, 1)]
    assert 99.0 not in out["x"]  # duplicated frame dropped, first kept


def test_dedupe_drops_whole_moments_not_rows(parsed):
    moments, _, _ = parsed
    # all entity rows of a kept moment survive together: no moment with a lone entity
    sizes = moments.group_by("event_id", "moment_idx").len()
    assert sizes["len"].min() >= 10


def test_normalize_offense_direction():
    df = pl.DataFrame(
        {
            "game_id": ["g"] * 4,
            "event_id": [1, 1, 2, 2],
            "entity": ["ball", "player", "ball", "player"],
            "x": [80.0, 90.0, 10.0, 20.0],
            "y": [10.0, 40.0, 10.0, 40.0],
        }
    )
    out = normalize_offense_direction(df)
    # event 1: ball on right half -> flipped; event 2 untouched
    assert out.filter(pl.col("event_id") == 1)["x"].to_list() == [14.0, 4.0]
    assert out.filter(pl.col("event_id") == 1)["y"].to_list() == [40.0, 10.0]
    assert out.filter(pl.col("event_id") == 2)["x"].to_list() == [10.0, 20.0]


def test_event_frames(parsed):
    moments, _, _ = parsed
    frames = event_frames(moments, 1)
    assert 0 < len(frames) <= 80
    f = frames[0]
    assert f["quarter"] == 1
    assert len(f["ball"]) == 3
    assert len(f["players"]) == 10
    # game clock never increases within an event
    clocks = [g["game_clock"] for g in frames]
    assert all(a >= b for a, b in zip(clocks, clocks[1:]))


def test_parse_to_parquet_roundtrip(tmp_path):
    counts = parse_to_parquet(FIXTURE, out_dir=tmp_path)
    assert counts["games"] == 1
    assert counts["moments"] > 0
    m = pl.read_parquet(tmp_path / "moments" / "0021500492.parquet")
    assert m.height == counts["moments"]
