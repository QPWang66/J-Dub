"""jdub CLI: download, parse, studio, viz."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def download(name: str = typer.Argument("01.01.2016.CHA.at.TOR")) -> None:
    """Download one game from the SportVU mirror (e.g. '01.01.2016.CHA.at.TOR')."""
    from jdub.data import download_game

    path = download_game(name)
    typer.echo(f"extracted {path}")


@app.command()
def parse(json_path: Path) -> None:
    """Parse raw SportVU JSON into Parquet tables under data/parquet/."""
    from jdub.data import parse_to_parquet

    counts = parse_to_parquet(json_path)
    typer.echo(f"wrote {counts}")


@app.command()
def detect(game_id: str) -> None:
    """Run M2 detection (matchups + atomic actions) for a parsed game."""
    from jdub.events import detect_to_parquet

    counts = detect_to_parquet(game_id)
    typer.echo(f"wrote {counts}")


@app.command()
def studio(port: int = 8000) -> None:
    """Run the jdub studio web viewer."""
    from jdub.studio import run

    typer.echo(f"jdub studio -> http://127.0.0.1:{port}")
    run(port=port)


@app.command()
def viz(
    game_id: str,
    event_id: int,
    out: Path = typer.Option(
        None, help="Output path (.mp4 or .gif). Default out/<game>_<event>.mp4"
    ),
) -> None:
    """Render one event to mp4/gif."""
    from jdub.viz import render_event

    if out is None:
        out = Path("out") / f"{game_id}_{event_id}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"wrote {render_event(game_id, event_id, out)}")
