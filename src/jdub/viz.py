"""Render one event to mp4/gif — same top-down view as studio's M1 layer."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib import animation
from matplotlib.patches import Arc, Circle, Rectangle

from jdub.data import PARQUET_DIR, event_frames, load_games, load_moments

TRAIL = 25
FPS = 25
MIN_SEP = 2.0  # ft; dots are ~1 ft radius, keep near-coincident players visibly distinct


def spread_overlaps(
    pts: list[tuple[float, float]], min_sep: float = MIN_SEP, iters: int = 8
) -> list[tuple[float, float]]:
    """Nudge points apart so no pair is closer than min_sep. Display-only, data untouched."""
    p = [list(q) for q in pts]
    for _ in range(iters):
        moved = False
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                dx, dy = p[j][0] - p[i][0], p[j][1] - p[i][1]
                d = math.hypot(dx, dy)
                if d >= min_sep:
                    continue
                if d < 1e-6:
                    dx, dy, d = 1.0, 0.0, 1.0  # exactly coincident: pick an arbitrary axis
                push, ux, uy = (min_sep - d) / 2, dx / d, dy / d
                p[i][0] -= ux * push
                p[i][1] -= uy * push
                p[j][0] += ux * push
                p[j][1] += uy * push
                moved = True
        if not moved:
            break
    return [(q[0], q[1]) for q in p]


def draw_court(ax: plt.Axes) -> None:
    kw = {"color": "#5a6472", "lw": 1.5, "fill": False}
    ax.add_patch(Rectangle((0, 0), 94, 50, **kw))
    ax.plot([47, 47], [0, 50], color=kw["color"], lw=1.5)
    ax.add_patch(Circle((47, 25), 6, **kw))
    for bx, d in ((5.25, 1), (88.75, -1)):  # hoop x, direction into court
        base = bx - d * 5.25
        ax.add_patch(Rectangle((min(base, base + d * 19), 17), 19, 16, **kw))
        ax.add_patch(Circle((base + d * 19, 25), 6, **kw))
        ax.add_patch(Circle((bx, 25), 0.75, **kw))
        ax.plot([base + d * 4] * 2, [22, 28], color=kw["color"], lw=1.5)
        ax.plot([base, base + d * 14], [3, 3], color=kw["color"], lw=1.5)
        ax.plot([base, base + d * 14], [47, 47], color=kw["color"], lw=1.5)
        a = math.degrees(math.asin(22 / 23.75))
        theta = (-a, a) if d == 1 else (180 - a, 180 + a)
        ax.add_patch(
            Arc((bx, 25), 47.5, 47.5, theta1=theta[0], theta2=theta[1], color=kw["color"], lw=1.5)
        )
    ax.set_xlim(-2, 96)
    ax.set_ylim(52, -2)  # y down, matching SportVU origin
    ax.set_aspect("equal")
    ax.axis("off")


def render_event(game_id: str, event_id: int, out: Path, parquet_dir: Path = PARQUET_DIR) -> Path:
    """Animate one event and save to out (.mp4 via ffmpeg, .gif via pillow)."""
    moments = load_moments(game_id, parquet_dir)
    frames = event_frames(moments, event_id)
    if not frames:
        raise ValueError(f"event {event_id} has no moments in game {game_id}")
    game = load_games(parquet_dir).filter(pl.col("game_id") == game_id).to_dicts()[0]
    home_id = game["home_team_id"]

    fig, ax = plt.subplots(figsize=(9.4, 5.4), facecolor="#1e232b")
    draw_court(ax)
    home_dots = ax.scatter([], [], s=180, color="#e4572e", zorder=3, label=game["home_abbr"])
    away_dots = ax.scatter([], [], s=180, color="#3f88c5", zorder=3, label=game["visitor_abbr"])
    (trail,) = ax.plot([], [], color="#f6ae2d", lw=2, alpha=0.6, zorder=4)
    ball = ax.scatter([], [], color="#f6ae2d", zorder=5)
    clock = ax.text(47, -0.8, "", ha="center", color="#dddddd", fontsize=11)
    ax.legend(loc="lower center", ncol=2, frameon=False, labelcolor="#dddddd")

    def update(i: int):
        f = frames[i]
        spread = spread_overlaps([(x, y) for _, _, x, y in f["players"]])
        home_xy = [p for (t, *_), p in zip(f["players"], spread) if t == home_id]
        away_xy = [p for (t, *_), p in zip(f["players"], spread) if t != home_id]
        home_dots.set_offsets(home_xy or [(-10, -10)])
        away_dots.set_offsets(away_xy or [(-10, -10)])
        pts = [g["ball"] for g in frames[max(0, i - TRAIL) : i + 1] if g["ball"]]
        trail.set_data([p[0] for p in pts], [p[1] for p in pts])
        if f["ball"]:
            ball.set_offsets([f["ball"][:2]])
            ball.set_sizes([60 + f["ball"][2] * 8])
        sc = f"  sc {f['shot_clock']:.1f}" if f["shot_clock"] is not None else ""
        clock.set_text(
            f"Q{f['quarter']} {int(f['game_clock'] // 60):02d}:{f['game_clock'] % 60:04.1f}{sc}"
        )
        return home_dots, away_dots, trail, ball, clock

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    writer = "pillow" if out.suffix == ".gif" else "ffmpeg"
    anim.save(out, writer=writer, fps=FPS, savefig_kwargs={"facecolor": "#1e232b"})
    plt.close(fig)
    return out
