"""M4: fact compiler + salience planner + grounded Chinese commentary.

The iron rule (PRD §6): the language layer only ever sees the fact stream,
never coordinates. Facts below the confidence bar are voiced with hedged
wording ("似乎") or dropped. The template renderer is hallucination-free by
construction; --llm swaps in `claude -p` for style, constrained to cite the
same fact ids, with the template as fallback.
"""

from __future__ import annotations

import json
import re
import subprocess

import polars as pl

from jdub.data import PARQUET_DIR

CONF_BAR = 0.7
MAX_FACTS = 5
TYPE_WEIGHT = {
    "screen": 3.0,
    "switch": 3.0,
    "drop": 3.0,
    "blitz": 3.0,
    "over": 2.5,
    "under": 2.5,
    "drive": 2.0,
    "handoff": 2.0,
    "cut": 1.5,
}
COVERAGE_ZH = {
    "drop": "沉退护框",
    "switch": "换防",
    "blitz": "上抢夹击",
    "over": "挤过掩护",
    "under": "从掩护下方绕过",
}


def _names(game_id: str) -> dict[int, str]:
    df = pl.read_parquet(PARQUET_DIR / "players" / f"{game_id}.parquet")
    return {r["player_id"]: r["lastname"] for r in df.iter_rows(named=True)}


def compile_facts(game_id: str, event_id: int) -> list[dict]:
    """Timestamped, confidence-scored fact stream for one possession."""
    facts: list[dict] = []
    names = _names(game_id)

    def who(pid) -> str:
        return names.get(pid, str(pid))

    actions = (
        pl.read_parquet(PARQUET_DIR / "actions" / f"{game_id}.parquet")
        .filter(pl.col("event_id") == event_id)
        .to_dicts()
    )
    for a in actions:
        facts.append(
            {
                "id": f"a{len(facts)}",
                "type": a["type"],
                "start_idx": a["start_idx"],
                "end_idx": a["end_idx"],
                "gc": a["gc_start"],
                "actors": {"p1": who(a["p1"]), "p2": who(a["p2"]) if a["p2"] else None},
                "confidence": a["confidence"],
            }
        )
    cov_path = PARQUET_DIR / "coverages" / f"{game_id}.parquet"
    if cov_path.exists():
        for c in pl.read_parquet(cov_path).filter(pl.col("event_id") == event_id).to_dicts():
            facts.append(
                {
                    "id": f"a{len(facts)}",
                    "type": c["coverage"],
                    "start_idx": c["start_idx"],
                    "end_idx": c["end_idx"],
                    "gc": c["gc_start"],
                    "actors": {
                        "p1": who(c["d2"]),
                        "p2": who(c["d1"]),
                        "handler": who(c["handler"]),
                        "screener": who(c["screener"]),
                    },
                    "confidence": c["confidence"],
                    "screen_start_idx": c["screen_start_idx"],
                }
            )
    pbp_path = PARQUET_DIR / "pbp" / f"{game_id}.parquet"
    if pbp_path.exists():
        row = pl.read_parquet(pbp_path).filter(pl.col("event_id") == event_id).to_dicts()
        if row:
            facts.append(
                {
                    "id": "outcome",
                    "type": "outcome",
                    "start_idx": max((f["end_idx"] for f in facts), default=0),
                    "end_idx": max((f["end_idx"] for f in facts), default=0),
                    "gc": None,
                    "actors": {},
                    "confidence": 1.0,
                    "desc": row[0]["desc"],
                }
            )
    return sorted(facts, key=lambda f: f["start_idx"])


def plan_salience(facts: list[dict], k: int = MAX_FACTS) -> list[dict]:
    """Pick the k facts most worth saying: outcome always, screens with their
    coverage as a pair, the rest by type weight x confidence."""
    outcome = [f for f in facts if f["type"] == "outcome"]
    rest = [f for f in facts if f["type"] != "outcome"]
    scored = sorted(
        rest, key=lambda f: TYPE_WEIGHT.get(f["type"], 1.0) * f["confidence"], reverse=True
    )
    picked: list[dict] = []
    for f in scored:
        if len(picked) >= k - len(outcome):
            break
        if f in picked:
            continue
        picked.append(f)
        if f["type"] in COVERAGE_ZH:  # pull in the screen this coverage answers
            twin = next(
                (
                    g
                    for g in rest
                    if g["type"] == "screen" and g["start_idx"] == f.get("screen_start_idx")
                ),
                None,
            )
            if twin and twin not in picked and len(picked) < k - len(outcome):
                picked.append(twin)
    return sorted(picked + outcome, key=lambda f: f["start_idx"])


def _hedge(conf: float, text: str) -> str:
    return text if conf >= CONF_BAR else f"似乎{text}"


def render_zh(facts: list[dict]) -> list[dict]:
    """Deterministic grounded sentences: [{text, start_idx, refs}]."""
    out: list[dict] = []
    covered_screens: set[int] = set()
    for f in facts:
        if f["type"] in COVERAGE_ZH:
            covered_screens.add(f.get("screen_start_idx", -1))
    for f in facts:
        a = f["actors"]
        t = f["type"]
        if t == "screen":
            if f["start_idx"] in covered_screens:
                continue  # voiced together with its coverage
            text = _hedge(f["confidence"], f"{a['p1']}给{a['p2']}做掩护。")
        elif t in COVERAGE_ZH:
            text = _hedge(
                f["confidence"],
                f"{a['screener']}上提给{a['handler']}做掩护,{a['p1']}选择{COVERAGE_ZH[t]}。",
            )
        elif t == "drive":
            text = _hedge(f["confidence"], f"{a['p1']}持球突破,直插禁区。")
        elif t == "handoff":
            text = _hedge(f["confidence"], f"{a['p1']}与{a['p2']}完成手递手交接。")
        elif t == "cut":
            text = _hedge(f["confidence"], f"{a['p1']}空切杀向篮下。")
        elif t == "outcome":
            text = f"回合终结:{f['desc']}"
        else:
            continue
        out.append({"text": text, "start_idx": f["start_idx"], "refs": [f["id"]]})
    return out


LLM_PROMPT = """你是一名克制、专业的篮球战术解说。仅根据下面的事实流写3-5句中文解说。
铁律:不得引入任何事实流之外的信息;置信度低于{bar}的事实必须用"似乎/疑似"等模糊措辞;
每句话末尾用方括号标注引用的事实id,例如 [a1,a3]。逐句换行输出,不要编号,不要其他内容。

事实流(JSON):
{facts}
"""


def render_llm(facts: list[dict]) -> list[dict] | None:
    """Style layer via the `claude` CLI; returns None on any failure (caller
    falls back to the template renderer)."""
    prompt = LLM_PROMPT.format(bar=CONF_BAR, facts=json.dumps(facts, ensure_ascii=False))
    try:
        r = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    by_id = {f["id"]: f for f in facts}
    out: list[dict] = []
    for line in r.stdout.strip().splitlines():
        m = re.search(r"\[([a-z0-9,\s]+)\]\s*$", line.strip())
        if not m:
            continue
        refs = [x.strip() for x in m.group(1).split(",") if x.strip() in by_id]
        if not refs:
            continue
        out.append(
            {
                "text": re.sub(r"\s*\[[a-z0-9,\s]+\]\s*$", "", line.strip()),
                "start_idx": min(by_id[x]["start_idx"] for x in refs),
                "refs": refs,
            }
        )
    return out or None


def generate(game_id: str, event_id: int, llm: bool = False) -> dict:
    facts = compile_facts(game_id, event_id)
    salient = plan_salience(facts)
    sentences = (render_llm(salient) if llm else None) or render_zh(salient)
    return {"facts": facts, "salient": [f["id"] for f in salient], "sentences": sentences}


def _main() -> None:  # smoke check: uv run python -m jdub.commentary <game> <event>
    import sys

    print(json.dumps(generate(sys.argv[1], int(sys.argv[2])), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    _main()
