"""M4: fact compiler + salience planner + grounded commentary (zh/en).

The iron rule (PRD §6): the language layer only ever sees the fact stream,
never coordinates. Facts below the confidence bar are voiced with hedged
wording ("似乎") or dropped. The template renderer is hallucination-free by
construction; --llm swaps in a local model (any OpenAI-compatible endpoint,
ollama by default) for style, constrained to cite the same fact ids, with
the template as fallback.
"""

from __future__ import annotations

import json
import os
import re

import polars as pl
import requests

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
    "post_up": 2.5,
    "offball_screen": 2.5,
    "iso": 2.2,
    "drive": 2.0,
    "handoff": 2.0,
    "cut": 1.5,
    "transition": 1.5,
    "pass": 0.5,  # context, rarely the story
}
COVERAGE = {
    "zh": {
        "drop": "沉退护框",
        "switch": "换防",
        "blitz": "上抢夹击",
        "over": "挤过掩护",
        "under": "从掩护下方绕过",
    },
    "en": {
        "drop": "drop back and protect the rim",
        "switch": "switch",
        "blitz": "blitz the ball handler",
        "over": "fight over the screen",
        "under": "go under the screen",
    },
}
TEMPLATES = {
    "zh": {
        "screen": "{p1}给{p2}做掩护。",
        "coverage": "{screener}上提给{handler}做掩护,{p1}选择{cov}。",
        "drive": "{p1}持球突破,直插禁区。",
        "handoff": "{p1}与{p2}完成手递手交接。",
        "cut": "{p1}空切杀向篮下。",
        "post_up": "{p1}低位背身要位。",
        "offball_screen": "{p1}给{p2}做无球掩护。",
        "iso": "{p1}单打,队友拉开空间。",
        "transition": "{p1}持球推转换。",
        "pass": "{p1}转移给{p2}。",
        "outcome": "回合终结:{desc}",
        "hedge": "似乎{text}",
    },
    "en": {
        "screen": "{p1} sets a screen for {p2}.",
        "coverage": "{screener} comes up to screen for {handler}; {p1} chooses to {cov}.",
        "drive": "{p1} drives hard into the paint.",
        "handoff": "{p1} hands off to {p2}.",
        "cut": "{p1} cuts to the basket.",
        "post_up": "{p1} posts up on the block.",
        "offball_screen": "{p1} sets an off-ball screen for {p2}.",
        "iso": "{p1} isolates, teammates spacing the floor.",
        "transition": "{p1} pushes the pace in transition.",
        "pass": "{p1} swings it to {p2}.",
        "outcome": "Possession ends: {desc}",
        "hedge": "It looks like {text}",
    },
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
        if f["type"] in COVERAGE["zh"]:  # pull in the screen this coverage answers
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


def render_text(facts: list[dict], lang: str = "zh") -> list[dict]:
    """Deterministic grounded sentences: [{text, start_idx, refs}]."""
    tpl = TEMPLATES[lang]
    cov = COVERAGE[lang]
    covered_screens = {f.get("screen_start_idx", -1) for f in facts if f["type"] in cov}
    out: list[dict] = []
    for f in facts:
        t = f["type"]
        if t == "screen" and f["start_idx"] in covered_screens:
            continue  # voiced together with its coverage
        if t in cov:
            text = tpl["coverage"].format(cov=cov[t], **f["actors"])
        elif t == "outcome":
            text = tpl["outcome"].format(desc=f["desc"])
        elif t in tpl:
            text = tpl[t].format(**f["actors"])
        else:
            continue
        if t != "outcome" and f["confidence"] < CONF_BAR:
            text = tpl["hedge"].format(text=text)
        out.append({"text": text, "start_idx": f["start_idx"], "refs": [f["id"]]})
    return out


LLM_PROMPTS = {
    "zh": """你是一名克制、专业的篮球战术解说。仅根据下面的事实流写3-5句中文解说。
铁律:不得引入任何事实流之外的信息;置信度低于{bar}的事实必须用"似乎/疑似"等模糊措辞;
每句话末尾用方括号标注引用的事实id,例如 [a1,a3]。逐句换行输出,不要编号,不要其他内容。

事实流(JSON):
{facts}
""",
    "en": """You are a restrained, professional basketball tactics commentator. Write 3-5 \
sentences of English commentary based ONLY on the fact stream below.
Iron rules: introduce nothing beyond the fact stream; hedge any fact with confidence \
below {bar} ("looks like", "appears to"); end every sentence with the cited fact ids \
in square brackets, e.g. [a1,a3]. One sentence per line, no numbering, nothing else.

Fact stream (JSON):
{facts}
""",
}


LLM_URL = os.environ.get("JDUB_LLM_URL", "http://localhost:11434/v1/chat/completions")
LLM_MODEL = os.environ.get("JDUB_LLM_MODEL", "qwen3:8b")


def render_llm(facts: list[dict], lang: str = "zh") -> list[dict] | None:
    """Style layer via a local OpenAI-compatible endpoint (`ollama serve` by
    default; override with JDUB_LLM_URL / JDUB_LLM_MODEL). Returns None on any
    failure (caller falls back to the template renderer)."""
    prompt = LLM_PROMPTS[lang].format(bar=CONF_BAR, facts=json.dumps(facts, ensure_ascii=False))
    try:
        r = requests.post(
            LLM_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=180,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return None
    text = re.sub(r"(?s)<think>.*?</think>", "", text)  # qwen3 reasoning block
    if not text.strip():
        return None
    by_id = {f["id"]: f for f in facts}
    out: list[dict] = []
    for line in text.strip().splitlines():
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


def generate(game_id: str, event_id: int, llm: bool = False, lang: str = "zh") -> dict:
    facts = compile_facts(game_id, event_id)
    salient = plan_salience(facts)
    sentences = (render_llm(salient, lang) if llm else None) or render_text(salient, lang)
    return {"facts": facts, "salient": [f["id"] for f in salient], "sentences": sentences}


def _main() -> None:  # smoke check: uv run python -m jdub.commentary <game> <event>
    import sys

    print(json.dumps(generate(sys.argv[1], int(sys.argv[2])), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    _main()
