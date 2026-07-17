from jdub.commentary import CONF_BAR, plan_salience, render_zh

FACTS = [
    {
        "id": "a0",
        "type": "screen",
        "start_idx": 10,
        "end_idx": 14,
        "gc": 700.0,
        "actors": {"p1": "Koufos", "p2": "Collison"},
        "confidence": 0.6,
    },
    {
        "id": "a1",
        "type": "drop",
        "start_idx": 12,
        "end_idx": 40,
        "gc": 699.0,
        "actors": {"p1": "Adams", "p2": "Morrow", "handler": "Collison", "screener": "Koufos"},
        "confidence": 1.0,
        "screen_start_idx": 10,
    },
    {
        "id": "a2",
        "type": "drive",
        "start_idx": 50,
        "end_idx": 70,
        "gc": 697.0,
        "actors": {"p1": "Collison", "p2": None},
        "confidence": 0.55,
    },
    {
        "id": "outcome",
        "type": "outcome",
        "start_idx": 70,
        "end_idx": 70,
        "gc": None,
        "actors": {},
        "confidence": 1.0,
        "desc": "Belinelli 21' Jump Shot",
    },
]


def test_salience_keeps_outcome_and_pairs_screen_with_coverage():
    picked = plan_salience(FACTS, k=3)
    ids = [f["id"] for f in picked]
    assert "outcome" in ids
    assert "a1" in ids  # highest-weight fact
    assert ids == sorted(ids, key=lambda i: next(f["start_idx"] for f in FACTS if f["id"] == i))


def test_render_grounded_and_hedged():
    sents = render_zh(FACTS)
    text = " ".join(s["text"] for s in sents)
    assert "Adams选择沉退护框" in text
    assert "似乎" in text  # drive at 0.55 < CONF_BAR must be hedged
    assert 0.55 < CONF_BAR
    # screen voiced inside its coverage sentence, not duplicated
    assert sum("做掩护" in s["text"] for s in sents) == 1
    # every sentence anchors to a fact frame
    assert all(isinstance(s["start_idx"], int) and s["refs"] for s in sents)
