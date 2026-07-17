import polars as pl

from jdub.robustness import _label_agree, f1, perturb


def _act(event_id, type_, p1, p2, s, e, **extra):
    return {
        "event_id": event_id,
        "type": type_,
        "p1": p1,
        "p2": p2,
        "start_idx": s,
        "end_idx": e,
        **extra,
    }


def test_f1_identity_and_miss():
    clean = [_act(1, "screen", 10, 20, 100, 120), _act(2, "drive", 30, None, 50, 80)]
    assert f1(clean, clean) == 1.0
    assert f1(clean, clean[:1]) < 1.0
    assert f1([], []) == 1.0
    # shifted beyond IoU bar = no match
    moved = [_act(1, "screen", 10, 20, 300, 320), clean[1]]
    assert f1(clean, moved) == 0.5


def test_label_agreement():
    clean = [_act(1, "coverage", 10, 20, 100, 120, label="drop")]
    flipped = [_act(1, "coverage", 10, 20, 102, 122, label="switch")]
    assert _label_agree(clean, clean) == 1.0
    assert _label_agree(clean, flipped) == 0.0


def test_perturb_zero_is_identity_and_sigma_moves_xy():
    m = pl.DataFrame({"x": [10.0, 20.0], "y": [5.0, 6.0], "z": [0.0, 0.0]})
    assert perturb(m, 0.0).equals(m)
    p = perturb(m, 1.0, seed=1)
    assert p["x"].to_list() != m["x"].to_list()
    assert p["z"].equals(m["z"])
