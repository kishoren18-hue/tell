"""Smoke test: every model trains and evaluates end to end on a small dataset."""
import json
from pathlib import Path

import numpy as np

from tell.data import direction_from_landing, to_samples
from tell.metrics import binomial_p
from tell.train import TrainCfg, evaluate

FIX = json.loads((Path(__file__).parent / "fixtures" / "parity.json").read_text())


def test_court_geometry_matches_app():
    assert direction_from_landing("left", 1.0, 6) == "straight"
    assert direction_from_landing("left", 6, 6) == "right"
    assert direction_from_landing("right", 2, 5) == "left"


def test_binomial():
    assert abs(binomial_p(9, 10, 0.5) - 11 / 1024) < 1e-12


def test_all_models_run():
    S = to_samples(FIX["dataset"])
    rep = evaluate(S, ["logreg", "lstm", "rallyworld"], [0, 200], TrainCfg(epochs=3), log=lambda *_: None)
    for name in ["logreg", "lstm", "rallyworld"]:
        for h in ["0", "200"]:
            r = rep["models"][name][h]
            assert 0 <= r["accuracy"] <= 1 and np.isfinite(r["ce_bits"])
    assert rep["models"]["rallyworld"]["world_forecast_error_200ms"] > 0
