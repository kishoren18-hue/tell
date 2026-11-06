"""The Python features must match the app's TypeScript features exactly."""
import json
from pathlib import Path

import numpy as np

from tell.features import sequence_features, summary_features

FIX = json.loads((Path(__file__).parent / "fixtures" / "parity.json").read_text())


def test_features_match_typescript():
    ds = FIX["dataset"]
    hands = {p["id"]: p["hand"] for p in ds["players"]}
    attacks = {a["id"]: a for a in ds["attacks"]}
    for e in FIX["expected"]:
        a = attacks[e["id"]]
        seq = sequence_features(a["frames"], hands[a["hitterId"]])
        np.testing.assert_allclose(seq, np.array(e["seq"]), atol=1e-6)
        np.testing.assert_allclose(summary_features(seq, 100), e["summary100"], atol=1e-6)
        np.testing.assert_allclose(summary_features(seq, 300), e["summary300"], atol=1e-6)


def test_lefty_mirroring_is_covered():
    ds = FIX["dataset"]
    hands = {p["id"]: p["hand"] for p in ds["players"]}
    assert "L" in hands.values()
