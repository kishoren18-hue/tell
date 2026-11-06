"""Load a Tell dataset export (the JSON the app writes from Film > Export dataset)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import sequence_features, summary_features

DIRECTIONS = ["left", "straight", "right"]
TAKEOFF = {
    "left": (1.0, -0.8), "middle": (4.5, -0.6), "right": (8.0, -0.8),
    "back-left": (1.5, -3.5), "back-middle": (4.5, -3.5), "back-right": (7.5, -3.5),
}
STRAIGHT_CONE = 12.0


def direction_from_landing(position: str, x: float, y: float) -> str:
    tx, ty = TAKEOFF[position]
    a = math.degrees(math.atan2(x - tx, y - ty))
    if a < -STRAIGHT_CONE:
        return "left"
    if a > STRAIGHT_CONE:
        return "right"
    return "straight"


def canonical_label(direction: str, hand: str) -> int:
    i = DIRECTIONS.index(direction)
    return 2 - i if hand == "L" else i


@dataclass
class Samples:
    """Usable attacks as arrays. Sequences are canonical (right-handed)."""
    seq: np.ndarray          # (N, T, FRAME_DIM)
    y: np.ndarray            # (N,) canonical class
    match: np.ndarray        # (N,) match id
    hitter: np.ndarray       # (N,) hitter id
    hand: np.ndarray         # (N,) 'R'/'L'
    landing: np.ndarray      # (N, 2) [lateral offset from takeoff, depth], canonical; NaN if unknown
    attack_ids: list = field(default_factory=list)
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.y)

    def summary(self, horizon_ms: float) -> np.ndarray:
        return np.stack([summary_features(s, horizon_ms) for s in self.seq])


def load_dataset(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("format") != "tell-dataset":
        raise ValueError(f"{path} is not a Tell dataset export")
    if d.get("version") != 1:
        raise ValueError(f"unsupported dataset version {d.get('version')}")
    return d


def to_samples(d: dict) -> Samples:
    hands = {p["id"]: p.get("hand", "R") for p in d["players"]}
    seqs, ys, ms, hs, hd, land, ids = [], [], [], [], [], [], []
    skipped = 0
    for a in d["attacks"]:
        hand = hands.get(a["hitterId"], "R")
        if not a.get("frames"):
            skipped += 1
            continue
        seq = sequence_features(a["frames"], hand)
        if seq is None:
            skipped += 1
            continue
        seqs.append(seq)
        ys.append(canonical_label(a["direction"], hand))
        ms.append(a["matchId"])
        hs.append(a["hitterId"])
        hd.append(hand)
        ids.append(a["id"])
        if a.get("landing"):
            tx, _ = TAKEOFF[a["position"]]
            dx = a["landing"]["x"] - tx
            land.append([-dx if hand == "L" else dx, a["landing"]["y"]])
        else:
            land.append([np.nan, np.nan])
    if not seqs:
        raise ValueError("no attacks with usable pose in this dataset")
    return Samples(np.stack(seqs), np.array(ys), np.array(ms), np.array(hs), np.array(hd),
                   np.array(land, dtype=float), ids, skipped)
