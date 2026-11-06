"""Feature extraction, a line-for-line port of src/lib/features.ts.

tests/test_parity.py checks this against numbers produced by the TypeScript
code, so the phone app and the desktop pipeline see identical features.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

SAMPLE_HZ = 30
CONTACT_INDEX = 36
WINDOW_FRAMES = 46

FEATURE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
N_JOINT_FEATS = len(FEATURE_JOINTS) * 3  # 39
ENGINEERED_NAMES = [
    "shoulder_yaw", "hip_yaw", "hip_shoulder_separation", "torso_pitch", "torso_roll",
    "hitting_elbow_angle", "hitting_shoulder_abduction", "wrist_rel_shoulder_x",
    "wrist_rel_shoulder_y", "wrist_rel_shoulder_z", "head_yaw", "off_wrist_height",
    "hip_image_x", "hip_image_y",
]
N_ENG = len(ENGINEERED_NAMES)  # 14
FRAME_DIM = N_JOINT_FEATS + N_ENG  # 53
SUMMARY_WINDOW = 10
SUMMARY_DIM = N_ENG * 5 + N_JOINT_FEATS * 2  # 148

LM = dict(nose=0, l_ear=7, r_ear=8, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14,
          l_wrist=15, r_wrist=16, l_hip=23, r_hip=24)

_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
          (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR_INDEX = list(range(33))
for _a, _b in _PAIRS:
    MIRROR_INDEX[_a], MIRROR_INDEX[_b] = _b, _a


def mirror_frame(f: dict) -> dict:
    """Mirror a left-handed hitter into the canonical right-handed frame."""
    w = np.asarray(f["w"], dtype=float).reshape(33, 3)[MIRROR_INDEX].copy()
    p = np.asarray(f["p"], dtype=float).reshape(33, 3)[MIRROR_INDEX].copy()
    w[:, 0] = -w[:, 0]
    p[:, 0] = 1 - p[:, 0]
    return {"w": w.reshape(-1).tolist(), "p": p.reshape(-1).tolist()}


def canonical_frames(frames: Sequence[Optional[dict]], hand: str) -> list:
    if hand == "R":
        return list(frames)
    return [mirror_frame(f) if f else None for f in frames]


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    if d < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / d)))


def frame_features(f: dict) -> list[float]:
    W = np.asarray(f["w"], dtype=float).reshape(33, 3)
    P = np.asarray(f["p"], dtype=float).reshape(33, 3)
    ls, rs = W[LM["l_shoulder"]], W[LM["r_shoulder"]]
    lh, rh = W[LM["l_hip"]], W[LM["r_hip"]]
    s_mid = (ls + rs) / 2
    h_mid = (lh + rh) / 2
    torso = s_mid - h_mid
    L = max(float(np.linalg.norm(torso)), 0.05)

    out: list[float] = []
    for i in FEATURE_JOINTS:
        out.extend(((W[i] - h_mid) / L).tolist())

    sv = rs - ls
    hv = rh - lh
    shoulder_yaw = math.atan2(sv[2], sv[0])
    hip_yaw = math.atan2(hv[2], hv[0])
    sep = shoulder_yaw - hip_yaw
    while sep > math.pi:
        sep -= 2 * math.pi
    while sep < -math.pi:
        sep += 2 * math.pi
    pitch = math.atan2(torso[2], -torso[1])
    roll = math.atan2(torso[0], -torso[1])

    re, rw = W[LM["r_elbow"]], W[LM["r_wrist"]]
    elbow = _angle_between(rs - re, rw - re)
    abduction = _angle_between(re - rs, h_mid - s_mid)
    wr = rw - rs

    nose = W[LM["nose"]]
    ear_mid = (W[LM["l_ear"]] + W[LM["r_ear"]]) / 2
    head_yaw = math.atan2(nose[0] - ear_mid[0], nose[2] - ear_mid[2])
    off_wrist_h = (W[LM["l_wrist"]][1] - s_mid[1]) / L

    hip_img_x = (P[LM["l_hip"], 0] + P[LM["r_hip"], 0]) / 2
    hip_img_y = (P[LM["l_hip"], 1] + P[LM["r_hip"], 1]) / 2

    out.extend([shoulder_yaw, hip_yaw, sep, pitch, roll, elbow, abduction,
                wr[0] / L, wr[1] / L, wr[2] / L, head_yaw, off_wrist_h, hip_img_x, hip_img_y])
    return [float(v) for v in out]


def interpolate(rows: Sequence[Optional[list]]) -> np.ndarray:
    n = len(rows)
    idx = [i for i, r in enumerate(rows) if r is not None]
    out = []
    for i in range(n):
        if rows[i] is not None:
            out.append(np.asarray(rows[i], dtype=float))
            continue
        lo = max((k for k in idx if k < i), default=-1)
        hi = min((k for k in idx if k > i), default=-1)
        if lo < 0:
            out.append(np.asarray(rows[hi], dtype=float))
        elif hi < 0:
            out.append(np.asarray(rows[lo], dtype=float))
        else:
            t = (i - lo) / (hi - lo)
            a = np.asarray(rows[lo], dtype=float)
            b = np.asarray(rows[hi], dtype=float)
            out.append(a + (b - a) * t)
    return np.stack(out)


def sequence_features(frames: Sequence[Optional[dict]], hand: str) -> Optional[np.ndarray]:
    """(T, FRAME_DIM) array, or None if fewer than half the frames have a detection."""
    canon = canonical_frames(frames, hand)
    raw = [frame_features(f) if f else None for f in canon]
    valid = sum(r is not None for r in raw)
    if valid < math.ceil(len(raw) / 2):
        return None
    return interpolate(raw)


def cutoff_index(horizon_ms: float) -> int:
    # JS Math.round rounds .5 up; Python round() rounds half to even. Match JS.
    return CONTACT_INDEX - int(math.floor(horizon_ms / 1000 * SAMPLE_HZ + 0.5))


def summary_features(seq: np.ndarray, horizon_ms: float) -> np.ndarray:
    e = cutoff_index(horizon_ms)
    s = max(0, e - SUMMARY_WINDOW + 1)
    win = seq[s:e + 1]
    W = len(win)
    out: list[float] = []
    for k in range(N_ENG):
        vals = win[:, N_JOINT_FEATS + k]
        out.extend([vals[-1], vals.mean(), (vals[-1] - vals[0]) / max(1, W - 1), vals.min(), vals.max()])
    last = seq[e]
    prev = seq[max(0, e - 3)]
    out.extend(last[:N_JOINT_FEATS].tolist())
    out.extend(((last[:N_JOINT_FEATS] - prev[:N_JOINT_FEATS]) / 3).tolist())
    return np.asarray(out, dtype=float)


def state_of(row: np.ndarray) -> np.ndarray:
    """Pose state used by the world models: joints + hip image xy (41 dims)."""
    return np.concatenate([row[:N_JOINT_FEATS], row[N_JOINT_FEATS + 12:N_JOINT_FEATS + 14]])
