"""Re-extract pose on a laptop with a bigger MediaPipe model.

The phone uses the "full" pose model on a 1.5 s window around each tagged
contact. On a laptop you can redo that with the "heavy" model, which is more
accurate on small, far-away players, and fill in attacks the phone skipped:

    python -m tell.extract export.json --match <match id> --video match.mp4 \
        --model pose_landmarker_heavy.task --out export-heavy.json

Download models from
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .features import CONTACT_INDEX, SAMPLE_HZ, WINDOW_FRAMES

TRACK_IDS = (11, 12, 23, 24)


def centroid(p: np.ndarray) -> np.ndarray:
    return p.reshape(33, 3)[list(TRACK_IDS), :2].mean(0)


def link_track(all_cands: list[list[dict]], tap: tuple[float, float], max_jump: float = 0.12) -> list:
    """Follow the tapped player outward from the contact frame (same logic as src/lib/extract.ts)."""
    out: list = [None] * len(all_cands)

    def pick(cands, ref, limit):
        best, bd = None, float("inf")
        for c in cands:
            d = float(np.hypot(*(c["c"] - ref)))
            if d < bd:
                best, bd = c, d
        return best if best is not None and bd <= limit else None

    start = CONTACT_INDEX
    first = pick(all_cands[start], np.array(tap), 0.25)
    if first is None:
        for d in range(1, 4):
            for s in (CONTACT_INDEX - d, CONTACT_INDEX + d):
                if first is None and 0 <= s < len(all_cands):
                    f = pick(all_cands[s], np.array(tap), 0.25)
                    if f is not None:
                        first, start = f, s
    if first is None:
        return out
    out[start] = first["frame"]
    for step in (-1, 1):
        ref, gap, i = first["c"], 0, start + step
        while 0 <= i < len(all_cands):
            nxt = pick(all_cands[i], ref, max_jump * (1 + gap))
            if nxt is not None:
                out[i], ref, gap = nxt["frame"], nxt["c"], 0
            else:
                gap += 1
            i += step
    return out


def read_frame(cap, t: float):
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_attack(landmarker, cap, contact_time: float, tap: tuple[float, float], ts0: int) -> tuple[list, int]:
    import mediapipe as mp

    cands_all = []
    ts = ts0
    for i in range(WINDOW_FRAMES):
        t = contact_time + (i - CONTACT_INDEX) / SAMPLE_HZ
        rgb = read_frame(cap, t)
        ts += 34
        if rgb is None:
            cands_all.append([])
            continue
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
        cands = []
        for k, pts in enumerate(res.pose_landmarks):
            p = np.array([[q.x, q.y, q.visibility if q.visibility is not None else 1.0] for q in pts]).reshape(-1)
            w = np.array([[q.x, q.y, q.z] for q in res.pose_world_landmarks[k]]).reshape(-1)
            cands.append({"frame": {"w": w.round(6).tolist(), "p": p.round(6).tolist()}, "c": centroid(p)})
        cands_all.append(cands)
    return link_track(cands_all, tap), ts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--match", required=True, help="match id the video belongs to")
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True, help="path to pose_landmarker_*.task")
    ap.add_argument("--only-missing", action="store_true", help="skip attacks that already have pose")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import cv2
    from mediapipe.tasks.python import BaseOptions, vision

    d = json.loads(Path(args.dataset).read_text())
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=6,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    todo = [a for a in d["attacks"] if a["matchId"] == args.match and a.get("hitterTap")
            and not (args.only_missing and a.get("frames"))]
    todo.sort(key=lambda a: a["contactTime"])
    ts = 0
    for n, a in enumerate(todo, 1):
        # A fresh landmarker per attack: VIDEO mode tracks across calls, and attacks are far apart in time.
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            frames, ts = extract_attack(lm, cap, a["contactTime"], (a["hitterTap"]["x"], a["hitterTap"]["y"]), ts)
        a["frames"] = frames
        print(f"[{n}/{len(todo)}] {a['contactTime']:.2f}s: hitter found in {sum(f is not None for f in frames)}/{WINDOW_FRAMES} frames")
    Path(args.out).write_text(json.dumps(d))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
