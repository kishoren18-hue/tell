"""Train RallyWorld on a whole dataset and export it to ONNX.

    python -m tell.export data/my-team.json --out rallyworld.onnx

The ONNX graph takes raw per-frame features (1, 46, 53) from
src/lib/features.ts (canonical, right-handed frame) and a cutoff frame index,
and returns calibrated direction probabilities, the landing mixture and the
next 0.5 s of pose. Normalization and temperature are baked in. A sidecar
JSON records the feature spec so a mismatched app can refuse to load it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import load_dataset, to_samples
from .features import CONTACT_INDEX, FRAME_DIM, SAMPLE_HZ, WINDOW_FRAMES
from .models import RallyWorld, states_from_features
from .train import TrainCfg, fit_temperature, inner_split, norm_stats, seq_logits, train_seq, cutoff_from_ms


class Exportable(nn.Module):
    def __init__(self, model: RallyWorld, mu: np.ndarray, sd: np.ndarray, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.register_buffer("temp", torch.tensor(float(temperature)))

    def forward(self, features: torch.Tensor, cutoff: torch.Tensor):
        o = self.model((features - self.mu) / self.sd, cutoff)
        state = states_from_features(features).gather(1, cutoff.view(-1, 1, 1).expand(-1, 1, 41))
        return (
            torch.softmax(o["dir"] / self.temp, -1),
            torch.softmax(o["mix_logits"], -1),
            o["mu"],
            o["log_sigma"].exp(),
            state + o["future"],
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", default="rallyworld.onnx")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--temperature-horizon", type=int, default=100)
    args = ap.parse_args(argv)

    S = to_samples(load_dataset(args.dataset))
    cfg = TrainCfg(epochs=args.epochs)
    all_idx = np.arange(len(S))
    tr, va = inner_split(all_idx, 0)
    mu, sd = norm_stats(S.seq[tr])
    probe = train_seq("rallyworld", S, tr, mu, sd, cfg)
    temp = fit_temperature(seq_logits(probe, S, va, mu, sd, cutoff_from_ms(args.temperature_horizon)), S.y[va])
    mu, sd = norm_stats(S.seq)
    model = train_seq("rallyworld", S, all_idx, mu, sd, cfg)
    wrapped = Exportable(model, mu, sd, temp).eval()

    x = torch.tensor(S.seq[:1], dtype=torch.float32)
    c = torch.tensor([cutoff_from_ms(100)], dtype=torch.long)
    with torch.no_grad():
        ref = [t.numpy() for t in wrapped(x, c)]
    torch.onnx.export(
        wrapped, (x, c), args.out,
        input_names=["features", "cutoff"],
        output_names=["direction_probs", "landing_weights", "landing_mean", "landing_std", "future_pose"],
        dynamic_axes={"features": {0: "batch"}, "cutoff": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    meta = {
        "model": "RallyWorld",
        "feature_version": 1,
        "frame_dim": FRAME_DIM,
        "window_frames": WINDOW_FRAMES,
        "contact_index": CONTACT_INDEX,
        "sample_hz": SAMPLE_HZ,
        "classes": ["left", "straight", "right"],
        "canonical_frame": "right-handed; mirror left-handed hitters before calling and swap left/right after",
        "temperature": temp,
        "trained_on_attacks": int(len(S)),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out)
        got = sess.run(None, {"features": x.numpy(), "cutoff": c.numpy()})
        worst = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
        print(f"onnxruntime check: max abs difference vs PyTorch = {worst:.2e}")
    except ImportError:
        print("onnxruntime not installed; skipped the export check")
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
