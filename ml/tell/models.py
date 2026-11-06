"""Sequence models for Tell.

PoseLSTM
    The tennis baseline: an LSTM over pose frames up to the cutoff, after
    Shimizu et al. (2019), "Prediction of Future Shot Direction using Pose and
    Position of Tennis Player". Predicts left / straight / right.

RallyWorld
    A small causal transformer world model. Every frame is a token; each token
    only sees the past. From every token it predicts
      1. the hitter's next 0.5 s of pose (15 future states)    -> world model
      2. where the ball will land, as a mixture of Gaussians    -> landing
      3. the shot direction                                     -> tell
    The pose-forecast loss needs no labels, so every filmed swing trains it.
    Directions are trained at random cutoffs (0-500 ms before contact), so one
    model reads at any horizon. The idea of predicting shot class and landing
    area from separate heads follows ShuttleNet (Wang et al., AAAI 2022).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import CONTACT_INDEX, FRAME_DIM, N_JOINT_FEATS, SAMPLE_HZ, WINDOW_FRAMES

STATE_DIM = N_JOINT_FEATS + 2
FORECAST = 15
N_MIX = 3


class PoseLSTM(nn.Module):
    def __init__(self, in_dim: int = FRAME_DIM, hidden: int = 64, k: int = 3, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, k)

    def forward(self, x: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(F.gelu(self.inp(x)))
        idx = cutoff.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return self.head(self.drop(h.gather(1, idx).squeeze(1)))


class RallyWorld(nn.Module):
    def __init__(self, in_dim: int = FRAME_DIM, d: int = 64, layers: int = 3, heads: int = 4,
                 k: int = 3, dropout: float = 0.1, max_len: int = WINDOW_FRAMES):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.dir_head = nn.Linear(d, k)
        self.land_head = nn.Linear(d, N_MIX * 5)  # logits, mu(2), log_sigma(2)
        self.dyn_head = nn.Linear(d, FORECAST * STATE_DIM)
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal", mask, persistent=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        h = self.inp(x) + self.pos[:, :T]
        return self.norm(self.enc(h, mask=self.causal[:T, :T]))

    def heads(self, h: torch.Tensor):
        lm = self.land_head(h).view(*h.shape[:-1], N_MIX, 5)
        return {
            "dir": self.dir_head(h),
            "mix_logits": lm[..., 0],
            "mu": lm[..., 1:3],
            "log_sigma": lm[..., 3:5].clamp(-3, 3),
            "future": self.dyn_head(h).view(*h.shape[:-1], FORECAST, STATE_DIM),
        }

    def forward(self, x: torch.Tensor, cutoff: torch.Tensor):
        """Outputs at the cutoff token for each item. Future states are residuals from the cutoff state."""
        h = self.encode(x)
        idx = cutoff.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return self.heads(h.gather(1, idx).squeeze(1))


def mdn_nll(mix_logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of target (B, 2) under a diagonal Gaussian mixture."""
    t = target.unsqueeze(-2)
    z = (t - mu) / log_sigma.exp()
    comp = -0.5 * (z ** 2).sum(-1) - log_sigma.sum(-1) - math.log(2 * math.pi)
    return -(torch.logsumexp(F.log_softmax(mix_logits, -1) + comp, -1))


def states_from_features(x: torch.Tensor) -> torch.Tensor:
    """(…, FRAME_DIM) standardized-free features -> (…, STATE_DIM) pose state."""
    return torch.cat([x[..., :N_JOINT_FEATS], x[..., N_JOINT_FEATS + 12:N_JOINT_FEATS + 14]], -1)


def cutoff_from_ms(ms: float) -> int:
    return CONTACT_INDEX - int(math.floor(ms / 1000 * SAMPLE_HZ + 0.5))
