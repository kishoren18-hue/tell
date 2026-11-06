"""Same honest metrics as the app (src/lib/ml/metrics.ts)."""
from __future__ import annotations

import math

import numpy as np


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def ce_bits(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log2(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean())


def info_gain(p: np.ndarray, y: np.ndarray, prior: np.ndarray) -> float:
    cm = ce_bits(p, y)
    cp = ce_bits(np.tile(prior, (len(y), 1)), y)
    return max(0.0, 1 - cm / cp) if cp > 1e-9 else 0.0


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = p.max(1)
    pred = p.argmax(1)
    b = np.minimum(bins - 1, (conf * bins).astype(int))
    out = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def binomial_p(successes: int, n: int, p0: float) -> float:
    """One-sided exact P(X >= successes)."""
    if n == 0:
        return 1.0
    if p0 <= 0:
        return 0.0 if successes > 0 else 1.0
    if p0 >= 1:
        return 1.0
    tot = 0.0
    for k in range(successes, n + 1):
        tot += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + k * math.log(p0) + (n - k) * math.log(1 - p0))
    return min(1.0, tot)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / den), min(1.0, (c + h) / den))


def summarize(p: np.ndarray, y: np.ndarray, k: int = 3) -> dict:
    counts = np.bincount(y, minlength=k)
    prior = (counts + 1) / (len(y) + k)
    majority = counts.max() / len(y)
    acc = accuracy(p, y)
    correct = int(round(acc * len(y)))
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "accuracy_ci": wilson(correct, len(y)),
        "majority_accuracy": float(majority),
        "p_value_vs_majority": binomial_p(correct, len(y), float(majority)),
        "ce_bits": ce_bits(p, y),
        "info_gain": info_gain(p, y, prior),
        "ece": ece(p, y),
    }
