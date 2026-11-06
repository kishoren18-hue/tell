"""Train and evaluate Tell models on a dataset export.

    python -m tell.train data/my-team.json --models logreg,lstm,rallyworld --out results.json

Evaluation follows the app: folds grouped by match when there are 3+ matches
(the model is never scored on a match it saw), temperature fit on held-out
training data only, and every number reported next to the "always guess the
most common direction" baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import Samples, load_dataset, to_samples
from .features import CONTACT_INDEX, N_JOINT_FEATS
from .metrics import summarize
from .models import FORECAST, PoseLSTM, RallyWorld, cutoff_from_ms, mdn_nll, states_from_features

HORIZONS = [0, 100, 200, 300, 400, 500]
K = 3


# ---------------------------------------------------------------- folds

def make_folds(groups: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 11):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) >= 3:
        order = rng.permutation(uniq)
        nf = min(k, len(order))
        bucket = {g: i % nf for i, g in enumerate(order)}
        f = np.array([bucket[g] for g in groups])
        return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(nf)], True
    assign = np.empty(len(y), dtype=int)
    ptr = 0
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for i in idx:
            assign[i] = ptr % k
            ptr += 1
    return [(np.where(assign != i)[0], np.where(assign == i)[0]) for i in range(k)], False


def inner_split(idx: np.ndarray, seed: int, frac: float = 0.2):
    rng = np.random.default_rng(seed)
    p = rng.permutation(idx)
    cut = max(1, int(len(p) * frac))
    return p[cut:], p[:cut]


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 5:
        return 1.0
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y)
    best, best_t = float("inf"), 1.0
    for T in np.exp(np.linspace(math.log(0.25), math.log(10), 80)):
        nll = F.cross_entropy(z / T, t).item()
        if nll < best:
            best, best_t = nll, float(T)
    return best_t


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- logistic baseline (same model as the app)

def run_logreg(S: Samples, folds, horizons, seed: int) -> dict[int, np.ndarray]:
    out = {}
    for h in horizons:
        X = S.summary(h)
        probs = np.zeros((len(S), K))
        for tr, te in folds:
            itr, iva = inner_split(tr, seed)
            temp = 1.0
            if len(iva) >= 5 and len(np.unique(S.y[itr])) > 1:
                sc = StandardScaler().fit(X[itr])
                m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[itr]), S.y[itr])
                temp = fit_temperature(full_logits(m, sc.transform(X[iva])), S.y[iva])
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), S.y[tr])
            probs[te] = softmax_np(full_logits(m, sc.transform(X[te])) / temp)
        out[h] = probs
    return out


def full_logits(m: LogisticRegression, X: np.ndarray) -> np.ndarray:
    z = np.full((len(X), K), -30.0)
    z[:, m.classes_] = m.decision_function(X) if len(m.classes_) > 2 else np.c_[-m.decision_function(X), m.decision_function(X)] / 2
    return z


# ---------------------------------------------------------------- sequence models

@dataclass
class TrainCfg:
    epochs: int = 150
    lr: float = 2e-3
    wd: float = 1e-2
    batch: int = 32
    w_land: float = 0.3
    w_dyn: float = 1.0
    seed: int = 0


def norm_stats(seq: np.ndarray):
    flat = seq.reshape(-1, seq.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0)
    sd[sd < 1e-6] = 1
    return mu, sd


def future_targets(states: torch.Tensor):
    """For each token t: states[t+1..t+F] - states[t], and a mask where the window is complete."""
    B, T, D = states.shape
    tgt = torch.zeros(B, T, FORECAST, D)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for t in range(T - FORECAST):
        tgt[:, t] = states[:, t + 1:t + 1 + FORECAST] - states[:, t:t + 1]
        mask[:, t] = True
    return tgt, mask


def train_seq(kind: str, S: Samples, idx: np.ndarray, mu, sd, cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    x = torch.tensor((S.seq[idx] - mu) / sd, dtype=torch.float32)
    y = torch.tensor(S.y[idx])
    land = torch.tensor(S.landing[idx], dtype=torch.float32)
    states = states_from_features(torch.tensor(S.seq[idx], dtype=torch.float32))
    fut, fmask = future_targets(states)
    model = RallyWorld() if kind == "rallyworld" else PoseLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    lo, hi = cutoff_from_ms(500), cutoff_from_ms(0)
    n = len(idx)
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(n)
        for b in range(0, n, cfg.batch):
            bi = torch.tensor(perm[b:b + cfg.batch])
            cut = torch.tensor(rng.integers(lo, hi + 1, size=len(bi)))
            if kind == "rallyworld":
                o = model.heads(model.encode(x[bi]))
                loss = F.cross_entropy(pick(o["dir"], cut), y[bi])
                lm = ~torch.isnan(land[bi]).any(1)
                if lm.any():
                    nll = mdn_nll(pick(o["mix_logits"], cut), pick(o["mu"], cut), pick(o["log_sigma"], cut),
                                  torch.nan_to_num(land[bi]))
                    loss = loss + cfg.w_land * nll[lm].mean() / 4
                dyn = ((o["future"] - fut[bi]) ** 2).mean((-1, -2))
                loss = loss + cfg.w_dyn * dyn[fmask[bi]].mean()
            else:
                loss = F.cross_entropy(model(x[bi], cut), y[bi])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def pick(t: torch.Tensor, cut: torch.Tensor) -> torch.Tensor:
    """Select token `cut[b]` from each sequence of a (B, T, ...) tensor."""
    idx = cut.view(-1, 1, *([1] * (t.dim() - 2))).expand(-1, 1, *t.shape[2:])
    return t.gather(1, idx).squeeze(1)


@torch.no_grad()
def seq_logits(model, S: Samples, idx: np.ndarray, mu, sd, cutoff: int) -> np.ndarray:
    x = torch.tensor((S.seq[idx] - mu) / sd, dtype=torch.float32)
    c = torch.full((len(idx),), cutoff, dtype=torch.long)
    out = model(x, c)
    return (out["dir"] if isinstance(out, dict) else out).numpy()


def run_seq(kind: str, S: Samples, folds, horizons, cfg: TrainCfg, log=print):
    probs = {h: np.zeros((len(S), K)) for h in horizons}
    world = []
    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        itr, iva = inner_split(tr, cfg.seed + fi)
        mu, sd = norm_stats(S.seq[itr])
        model = train_seq(kind, S, itr, mu, sd, cfg)
        for h in horizons:
            c = cutoff_from_ms(h)
            temp = fit_temperature(seq_logits(model, S, iva, mu, sd, c), S.y[iva])
            probs[h][te] = softmax_np(seq_logits(model, S, te, mu, sd, c) / temp)
        if kind == "rallyworld":
            world.append(forecast_errors(model, S, te, mu, sd))
        log(f"  {kind} fold {fi + 1}/{len(folds)} trained on {len(itr)} in {time.time() - t0:.1f}s")
    return probs, (np.concatenate(world) if world else None)


# ---------------------------------------------------------------- world-model evaluation

def joint_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mean per-joint distance in torso lengths, per sample. Arrays: (N, F, STATE_DIM)."""
    J = N_JOINT_FEATS // 3
    a = pred[..., :N_JOINT_FEATS].reshape(*pred.shape[:-1], J, 3)
    b = truth[..., :N_JOINT_FEATS].reshape(*truth.shape[:-1], J, 3)
    return np.linalg.norm(a - b, axis=-1).mean((-1, -2))


@torch.no_grad()
def forecast_errors(model: RallyWorld, S: Samples, idx: np.ndarray, mu, sd, horizon_ms: int = 200) -> np.ndarray:
    c = cutoff_from_ms(horizon_ms)
    x = torch.tensor((S.seq[idx] - mu) / sd, dtype=torch.float32)
    out = model(x, torch.full((len(idx),), c, dtype=torch.long))
    st = states_from_features(torch.tensor(S.seq[idx], dtype=torch.float32)).numpy()
    pred = st[:, c:c + 1] + out["future"].numpy()
    truth = st[:, c + 1:c + 1 + FORECAST]
    return joint_error(pred, truth)


def baseline_forecasts(S: Samples, horizon_ms: int = 200) -> dict[str, float]:
    c = cutoff_from_ms(horizon_ms)
    st = np.stack([states_from_features(torch.tensor(s, dtype=torch.float32)).numpy() for s in S.seq])
    truth = st[:, c + 1:c + 1 + FORECAST]
    still = np.repeat(st[:, c:c + 1], FORECAST, 1)
    vel = (st[:, c] - st[:, c - 3]) / 3
    cv = st[:, c:c + 1] + vel[:, None] * np.arange(1, FORECAST + 1)[None, :, None]
    return {"still": float(joint_error(still, truth).mean()), "constant_velocity": float(joint_error(cv, truth).mean())}


# ---------------------------------------------------------------- main

def evaluate(S: Samples, models: list[str], horizons: list[int], cfg: TrainCfg, log=print) -> dict:
    folds, grouped = make_folds(S.match, S.y, seed=cfg.seed + 11)
    log(f"{len(S)} attacks, {len(np.unique(S.match))} matches, {len(folds)} folds "
        f"({'grouped by match' if grouped else 'stratified, fewer than 3 matches'})")
    report = {"n": len(S), "skipped": S.skipped, "grouped_by_match": grouped, "horizons": horizons, "models": {}}
    for name in models:
        t0 = time.time()
        world = None
        if name == "logreg":
            probs = run_logreg(S, folds, horizons, cfg.seed)
        else:
            probs, world = run_seq(name, S, folds, horizons, cfg, log)
        res = {str(h): summarize(probs[h], S.y) for h in horizons}
        if world is not None:
            res["world_forecast_error_200ms"] = float(world.mean())
        report["models"][name] = res
        log(f"{name}: done in {time.time() - t0:.0f}s")
    report["world_baselines_200ms"] = baseline_forecasts(S)
    return report


def print_table(report: dict) -> None:
    hs = report["horizons"]
    print("\nAccuracy by cutoff before contact (majority baseline in brackets, * = p < 0.05)")
    print(f"{'model':<12}" + "".join(f"{('-' + str(h) + 'ms') if h else 'contact':>10}" for h in hs))
    for name, res in report["models"].items():
        row = ""
        for h in hs:
            r = res[str(h)]
            row += f"{r['accuracy'] * 100:>8.1f}{'*' if r['p_value_vs_majority'] < 0.05 else ' '} "
        print(f"{name:<12}{row}")
    any_h = report["models"][next(iter(report["models"]))][str(hs[0])]
    print(f"{'majority':<12}" + f"{any_h['majority_accuracy'] * 100:>8.1f}  " * len(hs))
    print("\nInformation gain over base rates (tell score scale, 0-100)")
    for name, res in report["models"].items():
        print(f"{name:<12}" + "".join(f"{res[str(h)]['info_gain'] * 100:>10.1f}" for h in hs))
    wb = report["world_baselines_200ms"]
    rw = report["models"].get("rallyworld", {}).get("world_forecast_error_200ms")
    print("\n0.5 s pose forecast from 200 ms before contact (mean joint error, torso lengths)")
    if rw is not None:
        print(f"  RallyWorld        {rw:.3f}")
    print(f"  hitter freezes    {wb['still']:.3f}\n  constant velocity {wb['constant_velocity']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Tell dataset export (.json)")
    ap.add_argument("--models", default="logreg,lstm,rallyworld")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args(argv)
    S = to_samples(load_dataset(args.dataset))
    cfg = TrainCfg(epochs=args.epochs, seed=args.seed)
    rep = evaluate(S, args.models.split(","), [int(h) for h in args.horizons.split(",")], cfg)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print_table(rep)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

__all__ = ["evaluate", "make_folds", "train_seq", "norm_stats", "TrainCfg", "CONTACT_INDEX"]
