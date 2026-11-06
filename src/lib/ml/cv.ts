// Cross-validation that doesn't flatter the model.
// - Folds are grouped by match when there are 3+ matches, so the model is
//   always scored on a match it never saw (same gym, lighting and camera spot
//   would otherwise leak).
// - Temperature scaling is fit on a held-out slice of each training fold,
//   never on the fold being scored.

import { Rng } from '../rng';
import { softmax, Standardizer, train, type TrainOptions, DEFAULT_TRAIN } from './models';

export interface Fold {
  train: number[];
  test: number[];
}

export function makeFolds(groups: string[], y: number[], k = 5, seed = 11): { folds: Fold[]; grouped: boolean } {
  const n = y.length;
  const uniq = Array.from(new Set(groups));
  const rng = new Rng(seed);
  if (uniq.length >= 3) {
    // Leave-one-match-out when there are only a few matches, otherwise k grouped folds.
    const g = rng.shuffle(uniq);
    const nf = Math.min(k, g.length);
    const bucket = new Map<string, number>();
    g.forEach((name, i) => bucket.set(name, i % nf));
    const folds: Fold[] = Array.from({ length: nf }, () => ({ train: [], test: [] }));
    for (let i = 0; i < n; i++) {
      const f = bucket.get(groups[i]) as number;
      folds.forEach((fold, j) => (j === f ? fold.test : fold.train).push(i));
    }
    return { folds: folds.filter((f) => f.test.length && f.train.length), grouped: true };
  }
  // Stratified k-fold.
  const byClass = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    if (!byClass.has(y[i])) byClass.set(y[i], []);
    byClass.get(y[i])!.push(i);
  }
  const assign = new Array<number>(n);
  let ptr = 0;
  for (const idx of byClass.values()) for (const i of rng.shuffle(idx)) assign[i] = ptr++ % k;
  const folds: Fold[] = Array.from({ length: k }, () => ({ train: [], test: [] }));
  for (let i = 0; i < n; i++) folds.forEach((fold, j) => (j === assign[i] ? fold.test : fold.train).push(i));
  return { folds: folds.filter((f) => f.test.length && f.train.length), grouped: false };
}

/** Fit a single temperature T > 0 that minimizes NLL of softmax(logits / T). */
export function fitTemperature(logits: number[][], y: number[]): number {
  if (logits.length < 5) return 1;
  const nll = (t: number) => {
    let s = 0;
    logits.forEach((z, i) => {
      const p = softmax(z.map((v) => v / t));
      s -= Math.log(Math.max(p[y[i]], 1e-9));
    });
    return s;
  };
  // Golden-section search on log T in [log 0.25, log 10].
  let a = Math.log(0.25);
  let b = Math.log(10);
  const gr = (Math.sqrt(5) - 1) / 2;
  let c = b - gr * (b - a);
  let d = a + gr * (b - a);
  for (let i = 0; i < 40; i++) {
    if (nll(Math.exp(c)) < nll(Math.exp(d))) b = d;
    else a = c;
    c = b - gr * (b - a);
    d = a + gr * (b - a);
  }
  return Math.exp((a + b) / 2);
}

export interface FittedModel {
  scaler: Standardizer;
  logits: (x: number[]) => number[];
  temperature: number;
  predict: (x: number[]) => number[];
}

/** Train with an inner 80/20 split to pick the temperature, then refit on everything. */
export function fitCalibrated(x: number[][], y: number[], k: number, opts: TrainOptions = DEFAULT_TRAIN): FittedModel {
  const rng = new Rng(opts.seed + 101);
  const idx = rng.shuffle(x.map((_, i) => i));
  const cut = Math.max(1, Math.floor(idx.length * 0.8));
  const inTr = idx.slice(0, cut);
  const inVa = idx.slice(cut);
  let temperature = 1;
  if (inVa.length >= 5) {
    const sc = Standardizer.fit(inTr.map((i) => x[i]));
    const m = train(inTr.map((i) => sc.apply(x[i])), inTr.map((i) => y[i]), k, opts);
    temperature = fitTemperature(inVa.map((i) => m.logits(sc.apply(x[i]))), inVa.map((i) => y[i]));
  }
  const scaler = Standardizer.fit(x);
  const model = train(x.map((r) => scaler.apply(r)), y, k, opts);
  const logits = (r: number[]) => model.logits(scaler.apply(r));
  return {
    scaler,
    logits,
    temperature,
    predict: (r: number[]) => softmax(logits(r).map((v) => v / temperature)),
  };
}

export interface OofResult {
  probs: number[][];
  grouped: boolean;
  nFolds: number;
}

/** Out-of-fold calibrated probabilities for every sample. */
export function outOfFold(
  x: number[][],
  y: number[],
  groups: string[],
  k: number,
  opts: TrainOptions = DEFAULT_TRAIN,
): OofResult {
  const { folds, grouped } = makeFolds(groups, y);
  const probs: number[][] = new Array(x.length);
  for (const f of folds) {
    const m = fitCalibrated(f.train.map((i) => x[i]), f.train.map((i) => y[i]), k, opts);
    for (const i of f.test) probs[i] = m.predict(x[i]);
  }
  return { probs, grouped, nFolds: folds.length };
}
