// Small classifiers that train in well under a second on a phone for a few
// hundred attacks. Both are full-batch Adam with L2 weight decay.

import { Rng } from '../rng';

export interface Classifier {
  kind: ModelKind;
  logits(x: number[]): number[];
}
export type ModelKind = 'logistic' | 'mlp';

export interface TrainOptions {
  kind: ModelKind;
  l2: number;
  iters: number;
  lr: number;
  hidden: number;
  seed: number;
  /** Per-sample weights (e.g. to balance classes). */
  weights?: number[];
}

export const DEFAULT_TRAIN: TrainOptions = {
  kind: 'logistic',
  l2: 0.05,
  iters: 300,
  lr: 0.05,
  hidden: 24,
  seed: 7,
};

export class Standardizer {
  mean: number[];
  std: number[];
  constructor(mean: number[], std: number[]) {
    this.mean = mean;
    this.std = std;
  }
  static fit(x: number[][]): Standardizer {
    const d = x[0].length;
    const mean = new Array<number>(d).fill(0);
    const sq = new Array<number>(d).fill(0);
    for (const r of x) for (let i = 0; i < d; i++) mean[i] += r[i];
    for (let i = 0; i < d; i++) mean[i] /= x.length;
    for (const r of x) for (let i = 0; i < d; i++) sq[i] += (r[i] - mean[i]) ** 2;
    const std = sq.map((s) => Math.sqrt(s / Math.max(1, x.length - 1)) || 1);
    return new Standardizer(mean, std.map((s) => (s < 1e-8 ? 1 : s)));
  }
  apply(r: number[]): number[] {
    return r.map((v, i) => (v - this.mean[i]) / this.std[i]);
  }
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

class Adam {
  m: Float64Array;
  v: Float64Array;
  t = 0;
  lr: number;
  constructor(n: number, lr: number) {
    this.m = new Float64Array(n);
    this.v = new Float64Array(n);
    this.lr = lr;
  }
  step(p: Float64Array, g: Float64Array) {
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    for (let i = 0; i < p.length; i++) {
      this.m[i] = b1 * this.m[i] + (1 - b1) * g[i];
      this.v[i] = b2 * this.v[i] + (1 - b2) * g[i] * g[i];
      p[i] -= (this.lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + 1e-8);
    }
  }
}

/** Multinomial logistic regression. x must already be standardized. */
export function trainLogistic(x: number[][], y: number[], k: number, o: TrainOptions): Classifier {
  const n = x.length;
  const d = x[0].length;
  const w = o.weights ?? new Array<number>(n).fill(1);
  const wsum = w.reduce((a, b) => a + b, 0);
  // params: W (d*k) then b (k)
  const p = new Float64Array(d * k + k);
  const g = new Float64Array(p.length);
  const opt = new Adam(p.length, o.lr);
  const z = new Array<number>(k);
  for (let it = 0; it < o.iters; it++) {
    g.fill(0);
    for (let s = 0; s < n; s++) {
      const xs = x[s];
      for (let c = 0; c < k; c++) {
        let v = p[d * k + c];
        for (let i = 0; i < d; i++) v += xs[i] * p[i * k + c];
        z[c] = v;
      }
      const pr = softmax(z);
      const ws = w[s] / wsum;
      for (let c = 0; c < k; c++) {
        const err = (pr[c] - (y[s] === c ? 1 : 0)) * ws;
        g[d * k + c] += err;
        for (let i = 0; i < d; i++) g[i * k + c] += err * xs[i];
      }
    }
    for (let i = 0; i < d * k; i++) g[i] += o.l2 * p[i];
    opt.step(p, g);
  }
  return {
    kind: 'logistic',
    logits(xs: number[]) {
      const out = new Array<number>(k);
      for (let c = 0; c < k; c++) {
        let v = p[d * k + c];
        for (let i = 0; i < d; i++) v += xs[i] * p[i * k + c];
        out[c] = v;
      }
      return out;
    },
  };
}

/** One-hidden-layer tanh MLP. x must already be standardized. */
export function trainMlp(x: number[][], y: number[], k: number, o: TrainOptions): Classifier {
  const n = x.length;
  const d = x[0].length;
  const h = o.hidden;
  const rng = new Rng(o.seed);
  const w = o.weights ?? new Array<number>(n).fill(1);
  const wsum = w.reduce((a, b) => a + b, 0);
  // layout: W1 (d*h), b1 (h), W2 (h*k), b2 (k)
  const oW1 = 0;
  const oB1 = d * h;
  const oW2 = oB1 + h;
  const oB2 = oW2 + h * k;
  const p = new Float64Array(oB2 + k);
  const s1 = Math.sqrt(1 / d);
  const s2 = Math.sqrt(1 / h);
  for (let i = 0; i < d * h; i++) p[oW1 + i] = rng.normal() * s1;
  for (let i = 0; i < h * k; i++) p[oW2 + i] = rng.normal() * s2;
  const g = new Float64Array(p.length);
  const opt = new Adam(p.length, o.lr * 0.4);
  const a = new Array<number>(h);
  const z = new Array<number>(k);
  const dh = new Array<number>(h);
  const forward = (xs: number[]) => {
    for (let j = 0; j < h; j++) {
      let v = p[oB1 + j];
      for (let i = 0; i < d; i++) v += xs[i] * p[oW1 + i * h + j];
      a[j] = Math.tanh(v);
    }
    for (let c = 0; c < k; c++) {
      let v = p[oB2 + c];
      for (let j = 0; j < h; j++) v += a[j] * p[oW2 + j * k + c];
      z[c] = v;
    }
  };
  for (let it = 0; it < o.iters; it++) {
    g.fill(0);
    for (let s = 0; s < n; s++) {
      const xs = x[s];
      forward(xs);
      const pr = softmax(z);
      const ws = w[s] / wsum;
      dh.fill(0);
      for (let c = 0; c < k; c++) {
        const err = (pr[c] - (y[s] === c ? 1 : 0)) * ws;
        g[oB2 + c] += err;
        for (let j = 0; j < h; j++) {
          g[oW2 + j * k + c] += err * a[j];
          dh[j] += err * p[oW2 + j * k + c];
        }
      }
      for (let j = 0; j < h; j++) {
        const dz = dh[j] * (1 - a[j] * a[j]);
        if (dz === 0) continue;
        g[oB1 + j] += dz;
        for (let i = 0; i < d; i++) g[oW1 + i * h + j] += dz * xs[i];
      }
    }
    for (let i = 0; i < d * h; i++) g[oW1 + i] += o.l2 * p[oW1 + i];
    for (let i = 0; i < h * k; i++) g[oW2 + i] += o.l2 * p[oW2 + i];
    opt.step(p, g);
  }
  return {
    kind: 'mlp',
    logits(xs: number[]) {
      forward(xs);
      return z.slice();
    },
  };
}

export function train(x: number[][], y: number[], k: number, o: TrainOptions): Classifier {
  return o.kind === 'mlp' ? trainMlp(x, y, k, o) : trainLogistic(x, y, k, o);
}

/** Inverse-frequency weights so the model doesn't just learn the most common shot. */
export function balancedWeights(y: number[], k: number): number[] {
  const counts = new Array<number>(k).fill(0);
  for (const c of y) counts[c]++;
  return y.map((c) => y.length / (k * Math.max(1, counts[c])));
}
