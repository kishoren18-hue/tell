// World model v0 (on-device). Predicts the next 0.5 s of the hitter's body
// from what it has seen so far, and where the ball will land.
//
// - Pose dynamics: PCA-compressed pose state, then a linear autoregressive
//   model z[t+1] = A [z[t], z[t-1], z[t-2]] trained on every frame of every
//   clip. It needs no labels, so every clip you film makes it better even
//   before anyone taps a landing spot.
// - Landing: ridge regression from the pre-contact summary features to the
//   landing point, with residual spread as an uncertainty ellipse.
//
// The heavier transformer world model (RallyWorld) lives in ml/tell/models.py.

import { N_JOINT_FEATS } from '../features';
import { ridge, ridgePredict, symEig, type Mat } from './linalg';
import { Standardizer } from './models';

export const FORECAST_STEPS = 15; // 0.5 s at 30 Hz
const AR_ORDER = 3;
const STATE_DIM = N_JOINT_FEATS + 2; // joints + hip image xy

/** Pose state for one frame: joint coords (torso units) + hip position in image. */
export function stateOf(row: number[]): number[] {
  return [...row.slice(0, N_JOINT_FEATS), row[N_JOINT_FEATS + 12], row[N_JOINT_FEATS + 13]];
}

export interface PoseDynamics {
  mean: number[];
  components: number[][]; // k x STATE_DIM
  A: Mat; // (k*AR_ORDER + 1) x k
  k: number;
}

function project(d: PoseDynamics, s: number[]): number[] {
  return d.components.map((c) => c.reduce((acc, v, i) => acc + v * (s[i] - d.mean[i]), 0));
}
function reconstruct(d: PoseDynamics, z: number[]): number[] {
  const s = d.mean.slice();
  d.components.forEach((c, j) => {
    for (let i = 0; i < s.length; i++) s[i] += c[i] * z[j];
  });
  return s;
}

/** Fit pose dynamics on a set of sequences (T x FRAME_DIM each). */
export function fitDynamics(seqs: number[][][], k = 12, lambda = 1e-2): PoseDynamics {
  const states = seqs.flatMap((s) => s.map(stateOf));
  const mean = new Array<number>(STATE_DIM).fill(0);
  for (const s of states) for (let i = 0; i < STATE_DIM; i++) mean[i] += s[i];
  for (let i = 0; i < STATE_DIM; i++) mean[i] /= states.length;
  const cov: Mat = Array.from({ length: STATE_DIM }, () => new Array<number>(STATE_DIM).fill(0));
  for (const s of states) {
    for (let i = 0; i < STATE_DIM; i++) {
      const di = s[i] - mean[i];
      for (let j = i; j < STATE_DIM; j++) cov[i][j] += di * (s[j] - mean[j]);
    }
  }
  for (let i = 0; i < STATE_DIM; i++) for (let j = i; j < STATE_DIM; j++) {
    cov[i][j] /= states.length;
    cov[j][i] = cov[i][j];
  }
  const { vectors } = symEig(cov);
  const dyn: PoseDynamics = { mean, components: vectors.slice(0, k), A: [], k };
  const X: Mat = [];
  const Y: Mat = [];
  for (const s of seqs) {
    const z = s.map((r) => project(dyn, stateOf(r)));
    for (let t = AR_ORDER - 1; t < z.length - 1; t++) {
      const input: number[] = [];
      for (let lag = 0; lag < AR_ORDER; lag++) input.push(...z[t - lag]);
      X.push(input);
      Y.push(z[t + 1]);
    }
  }
  dyn.A = ridge(X, Y, lambda);
  return dyn;
}

/** Roll the pose forward `steps` frames starting after frame `from`. Returns full states. */
export function forecastPose(d: PoseDynamics, seq: number[][], from: number, steps = FORECAST_STEPS): number[][] {
  const hist = [];
  for (let lag = AR_ORDER - 1; lag >= 0; lag--) hist.push(project(d, stateOf(seq[Math.max(0, from - lag)])));
  const out: number[][] = [];
  for (let s = 0; s < steps; s++) {
    const input: number[] = [];
    for (let lag = 0; lag < AR_ORDER; lag++) input.push(...hist[hist.length - 1 - lag]);
    const next = ridgePredict(d.A, input);
    hist.push(next);
    out.push(reconstruct(d, next));
  }
  return out;
}

/** Mean per-joint error (torso lengths) of a forecast against the truth. */
export function poseError(pred: number[][], truth: number[][]): number {
  let s = 0;
  let n = 0;
  for (let t = 0; t < Math.min(pred.length, truth.length); t++) {
    for (let j = 0; j < N_JOINT_FEATS / 3; j++) {
      const a = pred[t];
      const b = truth[t];
      s += Math.hypot(a[j * 3] - b[j * 3], a[j * 3 + 1] - b[j * 3 + 1], a[j * 3 + 2] - b[j * 3 + 2]);
      n++;
    }
  }
  return n ? s / n : 0;
}

/** Baselines a world model has to beat: freeze the pose, or keep it moving at its current velocity. */
export function staticForecast(seq: number[][], from: number, steps = FORECAST_STEPS): number[][] {
  const s = stateOf(seq[from]);
  return Array.from({ length: steps }, () => s.slice());
}
export function constantVelocityForecast(seq: number[][], from: number, steps = FORECAST_STEPS): number[][] {
  const s = stateOf(seq[from]);
  // Velocity averaged over the last 3 frames so a single noisy frame doesn't dominate.
  const back = Math.min(3, from);
  const p = stateOf(seq[from - back]);
  return Array.from({ length: steps }, (_, i) => s.map((v, j) => v + ((v - p[j]) / Math.max(1, back)) * (i + 1)));
}

export interface LandingModel {
  scaler: Standardizer;
  W: Mat;
  /** Residual standard deviation in meters along [dx, depth]. */
  spread: [number, number];
}

/** Landing target is [lateral offset from takeoff, depth] in the canonical frame. */
export function fitLanding(x: number[][], target: number[][], lambda = 30): LandingModel {
  const scaler = Standardizer.fit(x);
  const xs = x.map((r) => scaler.apply(r));
  const W = ridge(xs, target, lambda);
  const res = xs.map((r, i) => {
    const p = ridgePredict(W, r);
    return [p[0] - target[i][0], p[1] - target[i][1]];
  });
  const sd = (k: number) => Math.sqrt(res.reduce((a, r) => a + r[k] * r[k], 0) / Math.max(1, res.length - 1));
  return { scaler, W, spread: [sd(0), sd(1)] };
}

export function predictLanding(m: LandingModel, x: number[]): [number, number] {
  const p = ridgePredict(m.W, m.scaler.apply(x));
  return [p[0], p[1]];
}
