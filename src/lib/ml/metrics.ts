// Honest evaluation: every number is compared against what you'd get by
// always guessing the most common shot, and calibration is measured, not assumed.

export function accuracy(probs: number[][], y: number[]): number {
  let c = 0;
  probs.forEach((p, i) => {
    if (argmax(p) === y[i]) c++;
  });
  return probs.length ? c / probs.length : 0;
}

export function argmax(p: number[]): number {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}

/** Mean negative log-likelihood in bits. */
export function crossEntropyBits(probs: number[][], y: number[]): number {
  let s = 0;
  probs.forEach((p, i) => {
    s -= Math.log2(Math.max(p[y[i]], 1e-6));
  });
  return probs.length ? s / probs.length : 0;
}

export function brier(probs: number[][], y: number[]): number {
  let s = 0;
  probs.forEach((p, i) => {
    p.forEach((v, c) => {
      s += (v - (y[i] === c ? 1 : 0)) ** 2;
    });
  });
  return probs.length ? s / probs.length : 0;
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  count: number;
  confidence: number;
  accuracy: number;
}

/** Expected calibration error on top-label confidence. */
export function calibration(probs: number[][], y: number[], bins = 10): { ece: number; bins: ReliabilityBin[] } {
  const out: ReliabilityBin[] = Array.from({ length: bins }, (_, i) => ({
    lo: i / bins,
    hi: (i + 1) / bins,
    count: 0,
    confidence: 0,
    accuracy: 0,
  }));
  probs.forEach((p, i) => {
    const c = argmax(p);
    const conf = p[c];
    const b = Math.min(bins - 1, Math.floor(conf * bins));
    out[b].count++;
    out[b].confidence += conf;
    out[b].accuracy += c === y[i] ? 1 : 0;
  });
  let ece = 0;
  for (const b of out) {
    if (b.count) {
      b.confidence /= b.count;
      b.accuracy /= b.count;
      ece += (b.count / probs.length) * Math.abs(b.accuracy - b.confidence);
    }
  }
  return { ece, bins: out };
}

export function confusion(probs: number[][], y: number[], k: number): number[][] {
  const m = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  probs.forEach((p, i) => m[y[i]][argmax(p)]++);
  return m;
}

export function classCounts(y: number[], k: number): number[] {
  const c = new Array<number>(k).fill(0);
  for (const v of y) c[v]++;
  return c;
}

/** Wilson score interval for a proportion. */
export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 1];
  const p = successes / n;
  const den = 1 + (z * z) / n;
  const centre = p + (z * z) / (2 * n);
  const half = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, (centre - half) / den), Math.min(1, (centre + half) / den)];
}

function logChoose(n: number, k: number): number {
  let s = 0;
  for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i);
  return s;
}

/** One-sided exact binomial test: P(X >= successes) when each trial succeeds with prob p0. */
export function binomialPValue(successes: number, n: number, p0: number): number {
  if (n === 0) return 1;
  if (p0 <= 0) return successes > 0 ? 0 : 1;
  if (p0 >= 1) return 1;
  let total = 0;
  for (let k = successes; k <= n; k++) {
    total += Math.exp(logChoose(n, k) + k * Math.log(p0) + (n - k) * Math.log(1 - p0));
  }
  return Math.min(1, total);
}
