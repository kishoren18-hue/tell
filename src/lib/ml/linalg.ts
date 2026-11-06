// Minimal dense linear algebra, enough for ridge regression and PCA on
// matrices with a few hundred columns. Row-major number[][] throughout.

export type Mat = number[][];

export function zeros(r: number, c: number): Mat {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}

export function transpose(a: Mat): Mat {
  const r = a.length;
  const c = a[0].length;
  const t = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) t[j][i] = a[i][j];
  return t;
}

/** A^T A without forming the transpose. */
export function gram(a: Mat): Mat {
  const c = a[0].length;
  const g = zeros(c, c);
  for (const row of a) {
    for (let i = 0; i < c; i++) {
      const ri = row[i];
      if (ri === 0) continue;
      const gi = g[i];
      for (let j = i; j < c; j++) gi[j] += ri * row[j];
    }
  }
  for (let i = 0; i < c; i++) for (let j = 0; j < i; j++) g[i][j] = g[j][i];
  return g;
}

/** A^T B. */
export function atb(a: Mat, b: Mat): Mat {
  const c = a[0].length;
  const d = b[0].length;
  const out = zeros(c, d);
  for (let n = 0; n < a.length; n++) {
    const ra = a[n];
    const rb = b[n];
    for (let i = 0; i < c; i++) {
      const v = ra[i];
      if (v === 0) continue;
      const oi = out[i];
      for (let j = 0; j < d; j++) oi[j] += v * rb[j];
    }
  }
  return out;
}

export function matvec(a: Mat, x: number[]): number[] {
  return a.map((row) => row.reduce((s, v, i) => s + v * x[i], 0));
}

/** Solve S X = B for symmetric positive-definite S via Cholesky. */
export function cholSolve(s: Mat, b: Mat): Mat {
  const n = s.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = s[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(sum, 1e-12));
      else L[i][j] = sum / L[j][j];
    }
  }
  const m = b[0].length;
  const x = zeros(n, m);
  for (let col = 0; col < m; col++) {
    const y = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      let sum = b[i][col];
      for (let k = 0; k < i; k++) sum -= L[i][k] * y[k];
      y[i] = sum / L[i][i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i];
      for (let k = i + 1; k < n; k++) sum -= L[k][i] * x[k][col];
      x[i][col] = sum / L[i][i];
    }
  }
  return x;
}

/**
 * Ridge regression with an unpenalized intercept. Returns W ((d+1) x m),
 * last row is the intercept.
 */
export function ridge(x: Mat, y: Mat, lambda: number): Mat {
  const xa = x.map((r) => [...r, 1]);
  const g = gram(xa);
  const d = g.length;
  for (let i = 0; i < d - 1; i++) g[i][i] += lambda;
  g[d - 1][d - 1] += 1e-8;
  return cholSolve(g, atb(xa, y));
}

export function ridgePredict(w: Mat, x: number[]): number[] {
  const d = x.length;
  const m = w[0].length;
  const out = new Array<number>(m).fill(0);
  for (let j = 0; j < m; j++) {
    let s = w[d][j];
    for (let i = 0; i < d; i++) s += x[i] * w[i][j];
    out[j] = s;
  }
  return out;
}

/** Eigen-decomposition of a symmetric matrix (cyclic Jacobi). Eigenvalues sorted descending. */
export function symEig(a: Mat, sweeps = 60): { values: number[]; vectors: Mat } {
  const n = a.length;
  const m = a.map((r) => r.slice());
  const v = zeros(n, n);
  for (let i = 0; i < n; i++) v[i][i] = 1;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p][q] * m[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-15) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p];
          const mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k];
          const mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => m[j][j] - m[i][i]);
  return {
    values: order.map((i) => m[i][i]),
    // vectors[k] is the k-th eigenvector
    vectors: order.map((i) => v.map((row) => row[i])),
  };
}
