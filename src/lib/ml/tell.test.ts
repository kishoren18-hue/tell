import { describe, expect, it } from 'vitest';
import { directionFromLanding, attackAngle } from '../court';
import { FRAME_DIM, sequenceFeatures, summaryFeatures, SUMMARY_DIM } from '../features';
import { makeDemoDataset } from '../synth';
import { Rng } from '../rng';
import type { Dataset } from '../types';
import { binomialPValue, wilson } from './metrics';
import { analyze, fitLive, readAttack } from './tell';
import { cholSolve, symEig } from './linalg';

describe('court geometry', () => {
  it('labels straight, left and right shots from the hitter frame', () => {
    expect(directionFromLanding('left', { x: 1.0, y: 6 })).toBe('straight');
    expect(directionFromLanding('left', { x: 6, y: 6 })).toBe('right');
    expect(directionFromLanding('right', { x: 2, y: 5 })).toBe('left');
    expect(attackAngle('middle', { x: 4.5, y: 5 })).toBeCloseTo(0, 6);
  });
});

describe('linear algebra', () => {
  it('solves SPD systems', () => {
    const s = [[4, 1], [1, 3]];
    const x = cholSolve(s, [[1], [2]]);
    expect(4 * x[0][0] + x[1][0]).toBeCloseTo(1, 8);
    expect(x[0][0] + 3 * x[1][0]).toBeCloseTo(2, 8);
  });
  it('finds eigenvalues of a symmetric matrix', () => {
    const { values } = symEig([[2, 1], [1, 2]]);
    expect(values[0]).toBeCloseTo(3, 8);
    expect(values[1]).toBeCloseTo(1, 8);
  });
});

describe('statistics', () => {
  it('binomial p-value matches a hand calculation', () => {
    // P(X >= 9 | n = 10, p = 0.5) = 11/1024
    expect(binomialPValue(9, 10, 0.5)).toBeCloseTo(11 / 1024, 10);
  });
  it('wilson interval brackets the estimate', () => {
    const [lo, hi] = wilson(30, 100);
    expect(lo).toBeLessThan(0.3);
    expect(hi).toBeGreaterThan(0.3);
  });
});

describe('features', () => {
  const ds = makeDemoDataset({ matches: 1, attacksPerHitterPerMatch: 2 });
  it('builds fixed-size sequence and summary vectors', () => {
    const a = ds.attacks[0];
    const seq = sequenceFeatures(a.frames!, 'R')!;
    expect(seq.length).toBe(a.frames!.length);
    expect(seq[0].length).toBe(FRAME_DIM);
    expect(summaryFeatures(seq, 100).length).toBe(SUMMARY_DIM);
    expect(summaryFeatures(seq, 100).every(Number.isFinite)).toBe(true);
  });
  it('never looks past the cutoff', () => {
    const a = ds.attacks[1];
    const seq = sequenceFeatures(a.frames!, 'R')!;
    const tampered = seq.map((r, i) => (i > 31 ? r.map(() => 999) : r));
    // 100 ms cutoff = frame 33; 200 ms cutoff = frame 30, so frames after 31 must not matter at 200 ms.
    expect(summaryFeatures(tampered, 200)).toEqual(summaryFeatures(seq, 200));
  });
});

describe('end to end on synthetic data', () => {
  const ds = makeDemoDataset();
  const report = analyze(ds, { horizonsMs: [0, 200, 400] });

  it('beats always-guess-the-common-shot before contact', () => {
    const r = report.horizons.find((h) => h.horizonMs === 200)!;
    expect(r.accuracy).toBeGreaterThan(r.majorityAccuracy + 0.1);
    expect(r.pValue).toBeLessThan(0.01);
    expect(report.groupedByMatch).toBe(true);
  });

  it('ranks the hitter with the planted big tell above the one with none', () => {
    const s = (id: string) => report.hitters.find((h) => h.player.id === id)!.score!;
    expect(s('demo-p2')).toBeGreaterThan(s('demo-p11') + 15);
    expect(s('demo-p9')).toBeGreaterThan(s('demo-p11'));
  });

  it('gets more readable closer to contact', () => {
    const a0 = report.horizons.find((h) => h.horizonMs === 0)!.infoGain;
    const a400 = report.horizons.find((h) => h.horizonMs === 400)!.infoGain;
    expect(a0).toBeGreaterThan(a400);
  });

  it('world model beats constant-velocity and frozen-pose forecasts', () => {
    expect(report.world).not.toBeNull();
    expect(report.world!.model).toBeLessThan(report.world!.constantVelocity);
    expect(report.world!.model).toBeLessThan(report.world!.still);
  });

  it('live reads return probabilities, a landing guess and a 0.5 s ghost', () => {
    const live = fitLive(ds)!;
    const r = readAttack(live, ds.attacks[3], 'R')!;
    expect(r.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(r.ghost.length).toBe(15);
    expect(r.landing).not.toBeNull();
  });
});

describe('does not invent tells', () => {
  it('scores near zero when shot direction is random noise', () => {
    const ds: Dataset = makeDemoDataset({ seed: 5 });
    const rng = new Rng(3);
    const dirs = ['left', 'straight', 'right'] as const;
    ds.attacks = ds.attacks.map((a) => ({ ...a, direction: dirs[rng.int(3)], landing: undefined }));
    const r = analyze(ds, { horizonsMs: [100] });
    expect(r.horizons[0].infoGain).toBeLessThan(0.05);
    expect(r.horizons[0].pValue).toBeGreaterThan(0.01);
  });
});
