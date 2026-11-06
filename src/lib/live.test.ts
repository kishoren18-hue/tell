import { describe, expect, it } from 'vitest';
import { ContactDetector, PoseBuffer } from './live';
import { makeDemoDataset } from './synth';
import { CONTACT_INDEX, SAMPLE_HZ } from './types';

describe('live camera helpers', () => {
  const ds = makeDemoDataset({ matches: 1, attacksPerHitterPerMatch: 3, missingRate: 0 });

  it('finds contact within a couple of frames on synthetic swings', () => {
    let found = 0;
    let errSum = 0;
    for (const a of ds.attacks) {
      const hand = ds.players.find((p) => p.id === a.hitterId)!.hand;
      const det = new ContactDetector(hand);
      let hit: number | null = null;
      a.frames!.forEach((f, i) => {
        // Feed at 60 fps by repeating frames, like a phone camera would.
        for (const sub of [0, 0.5]) {
          const t = (i + sub) / SAMPLE_HZ;
          const r = det.push(t, f);
          if (r !== null && hit === null) hit = r;
        }
      });
      if (hit !== null) {
        found++;
        errSum += Math.abs(hit - CONTACT_INDEX / SAMPLE_HZ);
      }
    }
    expect(found / ds.attacks.length).toBeGreaterThan(0.9);
    expect(errSum / found).toBeLessThan(2.5 / SAMPLE_HZ);
  });

  it('rebuilds the 30 Hz model window from irregular camera timestamps', () => {
    const a = ds.attacks[0];
    const buf = new PoseBuffer();
    a.frames!.forEach((f, i) => buf.push(i / SAMPLE_HZ + 0.004 * Math.sin(i), f));
    const w = buf.window(CONTACT_INDEX / SAMPLE_HZ);
    expect(w.length).toBe(a.frames!.length);
    expect(w[CONTACT_INDEX]).toBe(a.frames![CONTACT_INDEX]);
    expect(buf.coverage(CONTACT_INDEX / SAMPLE_HZ)).toBe(1);
  });
});

describe('ball direction and smoothing', () => {
  it('reads the ball path after contact', async () => {
    const { ballDirection } = await import('./ball');
    const path = (vx: number) => Array.from({ length: 10 }, (_, i) => ({ t: 1 + i / 30, x: 0.5 + vx * (i / 30), y: 0.5 - 0.6 * (i / 30) }));
    expect(ballDirection(path(0.3), 1)?.direction).toBe('right');
    expect(ballDirection(path(-0.3), 1)?.direction).toBe('left');
    expect(ballDirection(path(0.02), 1)?.direction).toBe('straight');
    // Coming back towards the camera (a dig) gives no read.
    const back = Array.from({ length: 10 }, (_, i) => ({ t: 1 + i / 30, x: 0.5, y: 0.5 + 0.4 * (i / 30) }));
    expect(ballDirection(back, 1)).toBeNull();
  });

  it('one-euro filter damps jitter but follows a real move', async () => {
    const { OneEuro } = await import('./live');
    const f = new OneEuro();
    let out = [0];
    for (let i = 0; i < 60; i++) out = f.filter(i / 30, [0.5 + (i % 2 ? 0.01 : -0.01)]);
    expect(Math.abs(out[0] - 0.5)).toBeLessThan(0.006);
    for (let i = 60; i < 75; i++) out = f.filter(i / 30, [0.8]);
    expect(out[0]).toBeGreaterThan(0.75);
  });
});
