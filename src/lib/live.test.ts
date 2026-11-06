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
