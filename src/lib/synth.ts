// Synthetic demo data. This is NOT real volleyball footage: it is a
// parametric skeleton doing an approach, jump and arm swing, with "tells"
// planted at known strengths so you can see what the app does before you have
// filmed anything, and so the tests can check the pipeline recovers a tell
// that is really there (and doesn't invent one that isn't).
//
// Each demo hitter has a tell strength between 0 and 1. Strength scales how
// much their hips (early) and shoulders (mid swing) rotate towards the shot
// before contact. Every hitter also has a small late wrist cue right at
// contact, the way real arm swings commit at the last moment.

import { directionFromLanding, mirrorDirection, TAKEOFF } from './court';
import { mirrorFrame } from './pose';
import { Rng } from './rng';
import type { Attack, AttackPosition, Dataset, Direction, Player, PoseFrame } from './types';
import { CONTACT_INDEX, emptyDataset, SAMPLE_HZ, WINDOW_FRAMES } from './types';

type V3 = [number, number, number];

const TEMPLATE: Record<number, V3> = {
  0: [0, -0.68, 0.08],
  1: [-0.03, -0.71, 0.07], 2: [-0.035, -0.71, 0.07], 3: [-0.04, -0.71, 0.06],
  4: [0.03, -0.71, 0.07], 5: [0.035, -0.71, 0.07], 6: [0.04, -0.71, 0.06],
  7: [-0.075, -0.68, 0.0], 8: [0.075, -0.68, 0.0],
  9: [-0.02, -0.63, 0.07], 10: [0.02, -0.63, 0.07],
  11: [-0.19, -0.5, 0], 12: [0.19, -0.5, 0],
  23: [-0.12, 0, 0], 24: [0.12, 0, 0],
  25: [-0.12, 0.45, 0.03], 26: [0.12, 0.45, 0.03],
  27: [-0.12, 0.88, 0], 28: [0.12, 0.88, 0],
  29: [-0.12, 0.93, -0.05], 30: [0.12, 0.93, -0.05],
  31: [-0.12, 0.94, 0.12], 32: [0.12, 0.94, 0.12],
};

interface Key {
  t: number;
  up: V3;
  lo: V3;
}

const RIGHT_ARM: Key[] = [
  { t: -1.2, up: [0.03, 0.28, 0.06], lo: [0, 0.25, 0.1] },
  { t: -0.5, up: [0.03, 0.27, 0.0], lo: [0, 0.25, 0.05] },
  { t: -0.36, up: [0.05, 0.18, -0.2], lo: [0.02, 0.14, -0.2] },
  { t: -0.13, up: [0.11, -0.14, -0.2], lo: [-0.05, -0.2, -0.15] },
  { t: 0, up: [0.06, -0.28, 0.08], lo: [0.03, -0.27, 0.1] },
  { t: 0.3, up: [-0.05, 0.25, 0.1], lo: [-0.15, 0.15, 0.1] },
];
const LEFT_ARM: Key[] = [
  { t: -1.2, up: [-0.03, 0.28, -0.04], lo: [0, 0.25, 0.04] },
  { t: -0.5, up: [-0.03, 0.27, 0.04], lo: [0, 0.25, 0.08] },
  { t: -0.36, up: [-0.05, 0.18, -0.2], lo: [-0.02, 0.14, -0.2] },
  { t: -0.16, up: [-0.05, -0.28, 0.1], lo: [0, -0.26, 0.1] },
  { t: 0, up: [-0.08, 0.1, 0.15], lo: [0.05, 0.1, 0.15] },
  { t: 0.3, up: [-0.04, 0.27, 0.05], lo: [0, 0.25, 0.05] },
];

const smooth = (u: number) => u * u * (3 - 2 * u);
function interpKeys(keys: Key[], t: number): { up: V3; lo: V3 } {
  if (t <= keys[0].t) return keys[0];
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i];
    const b = keys[i + 1];
    if (t <= b.t) {
      const u = smooth((t - a.t) / (b.t - a.t));
      const mix = (p: V3, q: V3): V3 => [p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u, p[2] + (q[2] - p[2]) * u];
      return { up: mix(a.up, b.up), lo: mix(a.lo, b.lo) };
    }
  }
  return keys[keys.length - 1];
}
const bump = (t: number, centre: number, width: number) => Math.exp(-(((t - centre) / width) ** 2));
const ramp = (t: number, a: number, b: number) => smooth(Math.max(0, Math.min(1, (t - a) / (b - a))));

function rotY([x, y, z]: V3, th: number): V3 {
  const c = Math.cos(th);
  const s = Math.sin(th);
  return [x * c - z * s, y, x * s + z * c];
}

interface SwingParams {
  dir: number; // -1 left, 0 straight, 1 right (canonical)
  tell: number; // 0..1
  yawOffset: number;
  timing: number;
  scale: number;
  startX: number;
  noise: number;
}

function poseAt(t0: number, sp: SwingParams, rng: Rng): PoseFrame {
  const t = t0 - sp.timing;
  const d = sp.dir;
  const early = sp.tell * d;
  // Hip yaw: hips square up towards the shot during the plant.
  const hipYaw = sp.yawOffset + 0.2 * early * ramp(t, -0.55, -0.28);
  // Shoulder yaw: big cocking rotation, biased by the tell, plus a late universal cue.
  const shoulderYaw =
    hipYaw +
    -0.45 * bump(t, -0.14, 0.1) +
    0.32 * early * ramp(t, -0.4, -0.12) +
    0.12 * d * ramp(t, -0.08, 0);
  const pitch = 0.25 * (1 - ramp(t, -0.45, -0.3)) + 0.12 * bump(t, -0.38, 0.06);
  const crouch = bump(t, -0.36, 0.08);

  const w: V3[] = new Array(33);
  for (let i = 0; i < 33; i++) {
    const base = TEMPLATE[i];
    if (!base) continue;
    let p: V3 = [base[0] * sp.scale, base[1] * sp.scale, base[2] * sp.scale];
    const upper = i <= 22;
    if (upper) {
      p = rotY(p, shoulderYaw);
      p = [p[0], p[1] * Math.cos(pitch), p[2] - p[1] * Math.sin(pitch)];
    } else if (i >= 23 && i <= 24) {
      p = rotY(p, hipYaw);
    } else {
      p = rotY(p, hipYaw);
      if (i === 25 || i === 26) p = [p[0], p[1] - 0.1 * crouch, p[2] + 0.18 * crouch];
    }
    w[i] = p;
  }
  // Arms.
  const arm = (keys: Key[], sh: number, el: number, wr: number, hand: number[], lateral: number) => {
    const k = interpKeys(keys, t);
    const up = rotY(k.up, shoulderYaw);
    const lo = rotY([k.lo[0] + lateral, k.lo[1], k.lo[2]], shoulderYaw);
    const s = w[sh];
    const e: V3 = [s[0] + up[0] * sp.scale, s[1] + up[1] * sp.scale, s[2] + up[2] * sp.scale];
    const r: V3 = [e[0] + lo[0] * sp.scale, e[1] + lo[1] * sp.scale, e[2] + lo[2] * sp.scale];
    w[el] = e;
    w[wr] = r;
    for (const h of hand) w[h] = [r[0] + 0.02, r[1] + 0.04, r[2]];
  };
  arm(RIGHT_ARM, 12, 14, 16, [18, 20, 22], 0.1 * d * ramp(t, -0.07, 0));
  arm(LEFT_ARM, 11, 13, 15, [17, 19, 21], 0);

  // Image path: approach up the frame towards the net, jump, land.
  const approach = ramp(t, -1.2, -0.42);
  const hipX = sp.startX + 0.015 * early * approach + 0.02 * Math.sin(t * 9) * (1 - approach);
  const jump = Math.max(0, Math.sin(Math.PI * Math.max(0, Math.min(1, (t + 0.3) / 0.6))));
  const hipY = 0.8 - 0.16 * approach - 0.09 * jump + 0.02 * crouch;
  const S = 0.13;

  const wf: number[] = [];
  const pf: number[] = [];
  for (let i = 0; i < 33; i++) {
    const q = w[i];
    const n = sp.noise;
    const x = q[0] + rng.normal() * n;
    const y = q[1] + rng.normal() * n;
    const z = q[2] + rng.normal() * n;
    wf.push(x, y, z);
    pf.push(hipX + x * S, hipY + y * S, 0.85 + 0.1 * rng.next());
  }
  return { w: wf, p: pf };
}

export interface DemoHitter {
  player: Player;
  tell: number;
  position: AttackPosition;
  bias: [number, number, number];
}

export function demoHitters(): DemoHitter[] {
  return [
    { player: { id: 'demo-p2', name: 'Demo hitter 2', number: '2', hand: 'R' }, tell: 0.9, position: 'left', bias: [0.2, 0.3, 0.5] },
    { player: { id: 'demo-p5', name: 'Demo hitter 5', number: '5', hand: 'R' }, tell: 0.6, position: 'left', bias: [0.25, 0.35, 0.4] },
    { player: { id: 'demo-p7', name: 'Demo hitter 7', number: '7', hand: 'R' }, tell: 0.35, position: 'middle', bias: [0.35, 0.3, 0.35] },
    { player: { id: 'demo-p9', name: 'Demo hitter 9', number: '9', hand: 'L' }, tell: 0.7, position: 'right', bias: [0.45, 0.35, 0.2] },
    { player: { id: 'demo-p11', name: 'Demo hitter 11', number: '11', hand: 'R' }, tell: 0.0, position: 'right', bias: [0.4, 0.3, 0.3] },
    { player: { id: 'demo-p14', name: 'Demo hitter 14', number: '14', hand: 'R' }, tell: 0.15, position: 'back-middle', bias: [0.3, 0.4, 0.3] },
  ];
}

const DIR_OF: Direction[] = ['left', 'straight', 'right'];

function sampleCategorical(p: number[], rng: Rng): number {
  const u = rng.next() * p.reduce((a, b) => a + b, 0);
  let acc = 0;
  for (let i = 0; i < p.length; i++) {
    acc += p[i];
    if (u <= acc) return i;
  }
  return p.length - 1;
}

const MIRROR_POS: Record<AttackPosition, AttackPosition> = {
  left: 'right', right: 'left', middle: 'middle',
  'back-left': 'back-right', 'back-right': 'back-left', 'back-middle': 'back-middle',
};

export interface DemoOptions {
  seed?: number;
  matches?: number;
  attacksPerHitterPerMatch?: number;
  noise?: number;
  missingRate?: number;
}

/** Build a full synthetic dataset. Same seed, same data. */
export function makeDemoDataset(o: DemoOptions = {}): Dataset {
  const rng = new Rng(o.seed ?? 2026);
  const nMatches = o.matches ?? 5;
  const per = o.attacksPerHitterPerMatch ?? 10;
  const noise = o.noise ?? 0.03;
  const missing = o.missingRate ?? 0.04;
  const ds = emptyDataset();
  const hitters = demoHitters();
  ds.players = hitters.map((h) => h.player);
  for (let m = 0; m < nMatches; m++) {
    const matchId = `demo-m${m + 1}`;
    ds.matches.push({ id: matchId, name: `Demo match ${m + 1}`, date: `2026-10-${String(3 + m * 3).padStart(2, '0')}`, synthetic: true });
    // Each match has its own camera quirks: a small global yaw offset.
    const camYaw = rng.normal() * 0.06;
    for (const h of hitters) {
      for (let a = 0; a < per; a++) {
        // Work in the canonical (right-handed) frame, then mirror lefties.
        const canonPos = h.player.hand === 'L' ? MIRROR_POS[h.position] : h.position;
        const bias = h.player.hand === 'L' ? [h.bias[2], h.bias[1], h.bias[0]] : h.bias;
        const intended = sampleCategorical(bias, rng);
        const take = TAKEOFF[canonPos];
        const ang = intended === 0 ? -16 - rng.next() * 26 : intended === 2 ? 16 + rng.next() * 26 : (rng.next() - 0.5) * 18;
        const depth = 3.5 + rng.next() * 5;
        let lx = take.x + (depth - take.y) * Math.tan((ang * Math.PI) / 180);
        lx = Math.max(0.3, Math.min(8.7, lx));
        const canonLanding = { x: lx, y: depth };
        const canonDir = directionFromLanding(canonPos, canonLanding);
        const sp: SwingParams = {
          dir: intended - 1,
          tell: h.tell,
          yawOffset: camYaw + rng.normal() * 0.1,
          timing: rng.normal() * 0.02,
          scale: 0.95 + rng.next() * 0.12,
          startX: take.x / 9 * 0.6 + 0.2 + rng.normal() * 0.02,
          noise,
        };
        const frames: (PoseFrame | null)[] = [];
        for (let f = 0; f < WINDOW_FRAMES; f++) {
          const t = (f - CONTACT_INDEX) / SAMPLE_HZ;
          const fr = poseAt(t, sp, rng);
          frames.push(rng.next() < missing ? null : h.player.hand === 'L' ? mirrorFrame(fr) : fr);
        }
        const isL = h.player.hand === 'L';
        const attack: Attack = {
          id: `demo-a${m}-${h.player.id}-${a}`,
          matchId,
          hitterId: h.player.id,
          contactTime: 30 + a * 25 + rng.next() * 10,
          position: h.position,
          landing: isL ? { x: 9 - canonLanding.x, y: canonLanding.y } : canonLanding,
          direction: isL ? mirrorDirection(canonDir) : canonDir,
          shotType: rng.next() < 0.12 ? 'roll' : 'hard',
          outcome: (['kill', 'dug', 'blocked', 'error'] as const)[sampleCategorical([0.4, 0.35, 0.15, 0.1], rng)],
          frames,
          createdAt: Date.now(),
        };
        ds.attacks.push(attack);
      }
    }
  }
  return ds;
}

export { DIR_OF };
