import type { Hand, PoseFrame } from './types';

// MediaPipe Pose landmark indices (person's own left/right).
export const LM = {
  nose: 0,
  lEar: 7,
  rEar: 8,
  lShoulder: 11,
  rShoulder: 12,
  lElbow: 13,
  rElbow: 14,
  lWrist: 15,
  rWrist: 16,
  lHip: 23,
  rHip: 24,
  lKnee: 25,
  rKnee: 26,
  lAnkle: 27,
  rAnkle: 28,
} as const;

/** Joints used as raw features, in a fixed order shared with ml/tell/features.py. */
export const FEATURE_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

/** Skeleton edges for drawing. */
export const BONES: [number, number][] = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 11], [0, 12],
];

/** Index swap for mirroring a left-handed hitter into the canonical right-handed frame. */
const MIRROR_INDEX: number[] = (() => {
  const m = Array.from({ length: 33 }, (_, i) => i);
  const pairs: [number, number][] = [
    [1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18],
    [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32],
  ];
  for (const [a, b] of pairs) {
    m[a] = b;
    m[b] = a;
  }
  return m;
})();

export function mirrorFrame(f: PoseFrame): PoseFrame {
  const w = new Array<number>(99);
  const p = new Array<number>(99);
  for (let i = 0; i < 33; i++) {
    const j = MIRROR_INDEX[i];
    w[i * 3] = -f.w[j * 3];
    w[i * 3 + 1] = f.w[j * 3 + 1];
    w[i * 3 + 2] = f.w[j * 3 + 2];
    p[i * 3] = 1 - f.p[j * 3];
    p[i * 3 + 1] = f.p[j * 3 + 1];
    p[i * 3 + 2] = f.p[j * 3 + 2];
  }
  return { w, p };
}

/** Put every hitter in a right-handed frame so one model serves both hands. */
export function canonicalFrames(frames: (PoseFrame | null)[], hand: Hand): (PoseFrame | null)[] {
  if (hand === 'R') return frames;
  return frames.map((f) => (f ? mirrorFrame(f) : null));
}

export function joint(f: PoseFrame, i: number): [number, number, number] {
  return [f.w[i * 3], f.w[i * 3 + 1], f.w[i * 3 + 2]];
}

export function imgJoint(f: PoseFrame, i: number): [number, number] {
  return [f.p[i * 3], f.p[i * 3 + 1]];
}

/** Centroid of a pose in image space, used to follow one player across frames. */
export function imageCentroid(p: number[]): [number, number] {
  const ids = [LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip];
  let x = 0;
  let y = 0;
  for (const i of ids) {
    x += p[i * 3];
    y += p[i * 3 + 1];
  }
  return [x / ids.length, y / ids.length];
}
