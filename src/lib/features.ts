// Feature extraction. The exact same math lives in ml/tell/features.py and a
// parity test (ml/tests/test_parity.py) checks the two agree to 1e-6.
//
// Design notes, borrowed from tennis shot-direction work:
// - Use MediaPipe *world* landmarks (metric, hip-centred). In tennis, swapping
//   world coordinates for image coordinates cut cross-player direction accuracy
//   from 68% to 21% (Hazarika 2026), so world coords are the default here.
// - Keep a short sequence of pose up to the cutoff, like Shimizu et al. (2019)
//   who fed pose up to the moment of impact into an LSTM.
// - Mirror left-handed hitters so the hitting arm is always the right arm.

import { canonicalFrames, FEATURE_JOINTS, LM } from './pose';
import type { Hand, PoseFrame } from './types';
import { CONTACT_INDEX, SAMPLE_HZ } from './types';

export const N_JOINT_FEATS = FEATURE_JOINTS.length * 3; // 39
export const ENGINEERED_NAMES = [
  'shoulder_yaw',
  'hip_yaw',
  'hip_shoulder_separation',
  'torso_pitch',
  'torso_roll',
  'hitting_elbow_angle',
  'hitting_shoulder_abduction',
  'wrist_rel_shoulder_x',
  'wrist_rel_shoulder_y',
  'wrist_rel_shoulder_z',
  'head_yaw',
  'off_wrist_height',
  'hip_image_x',
  'hip_image_y',
];
export const N_ENG = ENGINEERED_NAMES.length; // 14
export const FRAME_DIM = N_JOINT_FEATS + N_ENG; // 53

/** Frames in the summary window that ends at the cutoff. */
export const SUMMARY_WINDOW = 10;
export const SUMMARY_DIM = N_ENG * 5 + N_JOINT_FEATS * 2; // 148

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mid = (a: V3, b: V3): V3 => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const norm = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function angleBetween(a: V3, b: V3): number {
  const d = norm(a) * norm(b);
  if (d < 1e-9) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b) / d)));
}
const J = (f: PoseFrame, i: number): V3 => [f.w[i * 3], f.w[i * 3 + 1], f.w[i * 3 + 2]];

/** Per-frame feature vector (FRAME_DIM) from one canonical (right-handed) frame. */
export function frameFeatures(f: PoseFrame): number[] {
  const ls = J(f, LM.lShoulder);
  const rs = J(f, LM.rShoulder);
  const lh = J(f, LM.lHip);
  const rh = J(f, LM.rHip);
  const sMid = mid(ls, rs);
  const hMid = mid(lh, rh);
  const torso = sub(sMid, hMid);
  const L = Math.max(norm(torso), 0.05);

  const out: number[] = [];
  for (const i of FEATURE_JOINTS) {
    const j = J(f, i);
    out.push((j[0] - hMid[0]) / L, (j[1] - hMid[1]) / L, (j[2] - hMid[2]) / L);
  }

  // World axes: x to the camera's right, y down, z away from the camera
  // (towards the net when filming from behind). Yaw is measured in the x-z plane.
  const sv = sub(rs, ls);
  const hv = sub(rh, lh);
  const shoulderYaw = Math.atan2(sv[2], sv[0]);
  const hipYaw = Math.atan2(hv[2], hv[0]);
  let sep = shoulderYaw - hipYaw;
  while (sep > Math.PI) sep -= 2 * Math.PI;
  while (sep < -Math.PI) sep += 2 * Math.PI;
  const pitch = Math.atan2(torso[2], -torso[1]);
  const roll = Math.atan2(torso[0], -torso[1]);

  const re = J(f, LM.rElbow);
  const rw = J(f, LM.rWrist);
  const elbowAngle = angleBetween(sub(rs, re), sub(rw, re));
  const abduction = angleBetween(sub(re, rs), sub(hMid, sMid));
  const wr = sub(rw, rs);

  const nose = J(f, LM.nose);
  const earMid = mid(J(f, LM.lEar), J(f, LM.rEar));
  const headYaw = Math.atan2(nose[0] - earMid[0], nose[2] - earMid[2]);
  const offWristH = (J(f, LM.lWrist)[1] - sMid[1]) / L;

  const hipImgX = (f.p[LM.lHip * 3] + f.p[LM.rHip * 3]) / 2;
  const hipImgY = (f.p[LM.lHip * 3 + 1] + f.p[LM.rHip * 3 + 1]) / 2;

  out.push(
    shoulderYaw, hipYaw, sep, pitch, roll, elbowAngle, abduction,
    wr[0] / L, wr[1] / L, wr[2] / L, headYaw, offWristH, hipImgX, hipImgY,
  );
  return out;
}

/**
 * Turn a window of frames into a dense (T x FRAME_DIM) matrix, filling
 * missing detections by linear interpolation. Returns null when fewer than
 * half the frames have a detection.
 */
export function sequenceFeatures(frames: (PoseFrame | null)[], hand: Hand): number[][] | null {
  const canon = canonicalFrames(frames, hand);
  const raw = canon.map((f) => (f ? frameFeatures(f) : null));
  const valid = raw.filter(Boolean).length;
  if (valid < Math.ceil(raw.length / 2)) return null;
  return interpolate(raw);
}

export function interpolate(rows: (number[] | null)[]): number[][] {
  const n = rows.length;
  const out: number[][] = new Array(n);
  const idx: number[] = [];
  rows.forEach((r, i) => r && idx.push(i));
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    if (r) {
      out[i] = r.slice();
      continue;
    }
    let lo = -1;
    let hi = -1;
    for (const k of idx) {
      if (k < i) lo = k;
      if (k > i) {
        hi = k;
        break;
      }
    }
    if (lo < 0) out[i] = (rows[hi] as number[]).slice();
    else if (hi < 0) out[i] = (rows[lo] as number[]).slice();
    else {
      const a = rows[lo] as number[];
      const b = rows[hi] as number[];
      const t = (i - lo) / (hi - lo);
      out[i] = a.map((v, d) => v + (b[d] - v) * t);
    }
  }
  return out;
}

/** Index of the last frame the model may look at, for a cutoff `horizonMs` before contact. */
export function cutoffIndex(horizonMs: number): number {
  return CONTACT_INDEX - Math.round((horizonMs / 1000) * SAMPLE_HZ);
}

/**
 * Fixed-length summary of the SUMMARY_WINDOW frames ending at the cutoff:
 * engineered features as last / mean / slope / min / max, plus joint
 * positions and joint velocities at the cutoff. Nothing after the cutoff is used.
 */
export function summaryFeatures(seq: number[][], horizonMs: number): number[] {
  const e = cutoffIndex(horizonMs);
  const s = Math.max(0, e - SUMMARY_WINDOW + 1);
  const win = seq.slice(s, e + 1);
  const W = win.length;
  const out: number[] = [];
  for (let k = 0; k < N_ENG; k++) {
    const d = N_JOINT_FEATS + k;
    const vals = win.map((r) => r[d]);
    const last = vals[W - 1];
    const mean = vals.reduce((a, b) => a + b, 0) / W;
    const slope = (vals[W - 1] - vals[0]) / Math.max(1, W - 1);
    out.push(last, mean, slope, Math.min(...vals), Math.max(...vals));
  }
  const last = seq[e];
  const prev = seq[Math.max(0, e - 3)];
  for (let d = 0; d < N_JOINT_FEATS; d++) out.push(last[d]);
  for (let d = 0; d < N_JOINT_FEATS; d++) out.push((last[d] - prev[d]) / 3);
  return out;
}
