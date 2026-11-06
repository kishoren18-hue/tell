// Real-time pieces for the live camera mode: a rolling pose buffer for the
// locked player, resampling into the 30 Hz window the models expect, and a
// contact detector that spots the moment of the arm swing's highest reach.

import { LM } from './pose';
import type { Hand, PoseFrame } from './types';
import { CONTACT_INDEX, SAMPLE_HZ, WINDOW_FRAMES } from './types';

export interface TimedFrame {
  t: number; // seconds
  frame: PoseFrame | null;
}

export class PoseBuffer {
  private items: TimedFrame[] = [];
  private keepSeconds: number;
  constructor(keepSeconds = 4) {
    this.keepSeconds = keepSeconds;
  }

  push(t: number, frame: PoseFrame | null) {
    this.items.push({ t, frame });
    const cut = t - this.keepSeconds;
    while (this.items.length && this.items[0].t < cut) this.items.shift();
  }

  clear() {
    this.items = [];
  }

  get latest(): TimedFrame | undefined {
    return this.items[this.items.length - 1];
  }

  /** Nearest detection to time t, if one exists within `tol` seconds. */
  at(t: number, tol = 0.045): PoseFrame | null {
    let best: TimedFrame | null = null;
    let bd = Infinity;
    // Items are time-ordered; a linear scan over ~4 s of frames is cheap.
    for (const it of this.items) {
      const d = Math.abs(it.t - t);
      if (d < bd) {
        bd = d;
        best = it;
      }
    }
    return best && bd <= tol ? best.frame : null;
  }

  /** The 46-frame model window with ball contact at `contactT`. Frames not seen yet are null. */
  window(contactT: number): (PoseFrame | null)[] {
    // 0.1 s tolerance keeps slow devices (10+ fps) usable; repeated frames are harmless.
    return Array.from({ length: WINDOW_FRAMES }, (_, i) => this.at(contactT + (i - CONTACT_INDEX) / SAMPLE_HZ, 0.1));
  }

  /** Time of the highest hitting-wrist point between t0 and t1, if the hand was above the head. */
  peakWrist(hand: Hand, t0: number, t1: number): number | null {
    const wrist = hand === 'R' ? LM.rWrist : LM.lWrist;
    let best: number | null = null;
    let by = Infinity;
    for (const it of this.items) {
      if (it.t < t0 || it.t > t1 || !it.frame) continue;
      const p = it.frame.p;
      const wy = p[wrist * 3 + 1];
      if (wy < p[LM.nose * 3 + 1] && wy < by) {
        by = wy;
        best = it.t;
      }
    }
    return best;
  }

  /** Share of the pre-contact frames that have a detection. */
  coverage(contactT: number): number {
    const w = this.window(contactT).slice(0, CONTACT_INDEX + 1);
    return w.filter(Boolean).length / w.length;
  }
}

/**
 * Finds contact as the top of the hitting hand's path: the wrist rises above
 * the head, peaks, then drops fast. Returns the time of the peak.
 */
export class ContactDetector {
  private hist: { t: number; y: number; raised: boolean }[] = [];
  private lastFire = -Infinity;
  hand: Hand;
  constructor(hand: Hand = 'R') {
    this.hand = hand;
  }

  reset() {
    this.hist = [];
  }

  /** Feed one frame (image coordinates). Returns a contact time when a swing just finished. */
  push(t: number, f: PoseFrame | null): number | null {
    if (!f) return null;
    const wrist = this.hand === 'R' ? LM.rWrist : LM.lWrist;
    const y = (i: number) => f.p[i * 3 + 1];
    const torso = Math.abs((y(LM.lHip) + y(LM.rHip)) / 2 - (y(LM.lShoulder) + y(LM.rShoulder)) / 2) || 0.05;
    const wy = y(wrist) / torso; // in torso lengths, y down
    const raised = y(wrist) < y(LM.nose) - 0.15 * torso;
    this.hist.push({ t, y: wy, raised });
    while (this.hist.length && this.hist[0].t < t - 1) this.hist.shift();
    if (t - this.lastFire < 1.2) return null;

    // Peak = the highest raised point in the last 0.4 s, not in the last 2 samples.
    const recent = this.hist.filter((h) => h.t >= t - 0.4);
    if (recent.length < 4) return null;
    let peak = recent[0];
    for (const h of recent) if (h.y < peak.y) peak = h;
    if (!peak.raised || peak === recent[recent.length - 1]) return null;
    const dropped = wy - peak.y;
    const dt = t - peak.t;
    // Arm swing: hand drops more than ~0.8 torso lengths within 0.25 s of the peak.
    if (dropped > 0.8 && dt > 0 && dt < 0.25) {
      this.lastFire = t;
      this.hist = [];
      return peak.t;
    }
    return null;
  }
}

/**
 * One Euro filter (Casiez et al., CHI 2012): smooths jitter when still and
 * keeps up when moving fast. Used only for drawing; models see raw pose.
 */
export class OneEuro {
  private x: Float64Array | null = null;
  private dx: Float64Array | null = null;
  private t = 0;
  private minCutoff: number;
  private beta: number;
  constructor(minCutoff = 1.4, beta = 0.6) {
    this.minCutoff = minCutoff;
    this.beta = beta;
  }
  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  reset() {
    this.x = null;
  }
  filter(t: number, v: number[]): number[] {
    if (!this.x || this.x.length !== v.length || t - this.t > 0.5) {
      this.x = Float64Array.from(v);
      this.dx = new Float64Array(v.length);
      this.t = t;
      return v.slice();
    }
    const dt = Math.max(1e-3, t - this.t);
    this.t = t;
    const ad = OneEuro.alpha(1, dt);
    const out = new Array<number>(v.length);
    for (let i = 0; i < v.length; i++) {
      const d = (v[i] - this.x[i]) / dt;
      this.dx![i] = ad * d + (1 - ad) * this.dx![i];
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx![i]), dt);
      this.x[i] = a * v[i] + (1 - a) * this.x[i];
      out[i] = this.x[i];
    }
    return out;
  }
}

/** Bounding box of a pose's confidently-seen landmarks, padded. Normalized image coords. */
export function poseBox(p: number[], pad = 0.18): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < 33; i++) {
    if (p[i * 3 + 2] < 0.4) continue;
    x0 = Math.min(x0, p[i * 3]);
    x1 = Math.max(x1, p[i * 3]);
    y0 = Math.min(y0, p[i * 3 + 1]);
    y1 = Math.max(y1, p[i * 3 + 1]);
  }
  if (!Number.isFinite(x0)) return null;
  const w = x1 - x0;
  const h = y1 - y0;
  return { x0: x0 - w * pad, y0: y0 - h * pad * 0.6, x1: x1 + w * pad, y1: y1 + h * pad * 0.4 };
}
