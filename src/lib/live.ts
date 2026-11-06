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
