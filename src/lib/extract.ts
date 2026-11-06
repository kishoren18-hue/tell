// Pose extraction in the browser with MediaPipe Pose Landmarker.
// Runs only on a 1.5 s window around each tagged contact, not the whole
// match, so a phone can keep up.

import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';
import { imageCentroid } from './pose';
import type { PoseFrame } from './types';
import { CONTACT_INDEX, SAMPLE_HZ, WINDOW_FRAMES } from './types';

const WASM_BASE = `${import.meta.env.BASE_URL}mediapipe/wasm`;
export const MODEL_URLS = {
  lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
  full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
};

const landmarkers: Partial<Record<'lite' | 'full', Promise<PoseLandmarker>>> = {};
const lastTs = new WeakMap<PoseLandmarker, number>();

export function getLandmarker(quality: 'lite' | 'full' = 'full'): Promise<PoseLandmarker> {
  if (!landmarkers[quality]) {
    const p = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      const make = (delegate: 'GPU' | 'CPU') =>
        PoseLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL_URLS[quality], delegate },
          runningMode: 'VIDEO',
          numPoses: 6,
          minPoseDetectionConfidence: 0.35,
          minPosePresenceConfidence: 0.35,
          minTrackingConfidence: 0.35,
        });
      // ?delegate=cpu forces the CPU path (useful on devices with a weak or emulated GPU).
      const forceCpu = typeof location !== 'undefined' && new URLSearchParams(location.search).get('delegate') === 'cpu';
      if (forceCpu) return await make('CPU');
      try {
        return await make('GPU');
      } catch {
        return await make('CPU');
      }
    })();
    landmarkers[quality] = p;
    p.catch(() => {
      delete landmarkers[quality];
    });
  }
  return landmarkers[quality]!;
}

export interface Detected {
  frame: PoseFrame;
  c: [number, number];
}

/** Detect every person in the current video frame. Timestamps are kept increasing per landmarker, as VIDEO mode requires. */
export function detectAll(lm: PoseLandmarker, video: HTMLVideoElement): Detected[] {
  const ts = Math.max((lastTs.get(lm) ?? 0) + 1, performance.now());
  lastTs.set(lm, ts);
  const res = lm.detectForVideo(video, ts);
  return res.landmarks.map((pts, k) => {
    const p: number[] = [];
    for (const q of pts) p.push(q.x, q.y, q.visibility ?? 1);
    const w: number[] = [];
    for (const q of res.worldLandmarks[k]) w.push(q.x, q.y, q.z);
    return { frame: { w, p }, c: imageCentroid(p) };
  });
}

function seek(video: HTMLVideoElement, t: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = Math.max(0, Math.min(video.duration || t, t));
    if (Math.abs(video.currentTime - target) < 1e-4 && video.readyState >= 2) {
      resolve();
      return;
    }
    const done = () => {
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', fail);
      resolve();
    };
    const fail = () => {
      video.removeEventListener('seeked', done);
      reject(new Error('The video could not be read at that point.'));
    };
    video.addEventListener('seeked', done);
    video.addEventListener('error', fail);
    video.currentTime = target;
  });
}

type Candidate = Detected;

/**
 * Sample WINDOW_FRAMES frames around `contactTime`, detect everyone in each,
 * then follow the player the user tapped (nearest to `tap` on the contact
 * frame) backwards and forwards by image position.
 */
export async function extractAttackPose(
  video: HTMLVideoElement,
  contactTime: number,
  tap: { x: number; y: number },
  onProgress?: (frac: number) => void,
): Promise<{ frames: (PoseFrame | null)[]; found: number }> {
  const lm = await getLandmarker();
  const wasPaused = video.paused;
  const restore = video.currentTime;
  video.pause();
  const all: Candidate[][] = [];
  for (let i = 0; i < WINDOW_FRAMES; i++) {
    const t = contactTime + (i - CONTACT_INDEX) / SAMPLE_HZ;
    await seek(video, t);
    const cands: Candidate[] = detectAll(lm, video);
    all.push(cands);
    onProgress?.((i + 1) / WINDOW_FRAMES);
  }
  await seek(video, restore);
  if (!wasPaused) void video.play();
  const frames = linkTrack(all, tap);
  return { frames, found: frames.filter(Boolean).length };
}

/** Greedy nearest-centroid linking from the contact frame outwards. */
export function linkTrack(all: Candidate[][], tap: { x: number; y: number }, maxJump = 0.12): (PoseFrame | null)[] {
  const out: (PoseFrame | null)[] = new Array(all.length).fill(null);
  const pick = (cands: Candidate[], ref: [number, number], limit: number) => {
    let best: Candidate | null = null;
    let bd = Infinity;
    for (const c of cands) {
      const d = Math.hypot(c.c[0] - ref[0], c.c[1] - ref[1]);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    return best && bd <= limit ? best : null;
  };
  // On the contact frame, the hitter is whoever is closest to the tap.
  let start = CONTACT_INDEX;
  let first = pick(all[start] ?? [], [tap.x, tap.y], 0.25);
  if (!first) {
    for (let d = 1; d <= 3 && !first; d++) {
      for (const s of [CONTACT_INDEX - d, CONTACT_INDEX + d]) {
        const f = pick(all[s] ?? [], [tap.x, tap.y], 0.25);
        if (f && !first) {
          first = f;
          start = s;
        }
      }
    }
  }
  if (!first) return out;
  out[start] = first.frame;
  for (const dir of [-1, 1]) {
    let ref = first.c;
    let gap = 0;
    for (let i = start + dir; i >= 0 && i < all.length; i += dir) {
      const next = pick(all[i], ref, maxJump * (1 + gap));
      if (next) {
        out[i] = next.frame;
        ref = next.c;
        gap = 0;
      } else gap++;
    }
  }
  return out;
}
