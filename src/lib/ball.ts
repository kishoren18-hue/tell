// Ball tracking with MediaPipe's EfficientDet-Lite0 object detector (COCO
// "sports ball"). Used for the red ball box, and to suggest which way the
// ball went after contact so labeling a swing is one tap instead of two.

import { FilesetResolver, ObjectDetector } from '@mediapipe/tasks-vision';
import type { Direction } from './types';

const WASM_BASE = `${import.meta.env.BASE_URL}mediapipe/wasm`;
const MODEL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite';

export interface BallBox {
  x: number; // centre, normalized
  y: number;
  w: number;
  h: number;
  score: number;
}

let detector: Promise<ObjectDetector> | null = null;
let lastTs = 0;

export function getBallDetector(cpu = false): Promise<ObjectDetector> {
  if (!detector) {
    detector = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
      const make = (delegate: 'GPU' | 'CPU') =>
        ObjectDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL, delegate },
          runningMode: 'VIDEO',
          scoreThreshold: 0.15,
          maxResults: 3,
          categoryAllowlist: ['sports ball'],
        });
      if (cpu) return make('CPU');
      try {
        return await make('GPU');
      } catch {
        return make('CPU');
      }
    })();
    detector.catch(() => {
      detector = null;
    });
  }
  return detector;
}

/** Best ball candidate in the frame, preferring one close to where the ball was last seen. */
export function detectBall(det: ObjectDetector, video: HTMLVideoElement, near?: { x: number; y: number } | null): BallBox | null {
  lastTs = Math.max(lastTs + 1, performance.now());
  const res = det.detectForVideo(video, lastTs);
  const W = video.videoWidth || 1;
  const H = video.videoHeight || 1;
  let best: BallBox | null = null;
  let bestScore = -Infinity;
  for (const d of res.detections) {
    const b = d.boundingBox;
    if (!b) continue;
    const box: BallBox = {
      x: (b.originX + b.width / 2) / W,
      y: (b.originY + b.height / 2) / H,
      w: b.width / W,
      h: b.height / H,
      score: d.categories[0]?.score ?? 0,
    };
    // A ball is small and roughly square; penalize anything else.
    const aspect = Math.max(box.w / Math.max(box.h, 1e-6), box.h / Math.max(box.w, 1e-6));
    let s = box.score - (aspect > 1.6 ? 0.3 : 0) - (box.w > 0.2 ? 0.3 : 0);
    if (near) s -= Math.hypot(box.x - near.x, box.y - near.y) * 0.8;
    if (s > bestScore) {
      bestScore = s;
      best = box;
    }
  }
  return best;
}

export interface BallPoint {
  t: number;
  x: number;
  y: number;
}

/**
 * Which way did the ball go after contact? Filming from behind the hitter, the
 * net is up the frame, so a ball heading away from the camera moves up the
 * image. Its sideways drift relative to that tells left / straight / right.
 * Returns null when there isn't a clean track to read.
 */
export function ballDirection(points: BallPoint[], contactT: number, straightDeg = 9): { direction: Direction; angle: number; n: number } | null {
  const after = points.filter((p) => p.t >= contactT - 0.02 && p.t <= contactT + 0.45).sort((a, b) => a.t - b.t);
  if (after.length < 3) return null;
  // Least-squares slope of x and y over time.
  const n = after.length;
  const mt = after.reduce((s, p) => s + p.t, 0) / n;
  const mx = after.reduce((s, p) => s + p.x, 0) / n;
  const my = after.reduce((s, p) => s + p.y, 0) / n;
  let stt = 0;
  let stx = 0;
  let sty = 0;
  for (const p of after) {
    stt += (p.t - mt) ** 2;
    stx += (p.t - mt) * (p.x - mx);
    sty += (p.t - mt) * (p.y - my);
  }
  if (stt < 1e-9) return null;
  const vx = stx / stt;
  const vy = sty / stt;
  if (vy > -0.05) return null; // not travelling away from the camera (dug, blocked back, or noise)
  const angle = (Math.atan2(vx, -vy) * 180) / Math.PI;
  const direction: Direction = angle < -straightDeg ? 'left' : angle > straightDeg ? 'right' : 'straight';
  return { direction, angle, n };
}
