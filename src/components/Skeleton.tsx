import { useEffect, useRef } from 'react';
import { N_JOINT_FEATS } from '../lib/features';

// Draws pose states (13 joints in torso units around the hips + hip image
// position) as a figure seen from behind, the way the camera saw it.

const BONES: [number, number][] = [
  [1, 2], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12], [0, 1], [0, 2],
];
const HIT_ARM: [number, number][] = [[2, 4], [4, 6]];

interface Props {
  states: number[][];
  /** Index of the frame to show. */
  frame: number;
  ghost?: number[][] | null;
  /** Index in `states` where the ghost starts (frame of the cutoff). */
  ghostFrom?: number;
  showGhostUntil?: number;
  hand?: 'R' | 'L';
  height?: number;
}

export function Skeleton({ states, frame, ghost, ghostFrom = 0, showGhostUntil = 0, hand = 'R', height = 260 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !states.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth;
    const H = height;
    c.width = W * dpr;
    c.height = H * dpr;
    const g = c.getContext('2d')!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const hx = states.map((s) => s[N_JOINT_FEATS]);
    const hy = states.map((s) => s[N_JOINT_FEATS + 1]);
    const mx = hx.reduce((a, b) => a + b, 0) / hx.length;
    const my = hy.reduce((a, b) => a + b, 0) / hy.length;
    const unit = H * 0.19;
    const travel = H * 1.1;
    const css = getComputedStyle(c);
    const ink = css.getPropertyValue('--ink').trim() || '#14213d';
    const ball = css.getPropertyValue('--ball').trim() || '#f2c230';
    const hot = css.getPropertyValue('--hot').trim() || '#d7263d';
    const floorY = H - 10;

    g.strokeStyle = css.getPropertyValue('--line-soft').trim() || '#c9d3e3';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, floorY);
    g.lineTo(W, floorY);
    g.stroke();

    const project = (s: number[]) => {
      const ox = W / 2 + (s[N_JOINT_FEATS] - mx) * travel;
      const oy = H * 0.6 + (s[N_JOINT_FEATS + 1] - my) * travel;
      return (j: number): [number, number] => [ox + s[j * 3] * unit, oy + s[j * 3 + 1] * unit];
    };
    const hitArm: [number, number][] = hand === 'L' ? [[1, 3], [3, 5]] : HIT_ARM;
    const offArm: [number, number][] = hand === 'L' ? [[2, 4], [4, 6]] : [[1, 3], [3, 5]];
    const draw = (s: number[], color: string, width: number, alpha: number, dash: number[] = []) => {
      const P = project(s);
      g.globalAlpha = alpha;
      g.setLineDash(dash);
      g.lineCap = 'round';
      g.strokeStyle = color;
      g.lineWidth = width;
      for (const [a, b] of [...BONES, ...offArm]) {
        const [x1, y1] = P(a);
        const [x2, y2] = P(b);
        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        g.stroke();
      }
      g.strokeStyle = dash.length ? color : hot;
      g.lineWidth = width + 1;
      for (const [a, b] of hitArm) {
        const [x1, y1] = P(a);
        const [x2, y2] = P(b);
        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        g.stroke();
      }
      const [nx, ny] = P(0);
      g.fillStyle = color;
      g.beginPath();
      g.arc(nx, ny - unit * 0.12, unit * 0.13, 0, Math.PI * 2);
      dash.length ? g.stroke() : g.fill();
      g.globalAlpha = 1;
      g.setLineDash([]);
    };

    // Faint trail of the last few frames so motion reads in a still.
    for (let k = Math.max(0, frame - 4); k < frame; k += 2) draw(states[k], ink, 1.5, 0.12);
    draw(states[Math.min(frame, states.length - 1)], ink, 3, 1);
    if (ghost && showGhostUntil > 0) {
      const g0 = Math.min(ghost.length, showGhostUntil);
            draw(ghost[g0 - 1], ball, 2.5, 0.95, [5, 4]);
    }
    void ghostFrom;
  }, [states, frame, ghost, ghostFrom, showGhostUntil, hand, height]);
  return <canvas ref={ref} className="skeleton" style={{ height }} aria-label="Hitter skeleton replay" role="img" />;
}
