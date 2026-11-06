import type { PointerEvent } from 'react';
import { TAKEOFF } from '../lib/court';
import type { AttackPosition } from '../lib/types';

// Top-down view of the opponent's half from behind the hitter, the same way
// round as a phone filming from behind your own end line. Net at the bottom.
const PX = 10; // svg units per meter
const X0 = 5;
const NET_Y = 100;
const toSvg = (x: number, y: number) => [X0 + x * PX, NET_Y - y * PX] as const;

export interface CourtDot {
  x: number;
  y: number;
  tone: 'ball' | 'hot' | 'ink' | 'line';
  r?: number;
}

interface Props {
  dots?: CourtDot[];
  takeoff?: AttackPosition;
  /** Probabilities for [left, straight, right]; draws three shaded lanes from the takeoff point. */
  lanes?: number[];
  /** Highlight one lane (0..2) with an outline. */
  laneOutline?: number;
  ellipse?: { x: number; y: number; sx: number; sy: number };
  onTap?: (p: { x: number; y: number }) => void;
  onLane?: (lane: number) => void;
  label?: string;
  compact?: boolean;
}

const LANE_EDGES = [-55, -12, 12, 55];

function lanePolygon(pos: AttackPosition, lane: number): string {
  const t = TAKEOFF[pos];
  const [ox, oy] = toSvg(t.x, t.y);
  const a0 = (LANE_EDGES[lane] * Math.PI) / 180;
  const a1 = (LANE_EDGES[lane + 1] * Math.PI) / 180;
  const R = 16;
  const pts = [[ox, oy]];
  for (let i = 0; i <= 8; i++) {
    const a = a0 + ((a1 - a0) * i) / 8;
    const [x, y] = toSvg(t.x + R * Math.sin(a), t.y + R * Math.cos(a));
    pts.push([x, y]);
  }
  return pts.map((p) => p.join(',')).join(' ');
}

export function Court({ dots = [], takeoff, lanes, laneOutline, ellipse, onTap, onLane, label, compact }: Props) {
  const handle = (e: PointerEvent<SVGSVGElement>) => {
    const svg = e.currentTarget;
    const r = svg.getBoundingClientRect();
    const vx = ((e.clientX - r.left) / r.width) * 100;
    const vy = ((e.clientY - r.top) / r.height) * 122;
    const x = (vx - X0) / PX;
    const y = (NET_Y - vy) / PX;
    if (onLane && takeoff) {
      const t = TAKEOFF[takeoff];
      const ang = (Math.atan2(x - t.x, y - t.y) * 180) / Math.PI;
      onLane(ang < -12 ? 0 : ang > 12 ? 2 : 1);
      return;
    }
    if (onTap && y > -0.5) onTap({ x: Math.max(0, Math.min(9, x)), y: Math.max(0.1, Math.min(10, y)) });
  };
  const interactive = !!(onTap || onLane);
  return (
    <figure className={`court ${compact ? 'court--compact' : ''}`}>
      <svg
        viewBox="0 0 100 122"
        role={interactive ? 'button' : 'img'}
        aria-label={label ?? 'Opponent court'}
        onPointerDown={interactive ? handle : undefined}
        className={interactive ? 'court__svg court__svg--tap' : 'court__svg'}
      >
        <defs>
          <clipPath id="court-clip">
            <rect x={X0} y={NET_Y - 9 * PX} width={9 * PX} height={9 * PX} />
          </clipPath>
        </defs>
        <rect x="0" y="0" width="100" height="122" className="court__free" />
        <rect x={X0} y={NET_Y - 9 * PX} width={9 * PX} height={9 * PX} className="court__floor" />
        <rect x={X0} y={NET_Y - 3 * PX} width={9 * PX} height={3 * PX} className="court__front" />
        <rect x={X0} y={NET_Y - 9 * PX} width={9 * PX} height={9 * PX} className="court__lines" />
        <line x1={X0} x2={X0 + 9 * PX} y1={NET_Y - 3 * PX} y2={NET_Y - 3 * PX} className="court__attack" />
        <rect x={X0} y={NET_Y} width={9 * PX} height={18} className="court__own" />
        <line x1={0} x2={100} y1={NET_Y} y2={NET_Y} className="court__net" />
        {takeoff && lanes && (
          <g clipPath="url(#court-clip)">
            {lanes.map((p, i) => (
              <polygon key={i} points={lanePolygon(takeoff, i)} className="court__lane" style={{ opacity: 0.06 + 0.6 * p }} />
            ))}
          </g>
        )}
        {takeoff && laneOutline !== undefined && (
          <g clipPath="url(#court-clip)">
            <polygon points={lanePolygon(takeoff, laneOutline)} className="court__lane-outline" />
          </g>
        )}
        {takeoff && onLane && !lanes && (
          <g clipPath="url(#court-clip)">
            {[0, 1, 2].map((i) => (
              <polygon key={i} points={lanePolygon(takeoff, i)} className={`court__guess court__guess--${i}`} />
            ))}
          </g>
        )}
        {ellipse && (
          <ellipse
            cx={toSvg(ellipse.x, ellipse.y)[0]}
            cy={toSvg(ellipse.x, ellipse.y)[1]}
            rx={Math.max(4, ellipse.sx * PX)}
            ry={Math.max(4, ellipse.sy * PX)}
            className="court__ellipse"
          />
        )}
        {takeoff && (() => {
          const [x, y] = toSvg(TAKEOFF[takeoff].x, TAKEOFF[takeoff].y);
          return <circle cx={x} cy={Math.min(y, 119)} r="2.6" className="court__takeoff" />;
        })()}
        {dots.map((d, i) => {
          const [x, y] = toSvg(d.x, d.y);
          return <circle key={i} cx={x} cy={y} r={d.r ?? 2.2} className={`court__dot court__dot--${d.tone}`} />;
        })}
      </svg>
      {label && <figcaption>{label}</figcaption>}
    </figure>
  );
}
