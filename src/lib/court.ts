import type { AttackPosition, Direction, Hand } from './types';

// Court frame used everywhere: the opponent's half seen from behind the
// hitter (the same view as a phone filming from behind your own end line).
// x: 0 = hitter's left sideline, 9 = hitter's right sideline (meters).
// y: 0 = net, 9 = opponent end line. Own side has negative y.
export const COURT_W = 9;
export const COURT_D = 9;

/** Where each attack position takes off, in the shared court frame. */
export const TAKEOFF: Record<AttackPosition, { x: number; y: number }> = {
  left: { x: 1.0, y: -0.8 },
  middle: { x: 4.5, y: -0.6 },
  right: { x: 8.0, y: -0.8 },
  'back-left': { x: 1.5, y: -3.5 },
  'back-middle': { x: 4.5, y: -3.5 },
  'back-right': { x: 7.5, y: -3.5 },
};

/** Angle of the attack in degrees: 0 = straight down the court, + = hitter's right. */
export function attackAngle(position: AttackPosition, landing: { x: number; y: number }): number {
  const t = TAKEOFF[position];
  return (Math.atan2(landing.x - t.x, landing.y - t.y) * 180) / Math.PI;
}

/** Cone half-width (degrees) that counts as straight. */
export const STRAIGHT_CONE = 12;

export function directionFromLanding(position: AttackPosition, landing: { x: number; y: number }): Direction {
  const a = attackAngle(position, landing);
  if (a < -STRAIGHT_CONE) return 'left';
  if (a > STRAIGHT_CONE) return 'right';
  return 'straight';
}

/**
 * Coaching name for a direction. Players say "line" and "cross", which depend
 * on where the hitter is. The model only ever sees left / straight / right.
 */
export function shotName(position: AttackPosition, d: Direction): string {
  if (d === 'straight') return position === 'left' || position === 'right' ? 'Line' : 'Straight';
  if (position === 'left' || position === 'back-left') return d === 'right' ? 'Cross' : 'Wide line';
  if (position === 'right' || position === 'back-right') return d === 'left' ? 'Cross' : 'Wide line';
  return d === 'left' ? 'To the left' : 'To the right';
}

/** Mirror a label when moving a left-handed hitter into the canonical frame (and back). */
export function mirrorDirection(d: Direction): Direction {
  return d === 'left' ? 'right' : d === 'right' ? 'left' : 'straight';
}

export function canonicalDirection(d: Direction, hand: Hand): Direction {
  return hand === 'L' ? mirrorDirection(d) : d;
}

/** Re-express a canonical 3-class probability vector in the real (unmirrored) frame. */
export function realProbs(p: number[], hand: Hand): number[] {
  return hand === 'L' ? [p[2], p[1], p[0]] : p;
}
