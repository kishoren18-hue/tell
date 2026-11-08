// Core data model for Tell. Everything here is JSON-serializable so a whole
// dataset can be exported, merged across phones, and loaded by the Python
// training pipeline in ml/ (see docs/DATA_FORMAT.md).

export const SAMPLE_HZ = 30;
/** Seconds of pose kept before contact. */
export const PRE_CONTACT_S = 1.2;
/** Seconds of pose kept after contact (follow-through, used by the world model). */
export const POST_CONTACT_S = 0.3;
export const CONTACT_INDEX = Math.round(PRE_CONTACT_S * SAMPLE_HZ); // 36
export const WINDOW_FRAMES = CONTACT_INDEX + Math.round(POST_CONTACT_S * SAMPLE_HZ) + 1; // 46

/** Direction of the attack in the hitter's own frame (hitter facing the net). */
export type Direction = 'left' | 'straight' | 'right';
export const DIRECTIONS: Direction[] = ['left', 'straight', 'right'];

export type Hand = 'R' | 'L';
export type AttackPosition = 'left' | 'middle' | 'right' | 'back-middle' | 'back-right' | 'back-left';
export type ShotType = 'hard' | 'roll' | 'tip';
export type Outcome = 'kill' | 'dug' | 'blocked' | 'error' | 'unknown';

/**
 * One sampled pose frame. `w` is MediaPipe world landmarks flattened
 * (33 x [x, y, z], meters, hip-centred). `p` is image landmarks flattened
 * (33 x [x, y, visibility], normalized 0..1). Missing detections are null.
 */
export interface PoseFrame {
  w: number[];
  p: number[];
}

// One stable profile across devices in each private workspace.
export const INDIVIDUAL_HITTER = 'individual-hitter';

export interface Player {
  id: string;
  name: string;
  number?: string;
  hand: Hand;
}

export interface DefenseReview {
  id: string;
  defender: string;
  role: 'libero' | 'blocker';
  /** Times in seconds in the source video. */
  contactTime: number;
  firstStepTime?: number;
  /** Defender's location in the video frame at contact, normalized to 0–1. */
  contactPoint: { x: number; y: number };
}

export interface Match {
  id: string;
  name: string;
  date: string;
  videoName?: string;
  videoSize?: number;
  synthetic?: boolean;
  /** Recorded with the live camera: pose only, no video file. */
  live?: boolean;
  defenseReviews?: DefenseReview[];
}

export interface Attack {
  id: string;
  matchId: string;
  hitterId: string;
  /** Time of ball contact in the source video, seconds. */
  contactTime: number;
  /** Where the user tapped the hitter on the contact frame (normalized). */
  hitterTap?: { x: number; y: number };
  position: AttackPosition;
  /** Landing / first-touch point on the opponent half, meters. x: 0 = hitter's left sideline, 9 = right. y: 0 = net, 9 = end line. */
  landing?: { x: number; y: number };
  /** Direction in the hitter's frame (derived from landing, or set by hand). */
  direction: Direction;
  shotType: ShotType;
  outcome: Outcome;
  /** WINDOW_FRAMES frames sampled at SAMPLE_HZ, contact at CONTACT_INDEX. */
  frames: (PoseFrame | null)[] | null;
  createdAt: number;
}

export interface Dataset {
  format: 'tell-dataset';
  version: 1;
  sampleHz: number;
  contactIndex: number;
  windowFrames: number;
  players: Player[];
  matches: Match[];
  attacks: Attack[];
}

export function emptyDataset(): Dataset {
  return {
    format: 'tell-dataset',
    version: 1,
    sampleHz: SAMPLE_HZ,
    contactIndex: CONTACT_INDEX,
    windowFrames: WINDOW_FRAMES,
    players: [],
    matches: [],
    attacks: [],
  };
}

export function uid(prefix = ''): string {
  const r = Math.random().toString(36).slice(2, 10);
  return `${prefix}${Date.now().toString(36)}${r}`;
}
