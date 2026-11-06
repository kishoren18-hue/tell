// The analysis that turns tagged attacks into tell scores.
//
// Tell score, per hitter: how much the model's pre-contact read reduces
// uncertainty about the shot, compared with only knowing how often shots go
// each way. 0 means your body gives nothing away; 100 means the model is
// certain and right every time. Formally 100 * (1 - CE_model / CE_prior)
// on out-of-fold predictions, the same idea as McFadden's pseudo-R².
//
// Read time: the earliest cutoff before contact at which the model still beats
// "always guess the most common shot" with p < 0.05.

import { canonicalDirection, mirrorDirection, realProbs, shotName, TAKEOFF } from '../court';
import { ENGINEERED_NAMES, N_JOINT_FEATS, sequenceFeatures, summaryFeatures, cutoffIndex } from '../features';
import { Rng } from '../rng';
import type { Attack, AttackPosition, Dataset, Direction, Hand, Player, PoseFrame } from '../types';
import { DIRECTIONS } from '../types';
import { fitCalibrated, outOfFold, type FittedModel } from './cv';
import {
  accuracy, argmax, binomialPValue, calibration, classCounts, confusion,
  crossEntropyBits, wilson, type ReliabilityBin,
} from './metrics';
import { DEFAULT_TRAIN, type ModelKind, type TrainOptions } from './models';
import {
  constantVelocityForecast, fitDynamics, fitLanding, forecastPose, FORECAST_STEPS,
  poseError, predictLanding, staticForecast, stateOf, type LandingModel, type PoseDynamics,
} from './world';

export const HORIZONS_MS = [0, 100, 200, 300, 400, 500];
export const DEFAULT_SCORE_HORIZON = 100;
export const MIN_ATTACKS_FOR_SCORE = 8;
export const MIN_ATTACKS_TO_TRAIN = 24;

export interface Usable {
  attack: Attack;
  hand: Hand;
  seq: number[][];
  label: number; // canonical class index
}

export function usableAttacks(ds: Dataset): { usable: Usable[]; skipped: { attack: Attack; reason: string }[] } {
  const hands = new Map(ds.players.map((p) => [p.id, p.hand] as const));
  const usable: Usable[] = [];
  const skipped: { attack: Attack; reason: string }[] = [];
  for (const a of ds.attacks) {
    const hand = hands.get(a.hitterId) ?? 'R';
    if (!a.frames) {
      skipped.push({ attack: a, reason: 'No pose yet' });
      continue;
    }
    const seq = sequenceFeatures(a.frames, hand);
    if (!seq) {
      skipped.push({ attack: a, reason: 'Hitter not visible in enough frames' });
      continue;
    }
    usable.push({ attack: a, hand, seq, label: DIRECTIONS.indexOf(canonicalDirection(a.direction, hand)) });
  }
  return { usable, skipped };
}

export interface HorizonResult {
  horizonMs: number;
  n: number;
  accuracy: number;
  accuracyCI: [number, number];
  majorityAccuracy: number;
  chance: number;
  pValue: number;
  ceModel: number;
  cePrior: number;
  infoGain: number; // 0..1
  ece: number;
  reliability: ReliabilityBin[];
  confusion: number[][];
}

export interface Cue {
  feature: string;
  label: string;
  effect: number; // standardized mean difference, right minus left
  detail: string;
}

export interface HitterScore {
  player: Player;
  n: number;
  score: number | null;
  scoreCI: [number, number] | null;
  accuracy: number;
  counts: Record<Direction, number>;
  cues: Cue[];
  lowData: boolean;
}

export interface AttackRead {
  attackId: string;
  /** Out-of-fold probabilities in the real (unmirrored) frame, at the score horizon. */
  probs: number[];
  predicted: Direction;
  correct: boolean;
}

export interface WorldEval {
  horizonMs: number;
  model: number;
  constantVelocity: number;
  still: number;
  n: number;
}

export interface Report {
  createdAt: number;
  kind: ModelKind;
  nAttacks: number;
  nSkipped: number;
  nMatches: number;
  groupedByMatch: boolean;
  scoreHorizonMs: number;
  classCounts: number[];
  horizons: HorizonResult[];
  readTimeMs: number | null;
  hitters: HitterScore[];
  reads: AttackRead[];
  world: WorldEval | null;
  landingMae: { model: number; baseline: number } | null;
  synthetic: boolean;
}

export function infoGain(probs: number[][], y: number[], prior: number[]): number {
  const ceM = crossEntropyBits(probs, y);
  const ceP = crossEntropyBits(y.map(() => prior), y);
  if (ceP <= 1e-9) return 0;
  return Math.max(0, 1 - ceM / ceP);
}

function bootstrapScore(probs: number[][], y: number[], prior: number[], rng: Rng, reps = 400): [number, number] {
  const vals: number[] = [];
  for (let r = 0; r < reps; r++) {
    const ps: number[][] = [];
    const ys: number[] = [];
    for (let i = 0; i < y.length; i++) {
      const j = rng.int(y.length);
      ps.push(probs[j]);
      ys.push(y[j]);
    }
    vals.push(infoGain(ps, ys, prior) * 100);
  }
  vals.sort((a, b) => a - b);
  return [vals[Math.floor(reps * 0.05)], vals[Math.floor(reps * 0.95)]];
}

const CUE_TEXT: Record<string, { label: string; deg: boolean; unit?: string }> = {
  shoulder_yaw: { label: 'Shoulder turn', deg: true },
  hip_yaw: { label: 'Hip turn', deg: true },
  hip_shoulder_separation: { label: 'Hip-shoulder separation', deg: true },
  torso_pitch: { label: 'Forward lean', deg: true },
  torso_roll: { label: 'Side lean', deg: true },
  hitting_elbow_angle: { label: 'Hitting elbow bend', deg: true },
  hitting_shoulder_abduction: { label: 'Hitting arm height', deg: true },
  wrist_rel_shoulder_x: { label: 'Hand position across the body', deg: false, unit: 'torso lengths' },
  wrist_rel_shoulder_y: { label: 'Hand height', deg: false, unit: 'torso lengths' },
  wrist_rel_shoulder_z: { label: 'Hand behind or in front of the shoulder', deg: false, unit: 'torso lengths' },
  head_yaw: { label: 'Head turn', deg: true },
  off_wrist_height: { label: 'Non-hitting arm height', deg: false, unit: 'torso lengths' },
  hip_image_x: { label: 'Approach line', deg: false, unit: 'of frame width' },
  hip_image_y: { label: 'Approach depth', deg: false, unit: 'of frame height' },
};

/**
 * Which body cues differ most between this hitter's two most common shots,
 * measured at the cutoff (so these are things visible before contact).
 */
export function hitterCues(items: Usable[], horizonMs: number): Cue[] {
  const e = cutoffIndex(horizonMs);
  const counts = [0, 1, 2].map((c) => items.filter((u) => u.label === c).length);
  const [ca, cb] = [0, 1, 2].sort((p, q) => counts[q] - counts[p]).slice(0, 2).sort();
  const groupA = items.filter((u) => u.label === ca);
  const groupB = items.filter((u) => u.label === cb);
  if (groupA.length < 4 || groupB.length < 4) return [];
  const hand = items[0].hand;
  const position = mostCommon(items.map((u) => u.attack.position));
  const nameOf = (c: number) => shotName(position, hand === 'L' ? mirrorDirection(DIRECTIONS[c]) : DIRECTIONS[c]).toLowerCase();
  const cues: Cue[] = [];
  ENGINEERED_NAMES.forEach((name, k) => {
    const d = N_JOINT_FEATS + k;
    const a = groupA.map((u) => u.seq[e][d]);
    const b = groupB.map((u) => u.seq[e][d]);
    const m = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length;
    const vr = (v: number[], mu: number) => v.reduce((s, x) => s + (x - mu) ** 2, 0) / Math.max(1, v.length - 1);
    const ma = m(a);
    const mb = m(b);
    const sd = Math.sqrt((vr(a, ma) * (a.length - 1) + vr(b, mb) * (b.length - 1)) / Math.max(1, a.length + b.length - 2)) || 1;
    const effect = (mb - ma) / sd;
    const t = CUE_TEXT[name];
    const diff = mb - ma;
    const amount = t.deg ? `${Math.abs((diff * 180) / Math.PI).toFixed(0)}°` : `${Math.abs(diff).toFixed(2)} ${t.unit}`;
    cues.push({
      feature: name,
      label: t.label,
      effect,
      detail: `${amount} apart between your ${nameOf(ca)} and ${nameOf(cb)} shots, ${horizonMs} ms before contact`,
    });
  });
  return cues.sort((p, q) => Math.abs(q.effect) - Math.abs(p.effect)).filter((c) => Math.abs(c.effect) >= 0.6).slice(0, 3);
}

function mostCommon<T>(xs: T[]): T {
  const m = new Map<T, number>();
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
  return [...m.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

export interface AnalyzeOptions {
  horizonsMs?: number[];
  scoreHorizonMs?: number;
  train?: Partial<TrainOptions>;
  onProgress?: (msg: string, frac: number) => void;
}

export function analyze(ds: Dataset, o: AnalyzeOptions = {}): Report {
  const horizons = o.horizonsMs ?? HORIZONS_MS;
  const scoreH = o.scoreHorizonMs ?? DEFAULT_SCORE_HORIZON;
  const opts: TrainOptions = { ...DEFAULT_TRAIN, ...o.train };
  const { usable, skipped } = usableAttacks(ds);
  if (usable.length < MIN_ATTACKS_TO_TRAIN) {
    throw new Error(`Tag at least ${MIN_ATTACKS_TO_TRAIN} attacks with pose before training (you have ${usable.length}).`);
  }
  const K = 3;
  const y = usable.map((u) => u.label);
  const groups = usable.map((u) => u.attack.matchId);
  const counts = classCounts(y, K);
  const prior = counts.map((c) => (c + 1) / (y.length + K));
  const majority = Math.max(...counts) / y.length;
  const progress = o.onProgress ?? (() => {});

  const results: HorizonResult[] = [];
  let scoreProbs: number[][] = [];
  let grouped = false;
  const allH = Array.from(new Set([...horizons, scoreH])).sort((a, b) => a - b);
  allH.forEach((h, hi) => {
    progress(`Reading ${h} ms before contact`, hi / (allH.length + 2));
    const x = usable.map((u) => summaryFeatures(u.seq, h));
    const oof = outOfFold(x, y, groups, K, opts);
    grouped = oof.grouped;
    if (h === scoreH) scoreProbs = oof.probs;
    if (!horizons.includes(h)) return;
    const acc = accuracy(oof.probs, y);
    const correct = Math.round(acc * y.length);
    const cal = calibration(oof.probs, y);
    results.push({
      horizonMs: h,
      n: y.length,
      accuracy: acc,
      accuracyCI: wilson(correct, y.length),
      majorityAccuracy: majority,
      chance: 1 / K,
      pValue: binomialPValue(correct, y.length, majority),
      ceModel: crossEntropyBits(oof.probs, y),
      cePrior: crossEntropyBits(y.map(() => prior), y),
      infoGain: infoGain(oof.probs, y, prior),
      ece: cal.ece,
      reliability: cal.bins,
      confusion: confusion(oof.probs, y, K),
    });
  });

  const significant = results.filter((r) => r.pValue < 0.05).map((r) => r.horizonMs);
  const readTimeMs = significant.length ? Math.max(...significant) : null;

  // Per-hitter scores at the score horizon.
  progress('Scoring hitters', allH.length / (allH.length + 2));
  const rng = new Rng(99);
  const byHitter = new Map<string, number[]>();
  usable.forEach((u, i) => {
    if (!byHitter.has(u.attack.hitterId)) byHitter.set(u.attack.hitterId, []);
    byHitter.get(u.attack.hitterId)!.push(i);
  });
  const hitters: HitterScore[] = [];
  for (const p of ds.players) {
    const idx = byHitter.get(p.id) ?? [];
    if (!idx.length) continue;
    const ps = idx.map((i) => scoreProbs[i]);
    const ys = idx.map((i) => y[i]);
    const enough = idx.length >= MIN_ATTACKS_FOR_SCORE;
    const realCounts: Record<Direction, number> = { left: 0, straight: 0, right: 0 };
    idx.forEach((i) => realCounts[usable[i].attack.direction]++);
    hitters.push({
      player: p,
      n: idx.length,
      score: enough ? infoGain(ps, ys, prior) * 100 : null,
      scoreCI: enough ? bootstrapScore(ps, ys, prior, rng) : null,
      accuracy: accuracy(ps, ys),
      counts: realCounts,
      cues: hitterCues(idx.map((i) => usable[i]), scoreH),
      lowData: idx.length < 20,
    });
  }
  hitters.sort((a, b) => (b.score ?? -1) - (a.score ?? -1));

  const reads: AttackRead[] = usable.map((u, i) => {
    const pr = realProbs(scoreProbs[i], u.hand);
    const predicted = DIRECTIONS[argmax(pr)];
    return { attackId: u.attack.id, probs: pr, predicted, correct: predicted === u.attack.direction };
  });

  // World model: forecast 0.5 s from 200 ms before contact, scored on held-out matches.
  progress('Testing the world model', (allH.length + 1) / (allH.length + 2));
  const world = evaluateWorld(usable);
  const landingMae = evaluateLanding(usable, scoreH);

  return {
    createdAt: Date.now(),
    kind: opts.kind,
    nAttacks: usable.length,
    nSkipped: skipped.length,
    nMatches: new Set(groups).size,
    groupedByMatch: grouped,
    scoreHorizonMs: scoreH,
    classCounts: counts,
    horizons: results,
    readTimeMs,
    hitters,
    reads,
    world,
    landingMae,
    synthetic: ds.matches.length > 0 && ds.matches.every((m) => m.synthetic),
  };
}

function splitByMatch(usable: Usable[]): [number[], number[]] {
  const matches = Array.from(new Set(usable.map((u) => u.attack.matchId)));
  if (matches.length >= 2) {
    const test = new Set(matches.slice(-Math.max(1, Math.floor(matches.length / 4))));
    const tr: number[] = [];
    const te: number[] = [];
    usable.forEach((u, i) => (test.has(u.attack.matchId) ? te : tr).push(i));
    return [tr, te];
  }
  const cut = Math.floor(usable.length * 0.75);
  return [usable.slice(0, cut).map((_, i) => i), usable.slice(cut).map((_, i) => i + cut)];
}

export function evaluateWorld(usable: Usable[], horizonMs = 200): WorldEval | null {
  const from = cutoffIndex(horizonMs);
  if (usable.length < 10 || from + FORECAST_STEPS >= usable[0].seq.length) return null;
  const [tr, te] = splitByMatch(usable);
  if (!te.length) return null;
  const dyn = fitDynamics(tr.map((i) => usable[i].seq));
  let m = 0;
  let cv = 0;
  let st = 0;
  for (const i of te) {
    const seq = usable[i].seq;
    const truth = seq.slice(from + 1, from + 1 + FORECAST_STEPS).map(stateOf);
    m += poseError(forecastPose(dyn, seq, from), truth);
    cv += poseError(constantVelocityForecast(seq, from), truth);
    st += poseError(staticForecast(seq, from), truth);
  }
  return { horizonMs, model: m / te.length, constantVelocity: cv / te.length, still: st / te.length, n: te.length };
}

function landingTarget(u: Usable): number[] | null {
  const l = u.attack.landing;
  if (!l) return null;
  const t = TAKEOFF[u.attack.position];
  const dx = l.x - t.x;
  return [u.hand === 'L' ? -dx : dx, l.y];
}

export function evaluateLanding(usable: Usable[], horizonMs: number): { model: number; baseline: number } | null {
  const withL = usable.filter((u) => u.attack.landing);
  if (withL.length < 20) return null;
  const [tr, te] = splitByMatch(withL);
  if (!te.length || tr.length < 10) return null;
  const x = withL.map((u) => summaryFeatures(u.seq, horizonMs));
  const t = withL.map((u) => landingTarget(u) as number[]);
  const lm = fitLanding(tr.map((i) => x[i]), tr.map((i) => t[i]));
  const mean = [0, 1].map((k) => tr.reduce((s, i) => s + t[i][k], 0) / tr.length);
  let em = 0;
  let eb = 0;
  for (const i of te) {
    const p = predictLanding(lm, x[i]);
    em += Math.hypot(p[0] - t[i][0], p[1] - t[i][1]);
    eb += Math.hypot(mean[0] - t[i][0], mean[1] - t[i][1]);
  }
  return { model: em / te.length, baseline: eb / te.length };
}

/** Models trained on everything, for live reads in the trainer and on new attacks. */
export interface LiveModels {
  horizonMs: number;
  classifier: FittedModel;
  dynamics: PoseDynamics;
  landing: LandingModel | null;
}

export function fitLive(ds: Dataset, horizonMs = DEFAULT_SCORE_HORIZON, train: Partial<TrainOptions> = {}): LiveModels | null {
  const { usable } = usableAttacks(ds);
  if (usable.length < MIN_ATTACKS_TO_TRAIN) return null;
  const x = usable.map((u) => summaryFeatures(u.seq, horizonMs));
  const y = usable.map((u) => u.label);
  const classifier = fitCalibrated(x, y, 3, { ...DEFAULT_TRAIN, ...train });
  const dynamics = fitDynamics(usable.map((u) => u.seq));
  const withL = usable.filter((u) => u.attack.landing);
  const landing = withL.length >= 20
    ? fitLanding(withL.map((u) => summaryFeatures(u.seq, horizonMs)), withL.map((u) => landingTarget(u) as number[]))
    : null;
  return { horizonMs, classifier, dynamics, landing };
}

export interface LiveRead {
  probs: number[]; // real frame
  landing: { x: number; y: number; sx: number; sy: number } | null;
  ghost: number[][]; // forecast states in the hitter's real frame
}

export function readAttack(m: LiveModels, attack: Attack, hand: Hand, horizonMs = m.horizonMs): LiveRead | null {
  if (!attack.frames) return null;
  return readFrames(m, attack.frames, attack.position, hand, horizonMs);
}

/** Read a 46-frame window (contact at CONTACT_INDEX) for a hitter, as of `horizonMs` before contact. */
export function readFrames(
  m: LiveModels,
  frames: (PoseFrame | null)[],
  position: AttackPosition,
  hand: Hand,
  horizonMs = m.horizonMs,
): LiveRead | null {
  const seq = sequenceFeatures(frames, hand);
  if (!seq) return null;
  const x = summaryFeatures(seq, horizonMs);
  const probs = realProbs(m.classifier.predict(x), hand);
  let landing: LiveRead['landing'] = null;
  if (m.landing) {
    const [dx, depth] = predictLanding(m.landing, x);
    const t = TAKEOFF[position];
    landing = {
      x: Math.max(0, Math.min(9, t.x + (hand === 'L' ? -dx : dx))),
      y: Math.max(0.3, Math.min(9, depth)),
      sx: m.landing.spread[0],
      sy: m.landing.spread[1],
    };
  }
  const from = cutoffIndex(horizonMs);
  let ghost = forecastPose(m.dynamics, seq, from);
  if (hand === 'L') ghost = ghost.map(mirrorState);
  return { probs, landing, ghost };
}

/** Undo the left-hand mirror on a pose state (joints in torso units + hip image xy). */
export function mirrorState(s: number[]): number[] {
  const out = s.slice();
  // FEATURE_JOINTS order: nose, lS, rS, lE, rE, lW, rW, lH, rH, lK, rK, lA, rA
  const swap = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11];
  for (let j = 0; j < 13; j++) {
    const src = swap[j];
    out[j * 3] = -s[src * 3];
    out[j * 3 + 1] = s[src * 3 + 1];
    out[j * 3 + 2] = s[src * 3 + 2];
  }
  out[N_JOINT_FEATS] = 1 - s[N_JOINT_FEATS];
  return out;
}
