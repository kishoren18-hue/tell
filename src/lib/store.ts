// Everything is stored on the device (IndexedDB). Videos are never uploaded
// or even stored: Tell keeps only the pose skeletons and your tags, which are
// small and don't show anyone's face.

import { get, set, del } from 'idb-keyval';
import type { Report } from './ml/tell';
import type { Dataset } from './types';
import { emptyDataset } from './types';

const KEY_DATASET = 'tell:dataset:v1';
const KEY_REPORT = 'tell:report:v1';
const KEY_PRACTICE = 'tell:practice:v1';

export async function loadDataset(): Promise<Dataset> {
  return ((await get(KEY_DATASET)) as Dataset | undefined) ?? emptyDataset();
}
export async function saveDataset(ds: Dataset): Promise<void> {
  await set(KEY_DATASET, ds);
}
export async function loadReport(): Promise<Report | null> {
  return ((await get(KEY_REPORT)) as Report | undefined) ?? null;
}
export async function saveReport(r: Report | null): Promise<void> {
  if (r) await set(KEY_REPORT, r);
  else await del(KEY_REPORT);
}

export interface PracticeStats {
  attempts: number;
  correct: number;
  modelCorrect: number;
  modelAttempts?: number;
  bestStreak: number;
  horizonMs: number;
}
export const EMPTY_PRACTICE: PracticeStats = { attempts: 0, correct: 0, modelCorrect: 0, modelAttempts: 0, bestStreak: 0, horizonMs: 0 };
export async function loadPractice(): Promise<PracticeStats> {
  return ((await get(KEY_PRACTICE)) as PracticeStats | undefined) ?? EMPTY_PRACTICE;
}
export async function savePractice(p: PracticeStats): Promise<void> {
  await set(KEY_PRACTICE, p);
}

/** Merge another export into this dataset (for combining phones or teammates). Same ids are kept once. */
export function mergeDatasets(a: Dataset, b: Dataset): Dataset {
  const byId = <T extends { id: string }>(xs: T[], ys: T[]) => {
    const m = new Map(xs.map((x) => [x.id, x]));
    for (const y of ys) if (!m.has(y.id)) m.set(y.id, y);
    return [...m.values()];
  };
  return {
    ...a,
    players: byId(a.players, b.players),
    matches: byId(a.matches, b.matches),
    attacks: byId(a.attacks, b.attacks),
  };
}

export function validateDataset(x: unknown): Dataset {
  const d = x as Dataset;
  if (!d || d.format !== 'tell-dataset' || !Array.isArray(d.attacks) || !Array.isArray(d.players)) {
    throw new Error('That file is not a Tell dataset export.');
  }
  if (d.version !== 1) throw new Error(`This file is dataset version ${d.version}; this app reads version 1.`);
  return d;
}
