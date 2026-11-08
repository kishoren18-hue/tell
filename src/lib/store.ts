// Everything is stored on the device (IndexedDB). Videos are never uploaded
// or even stored: Tell keeps only the pose skeletons and your tags, which are
// small and don't show anyone's face.

import { get, set, del, getMany, setMany } from 'idb-keyval';
import type { Report } from './ml/tell';
import type { Dataset } from './types';
import { emptyDataset } from './types';

const KEY_DATASET = 'tell:dataset:v1';
const KEY_REPORT = 'tell:report:v1';
const KEY_PRACTICE = 'tell:practice:v1';

const scoped = (key: string, workspace: string) => workspace === 'legacy' ? key : `${key}:${workspace}`;
export const datasetKey = (workspace: string) => scoped(KEY_DATASET, workspace);

export async function loadDataset(workspace: string): Promise<Dataset> {
  return (await get<Dataset>(datasetKey(workspace))) ?? emptyDataset();
}
export async function saveDataset(workspace: string, ds: Dataset): Promise<void> {
  await set(datasetKey(workspace), ds);
}
export async function loadReport(workspace: string): Promise<Report | null> {
  return (await get<Report>(scoped(KEY_REPORT, workspace))) ?? null;
}
export async function saveReport(workspace: string, r: Report | null): Promise<void> {
  const key = scoped(KEY_REPORT, workspace);
  if (r) await set(key, r);
  else await del(key);
}

/** Move guest data only to the first private workspace opened after signing in. */
export async function loadPrivateDataset(workspace: string): Promise<Dataset> {
  const migrate = async () => {
    const [personal, guest] = await getMany<Dataset | undefined>([datasetKey(workspace), datasetKey('guest')]);
    if (!guest || !(guest.players.length || guest.matches.length || guest.attacks.length)) return personal ?? emptyDataset();
    const merged = mergeDatasets(personal ?? emptyDataset(), guest);
    await setMany([[datasetKey(workspace), merged], [datasetKey('guest'), emptyDataset()]]);
    return merged;
  };
  return navigator.locks ? navigator.locks.request('tell:guest-migration', migrate) : migrate();
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
export async function loadPractice(workspace: string): Promise<PracticeStats> {
  return ((await get(scoped(KEY_PRACTICE, workspace))) as PracticeStats | undefined) ?? EMPTY_PRACTICE;
}
export async function savePractice(workspace: string, p: PracticeStats): Promise<void> {
  await set(scoped(KEY_PRACTICE, workspace), p);
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
