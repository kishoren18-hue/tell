import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { setMany } from 'idb-keyval';
import type { Report } from './lib/ml/tell';
import { datasetKey, loadDataset, loadPrivateDataset, loadReport, saveDataset, saveReport } from './lib/store';
import type { Dataset } from './lib/types';
import { emptyDataset } from './lib/types';

interface Ctx {
  workspace: string;
  individual: boolean;
  error: string | null;
  applySync: (before: Dataset, after: Dataset, key: string, state: unknown) => Promise<void>;
  ready: boolean;
  dataset: Dataset;
  getDataset: () => Dataset;
  report: Report | null;
  update: (fn: (d: Dataset) => Dataset) => void;
  replace: (d: Dataset) => void;
  setReport: (r: Report | null) => void;
  /** Report no longer matches the data (attacks were added or changed). */
  stale: boolean;
}

const C = createContext<Ctx | null>(null);

function fingerprint(d: Dataset) {
  return `${d.attacks.length}:${d.attacks.reduce((s, a) => s + (a.frames ? 1 : 0) + a.direction.length, 0)}:${d.players.map((p) => p.hand).join('')}`;
}

export function DataProvider({ workspace, children }: { workspace: string; children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const printKey = workspace === 'legacy' ? 'tell:report-print' : `tell:report-print:${workspace}`;
  const [dataset, setDataset] = useState<Dataset>(emptyDataset());
  const [report, setReportState] = useState<Report | null>(null);
  const [reportPrint, setReportPrint] = useState<string>('');
  const latest = useRef(dataset);

  useEffect(() => {
    let active = true;
    const load = workspace.startsWith('personal:') ? loadPrivateDataset(workspace) : loadDataset(workspace);
    void Promise.all([load, loadReport(workspace)]).then(([d, r]) => {
      if (!active) return;
      latest.current = d;
      setDataset(d);
      setReportState(r);
      setReportPrint(r ? localStorageGet(printKey) ?? '' : '');
      setReady(true);
    }).catch(() => {
      if (active) setError('Could not open local storage. Reload Tell to try again.');
    });
    return () => { active = false; };
  }, [workspace, printKey]);

  const getDataset = useCallback(() => latest.current, []);

  const applySync = useCallback(async (before: Dataset, after: Dataset, key: string, state: unknown) => {
    const current = latest.current;
    // Keep edits made while the network request was running.
    const merge = <T extends { id: string }>(initial: T[], remote: T[], now: T[]) => {
      const initialById = new Map(initial.map((item) => [item.id, item]));
      const currentById = new Map(now.map((item) => [item.id, item]));
      const result = new Map(remote.map((item) => [item.id, item]));
      for (const id of initialById.keys()) if (!currentById.has(id)) result.delete(id);
      for (const item of now) if (initialById.get(item.id) !== item) result.set(item.id, item);
      return [...result.values()];
    };
    const next = before === after ? current : {
      ...current,
      players: merge(before.players, after.players, current.players),
      matches: merge(before.matches, after.matches, current.matches),
      attacks: merge(before.attacks, after.attacks, current.attacks),
    };
    latest.current = next;
    if (next !== current) setDataset(next);
    await setMany([[datasetKey(workspace), next], [key, state]]);
  }, [workspace]);

  const replace = useCallback((d: Dataset) => {
    latest.current = d;
    setDataset(d);
    void saveDataset(workspace, d);
  }, [workspace]);

  const update = useCallback((fn: (d: Dataset) => Dataset) => {
    const next = fn(latest.current);
    latest.current = next;
    setDataset(next);
    void saveDataset(workspace, next);
  }, [workspace]);

  const setReport = useCallback((r: Report | null) => {
    setReportState(r);
    const fp = r ? fingerprint(latest.current) : '';
    setReportPrint(fp);
    localStorageSet(printKey, fp);
    void saveReport(workspace, r);
  }, [workspace, printKey]);

  const stale = !!report && reportPrint !== fingerprint(dataset);

  return <C.Provider value={{ workspace, individual: workspace === 'guest' || workspace.startsWith('personal:'), error, applySync, ready, dataset, getDataset, report, update, replace, setReport, stale }}>{children}</C.Provider>;
}

function localStorageGet(k: string): string | null {
  try {
    return localStorage.getItem(k);
  } catch {
    return null;
  }
}
function localStorageSet(k: string, v: string) {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* storage unavailable */
  }
}

export function useData(): Ctx {
  const c = useContext(C);
  if (!c) throw new Error('useData outside DataProvider');
  return c;
}
