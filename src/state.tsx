import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import type { Report } from './lib/ml/tell';
import { loadDataset, loadReport, saveDataset, saveReport } from './lib/store';
import type { Dataset } from './lib/types';
import { emptyDataset } from './lib/types';

interface Ctx {
  ready: boolean;
  dataset: Dataset;
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

export function DataProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [dataset, setDataset] = useState<Dataset>(emptyDataset());
  const [report, setReportState] = useState<Report | null>(null);
  const [reportPrint, setReportPrint] = useState<string>('');
  const latest = useRef(dataset);

  useEffect(() => {
    void (async () => {
      const [d, r] = await Promise.all([loadDataset(), loadReport()]);
      latest.current = d;
      setDataset(d);
      setReportState(r);
      setReportPrint(r ? localStorageGet('tell:report-print') ?? '' : '');
      setReady(true);
    })();
  }, []);

  const replace = useCallback((d: Dataset) => {
    latest.current = d;
    setDataset(d);
    void saveDataset(d);
  }, []);

  const update = useCallback((fn: (d: Dataset) => Dataset) => {
    const next = fn(latest.current);
    latest.current = next;
    setDataset(next);
    void saveDataset(next);
  }, []);

  const setReport = useCallback((r: Report | null) => {
    setReportState(r);
    const fp = r ? fingerprint(latest.current) : '';
    setReportPrint(fp);
    localStorageSet('tell:report-print', fp);
    void saveReport(r);
  }, []);

  const stale = !!report && reportPrint !== fingerprint(dataset);

  return <C.Provider value={{ ready, dataset, report, update, replace, setReport, stale }}>{children}</C.Provider>;
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
