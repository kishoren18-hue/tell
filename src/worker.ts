/// <reference lib="webworker" />
// Training runs off the main thread so the app stays responsive on a phone.
import { analyze, type AnalyzeOptions } from './lib/ml/tell';
import type { Dataset } from './lib/types';

export type WorkerIn = { type: 'analyze'; dataset: Dataset; options: Omit<AnalyzeOptions, 'onProgress'> };
export type WorkerOut =
  | { type: 'progress'; message: string; frac: number }
  | { type: 'done'; report: ReturnType<typeof analyze> }
  | { type: 'error'; message: string };

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  if (msg.type !== 'analyze') return;
  try {
    const report = analyze(msg.dataset, {
      ...msg.options,
      onProgress: (message, frac) => (self as unknown as Worker).postMessage({ type: 'progress', message, frac } satisfies WorkerOut),
    });
    (self as unknown as Worker).postMessage({ type: 'done', report } satisfies WorkerOut);
  } catch (err) {
    (self as unknown as Worker).postMessage({ type: 'error', message: (err as Error).message } satisfies WorkerOut);
  }
};
