// Writes the synthetic demo dataset (and the Python parity fixture) to disk.
//   npm run export-demo -- ml/data/demo.json        full demo dataset (300 attacks)
//   npm run export-demo -- --fixture                ml/tests/fixtures/parity.json
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { sequenceFeatures, summaryFeatures } from '../src/lib/features';
import { makeDemoDataset } from '../src/lib/synth';

const arg = process.argv[2] ?? 'ml/data/demo.json';
const round = (x: unknown): unknown =>
  typeof x === 'number' ? Math.round(x * 1e6) / 1e6 : Array.isArray(x) ? x.map(round) : x && typeof x === 'object'
    ? Object.fromEntries(Object.entries(x).map(([k, v]) => [k, round(v)])) : x;

if (arg === '--fixture') {
  // Rounded first, so TypeScript and Python compute features from identical inputs.
  const ds = round(makeDemoDataset({ seed: 77, matches: 3, attacksPerHitterPerMatch: 2 })) as ReturnType<typeof makeDemoDataset>;
  const hands = new Map(ds.players.map((p) => [p.id, p.hand]));
  const expected = ds.attacks.slice(0, 12).map((a) => {
    const seq = sequenceFeatures(a.frames!, hands.get(a.hitterId)!)!;
    return { id: a.id, seq, summary100: summaryFeatures(seq, 100), summary300: summaryFeatures(seq, 300) };
  });
  const out = 'ml/tests/fixtures/parity.json';
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify({ dataset: ds, expected }));
  console.log(`wrote ${out}`);
} else {
  const ds = round(makeDemoDataset());
  mkdirSync(dirname(arg), { recursive: true });
  writeFileSync(arg, JSON.stringify(ds));
  console.log(`wrote ${arg}`);
}
