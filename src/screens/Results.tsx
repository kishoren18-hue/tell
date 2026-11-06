import { useRef, useState } from 'react';
import type { Tab } from '../App';
import { Section, Progress, pct } from '../components/ui';
import { MIN_ATTACKS_TO_TRAIN, usableAttacks, type HorizonResult, type Report } from '../lib/ml/tell';
import type { ModelKind } from '../lib/ml/models';
import { useData } from '../state';
import type { WorkerIn, WorkerOut } from '../worker';

export function Results({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, report, setReport, stale } = useData();
  const [busy, setBusy] = useState<{ msg: string; frac: number } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [kind, setKind] = useState<ModelKind>(report?.kind ?? 'logistic');
  const [scoreH, setScoreH] = useState(report?.scoreHorizonMs ?? 100);
  const workerRef = useRef<Worker | null>(null);
  const { usable } = usableAttacks(dataset);

  const run = () => {
    setErr(null);
    setBusy({ msg: 'Starting', frac: 0 });
    workerRef.current?.terminate();
    const w = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.onmessage = (e: MessageEvent<WorkerOut>) => {
      const m = e.data;
      if (m.type === 'progress') setBusy({ msg: m.message, frac: m.frac });
      else if (m.type === 'done') {
        setReport(m.report);
        setBusy(null);
        w.terminate();
      } else {
        setErr(m.message);
        setBusy(null);
        w.terminate();
      }
    };
    w.postMessage({ type: 'analyze', dataset, options: { scoreHorizonMs: scoreH, train: { kind } } } satisfies WorkerIn);
  };

  const enough = usable.length >= MIN_ATTACKS_TO_TRAIN;

  return (
    <div className="screen">
      <div className="intro">
        <h1>Can a model read your hitters before contact?</h1>
        <p>
          Tell trains on every tagged attack and scores itself only on matches it hasn't seen. Every number below is
          compared with the dumb strategy of always guessing the most common direction.
        </p>
      </div>

      <Section title="Train">
        <div className="train-grid">
          <fieldset className="chips">
            <legend>Model</legend>
            <div className="seg">
              {(['logistic', 'mlp'] as const).map((k) => (
                <button key={k} className={`seg__btn ${kind === k ? 'seg__btn--on' : ''}`} aria-pressed={kind === k} onClick={() => setKind(k)}>
                  {k === 'logistic' ? 'Standard' : 'Deeper (slower)'}
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset className="chips">
            <legend>Score hitters at</legend>
            <div className="seg">
              {[0, 100, 200, 300].map((h) => (
                <button key={h} className={`seg__btn ${scoreH === h ? 'seg__btn--on' : ''}`} aria-pressed={scoreH === h} onClick={() => setScoreH(h)}>
                  {h === 0 ? 'Contact' : `${h} ms`}
                </button>
              ))}
            </div>
          </fieldset>
        </div>
        {busy ? (
          <div className="busy">
            <Progress value={busy.frac} label="Training" />
            <span className="muted small">{busy.msg}…</span>
          </div>
        ) : (
          <button className="btn btn--primary btn--block btn--big" onClick={run} disabled={!enough}>
            {report ? 'Train again' : 'Train'} on {usable.length} attacks
          </button>
        )}
        {!enough && (
          <p className="muted small">
            You need {MIN_ATTACKS_TO_TRAIN} attacks with pose to train ({usable.length} so far). Tag more in{' '}
            <button className="link" onClick={() => onGo('film')}>Film</button>, or load the demo data there.
          </p>
        )}
        {stale && !busy && <p className="notice">You've tagged or changed attacks since the last training. Train again to include them.</p>}
        {err && <p className="notice notice--bad">{err}</p>}
      </Section>

      {report && <ReportView report={report} onGo={onGo} />}
    </div>
  );
}

function ReportView({ report, onGo }: { report: Report; onGo: (t: Tab) => void }) {
  const at = report.horizons.find((h) => h.horizonMs === report.scoreHorizonMs) ?? report.horizons[0];
  const beats = at.pValue < 0.05;
  return (
    <>
      {report.synthetic && (
        <p className="notice">These results are from synthetic demo data with tells planted on purpose. They show what the app does, not how readable real players are.</p>
      )}
      <section className="verdict">
        <p className="verdict__big">
          {report.readTimeMs !== null ? (
            <>Readable {report.readTimeMs === 0 ? 'at contact' : `${report.readTimeMs} ms before contact`}</>
          ) : (
            <>No reliable read yet</>
          )}
        </p>
        <p className="verdict__sub">
          {report.scoreHorizonMs === 0 ? 'At contact' : `${report.scoreHorizonMs} ms before contact`} the model calls{' '}
          <b>{pct(at.accuracy)}</b> of {at.n} attacks. Always guessing the most common direction gets {pct(at.majorityAccuracy)}.{' '}
          {beats ? `That gap is real (${fmtP(at.pValue)}).` : `That gap could be luck (${fmtP(at.pValue)}). Tag more attacks.`}
        </p>
      </section>

      <Section title="Read curve">
        <ReadCurve rows={report.horizons} />
        <p className="muted small">
          Each point is a model that only sees the body up to that moment. Shaded band: 95% range. Dashed line: always
          guessing the most common direction.
        </p>
      </Section>

      <Section title="Can you trust its confidence?">
        <Reliability row={at} />
        <p className="muted small">Bars: how often the model was right when it said it was 30%, 40%… sure. Dashed line: perfectly honest confidence.</p>
        <p className="small">
          Average gap between how sure the model says it is and how often it's right: <b>{pct(at.ece)}</b>.{' '}
          {at.ece < 0.08 ? 'Its confidence means what it says.' : 'Take its percentages loosely until you have more data.'}
        </p>
      </Section>

      <Section title="World model">
        {report.world ? (
          <div className="stats">
            <Stat value={report.world.model.toFixed(2)} label="Forecast error, Tell" strong />
            <Stat value={report.world.still.toFixed(2)} label="If the hitter froze" />
            <Stat value={report.world.constantVelocity.toFixed(2)} label="If they kept moving the same way" />
          </div>
        ) : (
          <p className="muted small">Needs more attacks.</p>
        )}
        <p className="muted small">
          Starting {report.world?.horizonMs ?? 200} ms before contact, Tell predicts the hitter's next 0.5 s of movement.
          Error is the average joint miss in torso lengths, on {report.world?.n ?? 0} attacks from held-out matches. Lower is better.
        </p>
        {report.landingMae && (
          <p className="small">
            Landing guess: off by <b>{report.landingMae.model.toFixed(1)} m</b> on average, against{' '}
            {report.landingMae.baseline.toFixed(1)} m for always guessing the average landing spot.
          </p>
        )}
      </Section>

      <Section title="How this was tested">
        <ul className="tips">
          <li>
            {report.nAttacks} attacks from {report.nMatches} {report.nMatches === 1 ? 'match' : 'matches'}
            {report.nSkipped ? `, ${report.nSkipped} skipped because the hitter wasn't tracked well enough` : ''}.
          </li>
          <li>
            {report.groupedByMatch
              ? 'Each match was held out in turn: the model never saw the match it was scored on.'
              : 'Fewer than 3 matches, so attacks were split into 5 random folds. Same-match attacks can look alike, so expect real-world accuracy to be a bit lower.'}
          </li>
          <li>Confidence was tuned on a slice of the training data only, never on the attacks being scored.</li>
          <li>Lefties are mirrored so one model reads both hands.</li>
        </ul>
        <div className="row">
          <button className="btn" onClick={() => onGo('hitters')}>See hitter scores</button>
          <button className="btn" onClick={() => onGo('practice')}>Practice reading</button>
        </div>
      </Section>
    </>
  );
}

function fmtP(p: number) {
  if (p < 0.001) return 'p < 0.001';
  return `p = ${p.toFixed(3)}`;
}

function Stat({ value, label, strong }: { value: string; label: string; strong?: boolean }) {
  return (
    <div className={`stat ${strong ? 'stat--strong' : ''}`}>
      <span className="stat__v">{value}</span>
      <span className="stat__l">{label}</span>
    </div>
  );
}

function ReadCurve({ rows }: { rows: HorizonResult[] }) {
  const W = 320;
  const H = 180;
  const pad = { l: 34, r: 26, t: 12, b: 28 };
  const hs = rows.map((r) => r.horizonMs);
  const maxH = Math.max(...hs, 100);
  // Time runs left to right towards contact.
  const x = (h: number) => pad.l + ((maxH - h) / maxH) * (W - pad.l - pad.r);
  const y = (a: number) => pad.t + (1 - a) * (H - pad.t - pad.b);
  const sorted = [...rows].sort((a, b) => b.horizonMs - a.horizonMs);
  const band =
    sorted.map((r) => `${x(r.horizonMs)},${y(r.accuracyCI[1])}`).join(' ') +
    ' ' +
    [...sorted].reverse().map((r) => `${x(r.horizonMs)},${y(r.accuracyCI[0])}`).join(' ');
  const line = sorted.map((r) => `${x(r.horizonMs)},${y(r.accuracy)}`).join(' ');
  const base = rows[0]?.majorityAccuracy ?? 0.33;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Accuracy by time before contact">
      {[0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={pad.l} x2={W - pad.r} y1={y(g)} y2={y(g)} className="chart__grid" />
          <text x={pad.l - 6} y={y(g) + 3} className="chart__tick" textAnchor="end">{Math.round(g * 100)}%</text>
        </g>
      ))}
      {sorted.map((r) => (
        <text key={r.horizonMs} x={x(r.horizonMs)} y={H - 10} className="chart__tick" textAnchor="middle">
          {r.horizonMs === 0 ? 'contact' : `−${r.horizonMs}`}
        </text>
      ))}
      <polygon points={band} className="chart__band" />
      <line x1={pad.l} x2={W - pad.r} y1={y(base)} y2={y(base)} className="chart__base" />
      <polyline points={line} className="chart__line" />
      {sorted.map((r) => (
        <circle key={r.horizonMs} cx={x(r.horizonMs)} cy={y(r.accuracy)} r={3.5} className={r.pValue < 0.05 ? 'chart__pt' : 'chart__pt chart__pt--weak'} />
      ))}
    </svg>
  );
}

function Reliability({ row }: { row: HorizonResult }) {
  const bins = row.reliability.filter((b) => b.count > 0);
  const W = 320;
  const H = 120;
  const bw = (W - 20) / 10;
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} className="chart" role="img" aria-label="Stated confidence against actual accuracy">
      <line x1={10} y1={H} x2={W - 10} y2={10} className="chart__base" />
      {bins.map((b) => {
        const i = Math.round(b.lo * 10);
        const h = b.accuracy * (H - 10);
        return (
          <g key={i}>
            <rect x={10 + i * bw + 2} y={H - h} width={bw - 4} height={h} className="chart__bar" />
            <text x={10 + i * bw + bw / 2} y={H + 13} className="chart__tick" textAnchor="middle">{Math.round(b.lo * 100)}</text>
          </g>
        );
      })}
    </svg>
  );
}
