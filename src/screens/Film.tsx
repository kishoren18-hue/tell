import { useRef, useState } from 'react';
import type { Tab } from '../App';
import { Section, download } from '../components/ui';
import { mergeDatasets, validateDataset } from '../lib/store';
import { makeDemoDataset } from '../lib/synth';
import type { Hand, Player } from '../lib/types';
import { emptyDataset, INDIVIDUAL_HITTER, uid } from '../lib/types';
import { useData } from '../state';

export function Film({ onOpen, onGo }: { onOpen: (id: string) => void; onGo: (t: Tab) => void }) {
  const { dataset, individual, update, replace, setReport } = useData();
  const [name, setName] = useState('');
  const [confirmClear, setConfirmClear] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const realMatches = dataset.matches.filter((m) => !m.synthetic);
  const hasDemo = dataset.matches.some((m) => m.synthetic);

  const addMatch = () => {
    const id = uid('m');
    const label = name.trim() || `Match ${dataset.matches.length + 1}`;
    update((d) => ({ ...d, matches: [{ id, name: label, date: new Date().toISOString().slice(0, 10) }, ...d.matches] }));
    setName('');
    onOpen(id);
  };

  const loadDemo = () => {
    const demo = makeDemoDataset();
    if (individual) {
      demo.players = demo.players.slice(0, 1);
      demo.attacks = demo.attacks.filter((attack) => attack.hitterId === demo.players[0]?.id);
    }
    replace(mergeDatasets(dataset, demo));
    setReport(null);
    setMsg(`Demo data loaded: ${demo.attacks.length} synthetic attacks. Open Model to train on it.`);
  };

  const removeDemo = () => {
    const demoMatchIds = new Set(dataset.matches.filter((m) => m.synthetic).map((m) => m.id));
    update((d) => ({
      ...d,
      matches: d.matches.filter((m) => !m.synthetic),
      attacks: d.attacks.filter((a) => !demoMatchIds.has(a.matchId)),
      players: d.players.filter((p) => !p.id.startsWith('demo-')),
    }));
    setReport(null);
  };

  const exportData = () => {
    const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
    download(`tell-dataset-${new Date().toISOString().slice(0, 10)}.json`, blob);
  };

  const importData = async (f: File) => {
    try {
      const incoming = validateDataset(JSON.parse(await f.text()));
      if (individual) {
        const hitters = incoming.players.filter((p) => !p.id.startsWith('demo-'));
        const existing = dataset.players.find((p) => p.id === INDIVIDUAL_HITTER);
        const hitter = hitters[0];
        if (hitters.length > 1 || (existing && hitter && hitter.id !== existing.id)) {
          throw new Error('This export has a different hitter or a roster. Open a Team workspace to import it.');
        }
        if (hitter) {
          incoming.players = incoming.players.map((p) => p.id === hitter.id ? { ...p, id: INDIVIDUAL_HITTER } : p);
          incoming.attacks = incoming.attacks.map((a) => a.hitterId === hitter.id ? { ...a, hitterId: INDIVIDUAL_HITTER } : a);
        }
      }
      const before = dataset.attacks.length;
      const merged = mergeDatasets(dataset, incoming);
      replace(merged);
      setMsg(`Imported ${merged.attacks.length - before} new attacks.`);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  const counts = (matchId: string) => {
    const a = dataset.attacks.filter((x) => x.matchId === matchId);
    return { total: a.length, pose: a.filter((x) => x.frames && x.frames.filter(Boolean).length >= 23).length };
  };

  return (
    <div className="screen">
      <div className="intro">
        <h1>{individual ? 'Film a match. Tag your spikes.' : 'Film a match. Tag the spikes. See who gives their shot away.'}</h1>
        <p>
          Tell watches each hitter's approach and arm swing and guesses where the ball is going before contact. The
          better it guesses, the more readable you are.
        </p>
      </div>

      <RealSwings onGo={onGo} />

      {msg && (
        <p className="notice" role="status">
          {msg}
        </p>
      )}

      <Section title="Matches">
        <form
          className="row"
          onSubmit={(e) => {
            e.preventDefault();
            addMatch();
          }}
        >
          <input
            className="input"
            placeholder="Opponent or tournament"
            value={name}
            onChange={(e) => setName(e.target.value)}
            aria-label="Match name"
          />
          <button className="btn btn--primary" type="submit">
            Start match
          </button>
        </form>

        {dataset.matches.length === 0 ? (
          <div className="empty">
            <p>No matches yet. Start one above, or try Tell with demo data first.</p>
            <button className="btn" onClick={loadDemo}>
              Load demo data
            </button>
          </div>
        ) : (
          <ul className="list">
            {dataset.matches.map((m) => {
              const c = counts(m.id);
              return (
                <li key={m.id}>
                  <button className="list__item" onClick={() => onOpen(m.id)}>
                    <span className="list__main">
                      <strong>{m.name}</strong>
                      <span className="muted">
                        {m.synthetic ? 'Synthetic demo' : m.live ? `Live camera, ${m.date}` : m.date}
                        {!!m.defenseReviews?.length && ` · ${m.defenseReviews.length} defender ${m.defenseReviews.length === 1 ? 'review' : 'reviews'}`}
                      </span>
                    </span>
                    <span className="list__count">
                      <b>{c.total}</b> {c.total === 1 ? 'attack' : 'attacks'}
                      {c.total > c.pose && <em> ({c.total - c.pose} need pose)</em>}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        {hasDemo && (
          <div className="row row--end">
            {realMatches.length === 0 && (
              <button className="btn btn--quiet" onClick={() => onGo('results')}>
                Train on demo data
              </button>
            )}
            <button className="btn btn--quiet" onClick={removeDemo}>
              Remove demo data
            </button>
          </div>
        )}
      </Section>

      <Roster />

      <Section title="How to film">
        <ul className="tips">
          <li>Stand behind your own end line, as high as you can get (top of the bleachers is ideal).</li>
          <li>Hold the phone sideways and keep the whole net and both antennas in frame. A cheap tripod helps a lot.</li>
          <li>Use 60 fps if your phone offers it. Tell samples 30 frames a second either way.</li>
          <li>Get permission from players, and from parents for anyone under 18, before you film.</li>
        </ul>
      </Section>

      <Section title="Your data">
        <p className="muted small">
          Videos never leave this phone. Tell stores only skeletons and tags here on the device. Export to back up, move
          to another phone, or train the bigger models on a laptop.
        </p>
        <div className="row wrap">
          <button className="btn" onClick={exportData} disabled={!dataset.attacks.length}>
            Export dataset
          </button>
          <button className="btn" onClick={() => fileRef.current?.click()}>
            Import dataset
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importData(f);
              e.target.value = '';
            }}
          />
          {!hasDemo && dataset.matches.length > 0 && (
            <button className="btn btn--quiet" onClick={loadDemo}>
              Add demo data
            </button>
          )}
          {confirmClear ? (
            <>
              <button
                className="btn btn--danger"
                onClick={() => {
                  replace(emptyDataset());
                  setReport(null);
                  setConfirmClear(false);
                }}
              >
                Delete everything
              </button>
              <button className="btn btn--quiet" onClick={() => setConfirmClear(false)}>
                Keep my data
              </button>
            </>
          ) : (
            <button className="btn btn--quiet" onClick={() => setConfirmClear(true)} disabled={!dataset.attacks.length && !dataset.players.length}>
              Delete all data
            </button>
          )}
        </div>
      </Section>
    </div>
  );
}

function RealSwings({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, report } = useData();
  const real = dataset.attacks.filter((a) => !a.matchId.startsWith('demo-'));
  const withPose = real.filter((a) => a.frames && a.frames.filter(Boolean).length >= 23).length;
  const hitters = dataset.players.filter((p) => !p.id.startsWith('demo-')).length;
  const trained = !!report && !report.synthetic && report.nAttacks >= 24;
  const steps = [
    { done: hitters > 0, title: 'Add your hitters', body: 'Roster below. Mark lefties.', action: null as null | (() => void), cta: '' },
    {
      done: withPose >= 24,
      title: `Capture 24 real swings (${Math.min(withPose, 24)}/24)`,
      body: 'Live tab: Record and review. Or film a match and tag it here.',
      action: () => onGo('live'),
      cta: 'Open camera',
    },
    { done: trained, title: 'Train on your swings', body: 'Model tab. Takes a few seconds.', action: () => onGo('results'), cta: 'Train' },
    { done: trained, title: 'Live calls use your model', body: 'From then on every swing you save makes it sharper.', action: null, cta: '' },
  ];
  const next = steps.findIndex((x) => !x.done);
  if (next === -1) return null;
  return (
    <section className="checklist" aria-label="Getting real swings in">
      <h2>Get your real swings in</h2>
      <ol>
        {steps.map((st, i) => (
          <li key={st.title} className={st.done ? 'done' : i === next ? 'next' : ''}>
            <span className="checklist__mark" aria-hidden>{st.done ? '✓' : i + 1}</span>
            <span className="checklist__text">
              <b>{st.title}</b>
              <span>{st.body}</span>
            </span>
            {i === next && st.action && (
              <button className="btn btn--tiny btn--primary" onClick={st.action}>{st.cta}</button>
            )}
          </li>
        ))}
      </ol>
      {withPose > 0 && withPose < 24 && (
        <div className="progress" aria-label="Swings captured"><div className="progress__bar" style={{ width: `${(withPose / 24) * 100}%` }} /></div>
      )}
    </section>
  );
}

export function Roster() {
  const { dataset, individual, update } = useData();
  const [num, setNum] = useState('');
  const [pname, setPname] = useState('');
  const [hand, setHand] = useState<Hand>('R');
  const players = dataset.players.filter((p) => !p.id.startsWith('demo-'));

  const add = () => {
    if ((!num.trim() && !pname.trim()) || (individual && players.length)) return;
    const p: Player = { id: individual ? INDIVIDUAL_HITTER : uid('p'), name: pname.trim() || `#${num.trim()}`, number: num.trim() || undefined, hand };
    update((d) => ({ ...d, players: [...d.players, p] }));
    setNum('');
    setPname('');
    setHand('R');
  };

  return (
    <Section title={individual ? "Your hitter" : "Roster"}>
      {(!individual || players.length === 0) && <form
        className="roster-form"
        onSubmit={(e) => {
          e.preventDefault();
          add();
        }}
      >
        <input className="input input--num" inputMode="numeric" placeholder="#" value={num} onChange={(e) => setNum(e.target.value)} aria-label="Jersey number" />
        <input className="input" placeholder="Name" value={pname} onChange={(e) => setPname(e.target.value)} aria-label="Player name" />
        <div className="seg" role="group" aria-label="Hitting hand">
          {(['R', 'L'] as const).map((h) => (
            <button type="button" key={h} className={`seg__btn ${hand === h ? 'seg__btn--on' : ''}`} aria-pressed={hand === h} onClick={() => setHand(h)}>
              {h === 'R' ? 'Righty' : 'Lefty'}
            </button>
          ))}
        </div>
        <button className="btn" type="submit">
          Add
        </button>
      </form>}
      {players.length === 0 ? (
        <p className="muted small">{individual ? 'Add the hitter you want to track, including their hitting hand.' : 'Add your hitters so you can tag who hit each ball. Hitting hand matters because Tell mirrors lefties.'}</p>
      ) : (
        <ul className="roster">
          {players.map((p) => (
            <li key={p.id}>
              <span className="jersey">{p.number ?? '–'}</span>
              <span className="roster__name">{p.name}</span>
              <button
                className="btn btn--tiny"
                onClick={() => update((d) => ({ ...d, players: d.players.map((x) => (x.id === p.id ? { ...x, hand: x.hand === 'R' ? 'L' : 'R' } : x)) }))}
                aria-label={`Switch ${p.name} to ${p.hand === 'R' ? 'left' : 'right'} handed`}
              >
                {p.hand === 'R' ? 'Righty' : 'Lefty'}
              </button>
              {!dataset.attacks.some((a) => a.hitterId === p.id) && (
                <button className="btn btn--tiny btn--quiet" onClick={() => update((d) => ({ ...d, players: d.players.filter((x) => x.id !== p.id) }))}>
                  Remove
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}
