import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import { Court } from '../components/Court';
import { Chips, Progress, fmtTime } from '../components/ui';
import { directionFromLanding, shotName } from '../lib/court';
import { extractAttackPose } from '../lib/extract';
import { BONES } from '../lib/pose';
import type { Attack, AttackPosition, Direction, Outcome, ShotType } from '../lib/types';
import { CONTACT_INDEX, uid } from '../lib/types';
import { useData } from '../state';

const POSITIONS: { value: AttackPosition; label: string }[] = [
  { value: 'left', label: 'Left (4)' },
  { value: 'middle', label: 'Middle (3)' },
  { value: 'right', label: 'Right (2)' },
  { value: 'back-left', label: 'Back left' },
  { value: 'back-middle', label: 'Pipe' },
  { value: 'back-right', label: 'Back right (D)' },
];
const SHOTS: { value: ShotType; label: string }[] = [
  { value: 'hard', label: 'Hard swing' },
  { value: 'roll', label: 'Roll shot' },
  { value: 'tip', label: 'Tip' },
];
const OUTCOMES: { value: Outcome; label: string }[] = [
  { value: 'kill', label: 'Kill' },
  { value: 'dug', label: 'Dug' },
  { value: 'blocked', label: 'Blocked' },
  { value: 'error', label: 'Error' },
];

interface Draft {
  contactTime: number;
  tap?: { x: number; y: number };
  hitterId?: string;
  position: AttackPosition;
  landing?: { x: number; y: number };
  direction?: Direction;
  shotType: ShotType;
  outcome: Outcome;
}

type Job = { state: 'queued' } | { state: 'running'; frac: number } | { state: 'failed'; why: string };

export function Tag({ matchId, onBack }: { matchId: string; onBack: () => void }) {
  const { dataset, update } = useData();
  const match = dataset.matches.find((m) => m.id === matchId);
  const attacks = useMemo(
    () => dataset.attacks.filter((a) => a.matchId === matchId).sort((a, b) => a.contactTime - b.contactTime),
    [dataset.attacks, matchId],
  );
  const players = dataset.players;
  const [url, setUrl] = useState<string | null>(null);
  const [mode, setMode] = useState<'watch' | 'tap' | 'details'>('watch');
  const [draft, setDraft] = useState<Draft | null>(null);
  const [rate, setRate] = useState(1);
  const [paused, setPaused] = useState(true);
  const [now, setNow] = useState(0);
  const [jobs, setJobs] = useState<Record<string, Job>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [newPlayer, setNewPlayer] = useState('');
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const running = useRef(false);
  const last = useRef<{ hitterId?: string; position: AttackPosition }>({ position: 'left' });

  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
  }, [url]);

  const openFile = (f: File) => {
    if (url) URL.revokeObjectURL(url);
    setUrl(URL.createObjectURL(f));
    if (match && !match.synthetic) {
      update((d) => ({ ...d, matches: d.matches.map((m) => (m.id === matchId ? { ...m, videoName: f.name, videoSize: f.size } : m)) }));
    }
  };

  const [busyPose, setBusyPose] = useState<{ who: string; frac: number } | null>(null);
  const cancelPose = useRef(false);

  // Pose extraction queue. Runs on the visible video (phones won't reliably decode a hidden copy),
  // with an overlay showing progress while it scrubs through the 1.5 s around each contact.
  const pump = useCallback(async () => {
    if (running.current || !videoRef.current || !url || mode !== 'watch') return;
    const pending = dataset.attacks.filter((a) => a.matchId === matchId && a.frames === null && a.hitterTap && jobs[a.id]?.state !== 'failed');
    const next = pending[0];
    if (!next) return;
    running.current = true;
    cancelPose.current = false;
    const who = dataset.players.find((p) => p.id === next.hitterId);
    const label = who ? (who.number ? `#${who.number}` : who.name) : 'the hitter';
    setJobs((j) => ({ ...j, [next.id]: { state: 'running', frac: 0 } }));
    setBusyPose({ who: label, frac: 0 });
    try {
      const v = videoRef.current;
      if (v.readyState < 2) await new Promise((r) => v.addEventListener('loadeddata', r, { once: true }));
      const { frames, found } = await extractAttackPose(v, next.contactTime, next.hitterTap!, (frac) => {
        if (cancelPose.current) throw new Error('Stopped');
        setJobs((j) => ({ ...j, [next.id]: { state: 'running', frac } }));
        setBusyPose({ who: label, frac });
      });
      if (found < 10) {
        setJobs((j) => ({ ...j, [next.id]: { state: 'failed', why: `Found the hitter in only ${found} frames` } }));
        update((d) => ({ ...d, attacks: d.attacks.map((a) => (a.id === next.id ? { ...a, frames } : a)) }));
      } else {
        setJobs((j) => {
          const c = { ...j };
          delete c[next.id];
          return c;
        });
        update((d) => ({ ...d, attacks: d.attacks.map((a) => (a.id === next.id ? { ...a, frames } : a)) }));
      }
    } catch (e) {
      setJobs((j) => ({ ...j, [next.id]: { state: 'failed', why: (e as Error).message || 'Pose model failed to load' } }));
    } finally {
      running.current = false;
      setBusyPose(null);
    }
  }, [dataset.attacks, dataset.players, jobs, matchId, update, url, mode]);

  useEffect(() => {
    void pump();
  }, [pump]);

  const v = () => videoRef.current;
  const step = (dt: number) => {
    const el = v();
    if (!el) return;
    el.pause();
    el.currentTime = Math.max(0, Math.min(el.duration || 0, el.currentTime + dt));
  };
  const toggle = () => {
    const el = v();
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };
  const markContact = () => {
    const el = v();
    if (!el) return;
    el.pause();
    setDraft({
      contactTime: el.currentTime,
      hitterId: last.current.hitterId,
      position: last.current.position,
      shotType: 'hard',
      outcome: 'kill',
    });
    setMode('tap');
    setSelected(null);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
      if (e.key === ' ') {
        e.preventDefault();
        toggle();
      } else if (e.key === ',') step(-1 / 30);
      else if (e.key === '.') step(1 / 30);
      else if (e.key === 'ArrowLeft') step(-1);
      else if (e.key === 'ArrowRight') step(1);
      else if (e.key.toLowerCase() === 'm' && mode === 'watch') markContact();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  /** Normalized position inside the displayed video, accounting for letterboxing. */
  const videoPoint = (e: PointerEvent<HTMLElement>) => {
    const el = v()!;
    const r = el.getBoundingClientRect();
    const vw = el.videoWidth || r.width;
    const vh = el.videoHeight || r.height;
    const scale = Math.min(r.width / vw, r.height / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (r.width - dw) / 2;
    const oy = (r.height - dh) / 2;
    return { x: (e.clientX - r.left - ox) / dw, y: (e.clientY - r.top - oy) / dh };
  };

  const onOverlayTap = (e: PointerEvent<HTMLDivElement>) => {
    if (mode !== 'tap' || !draft) return;
    const p = videoPoint(e);
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return;
    setDraft({ ...draft, tap: p });
    setMode('details');
  };

  const save = () => {
    if (!draft || !draft.hitterId || !draft.tap) return;
    const direction = draft.landing ? directionFromLanding(draft.position, draft.landing) : draft.direction;
    if (!direction) return;
    const a: Attack = {
      id: uid('a'),
      matchId,
      hitterId: draft.hitterId,
      contactTime: draft.contactTime,
      hitterTap: draft.tap,
      position: draft.position,
      landing: draft.landing,
      direction,
      shotType: draft.shotType,
      outcome: draft.outcome,
      frames: null,
      createdAt: Date.now(),
    };
    last.current = { hitterId: draft.hitterId, position: draft.position };
    // iPhones only hand decoded frames to the pose model after a play() inside a tap.
    const el = v();
    if (el && el.paused) void el.play().then(() => el.pause()).catch(() => {});
    update((d) => ({ ...d, attacks: [...d.attacks, a] }));
    setJobs((j) => ({ ...j, [a.id]: { state: 'queued' } }));
    setDraft(null);
    setMode('watch');
  };

  const addPlayer = () => {
    const n = newPlayer.trim();
    if (!n) return;
    const id = uid('p');
    const isNum = /^\d+$/.test(n);
    update((d) => ({ ...d, players: [...d.players, { id, name: isNum ? `#${n}` : n, number: isNum ? n : undefined, hand: 'R' }] }));
    setDraft((dr) => (dr ? { ...dr, hitterId: id } : dr));
    setNewPlayer('');
  };

  // Draw the tracked skeleton on the contact frame of the selected attack, so you can check it followed the right player.
  useEffect(() => {
    const c = overlayRef.current;
    const el = v();
    if (!c || !el) return;
    const g = c.getContext('2d')!;
    const r = el.getBoundingClientRect();
    c.width = r.width;
    c.height = r.height;
    g.clearRect(0, 0, c.width, c.height);
    const a = attacks.find((x) => x.id === selected);
    const showTap = mode === 'details' && draft?.tap;
    const vw = el.videoWidth || r.width;
    const vh = el.videoHeight || r.height;
    const scale = Math.min(r.width / vw, r.height / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (r.width - dw) / 2;
    const oy = (r.height - dh) / 2;
    const P = (x: number, y: number): [number, number] => [ox + x * dw, oy + y * dh];
    if (showTap) {
      const [x, y] = P(draft!.tap!.x, draft!.tap!.y);
      g.strokeStyle = '#f2c230';
      g.lineWidth = 3;
      g.beginPath();
      g.arc(x, y, 16, 0, Math.PI * 2);
      g.stroke();
    }
    if (!a?.frames) return;
    const offset = Math.round((now - a.contactTime) * 30);
    const fi = CONTACT_INDEX + offset;
    const f = a.frames[Math.max(0, Math.min(a.frames.length - 1, fi))];
    if (!f || Math.abs(offset) > 9) return;
    g.strokeStyle = '#f2c230';
    g.lineWidth = 3;
    g.lineCap = 'round';
    for (const [i, j] of BONES) {
      const [x1, y1] = P(f.p[i * 3], f.p[i * 3 + 1]);
      const [x2, y2] = P(f.p[j * 3], f.p[j * 3 + 1]);
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    }
  }, [selected, now, attacks, mode, draft]);

  if (!match) return <p className="pad">This match no longer exists.</p>;

  const name = (id: string) => {
    const p = players.find((x) => x.id === id);
    return p ? (p.number ? `#${p.number}` : p.name) : 'Unknown';
  };
  const needPose = attacks.filter((a) => a.frames === null).length;
  const direction = draft?.landing ? directionFromLanding(draft.position, draft.landing) : draft?.direction;

  return (
    <div className="screen screen--tag">
      <div className="crumb">
        <button className="btn btn--quiet btn--tiny" onClick={onBack}>
          All matches
        </button>
        <h1 className="crumb__title">{match.name}</h1>
      </div>

      {match.synthetic ? (
        <p className="notice">This is a synthetic demo match, so there's no video. Its attacks show up in Model, Hitters and Practice.</p>
      ) : match.live ? (
        <p className="notice">Recorded with the live camera, so there's no video. Only the skeletons were kept. Capture more in Live.</p>
      ) : !url ? (
        <label className="dropzone">
          <input type="file" accept="video/*" onChange={(e) => e.target.files?.[0] && openFile(e.target.files[0])} />
          <strong>Open the video for this match</strong>
          <span>{match.videoName ? `Last time: ${match.videoName}` : 'From your camera roll or files. It stays on this phone.'}</span>
          {needPose > 0 && <span className="dropzone__warn">{needPose} tagged attacks still need pose from this video.</span>}
        </label>
      ) : (
        <>
          <div className="player">
            <video
              ref={videoRef}
              src={url}
              playsInline
              muted
              preload="auto"
              className="player__video"
              onPlay={() => setPaused(false)}
              onPause={() => setPaused(true)}
              onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)}
              onSeeked={(e) => setNow(e.currentTarget.currentTime)}
            />
            <canvas ref={overlayRef} className="player__overlay" aria-hidden />
            {mode === 'tap' && (
              <div className="player__tap" onPointerDown={onOverlayTap} role="button" aria-label="Tap the hitter">
                <span>Tap the hitter</span>
              </div>
            )}
            {busyPose && (
              <div className="player__busy" role="status">
                <span>Finding {busyPose.who} in the clip</span>
                <div className="progress progress--light"><div className="progress__bar" style={{ width: `${Math.round(busyPose.frac * 100)}%` }} /></div>
                <button className="btn btn--tiny" onClick={() => (cancelPose.current = true)}>Stop</button>
              </div>
            )}
          </div>
          <div className="transport">
            <button className="btn btn--icon" onClick={() => step(-1)} aria-label="Back 1 second">−1s</button>
            <button className="btn btn--icon" onClick={() => step(-1 / 30)} aria-label="Back 1 frame">−1f</button>
            <button className="btn btn--icon btn--play" onClick={toggle} aria-label={paused ? 'Play' : 'Pause'}>
              {paused ? 'Play' : 'Pause'}
            </button>
            <button className="btn btn--icon" onClick={() => step(1 / 30)} aria-label="Forward 1 frame">+1f</button>
            <button className="btn btn--icon" onClick={() => step(1)} aria-label="Forward 1 second">+1s</button>
            <button
              className="btn btn--icon"
              onClick={() => {
                const r = rate === 1 ? 0.5 : rate === 0.5 ? 0.25 : 1;
                setRate(r);
                if (v()) v()!.playbackRate = r;
              }}
              aria-label="Playback speed"
            >
              {rate}×
            </button>
            <span className="transport__time">{fmtTime(now)}</span>
          </div>
          {mode === 'watch' && (
            <button className="btn btn--primary btn--block btn--big" onClick={markContact}>
              Mark contact
            </button>
          )}
          {mode === 'watch' && <p className="muted small center">Pause on the frame where the hand meets the ball, then mark it. Keys: space, comma/period for frames, M to mark.</p>}
          {mode === 'tap' && (
            <div className="row row--end">
              <button className="btn btn--quiet" onClick={() => { setMode('watch'); setDraft(null); }}>
                Cancel
              </button>
            </div>
          )}
        </>
      )}

      {mode === 'details' && draft && (
        <div className="sheet" role="dialog" aria-label="Attack details">
          <div className="sheet__head">
            <h2>Attack at {fmtTime(draft.contactTime)}</h2>
            <button className="btn btn--quiet btn--tiny" onClick={() => setMode('tap')}>
              Re-tap hitter
            </button>
          </div>
          <fieldset className="chips">
            <legend>Hitter</legend>
            <div className="chips__row">
              {players.filter((p) => !p.id.startsWith('demo-')).map((p) => (
                <button
                  type="button"
                  key={p.id}
                  className={`chip chip--jersey ${draft.hitterId === p.id ? 'chip--on' : ''}`}
                  aria-pressed={draft.hitterId === p.id}
                  onClick={() => setDraft({ ...draft, hitterId: p.id })}
                >
                  {p.number ? `#${p.number}` : p.name}
                </button>
              ))}
              <form
                className="chip-add"
                onSubmit={(e) => {
                  e.preventDefault();
                  addPlayer();
                }}
              >
                <input className="input input--chip" placeholder="New #" value={newPlayer} onChange={(e) => setNewPlayer(e.target.value)} aria-label="Add hitter by number or name" />
              </form>
            </div>
          </fieldset>
          <Chips label="Attacked from" value={draft.position} options={POSITIONS} onChange={(position) => setDraft({ ...draft, position })} />
          <div className="field">
            <span className="field__label">Where did it go? Tap the first touch: floor, defender, or out.</span>
            <Court
              takeoff={draft.position}
              dots={draft.landing ? [{ ...draft.landing, tone: 'ball', r: 3 }] : []}
              onTap={(landing) => setDraft({ ...draft, landing, direction: undefined })}
              label={direction ? `${shotName(draft.position, direction)} (${direction} from the hitter)` : 'Tap the court'}
            />
            <div className="row wrap">
              <span className="muted small">Didn't see it land?</span>
              {(['left', 'straight', 'right'] as const).map((d) => (
                <button
                  key={d}
                  type="button"
                  className={`chip ${!draft.landing && draft.direction === d ? 'chip--on' : ''}`}
                  onClick={() => setDraft({ ...draft, landing: undefined, direction: d })}
                >
                  {d === 'left' ? 'Went left' : d === 'right' ? 'Went right' : 'Straight'}
                </button>
              ))}
            </div>
          </div>
          <Chips label="Shot" value={draft.shotType} options={SHOTS} onChange={(shotType) => setDraft({ ...draft, shotType })} />
          <Chips label="Result" value={draft.outcome} options={OUTCOMES} onChange={(outcome) => setDraft({ ...draft, outcome })} />
          <div className="row row--end">
            <button className="btn btn--quiet" onClick={() => { setDraft(null); setMode('watch'); }}>
              Discard
            </button>
            <button className="btn btn--primary" onClick={save} disabled={!draft.hitterId || !direction}>
              Save attack
            </button>
          </div>
          {!draft.hitterId && <p className="muted small">Pick or add the hitter to save.</p>}
        </div>
      )}

      <section className="section">
        <div className="section__head">
          <h2>Tagged attacks</h2>
          <span className="muted">{attacks.length}</span>
        </div>
        {attacks.length === 0 ? (
          <p className="muted small">Nothing tagged yet. About 60 attacks per match is typical; 300 across a few matches is enough for a first model.</p>
        ) : (
          <ul className="attacks">
            {attacks.map((a) => {
              const job = jobs[a.id];
              const got = a.frames ? a.frames.filter(Boolean).length : 0;
              return (
                <li key={a.id} className={selected === a.id ? 'attacks__on' : ''}>
                  <button
                    className="attacks__time"
                    onClick={() => {
                      setSelected(a.id);
                      if (v()) {
                        v()!.pause();
                        v()!.currentTime = a.contactTime;
                      }
                    }}
                  >
                    {fmtTime(a.contactTime)}
                  </button>
                  <span className="attacks__who">{name(a.hitterId)} <em>{shotName(a.position, a.direction)}</em></span>
                  <span className="attacks__pose">
                    {job?.state === 'running' ? (
                      <Progress value={job.frac} label="Finding the hitter" />
                    ) : job?.state === 'failed' ? (
                      <button className="btn btn--tiny" onClick={() => {
                        update((d) => ({ ...d, attacks: d.attacks.map((x) => (x.id === a.id ? { ...x, frames: null } : x)) }));
                        setJobs((j) => ({ ...j, [a.id]: { state: 'queued' } }));
                      }} title={job.why}>
                        Retry pose
                      </button>
                    ) : a.frames ? (
                      <span className={got >= 23 ? 'ok' : 'warn'}>{got >= 23 ? 'Pose ready' : `Weak (${got}/46)`}</span>
                    ) : (
                      <span className="muted">{url ? 'Queued' : 'Needs video'}</span>
                    )}
                  </span>
                  <button className="btn btn--tiny btn--quiet" aria-label="Delete attack" onClick={() => update((d) => ({ ...d, attacks: d.attacks.filter((x) => x.id !== a.id) }))}>
                    Delete
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        {Object.values(jobs).some((j) => j.state === 'failed') && (
          <p className="muted small">If pose keeps failing, check the hitter is fully in frame on the contact frame and tap the middle of their body.</p>
        )}
      </section>
    </div>
  );
}
