import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import type { Tab } from '../App';
import { Court } from '../components/Court';
import { Skeleton } from '../components/Skeleton';
import { pct } from '../components/ui';
import { directionFromLanding, shotName } from '../lib/court';
import { detectAll, getLandmarker, type Detected } from '../lib/extract';
import { sequenceFeatures } from '../lib/features';
import { ContactDetector, PoseBuffer } from '../lib/live';
import { fitLive, readFrames, usableAttacks, type LiveModels, type LiveRead } from '../lib/ml/tell';
import { FORECAST_STEPS, stateOf } from '../lib/ml/world';
import { BONES } from '../lib/pose';
import type { Attack, AttackPosition, Direction, Hand, PoseFrame } from '../lib/types';
import { DIRECTIONS, uid } from '../lib/types';
import { useData } from '../state';

// Live camera mode. Point the phone at the court, lock onto a hitter, and Tell
// reads them in real time: shot probabilities update every frame, and the
// world model draws where their body is going over the next half second.
// When the arm swing finishes, the swing is captured; tap where the ball went
// and it becomes training data. No video is stored.

const POSITIONS: { value: AttackPosition; label: string }[] = [
  { value: 'left', label: 'Left' },
  { value: 'middle', label: 'Middle' },
  { value: 'right', label: 'Right' },
  { value: 'back-middle', label: 'Pipe' },
];
const LIVE_HORIZON = 100;

type CamState = 'off' | 'starting' | 'on' | 'error';

interface Capture {
  t: number;
  frames: (PoseFrame | null)[];
  read: LiveRead | null;
  hand: Hand;
  position: AttackPosition;
}

export function Live({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, update } = useData();
  const [cam, setCam] = useState<CamState>('off');
  const [camErr, setCamErr] = useState('');
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [quality, setQuality] = useState<'lite' | 'full'>('lite');
  const [fps, setFps] = useState(0);
  const [people, setPeople] = useState(0);
  const [locked, setLocked] = useState(false);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [auto, setAuto] = useState(true);
  const [hitterId, setHitterId] = useState<string | undefined>(undefined);
  const [position, setPosition] = useState<AttackPosition>('left');
  const [capture, setCapture] = useState<Capture | null>(null);
  const [landing, setLanding] = useState<{ x: number; y: number } | undefined>();
  const [manualDir, setManualDir] = useState<Direction | undefined>();
  const [saved, setSaved] = useState<{ right: boolean; said: string; went: string } | null>(null);
  const [newPlayer, setNewPlayer] = useState('');

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runRef = useRef(0);
  const buffer = useRef(new PoseBuffer());
  const detector = useRef(new ContactDetector('R'));
  const lastDet = useRef<Detected[]>([]);
  const lock = useRef<{ c: [number, number]; lastSeen: number } | null>(null);
  const smooth = useRef<number[] | null>(null);
  const ghostRef = useRef<number[][] | null>(null);
  const settings = useRef({ auto, hand: 'R' as Hand, position, live: null as LiveModels | null, paused: false });

  const players = dataset.players.filter((p) => !p.id.startsWith('demo-'));
  const hand: Hand = dataset.players.find((p) => p.id === hitterId)?.hand ?? 'R';
  const nUsable = useMemo(() => usableAttacks(dataset).usable.length, [dataset]);
  const live = useMemo(() => fitLive(dataset, LIVE_HORIZON), [nUsable]); // eslint-disable-line react-hooks/exhaustive-deps
  const demoOnly = dataset.matches.length > 0 && dataset.attacks.every((a) => a.matchId.startsWith('demo-'));

  useEffect(() => {
    settings.current = { auto, hand, position, live, paused: !!capture };
    detector.current.hand = hand;
  }, [auto, hand, position, live, capture]);

  const stop = useCallback(() => {
    runRef.current++;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCam('off');
  }, []);

  useEffect(() => stop, [stop]);

  const openCapture = useCallback((tc: number) => {
    const s = settings.current;
    const frames = buffer.current.window(tc);
    if (buffer.current.coverage(tc) < 0.5) return;
    const read = s.live ? readFrames(s.live, frames, s.position, s.hand, LIVE_HORIZON) : null;
    setCapture({ t: tc, frames, read, hand: s.hand, position: s.position });
    setLanding(undefined);
    setManualDir(undefined);
    setSaved(null);
    window.setTimeout(() => document.querySelector('.live__sheet')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
  }, []);

  const loop = useCallback(async (run: number) => {
    const video = videoRef.current!;
    const lm = await getLandmarker(quality);
    let frames = 0;
    let fpsT = performance.now();
    let lastRead = 0;
    const step = () => {
      if (run !== runRef.current) return;
      const now = performance.now() / 1000;
      if (video.readyState >= 2 && !settings.current.paused) {
        const dets = detectAll(lm, video);
        lastDet.current = dets;
        // Follow the locked player by image position.
        let target: Detected | null = null;
        if (!lock.current && dets.length === 1) lock.current = { c: dets[0].c, lastSeen: now };
        if (lock.current) {
          let bd = Infinity;
          for (const d of dets) {
            const dist = Math.hypot(d.c[0] - lock.current.c[0], d.c[1] - lock.current.c[1]);
            if (dist < bd) {
              bd = dist;
              target = d;
            }
          }
          const gap = now - lock.current.lastSeen;
          if (target && bd < 0.12 + gap * 0.3) lock.current = { c: target.c, lastSeen: now };
          else target = null;
          if (gap > 1.2) {
            lock.current = null;
            buffer.current.clear();
            detector.current.reset();
          }
        }
        buffer.current.push(now, target?.frame ?? null);
        if (target && settings.current.auto) {
          const tc = detector.current.push(now, target.frame);
          if (tc !== null) window.setTimeout(() => openCapture(tc), Math.max(0, (tc + 0.34 - now) * 1000));
        }
        // Live read, ten times a second.
        const s = settings.current;
        if (s.live && target && now - lastRead > 0.1) {
          lastRead = now;
          const r = readFrames(s.live, buffer.current.window(now + LIVE_HORIZON / 1000), s.position, s.hand, LIVE_HORIZON);
          if (r) {
            const prev = smooth.current;
            smooth.current = prev ? prev.map((v, i) => 0.6 * v + 0.4 * r.probs[i]) : r.probs;
            ghostRef.current = r.ghost;
            setProbs(smooth.current);
          }
        }
        if (!target) ghostRef.current = null;
        draw(canvasRef.current, video, dets, target, ghostRef.current);
        frames++;
        if (performance.now() - fpsT > 1000) {
          setFps(frames);
          setPeople(dets.length);
          setLocked(!!lock.current);
          if (!lock.current) setProbs(null);
          frames = 0;
          fpsT = performance.now();
        }
      }
      const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(step);
      else requestAnimationFrame(step);
    };
    step();
  }, [quality, openCapture]);

  const start = async (face = facing) => {
    setCamErr('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setCam('error');
      setCamErr('This browser cannot open the camera here. Open Tell over https in Safari or Chrome.');
      return;
    }
    stop();
    setCam('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: face }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
      });
      streamRef.current = stream;
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      setCam('on');
      const run = ++runRef.current;
      void loop(run).catch((e) => {
        setCam('error');
        setCamErr(`The pose model failed to load (${(e as Error).message}). Check your connection and try again.`);
      });
    } catch (e) {
      setCam('error');
      const name = (e as Error).name;
      setCamErr(
        name === 'NotAllowedError'
          ? 'Camera permission was blocked. Allow camera access for this site in your browser settings, then try again.'
          : name === 'NotFoundError'
            ? 'No camera found on this device.'
            : `Could not start the camera (${(e as Error).message}).`,
      );
    }
  };

  const onTap = (e: PointerEvent<HTMLCanvasElement>) => {
    const v = videoRef.current;
    if (!v) return;
    const r = e.currentTarget.getBoundingClientRect();
    const p = toVideo(v, r, e.clientX, e.clientY);
    let best: Detected | null = null;
    let bd = 0.2;
    for (const d of lastDet.current) {
      const dist = Math.hypot(d.c[0] - p.x, d.c[1] - p.y);
      if (dist < bd) {
        bd = dist;
        best = d;
      }
    }
    if (best) {
      lock.current = { c: best.c, lastSeen: performance.now() / 1000 };
      buffer.current.clear();
      detector.current.reset();
      smooth.current = null;
      setLocked(true);
    }
  };

  const swingNow = () => {
    const now = performance.now() / 1000;
    const peak = buffer.current.peakWrist(hand, now - 0.6, now);
    const tc = peak ?? now - 0.08;
    window.setTimeout(() => openCapture(tc), Math.max(0, (tc + 0.34 - now) * 1000));
  };

  const save = () => {
    if (!capture || !hitterId) return;
    const direction = landing ? directionFromLanding(capture.position, landing) : manualDir;
    if (!direction) return;
    const day = new Date().toISOString().slice(0, 10);
    const matchId = `live-${day}`;
    const attack: Attack = {
      id: uid('a'),
      matchId,
      hitterId,
      contactTime: capture.t,
      position: capture.position,
      landing,
      direction,
      shotType: 'hard',
      outcome: 'unknown',
      frames: capture.frames,
      createdAt: Date.now(),
    };
    update((d) => ({
      ...d,
      matches: d.matches.some((m) => m.id === matchId)
        ? d.matches
        : [{ id: matchId, name: `Live session ${new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`, date: day, live: true }, ...d.matches],
      attacks: [...d.attacks, attack],
    }));
    const said = capture.read ? DIRECTIONS[argmax(capture.read.probs)] : null;
    setSaved({
      right: said === direction,
      said: said ? shotName(capture.position, said) : '',
      went: shotName(capture.position, direction),
    });
  };

  const addPlayer = () => {
    const n = newPlayer.trim();
    if (!n) return;
    const id = uid('p');
    const isNum = /^\d+$/.test(n);
    update((d) => ({ ...d, players: [...d.players, { id, name: isNum ? `#${n}` : n, number: isNum ? n : undefined, hand: 'R' }] }));
    setHitterId(id);
    setNewPlayer('');
  };

  const top = probs ? argmax(probs) : -1;
  const captureStates = useMemo(() => {
    if (!capture) return [];
    const seq = sequenceFeatures(capture.frames, 'R');
    return seq ? seq.map(stateOf) : [];
  }, [capture]);
  const [replayFrame, setReplayFrame] = useState(0);
  useEffect(() => {
    if (!captureStates.length) return;
    let f = 0;
    const id = window.setInterval(() => {
      f = (f + 1) % (captureStates.length + 15);
      setReplayFrame(Math.min(f, captureStates.length - 1));
    }, 1000 / 24);
    return () => window.clearInterval(id);
  }, [captureStates]);

  const captureDir = capture ? (landing ? directionFromLanding(capture.position, landing) : manualDir) : undefined;

  return (
    <div className="screen live">
      <div className="live__stage">
        <video ref={videoRef} className="live__video" playsInline muted />
        <canvas ref={canvasRef} className="live__canvas" onPointerDown={onTap} aria-label="Camera view. Tap a player to lock on." />
        {cam !== 'on' && (
          <div className="live__cover">
            <h1>Read hitters live</h1>
            <p>
              Point your phone at the court from behind the end line. Tell tracks the hitter, reads their swing as it
              happens, and draws where their body goes next.
            </p>
            {camErr && <p className="notice notice--bad">{camErr}</p>}
            <button className="btn btn--primary btn--big" onClick={() => void start()} disabled={cam === 'starting'}>
              {cam === 'starting' ? 'Starting camera…' : 'Start camera'}
            </button>
            <p className="muted small">Video is processed on this phone and never recorded or uploaded.</p>
          </div>
        )}
        {cam === 'on' && (
          <>
            <div className="live__hud">
              <span className="live__pill live__pill--rec">Live</span>
              <span className="live__pill">{fps} fps</span>
              <span className="live__pill">{people} {people === 1 ? 'person' : 'people'}</span>
              <span className={`live__pill ${locked ? 'live__pill--lock' : ''}`}>{locked ? 'Locked on' : 'Tap the hitter'}</span>
            </div>
            {live && probs && (
              <div className="live__read" aria-live="polite">
                <span className="live__call">{shotName(position, DIRECTIONS[top])}</span>
                <span className="live__conf">{pct(probs[top])}</span>
                <div className="live__bars">
                  {(['left', 'straight', 'right'] as const).map((d, i) => (
                    <span key={d} className={`live__bar ${i === top ? 'live__bar--on' : ''}`}>
                      <i style={{ height: `${Math.round(probs[i] * 100)}%` }} />
                      <b>{d === 'left' ? 'L' : d === 'right' ? 'R' : 'S'}</b>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {cam === 'on' && (
        <div className="live__controls">
          <div className="row wrap">
            <button className="btn btn--primary" onClick={swingNow} disabled={!locked}>Capture swing</button>
            <label className="toggle">
              <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Auto-capture on contact
            </label>
          </div>
          <div className="row wrap">
            <div className="seg" role="group" aria-label="Attack position">
              {POSITIONS.map((p) => (
                <button key={p.value} className={`seg__btn ${position === p.value ? 'seg__btn--on' : ''}`} aria-pressed={position === p.value} onClick={() => setPosition(p.value)}>
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="row wrap">
            <button className="btn btn--tiny" onClick={() => { const f = facing === 'environment' ? 'user' : 'environment'; setFacing(f); void start(f); }}>
              {facing === 'environment' ? 'Use front camera' : 'Use back camera'}
            </button>
            <button className="btn btn--tiny" onClick={() => { stop(); setQuality(quality === 'lite' ? 'full' : 'lite'); }}>
              {quality === 'lite' ? 'Switch to accurate tracking' : 'Switch to fast tracking'}
            </button>
            <button className="btn btn--tiny btn--quiet" onClick={stop}>Stop camera</button>
          </div>
          {fps > 0 && fps < 8 && !location.search.includes('delegate=cpu') && (
            <p className="notice">
              Tracking is slow on this device ({fps} fps).{' '}
              <a href="?delegate=cpu" className="link">Try CPU mode</a>, which is faster on some phones and laptops.
            </p>
          )}
          <p className="muted small">
            {live
              ? `Reading with a model trained on ${nUsable} attacks${demoOnly ? ' of synthetic demo data, so treat the calls as a demo until you capture real swings' : ''}. Yellow dashes: where Tell expects the body to be over the next half second.`
              : `No model yet: capture and label ${Math.max(0, 24 - nUsable)} more swings (or load demo data in Film) and live reads switch on.`}
          </p>
        </div>
      )}

      {capture && (
        <div className="sheet live__sheet" role="dialog" aria-label="Captured swing">
          <div className="sheet__head">
            <h2>Swing captured</h2>
            <button className="btn btn--quiet btn--tiny" onClick={() => setCapture(null)}>Discard</button>
          </div>
          {captureStates.length > 0 && (
            <Skeleton states={captureStates} frame={replayFrame} hand={capture.hand} ghost={capture.read?.ghost} showGhostUntil={FORECAST_STEPS} height={200} />
          )}
          {capture.read && !saved && (
            <p className="small">
              Tell's call 100 ms before contact: <b>{shotName(capture.position, DIRECTIONS[argmax(capture.read.probs)])}</b>{' '}
              ({pct(Math.max(...capture.read.probs))} sure). Where did it actually go?
            </p>
          )}
          {saved ? (
            <div className={`verdict ${saved.right ? '' : 'verdict--miss'}`}>
              <p className="verdict__big">{saved.said ? (saved.right ? 'Tell read it' : 'Tell missed it') : 'Saved'}</p>
              <p className="verdict__sub">
                It went {saved.went.toLowerCase()}{saved.said ? `; Tell said ${saved.said.toLowerCase()}` : ''}. Saved to today's live session, so the
                next training run learns from it.
              </p>
              <div className="row">
                <button className="btn" onClick={() => setCapture(null)}>Back to camera</button>
                <button className="btn btn--quiet" onClick={() => onGo('results')}>Retrain</button>
              </div>
            </div>
          ) : (
            <>
              <fieldset className="chips">
                <legend>Hitter</legend>
                <div className="chips__row">
                  {players.map((p) => (
                    <button key={p.id} type="button" className={`chip chip--jersey ${hitterId === p.id ? 'chip--on' : ''}`} onClick={() => setHitterId(p.id)}>
                      {p.number ? `#${p.number}` : p.name}
                    </button>
                  ))}
                  <form className="chip-add" onSubmit={(e) => { e.preventDefault(); addPlayer(); }}>
                    <input className="input input--chip" placeholder="New #" value={newPlayer} onChange={(e) => setNewPlayer(e.target.value)} aria-label="Add hitter" />
                  </form>
                </div>
              </fieldset>
              <Court
                takeoff={capture.position}
                lanes={capture.read?.probs}
                dots={landing ? [{ ...landing, tone: 'ball', r: 3 }] : []}
                onTap={(p) => { setLanding(p); setManualDir(undefined); }}
                label={captureDir ? `${shotName(capture.position, captureDir)}` : 'Tap where the ball landed or was first touched'}
              />
              <div className="row wrap">
                {(['left', 'straight', 'right'] as const).map((d) => (
                  <button key={d} className={`chip ${!landing && manualDir === d ? 'chip--on' : ''}`} onClick={() => { setLanding(undefined); setManualDir(d); }}>
                    {d === 'left' ? 'Went left' : d === 'right' ? 'Went right' : 'Straight'}
                  </button>
                ))}
              </div>
              <div className="row row--end">
                <button className="btn btn--primary" disabled={!hitterId || !captureDir} onClick={save}>Save swing</button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function argmax(p: number[]) {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}

function videoBox(v: HTMLVideoElement, r: { width: number; height: number }) {
  const vw = v.videoWidth || r.width;
  const vh = v.videoHeight || r.height;
  // object-fit: cover
  const scale = Math.max(r.width / vw, r.height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  return { ox: (r.width - dw) / 2, oy: (r.height - dh) / 2, dw, dh, aspect: vh / vw };
}

function toVideo(v: HTMLVideoElement, r: DOMRect, cx: number, cy: number) {
  const b = videoBox(v, r);
  return { x: (cx - r.left - b.ox) / b.dw, y: (cy - r.top - b.oy) / b.dh };
}

function draw(c: HTMLCanvasElement | null, v: HTMLVideoElement, dets: Detected[], target: Detected | null, ghost: number[][] | null) {
  if (!c) return;
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr)) {
    c.width = Math.round(r.width * dpr);
    c.height = Math.round(r.height * dpr);
  }
  const g = c.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  const b = videoBox(v, r);
  const P = (x: number, y: number): [number, number] => [b.ox + x * b.dw, b.oy + y * b.dh];
  const bones = (p: number[], color: string, width: number, dash: number[] = []) => {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.lineCap = 'round';
    g.setLineDash(dash);
    for (const [i, j] of BONES) {
      const [x1, y1] = P(p[i * 3], p[i * 3 + 1]);
      const [x2, y2] = P(p[j * 3], p[j * 3 + 1]);
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
      g.stroke();
    }
    g.setLineDash([]);
  };
  for (const d of dets) if (d !== target) bones(d.frame.p, 'rgba(255,255,255,0.55)', 2);
  if (!target) return;
  bones(target.frame.p, '#f2c230', 4);
  // World-model ghost: torso-unit pose states mapped back onto the image around the hips.
  if (ghost) {
    const p = target.frame.p;
    const torsoImg = Math.abs((p[23 * 3 + 1] + p[24 * 3 + 1]) / 2 - (p[11 * 3 + 1] + p[12 * 3 + 1]) / 2) || 0.05;
    const hip0 = [(p[23 * 3] + p[24 * 3]) / 2, (p[23 * 3 + 1] + p[24 * 3 + 1]) / 2];
    const startHip = ghost[0] ? [ghost[0][39], ghost[0][40]] : hip0;
    const order = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    for (const k of [4, 9, 14]) {
      const s = ghost[Math.min(k, ghost.length - 1)];
      if (!s) continue;
      const hx = hip0[0] + (s[39] - startHip[0]);
      const hy = hip0[1] + (s[40] - startHip[1]);
      const img = new Array(99).fill(0);
      order.forEach((lmIdx, j) => {
        img[lmIdx * 3] = hx + s[j * 3] * torsoImg * b.aspect;
        img[lmIdx * 3 + 1] = hy + s[j * 3 + 1] * torsoImg;
      });
      g.globalAlpha = k === 14 ? 0.95 : 0.4;
      bones(img, '#f2c230', 2.5, [6, 5]);
      g.globalAlpha = 1;
    }
  }
}
