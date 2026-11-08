import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import type { Tab } from '../App';
import { Court } from '../components/Court';
import { pct } from '../components/ui';
import { ballDirection, detectBall, getBallDetector, type BallBox, type BallPoint } from '../lib/ball';
import { directionFromLanding, shotName } from '../lib/court';
import { detectAll, getLandmarker, type Detected } from '../lib/extract';
import { ContactDetector, OneEuro, PoseBuffer, poseBox } from '../lib/live';
import { fitLive, readFrames, usableAttacks, type LiveModels, type LiveRead } from '../lib/ml/tell';
import { BONES } from '../lib/pose';
import type { Attack, AttackPosition, Direction, Hand, PoseFrame } from '../lib/types';
import { INDIVIDUAL_HITTER, DIRECTIONS, uid } from '../lib/types';
import { useData } from '../state';

// Two ways to use the camera:
//  Read:   live calls while you watch. Tell tracks the hitter, shows its call
//          as they swing, and draws the world model's forecast of the next
//          half second. Swings are captured automatically at contact.
//  Record: like Smashspeed. Record a rep, then review it frame by frame with
//          red boxes on the hitter and the ball, Tell's call, and where the
//          ball actually went. Confirm and it becomes training data.
// Video only lives in memory for the review; it is never uploaded or saved.

const POSITIONS: { value: AttackPosition; label: string }[] = [
  { value: 'left', label: 'Left' },
  { value: 'middle', label: 'Middle' },
  { value: 'right', label: 'Right' },
  { value: 'back-middle', label: 'Pipe' },
];
const HORIZON = 100;
const MAX_RECORD_S = 12;
const RED = '#e5383b';
const YELLOW = '#f2c230';

type Cam = { state: 'off' } | { state: 'starting' | 'loading'; note: string } | { state: 'on' } | { state: 'error'; note: string };

interface TimelineEntry {
  t: number; // seconds, performance clock
  p: number[] | null; // smoothed target landmarks (image)
  others: number[][];
  ball: BallBox | null;
}

export interface SwingDraft {
  contactT: number;
  frames: (PoseFrame | null)[];
  read: LiveRead | null;
  hand: Hand;
  position: AttackPosition;
  suggestion: { direction: Direction; angle: number } | null;
}

export function Live({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, individual } = useData();
  const [cam, setCam] = useState<Cam>({ state: 'off' });
  const [mode, setMode] = useState<'read' | 'record'>('read');
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [fps, setFps] = useState(0);
  const [people, setPeople] = useState(0);
  const [locked, setLocked] = useState(false);
  const [probs, setProbs] = useState<number[] | null>(null);
  const [selectedHitter, setHitterId] = useState<string | undefined>(undefined);
  const [position, setPosition] = useState<AttackPosition>('left');
  const [draft, setDraft] = useState<SwingDraft | null>(null);
  const [recording, setRecording] = useState<{ started: number } | null>(null);
  const [recElapsed, setRecElapsed] = useState(0);
  const [review, setReview] = useState<{ url: string | null; timeline: TimelineEntry[]; start: number; buffer: PoseBuffer; ball: BallPoint[]; contactT: number | null } | null>(null);
  const [ballOn, setBallOn] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runRef = useRef(0);
  const buffer = useRef(new PoseBuffer(4));
  const recBuffer = useRef<PoseBuffer | null>(null);
  const timeline = useRef<TimelineEntry[]>([]);
  const ballPts = useRef<BallPoint[]>([]);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const detector = useRef(new ContactDetector('R'));
  const smoothF = useRef(new OneEuro());
  const lastDet = useRef<Detected[]>([]);
  const lock = useRef<{ c: [number, number]; lastSeen: number; last: Detected } | null>(null);
  const smoothProbs = useRef<number[] | null>(null);
  const ghostRef = useRef<number[][] | null>(null);
  const recContact = useRef<number | null>(null);
  const settings = useRef({ hand: 'R' as Hand, position, live: null as LiveModels | null, paused: false, mode, recording: false, ballOn });

  const players = dataset.players.filter((p) => !p.id.startsWith('demo-'));
  const hitterId = individual ? players[0]?.id : selectedHitter;
  const hand: Hand = dataset.players.find((p) => p.id === hitterId)?.hand ?? 'R';
  const realUsable = useMemo(() => usableAttacks({ ...dataset, attacks: dataset.attacks.filter((a) => !a.matchId.startsWith('demo-')) }).usable.length, [dataset]);
  const allUsable = useMemo(() => usableAttacks(dataset).usable.length, [dataset]);
  // Live calls come from your own swings once there are enough; demo data only until then.
  const live = useMemo(() => {
    if (realUsable >= 24) return { model: fitLive({ ...dataset, attacks: dataset.attacks.filter((a) => !a.matchId.startsWith('demo-')) }, HORIZON), source: 'yours' as const, n: realUsable };
    if (allUsable >= 24) return { model: fitLive(dataset, HORIZON), source: 'demo' as const, n: allUsable };
    return null;
  }, [realUsable, allUsable]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    settings.current = { hand, position, live: live?.model ?? null, paused: !!draft || !!review, mode, recording: !!recording, ballOn };
    detector.current.hand = hand;
  }, [hand, position, live, draft, review, mode, recording, ballOn]);

  const stopCamera = useCallback(() => {
    runRef.current++;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCam({ state: 'off' });
  }, []);
  useEffect(() => stopCamera, [stopCamera]);

  const buildDraft = useCallback((buf: PoseBuffer, contactT: number, ball: BallPoint[]): SwingDraft | null => {
    if (buf.coverage(contactT) < 0.5) return null;
    const s = settings.current;
    const frames = buf.window(contactT);
    const read = s.live ? readFrames(s.live, frames, s.position, s.hand, HORIZON) : null;
    const sug = ballDirection(ball, contactT);
    return { contactT, frames, read, hand: s.hand, position: s.position, suggestion: sug ? { direction: sug.direction, angle: sug.angle } : null };
  }, []);

  const finishRecording = useCallback(() => {
    const r = recorder.current;
    const start = recordingStart.current;
    const done = (url: string | null) => {
      setReview({ url, timeline: timeline.current.slice(), start, buffer: recBuffer.current!, ball: ballPts.current.slice(), contactT: recContact.current });
      setRecording(null);
      // Free the camera while reviewing; it restarts when you go back.
      runRef.current++;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setCam({ state: 'off' });
    };
    if (r && r.state !== 'inactive') {
      r.onstop = () => done(chunks.current.length ? URL.createObjectURL(new Blob(chunks.current, { type: r.mimeType })) : null);
      r.stop();
    } else done(null);
  }, []);
  const recordingStart = useRef(0);

  const loop = useCallback(async (run: number, cpu: boolean) => {
    const video = videoRef.current!;
    setCam({ state: 'loading', note: 'Loading the pose model. The first time takes a few seconds (about 20 MB).' });
    let lm = await getLandmarker('lite', cpu);
    let ballDet: Awaited<ReturnType<typeof getBallDetector>> | null = null;
    getBallDetector(cpu).then((d) => (ballDet = d)).catch(() => (ballDet = null));
    if (run !== runRef.current) return;
    setCam({ state: 'on' });
    let frames = 0;
    let fpsT = performance.now();
    let lastRead = 0;
    let failures = 0;
    let n = 0;
    const step = async () => {
      if (run !== runRef.current) return;
      const now = performance.now() / 1000;
      const s = settings.current;
      if (video.readyState >= 2 && video.videoWidth > 0 && !s.paused) {
        let dets: Detected[] = [];
        try {
          dets = detectAll(lm, video);
          failures = 0;
        } catch {
          // GPU path broke on this device: switch to CPU once and keep going.
          if (++failures > 2 && !cpu) {
            cpu = true;
            lm = await getLandmarker('lite', true);
          }
        }
        lastDet.current = dets;
        let target: Detected | null = null;
        if (!lock.current && dets.length === 1) lock.current = { c: dets[0].c, lastSeen: now, last: dets[0] };
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
          if (target && bd < 0.12 + gap * 0.4) lock.current = { c: target.c, lastSeen: now, last: target };
          else target = null;
          if (gap > 1.5) {
            lock.current = null;
            buffer.current.clear();
            detector.current.reset();
            smoothF.current.reset();
          }
        }
        buffer.current.push(now, target?.frame ?? null);
        if (s.recording) recBuffer.current?.push(now, target?.frame ?? null);

        // Ball: every other frame keeps the frame rate up.
        let ball: BallBox | null = null;
        if (s.ballOn && ballDet && (s.recording || s.mode === 'record' || n % 2 === 0)) {
          try {
            const lastB = ballPts.current[ballPts.current.length - 1];
            ball = detectBall(ballDet, video, lastB && now - lastB.t < 0.3 ? lastB : null);
          } catch {
            ball = null;
          }
          if (ball) {
            ballPts.current.push({ t: now, x: ball.x, y: ball.y });
            if (ballPts.current.length > 400) ballPts.current.splice(0, 100);
          }
        }
        n++;

        // Smooth what we draw, never what the model reads.
        const shown = lock.current ? (target ?? (now - lock.current.lastSeen < 0.25 ? lock.current.last : null)) : null;
        const sp = shown ? smoothF.current.filter(now, shown.frame.p) : null;

        if (s.recording) {
          timeline.current.push({ t: now, p: sp, others: dets.filter((d) => d !== target).map((d) => d.frame.p), ball });
          if (target) {
            const tc = detector.current.push(now, target.frame);
            if (tc !== null && recContact.current === null) {
              recContact.current = tc;
              window.setTimeout(() => finishRecording(), 700);
            }
          }
          if (now - recordingStart.current > MAX_RECORD_S) finishRecording();
        } else if (s.mode === 'read' && target) {
          const tc = detector.current.push(now, target.frame);
          if (tc !== null) {
            window.setTimeout(() => {
              const d = buildDraft(buffer.current, tc, ballPts.current);
              if (d) setDraft(d);
            }, Math.max(0, (tc + 0.5 - now) * 1000));
          }
        }

        if (s.mode === 'read' && s.live && target && now - lastRead > 0.1) {
          lastRead = now;
          const r = readFrames(s.live, buffer.current.window(now + HORIZON / 1000), s.position, s.hand, HORIZON);
          if (r) {
            const prev = smoothProbs.current;
            smoothProbs.current = prev ? prev.map((v, i) => 0.7 * v + 0.3 * r.probs[i]) : r.probs;
            ghostRef.current = r.ghost;
            setProbs(smoothProbs.current);
          }
        }
        if (!shown) ghostRef.current = null;
        drawOverlay(canvasRef.current, video, 'cover', {
          others: dets.filter((d) => d !== target).map((d) => d.frame.p),
          target: sp,
          ball,
          trail: ballPts.current.filter((b) => now - b.t < 0.6),
          ghost: s.mode === 'read' ? ghostRef.current : null,
          recording: s.recording,
        });
        frames++;
        if (performance.now() - fpsT > 1000) {
          setFps(frames);
          setPeople(dets.length);
          setLocked(!!lock.current);
          if (!lock.current) {
            setProbs(null);
            smoothProbs.current = null;
          }
          frames = 0;
          fpsT = performance.now();
        }
      }
      const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };
      if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => void step());
      else requestAnimationFrame(() => void step());
    };
    void step();
  }, [buildDraft, finishRecording]);

  const start = async (face = facing) => {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setCam({ state: 'error', note: 'The camera only works over https. Open tell-volleyball.vercel.app in Safari or Chrome.' });
      return;
    }
    stopCamera();
    setCam({ state: 'starting', note: 'Asking for the camera…' });
    const attempts: MediaStreamConstraints[] = [
      { audio: false, video: { facingMode: { ideal: face }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } } },
      { audio: false, video: { facingMode: face } },
      { audio: false, video: true },
    ];
    let stream: MediaStream | null = null;
    let lastErr: Error | null = null;
    for (const c of attempts) {
      try {
        stream = await navigator.mediaDevices.getUserMedia(c);
        break;
      } catch (e) {
        lastErr = e as Error;
        if (lastErr.name === 'NotAllowedError') break;
      }
    }
    if (!stream) {
      const name = lastErr?.name;
      setCam({
        state: 'error',
        note:
          name === 'NotAllowedError'
            ? 'Camera access is blocked. On iPhone: Settings > Safari > Camera > Allow (or tap "aA" in the address bar > Website Settings). Then reload.'
            : name === 'NotFoundError'
              ? 'No camera found on this device.'
              : `The camera wouldn't start (${lastErr?.message ?? 'unknown error'}). Close other apps using it and try again.`,
      });
      return;
    }
    streamRef.current = stream;
    const v = videoRef.current!;
    // iOS needs these set on the element itself before play().
    v.muted = true;
    v.setAttribute('muted', '');
    v.setAttribute('playsinline', '');
    v.srcObject = stream;
    try {
      await v.play();
    } catch {
      /* some browsers resolve play late; the loop waits for frames */
    }
    const run = ++runRef.current;
    const cpu = new URLSearchParams(location.search).get('delegate') === 'cpu';
    void loop(run, cpu).catch((e) => setCam({ state: 'error', note: `The pose model didn't load (${(e as Error).message}). Check your connection and try again.` }));
  };

  const onTap = (e: PointerEvent<HTMLCanvasElement>) => {
    const v = videoRef.current;
    if (!v) return;
    const r = e.currentTarget.getBoundingClientRect();
    const p = fromScreen(v, r, 'cover', e.clientX, e.clientY);
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
      lock.current = { c: best.c, lastSeen: performance.now() / 1000, last: best };
      buffer.current.clear();
      detector.current.reset();
      smoothF.current.reset();
      smoothProbs.current = null;
      setLocked(true);
    }
  };

  const startRecording = () => {
    timeline.current = [];
    ballPts.current = [];
    recContact.current = null;
    recBuffer.current = new PoseBuffer(MAX_RECORD_S + 2);
    detector.current.reset();
    chunks.current = [];
    recordingStart.current = performance.now() / 1000;
    const stream = streamRef.current;
    const MR = (window as unknown as { MediaRecorder?: typeof MediaRecorder }).MediaRecorder;
    if (stream && MR) {
      const type = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((t) => MR.isTypeSupported?.(t));
      try {
        const r = new MR(stream, type ? { mimeType: type, videoBitsPerSecond: 4_000_000 } : undefined);
        r.ondataavailable = (ev) => ev.data.size && chunks.current.push(ev.data);
        r.start(250);
        recorder.current = r;
      } catch {
        recorder.current = null;
      }
    }
    setRecording({ started: Date.now() });
  };

  useEffect(() => {
    if (!recording) return;
    const id = window.setInterval(() => setRecElapsed((Date.now() - recording.started) / 1000), 100);
    return () => window.clearInterval(id);
  }, [recording]);

  const closeReview = () => {
    if (review?.url) URL.revokeObjectURL(review.url);
    setReview(null);
    setDraft(null);
    window.setTimeout(() => void start(), 50);
  };

  useEffect(() => {
    if (draft) window.setTimeout(() => document.querySelector('.swing-sheet')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
  }, [draft]);

  const top = probs ? argmax(probs) : -1;
  const modelNote = live
    ? live.source === 'yours'
      ? `Calls come from your own ${live.n} labeled swings.`
      : `Calls come from synthetic demo data until you've labeled 24 real swings (${realUsable} so far).`
    : `Label ${Math.max(0, 24 - realUsable)} more swings and live calls switch on.`;

  if (review) {
    return (
      <RecordReview
        review={review}
        hitterId={hitterId}
        setHitterId={setHitterId}
        players={players}
        build={(t) => buildDraft(review.buffer, t, review.ball)}
        onClose={closeReview}
        onGo={onGo}
      />
    );
  }

  return (
    <div className="screen live">
      <div className={`live__stage ${recording ? 'live__stage--rec' : ''}`}>
        <video ref={videoRef} className="live__video" playsInline muted autoPlay />
        <canvas ref={canvasRef} className="live__canvas" onPointerDown={onTap} aria-label="Camera view. Tap a player to lock on." />
        {cam.state !== 'on' && (
          <div className="live__cover">
            <h1>Point it at a hitter</h1>
            <p>
              Stand behind the end line and hold the phone sideways. Tell boxes the hitter, reads the swing as it happens,
              and catches the moment of contact.
            </p>
            {cam.state === 'error' && <p className="notice notice--bad">{cam.note}</p>}
            {(cam.state === 'starting' || cam.state === 'loading') && <p className="live__status">{cam.note}</p>}
            <button className="btn btn--ball btn--big" onClick={() => void start()} disabled={cam.state === 'starting' || cam.state === 'loading'}>
              {cam.state === 'starting' || cam.state === 'loading' ? 'Starting…' : 'Start camera'}
            </button>
            <p className="live__fine">Video stays on this phone. Nothing is uploaded.</p>
          </div>
        )}
        {cam.state === 'on' && (
          <>
            <div className="live__hud">
              {recording ? (
                <span className="live__pill live__pill--rec">Recording {recElapsed.toFixed(1)}s</span>
              ) : (
                <span className="live__pill live__pill--live">{mode === 'read' ? 'Live' : 'Ready'}</span>
              )}
              <span className="live__pill">{fps} fps</span>
              <span className={`live__pill ${locked ? 'live__pill--lock' : ''}`}>{locked ? 'Hitter locked' : people ? 'Tap the hitter' : 'Looking for players'}</span>
            </div>
            {mode === 'read' && live && probs && locked && (
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
            {mode === 'record' && (
              <div className="live__recbar">
                {recording ? (
                  <button className="recbtn recbtn--on" onClick={finishRecording} aria-label="Stop recording">
                    <span />
                  </button>
                ) : (
                  <button className="recbtn" onClick={startRecording} aria-label="Record a swing">
                    <span />
                  </button>
                )}
                <span className="live__rechint">{recording ? 'Hit the ball. Stops by itself after contact.' : locked ? 'Tap record, then hit' : 'Tap the hitter first'}</span>
              </div>
            )}
          </>
        )}
      </div>

      <div className="live__controls">
        <div className="seg seg--wide" role="tablist" aria-label="Camera mode">
          <button role="tab" aria-selected={mode === 'read'} className={`seg__btn ${mode === 'read' ? 'seg__btn--on' : ''}`} onClick={() => setMode('read')} disabled={!!recording}>
            Live read
          </button>
          <button role="tab" aria-selected={mode === 'record'} className={`seg__btn ${mode === 'record' ? 'seg__btn--on' : ''}`} onClick={() => setMode('record')} disabled={!!recording}>
            Record and review
          </button>
        </div>
        <div className="live__row">
          <span className="field__label">Hitter</span>
          <div className="chips__row">
            {players.length === 0 && <span className="muted small">Add hitters in Film &gt; Roster, or when you save a swing.</span>}
            {players.map((p) => (
              <button key={p.id} className={`chip chip--jersey ${hitterId === p.id ? 'chip--on' : ''}`} onClick={() => setHitterId(p.id)}>
                {p.number ? `#${p.number}` : p.name}
              </button>
            ))}
          </div>
        </div>
        <div className="live__row">
          <span className="field__label">Hitting from</span>
          <div className="seg" role="group" aria-label="Attack position">
            {POSITIONS.map((p) => (
              <button key={p.value} className={`seg__btn ${position === p.value ? 'seg__btn--on' : ''}`} aria-pressed={position === p.value} onClick={() => setPosition(p.value)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
        <p className={`model-note ${live?.source === 'yours' ? 'model-note--yours' : ''}`}>{modelNote}</p>
        {cam.state === 'on' && (
          <div className="row wrap">
            <button className="btn btn--tiny" onClick={() => { const f = facing === 'environment' ? 'user' : 'environment'; setFacing(f); void start(f); }}>
              {facing === 'environment' ? 'Front camera' : 'Back camera'}
            </button>
            <label className="toggle small">
              <input type="checkbox" checked={ballOn} onChange={(e) => setBallOn(e.target.checked)} /> Track the ball
            </label>
            <button className="btn btn--tiny btn--quiet" onClick={stopCamera}>Stop camera</button>
          </div>
        )}
        {cam.state === 'on' && fps > 0 && fps < 10 && !location.search.includes('delegate=cpu') && (
          <p className="notice small">
            Running at {fps} fps, which is too slow to catch a swing. <a href="?delegate=cpu" className="link">Try CPU mode</a>, or turn off ball tracking.
          </p>
        )}
      </div>

      {draft && (
        <SwingSheet
          key={draft.contactT}
          draft={draft}
          hitterId={hitterId}
          setHitterId={setHitterId}
          players={players}
          onClose={() => setDraft(null)}
          onGo={onGo}
          source="live"
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------- review after recording

function RecordReview({
  review,
  hitterId,
  setHitterId,
  players,
  build,
  onClose,
  onGo,
}: {
  review: { url: string | null; timeline: TimelineEntry[]; start: number; ball: BallPoint[]; contactT: number | null };
  hitterId: string | undefined;
  setHitterId: (id: string) => void;
  players: { id: string; name: string; number?: string }[];
  build: (t: number) => SwingDraft | null;
  onClose: () => void;
  onGo: (t: Tab) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [contactT, setContactT] = useState<number | null>(review.contactT);
  const [draft, setDraft] = useState<SwingDraft | null>(() => (review.contactT !== null ? build(review.contactT) : null));
  const [now, setNow] = useState(0);
  const [playing, setPlaying] = useState(false);
  const duration = review.timeline.length ? review.timeline[review.timeline.length - 1].t - review.start : 0;

  // Draw the stored detections in sync with playback.
  useEffect(() => {
    let id = 0;
    const tick = () => {
      const v = videoRef.current;
      const t = review.start + (v ? v.currentTime : now);
      const e = nearest(review.timeline, t);
      if (v && canvasRef.current && e) {
        drawOverlay(canvasRef.current, v, 'contain', {
          others: e.others,
          target: e.p,
          ball: e.ball,
          trail: review.ball.filter((b) => b.t <= t && t - b.t < 0.6),
          ghost: null,
          recording: false,
          contact: contactT !== null && Math.abs(t - contactT) < 0.05,
        });
      }
      if (v) setNow(v.currentTime);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [review, contactT]); // eslint-disable-line react-hooks/exhaustive-deps

  const jumpTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, t);
  };

  const setContactHere = () => {
    const t = review.start + now;
    setContactT(t);
    const d = build(t);
    setDraft(d);
  };

  return (
    <div className="screen review">
      <div className="crumb">
        <button className="btn btn--quiet btn--tiny" onClick={onClose}>Back to camera</button>
        <h1 className="crumb__title">Review the swing</h1>
      </div>
      <div className="review__stage">
        {review.url ? (
          <video
            ref={videoRef}
            src={review.url}
            className="review__video"
            playsInline
            muted
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onLoadedMetadata={() => contactT !== null && jumpTo(contactT - review.start - 0.4)}
          />
        ) : (
          <div className="review__novideo">This browser can't record video, but the skeleton and ball track were saved.</div>
        )}
        <canvas ref={canvasRef} className="review__canvas" />
      </div>
      {review.url && (
        <div className="transport">
          <button className="btn btn--icon" onClick={() => jumpTo(now - 1 / 30)}>−1f</button>
          <button className="btn btn--icon btn--play" onClick={() => { const v = videoRef.current; if (v) { if (v.paused) { v.playbackRate = 0.5; void v.play(); } else v.pause(); } }}>
            {playing ? 'Pause' : 'Play ½×'}
          </button>
          <button className="btn btn--icon" onClick={() => jumpTo(now + 1 / 30)}>+1f</button>
          {contactT !== null && <button className="btn btn--tiny" onClick={() => jumpTo(contactT - review.start)}>Go to contact</button>}
          <span className="transport__time">{now.toFixed(2)}s / {duration.toFixed(1)}s</span>
        </div>
      )}
      {contactT === null ? (
        <div className="notice">
          Tell didn't spot the contact. Step to the frame where the hand meets the ball, then{' '}
          <button className="btn btn--tiny btn--primary" onClick={setContactHere}>Set contact here</button>
        </div>
      ) : !draft ? (
        <div className="notice notice--bad">
          The hitter wasn't tracked for long enough before contact. Make sure they stay in frame for the whole approach, then record again.
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn btn--tiny" onClick={setContactHere}>Use this frame as contact</button>
          </div>
        </div>
      ) : (
        <SwingSheet key={draft.contactT} draft={draft} hitterId={hitterId} setHitterId={setHitterId} players={players} onClose={onClose} onGo={onGo} source="record" />
      )}
    </div>
  );
}

// ---------------------------------------------------------------- labeling + verdict

function SwingSheet({
  draft,
  hitterId,
  setHitterId,
  players,
  onClose,
  onGo,
  source,
}: {
  draft: SwingDraft;
  hitterId: string | undefined;
  setHitterId: (id: string) => void;
  players: { id: string; name: string; number?: string }[];
  onClose: () => void;
  onGo: (t: Tab) => void;
  source: 'live' | 'record';
}) {
  const { individual, update } = useData();
  const [landing, setLanding] = useState<{ x: number; y: number } | undefined>();
  const [manual, setManual] = useState<Direction | undefined>(draft.suggestion?.direction);
  const [newPlayer, setNewPlayer] = useState('');
  const [saved, setSaved] = useState<{ right: boolean | null; went: string; said: string } | null>(null);
  const direction = landing ? directionFromLanding(draft.position, landing) : manual;
  const said = draft.read ? DIRECTIONS[argmax(draft.read.probs)] : null;

  const addPlayer = () => {
    const n = newPlayer.trim();
    if (!n || (individual && players.length)) return;
    const id = individual ? INDIVIDUAL_HITTER : uid('p');
    const isNum = /^\d+$/.test(n);
    update((d) => ({ ...d, players: [...d.players, { id, name: isNum ? `#${n}` : n, number: isNum ? n : undefined, hand: 'R' }] }));
    setHitterId(id);
    setNewPlayer('');
  };

  const save = () => {
    if (!hitterId || !direction) return;
    const day = new Date().toISOString().slice(0, 10);
    const matchId = `live-${day}`;
    const attack: Attack = {
      id: uid('a'),
      matchId,
      hitterId,
      contactTime: draft.contactT,
      position: draft.position,
      landing,
      direction,
      shotType: 'hard',
      outcome: 'unknown',
      frames: draft.frames,
      createdAt: Date.now(),
    };
    update((d) => ({
      ...d,
      matches: d.matches.some((m) => m.id === matchId)
        ? d.matches
        : [{ id: matchId, name: `Camera session ${new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`, date: day, live: true }, ...d.matches],
      attacks: [...d.attacks, attack],
    }));
    setSaved({ right: said ? said === direction : null, went: shotName(draft.position, direction), said: said ? shotName(draft.position, said) : '' });
  };

  if (saved) {
    return (
      <div className={`verdict ${saved.right === false ? 'verdict--miss' : ''}`}>
        <p className="verdict__big">{saved.right === null ? 'Swing saved' : saved.right ? 'Tell read it' : 'Tell missed it'}</p>
        <p className="verdict__sub">
          It went {saved.went.toLowerCase()}{saved.said ? `. Tell said ${saved.said.toLowerCase()}` : ''}. Saved as training data.
        </p>
        <div className="row wrap">
          <button className="btn btn--ball" onClick={onClose}>{source === 'record' ? 'Record the next one' : 'Back to camera'}</button>
          <button className="btn btn--onblue" onClick={() => onGo('results')}>Retrain the model</button>
        </div>
      </div>
    );
  }

  return (
    <div className="sheet swing-sheet" role="dialog" aria-label="Label this swing">
      <div className="sheet__head">
        <h2>Where did it go?</h2>
        <button className="btn btn--quiet btn--tiny" onClick={onClose}>Discard</button>
      </div>
      <div className="swing-calls">
        <div className="swing-call">
          <span className="swing-call__label">Tell's call, 0.1 s before contact</span>
          <span className="swing-call__value">{said ? `${shotName(draft.position, said)} ${pct(Math.max(...draft.read!.probs))}` : 'No model yet'}</span>
        </div>
        <div className="swing-call">
          <span className="swing-call__label">Ball tracker</span>
          <span className="swing-call__value">{draft.suggestion ? shotName(draft.position, draft.suggestion.direction) : 'Lost the ball'}</span>
        </div>
      </div>
      <div className="row wrap">
        {(['left', 'straight', 'right'] as const).map((d) => (
          <button key={d} className={`chip ${!landing && manual === d ? 'chip--on' : ''}`} onClick={() => { setLanding(undefined); setManual(d); }}>
            {shotName(draft.position, d)}
          </button>
        ))}
      </div>
      <Court
        compact
        takeoff={draft.position}
        lanes={draft.read?.probs}
        dots={landing ? [{ ...landing, tone: 'ball', r: 3 }] : []}
        onTap={(p) => { setLanding(p); setManual(undefined); }}
        label={landing ? `Landed: ${shotName(draft.position, directionFromLanding(draft.position, landing))}` : 'Optional: tap the exact landing spot'}
      />
      <fieldset className="chips">
        <legend>Hitter</legend>
        <div className="chips__row">
          {players.map((p) => (
            <button key={p.id} type="button" className={`chip chip--jersey ${hitterId === p.id ? 'chip--on' : ''}`} onClick={() => setHitterId(p.id)}>
              {p.number ? `#${p.number}` : p.name}
            </button>
          ))}
          {(!individual || players.length === 0) && <form className="chip-add" onSubmit={(e) => { e.preventDefault(); addPlayer(); }}>
            <input className="input input--chip" placeholder="New #" value={newPlayer} onChange={(e) => setNewPlayer(e.target.value)} aria-label="Add hitter" />
          </form>}
        </div>
      </fieldset>
      <button className="btn btn--primary btn--big" disabled={!hitterId || !direction} onClick={save}>
        {!hitterId ? 'Pick the hitter to save' : !direction ? 'Pick where it went' : `Save: ${shotName(draft.position, direction).toLowerCase()}`}
      </button>
    </div>
  );
}

// ---------------------------------------------------------------- drawing

function argmax(p: number[]) {
  let b = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i;
  return b;
}

function nearest(tl: TimelineEntry[], t: number): TimelineEntry | null {
  if (!tl.length) return null;
  let lo = 0;
  let hi = tl.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (tl[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  const a = tl[Math.max(0, lo - 1)];
  const b = tl[lo];
  return Math.abs(a.t - t) < Math.abs(b.t - t) ? a : b;
}

type Fit = 'cover' | 'contain';
function videoBox(v: HTMLVideoElement, r: { width: number; height: number }, fit: Fit) {
  const vw = v.videoWidth || r.width;
  const vh = v.videoHeight || r.height;
  const scale = fit === 'cover' ? Math.max(r.width / vw, r.height / vh) : Math.min(r.width / vw, r.height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  return { ox: (r.width - dw) / 2, oy: (r.height - dh) / 2, dw, dh, aspect: vh / vw };
}

function fromScreen(v: HTMLVideoElement, r: DOMRect, fit: Fit, cx: number, cy: number) {
  const b = videoBox(v, r, fit);
  return { x: (cx - r.left - b.ox) / b.dw, y: (cy - r.top - b.oy) / b.dh };
}

interface OverlayInput {
  others: number[][];
  target: number[] | null;
  ball: BallBox | null;
  trail: BallPoint[];
  ghost: number[][] | null;
  recording: boolean;
  contact?: boolean;
}

function drawOverlay(c: HTMLCanvasElement | null, v: HTMLVideoElement, fit: Fit, o: OverlayInput) {
  if (!c) return;
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(r.width * dpr) || c.height !== Math.round(r.height * dpr)) {
    c.width = Math.round(r.width * dpr);
    c.height = Math.round(r.height * dpr);
  }
  const g = c.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, r.width, r.height);
  const b = videoBox(v, r, fit);
  const P = (x: number, y: number): [number, number] => [b.ox + x * b.dw, b.oy + y * b.dh];
  const bones = (p: number[], color: string, width: number, dash: number[] = []) => {
    g.strokeStyle = color;
    g.lineWidth = width;
    g.lineCap = 'round';
    g.lineJoin = 'round';
    g.setLineDash(dash);
    g.beginPath();
    for (const [i, j] of BONES) {
      if (p[i * 3 + 2] < 0.3 || p[j * 3 + 2] < 0.3) continue;
      const [x1, y1] = P(p[i * 3], p[i * 3 + 1]);
      const [x2, y2] = P(p[j * 3], p[j * 3 + 1]);
      g.moveTo(x1, y1);
      g.lineTo(x2, y2);
    }
    g.stroke();
    g.setLineDash([]);
  };
  // Everyone else: thin white boxes, so you can see who Tell can lock onto.
  for (const p of o.others) {
    const bx = poseBox(p);
    if (!bx) continue;
    const [x0, y0] = P(bx.x0, bx.y0);
    const [x1, y1] = P(bx.x1, bx.y1);
    g.strokeStyle = 'rgba(255,255,255,0.55)';
    g.lineWidth = 1.5;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
  if (o.target) {
    const bx = poseBox(o.target);
    if (bx) {
      const [x0, y0] = P(bx.x0, bx.y0);
      const [x1, y1] = P(bx.x1, bx.y1);
      const L = Math.min(22, (x1 - x0) / 3);
      g.strokeStyle = o.contact ? YELLOW : RED;
      g.lineWidth = 3;
      g.beginPath();
      for (const [cx, cy, sx, sy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
        g.moveTo(cx, cy + sy * L);
        g.lineTo(cx, cy);
        g.lineTo(cx + sx * L, cy);
      }
      g.stroke();
      g.strokeStyle = o.contact ? 'rgba(242,194,48,0.5)' : 'rgba(229,56,59,0.45)';
      g.lineWidth = 1.5;
      g.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const label = o.contact ? 'Contact' : 'Hitter';
      g.font = '600 12px Barlow, system-ui, sans-serif';
      const tw = g.measureText(label).width + 12;
      g.fillStyle = o.contact ? YELLOW : RED;
      g.fillRect(x0, y0 - 20, tw, 18);
      g.fillStyle = o.contact ? '#14213d' : '#fff';
      g.fillText(label, x0 + 6, y0 - 7);
    }
    bones(o.target, YELLOW, 3.5);
    // Hitting arm emphasized.
    g.strokeStyle = RED;
    g.lineWidth = 4;
    g.beginPath();
    const arm = [12, 14, 16];
    arm.forEach((i, k) => {
      const [x, y] = P(o.target![i * 3], o.target![i * 3 + 1]);
      if (k === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
  }
  if (o.ghost && o.target) {
    const p = o.target;
    const torsoImg = Math.abs((p[23 * 3 + 1] + p[24 * 3 + 1]) / 2 - (p[11 * 3 + 1] + p[12 * 3 + 1]) / 2) || 0.05;
    const hip0 = [(p[23 * 3] + p[24 * 3]) / 2, (p[23 * 3 + 1] + p[24 * 3 + 1]) / 2];
    const startHip = [o.ghost[0][39], o.ghost[0][40]];
    const order = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
    const s = o.ghost[o.ghost.length - 1];
    const hx = hip0[0] + (s[39] - startHip[0]);
    const hy = hip0[1] + (s[40] - startHip[1]);
    const img = new Array(99).fill(0);
    order.forEach((lmIdx, j) => {
      img[lmIdx * 3] = hx + s[j * 3] * torsoImg * b.aspect;
      img[lmIdx * 3 + 1] = hy + s[j * 3 + 1] * torsoImg;
      img[lmIdx * 3 + 2] = 1;
    });
    g.globalAlpha = 0.85;
    bones(img, '#ffffff', 2.5, [6, 5]);
    g.globalAlpha = 1;
  }
  // Ball: trail plus a red box.
  if (o.trail.length > 1) {
    g.strokeStyle = 'rgba(242,194,48,0.85)';
    g.lineWidth = 3;
    g.beginPath();
    o.trail.forEach((pt, i) => {
      const [x, y] = P(pt.x, pt.y);
      if (i === 0) g.moveTo(x, y);
      else g.lineTo(x, y);
    });
    g.stroke();
  }
  if (o.ball) {
    const [cx, cy] = P(o.ball.x, o.ball.y);
    const w = Math.max(14, o.ball.w * b.dw) + 8;
    const h = Math.max(14, o.ball.h * b.dh) + 8;
    g.strokeStyle = RED;
    g.lineWidth = 2.5;
    g.strokeRect(cx - w / 2, cy - h / 2, w, h);
    g.font = '600 11px Barlow, system-ui, sans-serif';
    g.fillStyle = RED;
    g.fillRect(cx - w / 2, cy - h / 2 - 16, 32, 15);
    g.fillStyle = '#fff';
    g.fillText('Ball', cx - w / 2 + 5, cy - h / 2 - 5);
  }
  if (o.recording) {
    g.strokeStyle = RED;
    g.lineWidth = 6;
    g.strokeRect(3, 3, r.width - 6, r.height - 6);
  }
}
