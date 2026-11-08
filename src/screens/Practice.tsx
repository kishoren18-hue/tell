import { useEffect, useMemo, useRef, useState } from 'react';
import type { Tab } from '../App';
import { Court } from '../components/Court';
import { Skeleton } from '../components/Skeleton';
import { pct } from '../components/ui';
import { shotName } from '../lib/court';
import { cutoffIndex, sequenceFeatures } from '../lib/features';
import { fitLive, readAttack, type LiveModels, type LiveRead } from '../lib/ml/tell';
import { stateOf, FORECAST_STEPS } from '../lib/ml/world';
import { Rng } from '../lib/rng';
import { EMPTY_PRACTICE, loadPractice, savePractice, type PracticeStats } from '../lib/store';
import type { Attack, Dataset, Direction, Player } from '../lib/types';
import { DIRECTIONS } from '../lib/types';
import { useData } from '../state';

// Practice mode for liberos and blockers: watch the hitter's skeleton up to a
// cutoff, call the shot, then see what happened and what the model thought.
// Gets harder (earlier cutoff) as you get it right.

type Phase = 'playing' | 'guess' | 'reveal';
const LEVELS = [0, 100, 200, 300, 400];

export function Practice({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, workspace } = useData();
  const playable = useMemo(
    () => dataset.attacks.filter((a) => a.frames && a.frames.filter(Boolean).length >= 23),
    [dataset.attacks],
  );
  const [stats, setStats] = useState<PracticeStats>(EMPTY_PRACTICE);
  const [hitter, setHitter] = useState<string>('all');
  const [round, setRound] = useState(0);
  const [phase, setPhase] = useState<Phase>('playing');
  const [frame, setFrame] = useState(0);
  const [guess, setGuess] = useState<Direction | null>(null);
  const [streak, setStreak] = useState(0);
  const [misses, setMisses] = useState(0);
  const [slow, setSlow] = useState(false);
  const cache = useRef(new Map<string, LiveModels | null>());
  const rng = useRef(new Rng(Date.now() % 100000));

  useEffect(() => {
    void loadPractice(workspace).then(setStats);
  }, [workspace]);

  const pool = hitter === 'all' ? playable : playable.filter((a) => a.hitterId === hitter);
  const attack: Attack | undefined = useMemo(() => (pool.length ? pool[rng.current.int(pool.length)] : undefined), [round, pool.length, hitter]); // eslint-disable-line react-hooks/exhaustive-deps
  const player: Player | undefined = dataset.players.find((p) => p.id === attack?.hitterId);
  const hand = player?.hand ?? 'R';
  const level = stats.horizonMs;
  const cut = cutoffIndex(level);

  // Display states in the real (unmirrored) frame.
  const states = useMemo(() => {
    if (!attack?.frames) return [];
    const seq = sequenceFeatures(attack.frames, 'R');
    return seq ? seq.map(stateOf) : [];
  }, [attack]);

  // The model's read comes from models trained without this attack's match, so it is a fair opponent.
  const read: LiveRead | null = useMemo(() => {
    if (!attack) return null;
    const key = `${attack.matchId}:${level}`;
    if (!cache.current.has(key)) {
      const others: Dataset = { ...dataset, attacks: dataset.attacks.filter((a) => a.matchId !== attack.matchId) };
      cache.current.set(key, fitLive(others, level));
    }
    const m = cache.current.get(key);
    return m ? readAttack(m, attack, hand, level) : null;
  }, [attack, level, dataset, hand]);

  // Playback.
  useEffect(() => {
    if (!states.length) return;
    setFrame(0);
    setPhase('playing');
    setGuess(null);
    let f = 0;
    const fps = slow ? 12 : 30;
    const id = window.setInterval(() => {
      f++;
      if (f >= cut) {
        setFrame(cut);
        setPhase('guess');
        window.clearInterval(id);
        return;
      }
      setFrame(f);
    }, 1000 / fps);
    return () => window.clearInterval(id);
  }, [states, cut, slow, round]);

  // After the guess, play out to the end so you see the swing.
  useEffect(() => {
    if (phase !== 'reveal' || !states.length) return;
    let f = cut;
    const id = window.setInterval(() => {
      f++;
      if (f >= states.length) {
        window.clearInterval(id);
        return;
      }
      setFrame(f);
    }, 1000 / 20);
    return () => window.clearInterval(id);
  }, [phase, cut, states.length]);

  if (playable.length < 1) {
    return (
      <div className="screen">
        <div className="intro">
          <h1>Practice reading hitters</h1>
          <p>Watch a hitter's swing, freeze before contact, and call the shot. You'll need some tagged attacks with pose first.</p>
        </div>
        <button className="btn btn--primary" onClick={() => onGo('film')}>Tag attacks or load demo data</button>
      </div>
    );
  }

  const answerLane = attack ? DIRECTIONS.indexOf(attack.direction) : 1;
  const modelGuess = read ? DIRECTIONS[read.probs.indexOf(Math.max(...read.probs))] : null;

  const submit = (lane: number) => {
    if (phase !== 'guess' || !attack) return;
    const g = DIRECTIONS[lane];
    setGuess(g);
    setPhase('reveal');
    const right = g === attack.direction;
    const modelRight = modelGuess === attack.direction;
    const nextStreak = right ? streak + 1 : 0;
    const nextMiss = right ? 0 : misses + 1;
    let horizonMs = level;
    let s = nextStreak;
    let m = nextMiss;
    if (nextStreak >= 3) {
      horizonMs = LEVELS[Math.min(LEVELS.length - 1, LEVELS.indexOf(level) + 1)];
      s = 0;
    } else if (nextMiss >= 2) {
      horizonMs = LEVELS[Math.max(0, LEVELS.indexOf(level) - 1)];
      m = 0;
    }
    setStreak(s);
    setMisses(m);
    const next: PracticeStats = {
      attempts: stats.attempts + 1,
      correct: stats.correct + (right ? 1 : 0),
      modelCorrect: stats.modelCorrect + (modelRight ? 1 : 0),
      modelAttempts: (stats.modelAttempts ?? 0) + (modelGuess ? 1 : 0),
      bestStreak: Math.max(stats.bestStreak, nextStreak),
      horizonMs,
    };
    setStats(next);
    void savePractice(workspace, next);
  };

  const hitterName = player ? (player.number ? `#${player.number}` : player.name) : '';
  const hitters = dataset.players.filter((p) => playable.some((a) => a.hitterId === p.id));

  return (
    <div className="screen">
      <div className="practice-head">
        <div>
          <h1 className="h-tight">Call the shot</h1>
          <p className="muted small">
            Freezes {level === 0 ? 'at contact' : `${level} ms before contact`}. Three right in a row and it freezes earlier.
          </p>
        </div>
        <select className="input input--select" value={hitter} onChange={(e) => { setHitter(e.target.value); setRound((r) => r + 1); }} aria-label="Hitter to practice against">
          <option value="all">All hitters</option>
          {hitters.map((p) => (
            <option key={p.id} value={p.id}>{p.number ? `#${p.number}` : p.name}</option>
          ))}
        </select>
      </div>

      <div className="stage">
        <div className="stage__meta">
          <span>{hitterName}</span>
          <span>{attack ? positionLabel(attack.position) : ''}</span>
          <span>{phase === 'playing' ? 'Watching…' : phase === 'guess' ? 'Your call' : guess === attack?.direction ? 'You read it' : 'Missed it'}</span>
        </div>
        {states.length > 0 && (
          <Skeleton
            states={states}
            frame={frame}
            hand={hand}
            ghost={phase === 'reveal' ? read?.ghost : null}
            showGhostUntil={phase === 'reveal' ? Math.max(1, Math.min(FORECAST_STEPS, frame - cut)) : 0}
          />
        )}
      </div>

      {attack && (
        <Court
          takeoff={attack.position}
          onLane={phase === 'guess' ? submit : undefined}
          lanes={phase === 'reveal' && read ? read.probs : undefined}
          laneOutline={phase === 'reveal' ? answerLane : undefined}
          dots={phase === 'reveal' && attack.landing ? [{ ...attack.landing, tone: 'ball', r: 3.2 }] : []}
          label={
            phase === 'guess'
              ? 'Tap the lane you think the ball goes down'
              : phase === 'reveal'
                ? `It went ${shotName(attack.position, attack.direction).toLowerCase()}. You said ${guess ? shotName(attack.position, guess).toLowerCase() : '…'}.${modelGuess ? ` Tell said ${shotName(attack.position, modelGuess).toLowerCase()} (${pct(Math.max(...read!.probs))} sure).` : ''}`
                : 'Watch the approach'
          }
        />
      )}

      <div className="row row--between">
        <label className="toggle">
          <input type="checkbox" checked={slow} onChange={(e) => setSlow(e.target.checked)} /> Slow motion
        </label>
        <button className="btn btn--primary" onClick={() => setRound((r) => r + 1)} disabled={phase === 'playing'}>
          {phase === 'reveal' ? 'Next hitter' : 'Skip'}
        </button>
      </div>

      <div className="stats">
        <div className="stat stat--strong">
          <span className="stat__v">{stats.attempts ? pct(stats.correct / stats.attempts) : '–'}</span>
          <span className="stat__l">You, over {stats.attempts} calls</span>
        </div>
        <div className="stat">
          <span className="stat__v">{stats.modelAttempts ? pct(stats.modelCorrect / stats.modelAttempts) : '–'}</span>
          <span className="stat__l">{stats.modelAttempts ? 'Tell, same attacks' : 'Tell needs a second match to compete'}</span>
        </div>
        <div className="stat">
          <span className="stat__v">{stats.bestStreak}</span>
          <span className="stat__l">Best streak</span>
        </div>
      </div>
      {phase === 'reveal' && read && <p className="muted small">Yellow dashed figure: Tell's forecast of the next half second of the swing.</p>}
      <button className="btn btn--quiet btn--tiny" onClick={() => { setStats(EMPTY_PRACTICE); void savePractice(workspace, EMPTY_PRACTICE); }}>
        Reset practice stats
      </button>
    </div>
  );
}

function positionLabel(p: Attack['position']) {
  return { left: 'Left side', middle: 'Middle', right: 'Right side', 'back-left': 'Back left', 'back-middle': 'Pipe', 'back-right': 'Back right' }[p];
}
