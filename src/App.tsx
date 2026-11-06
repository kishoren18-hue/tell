import { useState } from 'react';
import { DataProvider, useData } from './state';
import { Film } from './screens/Film';
import { Tag } from './screens/Tag';
import { Results } from './screens/Results';
import { Hitters } from './screens/Hitters';
import { Practice } from './screens/Practice';
import { Live } from './screens/Live';

type Tab = 'film' | 'live' | 'results' | 'hitters' | 'practice';

const TABS: { id: Tab; label: string }[] = [
  { id: 'film', label: 'Film' },
  { id: 'live', label: 'Live' },
  { id: 'results', label: 'Model' },
  { id: 'hitters', label: 'Hitters' },
  { id: 'practice', label: 'Practice' },
];

function Shell() {
  const { ready } = useData();
  const [tab, setTab] = useState<Tab>('film');
  const [matchId, setMatchId] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="topbar">
        <button className="wordmark" onClick={() => { setTab('film'); setMatchId(null); }} aria-label="Tell home">
          Tell
        </button>
        <span className="topbar__tag">How readable is your spike?</span>
      </header>
      <main className={`main ${tab === 'live' ? 'main--wide' : ''}`} id="main">
        {!ready ? (
          <p className="muted pad">Loading your matches…</p>
        ) : tab === 'film' ? (
          matchId ? <Tag matchId={matchId} onBack={() => setMatchId(null)} /> : <Film onOpen={setMatchId} onGo={setTab} />
        ) : tab === 'live' ? (
          <Live onGo={setTab} />
        ) : tab === 'results' ? (
          <Results onGo={setTab} />
        ) : tab === 'hitters' ? (
          <Hitters onGo={setTab} />
        ) : (
          <Practice onGo={setTab} />
        )}
      </main>
      <nav className="tabbar" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tabbar__btn ${tab === t.id ? 'tabbar__btn--on' : ''}`}
            aria-current={tab === t.id ? 'page' : undefined}
            onClick={() => setTab(t.id)}
          >
            <TabIcon id={t.id} />
            <span>{t.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

function TabIcon({ id }: { id: Tab }) {
  const common = { width: 22, height: 22, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true };
  if (id === 'film')
    return (
      <svg {...common}>
        <rect x="3" y="6" width="13" height="12" rx="2" />
        <path d="M16 10l5-3v10l-5-3" />
      </svg>
    );
  if (id === 'live')
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="3.2" />
        <path d="M5.6 5.6a9 9 0 0 0 0 12.8M18.4 5.6a9 9 0 0 1 0 12.8M8.5 8.5a5 5 0 0 0 0 7M15.5 8.5a5 5 0 0 1 0 7" />
      </svg>
    );
  if (id === 'results')
    return (
      <svg {...common}>
        <path d="M4 19V5M4 19h16" />
        <path d="M7 15l4-4 3 3 5-6" />
      </svg>
    );
  if (id === 'hitters')
    return (
      <svg {...common}>
        <path d="M8 4l-4 3 2 3 2-1v11h8V9l2 1 2-3-4-3h-2a2 2 0 0 1-4 0z" />
      </svg>
    );
  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 4c2 3 2 13 0 16M4.5 9c4 1 11 1 15 0M5 15.5c4-1.2 10-1.2 14 0" />
    </svg>
  );
}

export default function App() {
  return (
    <DataProvider>
      <Shell />
    </DataProvider>
  );
}

export type { Tab };
