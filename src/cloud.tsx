import type { User } from '@supabase/supabase-js';
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { createTeam, joinTeam, leaveTeam, listTeams, supabase, syncTeam, type Team } from './lib/sync';
import { useData } from './state';

interface Cloud {
  user: User | null;
  teams: Team[];
  team: Team | null;
  setTeam: (id: string | null) => void;
  status: 'idle' | 'syncing' | 'error' | 'offline';
  message: string;
  lastSync: number | null;
  syncNow: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, name: string) => Promise<'confirm' | 'done'>;
  signOut: () => Promise<void>;
  create: (name: string) => Promise<Team>;
  join: (code: string) => Promise<Team>;
  leave: (id: string) => Promise<void>;
}

const C = createContext<Cloud | null>(null);
const TEAM_KEY = 'tell:active-team';

function ls(k: string, v?: string | null) {
  try {
    if (v === undefined) return localStorage.getItem(k);
    if (v === null) localStorage.removeItem(k);
    else localStorage.setItem(k, v);
  } catch {
    /* storage unavailable */
  }
  return null;
}

export function CloudProvider({ children }: { children: ReactNode }) {
  const { dataset, replace, ready } = useData();
  const [user, setUser] = useState<User | null>(null);
  const [teams, setTeams] = useState<Team[]>([]);
  const [teamId, setTeamId] = useState<string | null>(() => ls(TEAM_KEY));
  const [status, setStatus] = useState<Cloud['status']>('idle');
  const [message, setMessage] = useState('');
  const [lastSync, setLastSync] = useState<number | null>(null);
  const busy = useRef(false);
  const latest = useRef(dataset);
  latest.current = dataset;

  useEffect(() => {
    const sb = supabase();
    void sb.auth.getSession().then(({ data }) => setUser(data.session?.user ?? null));
    const { data } = sb.auth.onAuthStateChange((_e, session) => setUser(session?.user ?? null));
    return () => data.subscription.unsubscribe();
  }, []);

  const refreshTeams = useCallback(async () => {
    if (!user) {
      setTeams([]);
      return;
    }
    try {
      const t = await listTeams();
      setTeams(t);
      if (t.length && !t.some((x) => x.id === teamId)) {
        setTeamId(t[0].id);
        ls(TEAM_KEY, t[0].id);
      }
    } catch (e) {
      setStatus('error');
      setMessage((e as Error).message);
    }
  }, [user, teamId]);

  useEffect(() => {
    void refreshTeams();
  }, [refreshTeams]);

  const team = teams.find((t) => t.id === teamId) ?? null;

  const syncNow = useCallback(async () => {
    if (!user || !team || busy.current) return;
    if (!navigator.onLine) {
      setStatus('offline');
      setMessage('Offline. Your swings are saved on this phone and will sync later.');
      return;
    }
    busy.current = true;
    setStatus('syncing');
    try {
      const r = await syncTeam(team.id, latest.current, (m) => setMessage(m));
      if (r.pulled > 0) replace(r.dataset);
      setLastSync(Date.now());
      setStatus('idle');
      setMessage(
        r.pulled > 0 || r.pushed > 0
          ? `Synced: ${Math.max(0, r.pulled)} new from the team, ${r.pushed} uploaded.`
          : 'Up to date.',
      );
    } catch (e) {
      setStatus('error');
      setMessage(`Sync failed: ${(e as Error).message}`);
    } finally {
      busy.current = false;
    }
  }, [user, team, replace]);

  // Sync shortly after any change, and when the app opens.
  useEffect(() => {
    if (!ready || !user || !team) return;
    const id = window.setTimeout(() => void syncNow(), 4000);
    return () => window.clearTimeout(id);
  }, [dataset, ready, user, team, syncNow]);

  const value: Cloud = {
    user,
    teams,
    team,
    setTeam: (id) => {
      setTeamId(id);
      ls(TEAM_KEY, id);
    },
    status,
    message,
    lastSync,
    syncNow,
    signIn: async (email, password) => {
      const { error } = await supabase().auth.signInWithPassword({ email, password });
      if (error) throw new Error(error.message === 'Invalid login credentials' ? 'Wrong email or password.' : error.message);
    },
    signUp: async (email, password, name) => {
      const { data, error } = await supabase().auth.signUp({
        email,
        password,
        options: { data: { name }, emailRedirectTo: `${location.origin}/?signed-up=1` },
      });
      if (error) throw new Error(error.message);
      return data.session ? 'done' : 'confirm';
    },
    signOut: async () => {
      await supabase().auth.signOut();
      setTeams([]);
    },
    create: async (name) => {
      const t = await createTeam(name, (user?.user_metadata?.name as string) || undefined);
      await refreshTeams();
      setTeamId(t.id);
      ls(TEAM_KEY, t.id);
      return t;
    },
    join: async (code) => {
      const t = await joinTeam(code, (user?.user_metadata?.name as string) || undefined);
      await refreshTeams();
      setTeamId(t.id);
      ls(TEAM_KEY, t.id);
      return t;
    },
    leave: async (id) => {
      await leaveTeam(id);
      if (teamId === id) {
        setTeamId(null);
        ls(TEAM_KEY, null);
      }
      await refreshTeams();
    },
  };
  return <C.Provider value={value}>{children}</C.Provider>;
}

export function useCloud(): Cloud {
  const c = useContext(C);
  if (!c) throw new Error('useCloud outside CloudProvider');
  return c;
}
