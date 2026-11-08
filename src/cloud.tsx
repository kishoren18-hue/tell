import type { User } from '@supabase/supabase-js';
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { createTeam, joinTeam, leaveTeam, listTeams, supabase, syncWorkspace, type Team } from './lib/sync';
import { DataProvider, useData } from './state';
import { loadDataset } from './lib/store';

export type Mode = 'individual' | 'team' | 'legacy';

interface Cloud {
  user: User | null;
  mode: Mode | null;
  setMode: (mode: Mode | null) => void;
  hasLegacy: boolean;
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
const MODE_KEY = 'tell:mode';

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
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [memberships, setMemberships] = useState<{ owner: string; teams: Team[] } | null>(null);
  const [teamId, setTeamId] = useState<string | null>(() => ls(TEAM_KEY));
  const [mode, setMode] = useState<Mode | null>(() => {
    const saved = ls(MODE_KEY);
    return saved === 'individual' || saved === 'team' || saved === 'legacy' ? saved : null;
  });
  const [hasLegacy, setHasLegacy] = useState(false);
  const [teamError, setTeamError] = useState<{ owner: string; message: string } | null>(null);
  const userId = user?.id;

  useEffect(() => {
    const { data } = supabase().auth.onAuthStateChange((_e, session) => {
      setUser(session?.user ?? null);
      setAuthReady(true);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    void loadDataset('legacy').then((d) => {
      setHasLegacy(Boolean(d.players.length || d.matches.length || d.attacks.length));
    }).catch(() => {});
  }, []);

  const refreshTeams = useCallback(async () => {
    if (!userId) return;
    const teams = await listTeams();
    setMemberships({ owner: userId, teams });
    setTeamError(null);
  }, [userId]);

  useEffect(() => {
    if (mode === 'team' && userId) void refreshTeams().catch((e: Error) => setTeamError({ owner: userId, message: e.message }));
  }, [mode, userId, refreshTeams]);

  const teams = memberships?.owner === userId ? memberships?.teams ?? [] : [];
  const team = mode === 'team' ? teams.find((t) => t.id === teamId) ?? null : null;
  const workspace = mode === 'individual' ? (userId ? `personal:${userId}` : 'guest')
    : mode === 'legacy' ? 'legacy'
    : team ? `team:${userId}:${team.id}` : 'welcome';

  const selectTeam = (id: string | null) => {
    setTeamId(id);
    ls(TEAM_KEY, id);
  };
  const selectMode = (next: Mode | null) => {
    setMode(next);
    ls(MODE_KEY, next);
  };

  const error = mode === 'team' && teamError?.owner === userId ? teamError?.message ?? '' : '';
  const value: Cloud = {
    user, teams, team, mode, hasLegacy,
    setMode: selectMode,
    setTeam: selectTeam,
    status: error ? 'error' : 'idle',
    message: error,
    lastSync: null,
    syncNow: async () => {},
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
      const { error } = await supabase().auth.signOut();
      if (error) throw new Error(error.message);
      setMemberships(null);
      selectMode(null);
    },
    create: async (name) => {
      const t = await createTeam(name, (user?.user_metadata?.name as string) || undefined);
      await refreshTeams();
      selectTeam(t.id);
      return t;
    },
    join: async (code) => {
      const t = await joinTeam(code, (user?.user_metadata?.name as string) || undefined);
      await refreshTeams();
      selectTeam(t.id);
      return t;
    },
    leave: async (id) => {
      await leaveTeam(id);
      if (teamId === id) {
        selectTeam(null);
      }
      await refreshTeams();
    },
  };
  if (!authReady) return <p className="muted pad">Opening Tell…</p>;
  return (
    <DataProvider key={workspace} workspace={workspace}>
      <WorkspaceCloud value={value}>{children}</WorkspaceCloud>
    </DataProvider>
  );
}

const activeSyncs = new Set<string>();

function WorkspaceCloud({ value, children }: { value: Cloud; children: ReactNode }) {
  const { dataset, ready, workspace, applySync, getDataset } = useData();
  const [status, setStatus] = useState<Cloud['status']>('idle');
  const [message, setMessage] = useState('');
  const [lastSync, setLastSync] = useState<number | null>(null);
  const mounted = useRef(false);
  const userId = value.user?.id;
  const teamId = value.team?.id;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const syncNow = useCallback(async () => {
    if (!ready || !userId || (!teamId && !workspace.startsWith('personal:')) || activeSyncs.has(workspace)) return;
    if (!navigator.onLine) {
      setStatus('offline');
      setMessage('Offline. Swings are saved on this device and will sync when you reconnect.');
      return;
    }
    activeSyncs.add(workspace);
    setStatus('syncing');
    const before = getDataset();
    try {
      const result = await syncWorkspace(teamId ? { teamId } : { userId }, workspace, before, (msg) => {
        if (mounted.current) setMessage(msg);
      });
      if (!mounted.current) return;
      await applySync(before, result.dataset, result.key, result.state);
      setLastSync(Date.now());
      setStatus('idle');
      setMessage('Up to date.');
    } catch (e) {
      if (!mounted.current) return;
      setStatus('error');
      setMessage(`Sync failed: ${(e as Error).message}`);
    } finally {
      activeSyncs.delete(workspace);
    }
  }, [ready, userId, teamId, workspace, applySync, getDataset]);

  useEffect(() => {
    const timer = window.setTimeout(() => void syncNow(), 1500);
    return () => window.clearTimeout(timer);
  }, [dataset, syncNow]);

  useEffect(() => {
    if (!ready || !userId || (!teamId && !workspace.startsWith('personal:'))) return;
    const refresh = () => void syncNow();
    window.addEventListener('online', refresh);
    window.addEventListener('focus', refresh);
    const timer = window.setInterval(refresh, 30000);
    return () => {
      window.removeEventListener('online', refresh);
      window.removeEventListener('focus', refresh);
      window.clearInterval(timer);
    };
  }, [ready, userId, teamId, workspace, syncNow]);

  return <C.Provider value={{ ...value, status: value.message ? value.status : status, message: value.message || message, lastSync, syncNow }}>{children}</C.Provider>;
}

export function useCloud(): Cloud {
  const c = useContext(C);
  if (!c) throw new Error('useCloud outside CloudProvider');
  return c;
}
