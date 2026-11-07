// Cloud sync with Supabase. The app stays local-first: everything works
// offline in IndexedDB, and when you're signed in and on a team, players,
// matches and attacks sync to the team so phones and laptops share one dataset.
//
// Deletions are detected by diffing against the set of ids that existed after
// the last sync, so the rest of the app doesn't need to know sync exists.

import { createClient, type SupabaseClient } from '@supabase/supabase-js';
import { get } from 'idb-keyval';
import type { Attack, Dataset, Match, Player } from './types';

const URL = import.meta.env.VITE_SUPABASE_URL ?? 'https://fjjgdhezagebqliomjro.supabase.co';
// Publishable key: safe to ship in the browser; row-level security does the protecting.
const KEY = import.meta.env.VITE_SUPABASE_KEY ?? 'sb_publishable_wav5rZtmtMpGVrvJdziwQw__dZJf_kD';

let client: SupabaseClient | null = null;
export function supabase(): SupabaseClient {
  if (!client) client = createClient(URL, KEY, { auth: { persistSession: true, autoRefreshToken: true, storageKey: 'tell-auth' } });
  return client;
}

export interface Team {
  id: string;
  name: string;
  join_code: string;
  role?: 'owner' | 'member';
}

export async function listTeams(): Promise<Team[]> {
  const sb = supabase();
  const { data: auth } = await sb.auth.getUser();
  if (!auth.user) return [];
  const { data, error } = await sb.from('tell_members').select('role, tell_teams(id, name, join_code)').eq('user_id', auth.user.id);
  if (error) throw new Error(error.message);
  return (data ?? [])
    .map((r) => {
      const t = r.tell_teams as unknown as Team | null;
      return t ? { ...t, role: r.role as Team['role'] } : null;
    })
    .filter(Boolean) as Team[];
}

export async function createTeam(name: string, memberName?: string): Promise<Team> {
  const { data, error } = await supabase().rpc('tell_create_team', { team_name: name, member_name: memberName ?? null });
  if (error) throw new Error(error.message);
  return { ...(data as Team), role: 'owner' };
}

export async function joinTeam(code: string, memberName?: string): Promise<Team> {
  const { data, error } = await supabase().rpc('tell_join_team', { code, member_name: memberName ?? null });
  if (error) throw new Error(error.message.includes('No team') ? 'No team has that code. Check it with whoever shared it.' : error.message);
  return { ...(data as Team), role: 'member' };
}

export async function leaveTeam(teamId: string): Promise<void> {
  const sb = supabase();
  const { data: auth } = await sb.auth.getUser();
  const { error } = await sb.from('tell_members').delete().eq('team_id', teamId).eq('user_id', auth.user!.id);
  if (error) throw new Error(error.message);
}

type Kind = 'player' | 'match' | 'attack';
type Item = Player | Match | Attack;

interface SyncState {
  lastPull: string; // ISO time of the newest row seen
  hashes: Record<string, string>; // key -> hash of what the server has
  known: string[]; // keys present after the last sync
}

export type SyncScope = { teamId: string } | { userId: string };
const keyOf = (kind: Kind, id: string) => `${kind}:${id}`;

/** Round floats so an attack's pose payload is about half the size on the wire. */
function compact<T>(x: T): T {
  if (typeof x === 'number') return (Math.round(x * 1e4) / 1e4) as T;
  if (Array.isArray(x)) return x.map(compact) as T;
  if (x && typeof x === 'object') return Object.fromEntries(Object.entries(x).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, compact(v)])) as T;
  return x;
}

function hash(s: string): string {
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 2654435761);
    h2 = Math.imul(h2 ^ c, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
}

function localItems(ds: Dataset): Map<string, { kind: Kind; item: Item; json: string }> {
  const m = new Map<string, { kind: Kind; item: Item; json: string }>();
  const add = (kind: Kind, item: Item) => {
    if (item.id.startsWith('demo-')) return; // demo data stays on the device
    const c = compact(item);
    m.set(keyOf(kind, item.id), { kind, item: c, json: JSON.stringify(c) });
  };
  ds.players.forEach((p) => add('player', p));
  ds.matches.forEach((x) => add('match', x));
  ds.attacks.forEach((a) => add('attack', a));
  return m;
}

export interface SyncResult {
  dataset: Dataset;
  pulled: number;
  pushed: number;
  deleted: number;
  key: string;
  state: SyncState;
}

/** Sync one workspace. The caller saves the merged data and cursor together. */
export async function syncWorkspace(scope: SyncScope, workspace: string, ds: Dataset, onProgress?: (msg: string) => void): Promise<SyncResult> {
  const sb = supabase();
  const personal = 'userId' in scope;
  const tableName = personal ? 'tell_personal_items' : 'tell_items';
  const ownerColumn = personal ? 'user_id' : 'team_id';
  const ownerId = personal ? scope.userId : scope.teamId;
  const key = `tell:sync:${workspace}`;
  const st: SyncState = ((await get(key)) as SyncState | undefined) ?? { lastPull: '1970-01-01T00:00:00Z', hashes: {}, known: [] };
  const local = localItems(ds);
  const localHash = new Map([...local].map(([k, v]) => [k, hash(v.json)]));

  // 1. Pull everything that changed since last time.
  onProgress?.('Downloading workspace data');
  const remote: { kind: Kind; id: string; payload: Item; deleted: boolean; updated_at: string }[] = [];
  for (let from = 0; ; from += 200) {
    const { data, error } = await sb
      .from(tableName)
      .select('kind, id, payload, deleted, updated_at')
      .eq(ownerColumn, ownerId)
      .gt('updated_at', st.lastPull)
      .order('updated_at', { ascending: true })
      .order('kind').order('id')
      .range(from, from + 199);
    if (error) throw new Error(error.message);
    remote.push(...((data ?? []) as typeof remote));
    if (!data || data.length < 200) break;
  }

  const players = new Map(ds.players.map((p) => [p.id, p]));
  const matches = new Map(ds.matches.map((m) => [m.id, m]));
  const attacks = new Map(ds.attacks.map((a) => [a.id, a]));
  const table = { player: players, match: matches, attack: attacks } as const;
  let pulled = 0;
  let lastPull = st.lastPull;
  for (const r of remote) {
    const k = keyOf(r.kind, r.id);
    const pendingLocal = localHash.has(k) && localHash.get(k) !== st.hashes[k];
    if (r.updated_at > lastPull) lastPull = r.updated_at;
    if (pendingLocal && st.hashes[k] !== undefined) continue; // our unsynced edit wins; it is pushed below
    if (!localHash.has(k) && st.known.includes(k) && st.hashes[k] !== undefined && !r.deleted) continue; // pending local deletion
    const map = table[r.kind] as Map<string, Item>;
    if (r.deleted) {
      if (map.delete(r.id)) pulled++;
      delete st.hashes[k];
      localHash.delete(k);
      local.delete(k);
    } else {
      map.set(r.id, r.payload);
      const h = hash(JSON.stringify(compact(r.payload)));
      if (localHash.get(k) === h) pulled--; // our own earlier upload coming back
      st.hashes[k] = h;
      localHash.set(k, h);
      local.set(k, { kind: r.kind, item: r.payload, json: JSON.stringify(r.payload) });
      pulled++;
    }
  }

  // 2. Push new and edited items, and tombstones for local deletions.
  const toPush: { kind: Kind; id: string; payload: Item; deleted: boolean }[] = [];
  for (const [k, v] of local) if (st.hashes[k] !== localHash.get(k)) toPush.push({ kind: v.kind, id: v.item.id, payload: v.item, deleted: false });
  const nowKeys = new Set(local.keys());
  const gone = st.known.filter((k) => !nowKeys.has(k) && st.hashes[k] !== undefined);
  for (const k of gone) {
    const [kind, ...rest] = k.split(':');
    toPush.push({ kind: kind as Kind, id: rest.join(':'), payload: {} as Item, deleted: true });
  }
  let pushed = 0;
  // Players and matches first so teammates never see an attack without its hitter.
  const order: Record<Kind, number> = { player: 0, match: 1, attack: 2 };
  toPush.sort((a, b) => order[a.kind] - order[b.kind]);
  for (let i = 0; i < toPush.length; i += 25) {
    const batch = toPush.slice(i, i + 25);
    onProgress?.(`Uploading ${Math.min(i + 25, toPush.length)} of ${toPush.length}`);
    const { error } = await sb.from(tableName).upsert(batch.map((item) => ({ ...item, [ownerColumn]: ownerId })), { onConflict: `${ownerColumn},kind,id` });
    if (error) throw new Error(error.message);
    for (const b of batch) {
      const k = keyOf(b.kind, b.id);
      if (b.deleted) delete st.hashes[k];
      else st.hashes[k] = localHash.get(k)!;
      pushed++;
    }
  }

  st.known = [...local.keys()];
  st.lastPull = lastPull;
  return {
    key,
    state: st,
    dataset: pulled === 0 ? ds : { ...ds, players: [...players.values()] as Player[], matches: [...matches.values()] as Match[], attacks: [...attacks.values()] as Attack[] },
    pulled,
    pushed: pushed - gone.length,
    deleted: gone.length,
  };
}
