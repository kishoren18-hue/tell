import { useState } from 'react';
import { Section } from '../components/ui';
import { useCloud } from '../cloud';

export function Account({ onDone, onHome }: { onDone: () => void; onHome: () => void }) {
  const cloud = useCloud();
  return (
    <div className="screen">
      <div className="crumb">
        <button className="btn btn--quiet btn--tiny" onClick={onDone}>Back</button>
        <h1 className="crumb__title">{cloud.mode === 'team' ? 'Team setup' : cloud.user ? 'Account' : 'Sign in'}</h1>
        <button className="btn btn--quiet btn--tiny" onClick={onHome}>Switch mode</button>
      </div>
      {cloud.user ? <SignedIn onDone={onDone} /> : <SignInForm />}
    </div>
  );
}

function SignInForm() {
  const cloud = useCloud();
  const [mode, setMode] = useState<'in' | 'up'>('in');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null);

  const submit = async () => {
    setBusy(true);
    setMsg(null);
    try {
      if (mode === 'in') await cloud.signIn(email.trim(), password);
      else {
        const r = await cloud.signUp(email.trim(), password, name.trim());
        if (r === 'confirm') {
          setMsg({ text: `Almost done. We sent a confirmation link to ${email.trim()}. Open it, then come back here and sign in.` });
          setMode('in');
        }
      }
    } catch (e) {
      setMsg({ text: (e as Error).message, bad: true });
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <p className="muted">
        {cloud.mode === 'team'
          ? 'Sign in to create or join a team and share swings with teammates.'
          : 'Sign in to sync Individual swings privately across your devices. Guest swings move into your private workspace when you open it.'}
      </p>
      <div className="seg" role="group" aria-label="Sign in or create account">
        <button className={`seg__btn ${mode === 'in' ? 'seg__btn--on' : ''}`} onClick={() => setMode('in')}>Sign in</button>
        <button className={`seg__btn ${mode === 'up' ? 'seg__btn--on' : ''}`} onClick={() => setMode('up')}>Create account</button>
      </div>
      <form
        className="form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        {mode === 'up' && (
          <label className="field">
            <span className="field__label">Your name</span>
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" required />
          </label>
        )}
        <label className="field">
          <span className="field__label">Email</span>
          <input className="input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" required />
        </label>
        <label className="field">
          <span className="field__label">Password</span>
          <input
            className="input"
            type="password"
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={mode === 'in' ? 'current-password' : 'new-password'}
            required
          />
        </label>
        {msg && <p className={`notice ${msg.bad ? 'notice--bad' : ''}`}>{msg.text}</p>}
        <button className="btn btn--primary btn--big" type="submit" disabled={busy}>
          {busy ? 'One moment…' : mode === 'in' ? 'Sign in' : 'Create account'}
        </button>
      </form>
    </>
  );
}

function SignedIn({ onDone }: { onDone: () => void }) {
  const cloud = useCloud();
  const [teamName, setTeamName] = useState('');
  const [code, setCode] = useState('');
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmLeave, setConfirmLeave] = useState<string | null>(null);

  const run = async (fn: () => Promise<unknown>, ok?: string) => {
    setBusy(true);
    setMsg(null);
    try {
      await fn();
      if (ok) setMsg({ text: ok });
    } catch (e) {
      setMsg({ text: (e as Error).message, bad: true });
    } finally {
      setBusy(false);
    }
  };

  const share = async (c: string, name: string) => {
    const text = `Join ${name} on Tell: open ${location.origin}, sign in, and enter team code ${c}`;
    const nav = navigator as Navigator & { share?: (d: ShareData) => Promise<void> };
    if (nav.share) {
      try {
        await nav.share({ title: 'Join my Tell team', text });
        return;
      } catch {
        /* fall through to copy */
      }
    }
    try {
      await navigator.clipboard.writeText(text);
      setMsg({ text: 'Invite copied. Paste it in your team chat.' });
    } catch {
      setMsg({ text: `Team code: ${c}` });
    }
  };

  return (
    <>
      <div className="account-card">
        <div>
          <strong>{(cloud.user!.user_metadata?.name as string) || cloud.user!.email}</strong>
          <span className="muted small">{cloud.user!.email}</span>
        </div>
        <button className="btn btn--tiny" onClick={() => void run(cloud.signOut)}>Sign out</button>
      </div>

      {msg && <p className={`notice ${msg.bad ? 'notice--bad' : ''}`}>{msg.text}</p>}

      {cloud.mode === 'individual' && (
        <Section title="Private Individual sync">
          <SyncStatus />
          <p className="muted small">Your tagged swings sync to this account. Original videos stay on your device.</p>
        </Section>
      )}

      {cloud.mode === 'team' && <>
        {cloud.message && cloud.status === 'error' && !cloud.team && <p className="notice notice--bad">{cloud.message}</p>}
        <Section title="Your teams">
          {cloud.teams.length === 0 ? (
            <p className="muted small">You're not on a team yet. Create one for your squad, or join with the code a teammate sends you.</p>
          ) : (
            <ul className="list">
              {cloud.teams.map((t) => (
                <li key={t.id} className={`team ${cloud.team?.id === t.id ? 'team--on' : ''}`}>
                  <button className="team__main" onClick={() => { cloud.setTeam(t.id); if (cloud.team) onDone(); }} aria-pressed={cloud.team?.id === t.id}>
                    <strong>{t.name}</strong>
                    <span className="muted small">{cloud.team?.id === t.id ? 'Current team' : 'Open team workspace'}</span>
                  </button>
                  <span className="team__code" aria-label={`Team code ${t.join_code}`}>{t.join_code}</span>
                  <div className="team__acts">
                  <button className="btn btn--tiny" onClick={() => void share(t.join_code, t.name)}>Invite</button>
                  {confirmLeave === t.id ? (
                    <button className="btn btn--tiny btn--danger" onClick={() => void run(() => cloud.leave(t.id), `You left ${t.name}.`)}>Leave team</button>
                  ) : (
                    <button className="btn btn--tiny btn--quiet" onClick={() => setConfirmLeave(t.id)}>Leave</button>
                  )}
                  </div>
                </li>
              ))}
            </ul>
          )}
          {cloud.team && <SyncStatus />}
        </Section>

        <Section title="Create a team">
          <form className="row" onSubmit={(e) => { e.preventDefault(); if (teamName.trim()) void run(() => cloud.create(teamName.trim()).then(() => setTeamName('')), 'Team created. Tap Invite to send teammates the code.'); }}>
            <input className="input" placeholder="Team name" value={teamName} onChange={(e) => setTeamName(e.target.value)} aria-label="Team name" />
            <button className="btn btn--primary" type="submit" disabled={busy || !teamName.trim()}>Create</button>
          </form>
        </Section>

        <Section title="Join a team">
          <form className="row" onSubmit={(e) => { e.preventDefault(); if (code.trim()) void run(() => cloud.join(code.trim()).then(() => setCode('')), 'Joined. Opening the team workspace.'); }}>
            <input className="input input--code" placeholder="6-letter code" value={code} onChange={(e) => setCode(e.target.value.toUpperCase())} maxLength={6} aria-label="Team code" autoCapitalize="characters" />
            <button className="btn" type="submit" disabled={busy || code.trim().length < 6}>Join</button>
          </form>
        </Section>

        <p className="muted small">
          Synced: your roster, matches and swing skeletons with their labels. Never synced: videos (they never leave the
          phone) and the demo data.
        </p>
      </>}
    </>
  );
}

function SyncStatus() {
  const cloud = useCloud();
  return (
    <div className="sync-row">
      <span className={`dot dot--${cloud.status}`} aria-hidden />
      <span className="small" role="status">{cloud.message || (cloud.lastSync ? 'Up to date.' : 'Waiting to sync…')}</span>
      <button className="btn btn--tiny" onClick={() => void cloud.syncNow()} disabled={cloud.status === 'syncing'}>Sync now</button>
    </div>
  );
}
