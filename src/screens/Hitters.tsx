import type { Tab } from '../App';
import { Court, type CourtDot } from '../components/Court';
import { download, pct } from '../components/ui';
import { MIN_ATTACKS_FOR_SCORE, type HitterScore, type Report } from '../lib/ml/tell';
import { useData } from '../state';

export function Hitters({ onGo }: { onGo: (t: Tab) => void }) {
  const { dataset, report, stale } = useData();
  if (!report) {
    return (
      <div className="screen">
        <div className="intro">
          <h1>Tell scores</h1>
          <p>Train a model first. Each hitter then gets a score from 0 (gives nothing away) to 100 (the model always knows).</p>
        </div>
        <button className="btn btn--primary" onClick={() => onGo('results')}>Go to Model</button>
      </div>
    );
  }
  const readsById = new Map(report.reads.map((r) => [r.attackId, r]));
  return (
    <div className="screen">
      <div className="intro">
        <h1>Tell scores</h1>
        <p>
          How much the model's read {report.scoreHorizonMs ? `${report.scoreHorizonMs} ms before contact` : 'at contact'} narrows down
          where your ball goes. 0 means your body gives nothing away. Low is good for hitters; liberos want to know who's high.
        </p>
      </div>
      {report.synthetic && <p className="notice">Synthetic demo hitters with planted tells, not real players.</p>}
      {stale && <p className="notice">New tags since this was trained. Retrain in Model to update the scores.</p>}
      <ol className="hitters">
        {report.hitters.map((h) => (
          <HitterCard
            key={h.player.id}
            h={h}
            report={report}
            dots={dataset.attacks
              .filter((a) => a.hitterId === h.player.id && a.landing)
              .map((a) => ({ x: a.landing!.x, y: a.landing!.y, tone: readsById.get(a.id)?.correct ? 'hot' : 'ink' }) as CourtDot)}
          />
        ))}
      </ol>
      <p className="muted small">Court dots: red means the model called that attack before contact, dark means it didn't.</p>
    </div>
  );
}

function HitterCard({ h, report, dots }: { h: HitterScore; report: Report; dots: CourtDot[] }) {
  const label = h.player.number ? `#${h.player.number}` : h.player.name;
  const total = h.counts.left + h.counts.straight + h.counts.right;
  return (
    <li className="hitter">
      <div className="hitter__score">
        <span className="hitter__num" aria-label={h.score === null ? 'No score yet' : `Tell score ${Math.round(h.score)}`}>
          {h.score === null ? '–' : Math.round(h.score)}
        </span>
        <span className="hitter__cap">Tell score</span>
      </div>
      <div className="hitter__body">
        <h2 className="hitter__name">
          {h.player.name !== label ? `${label} ${h.player.name}` : label}
          {h.player.hand === 'L' && <span className="pill">Lefty</span>}
        </h2>
        {h.score === null ? (
          <p className="small">Tag {MIN_ATTACKS_FOR_SCORE - h.n} more attacks to get a score.</p>
        ) : (
          <p className="small">
            Likely between {Math.round(h.scoreCI![0])} and {Math.round(h.scoreCI![1])}. The model called {pct(h.accuracy)} of {h.n} attacks.
            {h.lowData && ' Under 20 attacks, so this will move.'}
          </p>
        )}
        <div className="split" aria-label="Shot directions">
          {(['left', 'straight', 'right'] as const).map((d) => (
            <span key={d} style={{ flexGrow: Math.max(1, h.counts[d]) }} className={`split__seg split__seg--${d}`}>
              {d === 'left' ? 'L' : d === 'right' ? 'R' : 'S'} {total ? Math.round((h.counts[d] / total) * 100) : 0}%
            </span>
          ))}
        </div>
        {h.cues.length > 0 && (
          <div className="cues">
            <span className="cues__title">Biggest giveaways we measured</span>
            <ul>
              {h.cues.map((c) => (
                <li key={c.feature}>
                  <b>{c.label}</b>: {c.detail}
                </li>
              ))}
            </ul>
          </div>
        )}
        {dots.length > 0 && <Court compact dots={dots} label="Where this hitter's attacks went" />}
        {h.score !== null && (
          <button className="btn btn--tiny" onClick={() => void shareCard(h, report)}>
            Share card
          </button>
        )}
      </div>
    </li>
  );
}

async function shareCard(h: HitterScore, report: Report) {
  const c = document.createElement('canvas');
  c.width = 1080;
  c.height = 1350;
  const g = c.getContext('2d')!;
  await document.fonts?.ready;
  g.fillStyle = '#2457a6';
  g.fillRect(0, 0, 1080, 1350);
  // Court lines.
  g.strokeStyle = 'rgba(255,255,255,0.9)';
  g.lineWidth = 10;
  g.strokeRect(90, 90, 900, 900);
  g.lineWidth = 6;
  g.beginPath();
  g.moveTo(90, 690);
  g.lineTo(990, 690);
  g.stroke();
  g.fillStyle = '#ffffff';
  g.textAlign = 'center';
  g.font = '700 520px "Barlow Condensed", "Arial Narrow", sans-serif';
  g.fillText(String(Math.round(h.score ?? 0)), 540, 640);
  g.font = '600 64px "Barlow Condensed", "Arial Narrow", sans-serif';
  g.fillText('Tell score', 540, 790);
  g.fillStyle = '#f2c230';
  g.font = '700 88px "Barlow Condensed", "Arial Narrow", sans-serif';
  const who = h.player.number ? `#${h.player.number} ${h.player.name.startsWith('#') ? '' : h.player.name}` : h.player.name;
  g.fillText(who.trim(), 540, 1110);
  g.fillStyle = 'rgba(255,255,255,0.85)';
  g.font = '500 44px Barlow, Arial, sans-serif';
  g.fillText(`Model called ${Math.round(h.accuracy * 100)}% of ${h.n} attacks ${report.scoreHorizonMs ? `${report.scoreHorizonMs} ms before contact` : 'at contact'}`, 540, 1190);
  g.font = '500 38px Barlow, Arial, sans-serif';
  g.fillText(report.synthetic ? 'Synthetic demo data' : 'Can you read my spike?', 540, 1270);
  const blob: Blob = await new Promise((r) => c.toBlob((b) => r(b!), 'image/png'));
  const file = new File([blob], `tell-score-${h.player.number ?? h.player.id}.png`, { type: 'image/png' });
  const nav = navigator as Navigator & { canShare?: (d: unknown) => boolean };
  if (nav.canShare?.({ files: [file] })) {
    try {
      await nav.share({ files: [file], title: 'My tell score' });
      return;
    } catch {
      /* fall back to download */
    }
  }
  download(file.name, blob);
}
