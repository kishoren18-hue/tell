import { useMemo, useState } from 'react';
import { sequenceFeatures } from '../lib/features';
import { stateOf } from '../lib/ml/world';
import { makeDemoDataset } from '../lib/synth';
import { CONTACT_INDEX, SAMPLE_HZ } from '../lib/types';
import { Skeleton } from './Skeleton';

/** The same replay renderer and synthetic swing used by Tell's practice tool. */
export function SwingPreview() {
  const [frame, setFrame] = useState(CONTACT_INDEX - 6);
  const states = useMemo(() => {
    const attack = makeDemoDataset({ matches: 1, attacksPerHitterPerMatch: 1, missingRate: 0 }).attacks[0];
    const sequence = attack.frames && sequenceFeatures(attack.frames, 'R');
    return sequence ? sequence.map(stateOf) : [];
  }, []);
  const before = (CONTACT_INDEX - frame) / SAMPLE_HZ;
  return (
    <figure className="swing-preview">
      <figcaption>
        <span>Swing replay</span>
        <span>Demo data</span>
      </figcaption>
      <div className="swing-preview__stage">
        <Skeleton states={states} frame={frame} height={180} />
        <span className="swing-preview__time">{before > 0 ? `${before.toFixed(2)}s before contact` : before < 0 ? `${Math.abs(before).toFixed(2)}s after contact` : 'Contact'}</span>
      </div>
      <div className="swing-preview__scrub">
        <label htmlFor="demo-frame">Scrub the swing</label>
        <input id="demo-frame" type="range" min={0} max={Math.max(0, states.length - 1)} value={frame} onChange={(e) => setFrame(Number(e.target.value))} aria-valuetext={before > 0 ? `${before.toFixed(2)} seconds before contact` : before < 0 ? `${Math.abs(before).toFixed(2)} seconds after contact` : 'Contact'} />
      </div>
    </figure>
  );
}
