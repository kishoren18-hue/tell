# Plan

## The first two weeks

The goal is to find out if real club hitters are readable before contact, and how early.

1. **Film five of our own matches** from behind the end line, as high up as possible, phone sideways on a tripod. 60 fps if the phone has it.
2. **Tag about 300 attacks** in the app: mark contact, tap the hitter, tap where the ball went. That's roughly 60 a match, about 20 minutes of tagging per match once you're quick. Pose extraction runs automatically on the phone.
3. **Train** in the Model tab. It holds out each match in turn, so the score is for matches the model hasn't seen.
4. **Decide.** The bar is the Model tab saying the gap over "always guess the most common direction" is real (p < 0.05) at 100 ms or more before contact. If it does, there's a product. If it doesn't at 300 attacks, look at the read curve: a read only at contact means the tell is in the arm swing, and the next fix is better pose (heavier model on a laptop with `ml/tell/extract.py`, or a higher camera angle).

Beating a coin flip isn't enough: with three directions a coin flip is 33%, and "always guess cross" already beats that. Everything in Tell is scored against the most-common-direction baseline instead.

## Metric: downloads

- A tell score is a number players will screenshot and argue about. The Hitters tab has a share card sized for Instagram stories.
- Start with our own team, then bring it to club tournaments. Every team that tags its matches grows the dataset.
- Track installs (people adding it to the home screen) and weekly active taggers.

## The data moat

Pro footage is the only volleyball video that exists in bulk. Every club match tagged in Tell adds labeled youth and club attacks: who hit, from where, body pose, and where the ball went. Nobody else has that. The exports are skeletons, not video, so they're small and safer to pool, with consent.

## Model roadmap

| Data size | What should win | Why |
| --- | --- | --- |
| up to ~500 attacks | Logistic regression on engineered features (in the app) | Few parameters, can't overfit much |
| 1,000+ attacks | RallyWorld, the transformer world model in `ml/` | Learns its own features and uses the full motion sequence |
| 10,000+ | RallyWorld pretrained on unlabeled swings, then fine-tuned | The pose-forecast objective needs no tags |

Next steps, in order:

1. **Ball tracking** (TrackNet-style) to label landing spots automatically. This removes most of the manual tagging.
2. **Automatic contact detection**: hand-ball proximity from the tracker, so tagging is just confirming.
3. **Run the ONNX RallyWorld in the app** (`onnxruntime-web`) once it beats the logistic model on real data.
4. **Native apps** with Capacitor, wrapping the same web app, if App Store presence matters for downloads.
5. **Baseball**: pitch-type tells from the pitcher's windup. Same engine, new skeleton features. Families already pay coaches to fix tipping pitches.
