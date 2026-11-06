# Prior art, and what Tell takes from it

Tell predicts where a volleyball attack is going from the hitter's body before contact. Nobody has shipped that for volleyball, but net and racket sports have worked on the same problem. Here's what exists and what Tell borrows from it. No code was copied from these projects; the methods are re-implemented here.

## Tennis

**Shimizu et al., "Prediction of Future Shot Direction using Pose and Position of Tennis Player" (ACM MMSports 2019).** This is the closest prior work. They built a shot-direction dataset from YouTube match video, extracted skeletons with OpenPose, and predicted three classes (right cross, left cross, straight) with an LSTM over the pose sequence up to the moment of impact, plus player position.
- Tell uses the same three-class framing (left / straight / right in the hitter's own frame).
- The `lstm` model in `ml/tell/models.py` is the same idea, used as a baseline.

**Hazarika, "Multi-Task Tennis Stroke Biomechanics Analysis Using MediaPipe Pose" (arXiv 2606.15992, 2026).** Uses MediaPipe Pose Landmarker and a transformer with separate heads for stroke type, direction and posture. Two findings shaped Tell:
- Swapping MediaPipe *world* landmarks for image landmarks cut cross-player direction accuracy from 68% to 21%. Tell uses world landmarks for body features and image coordinates only for the approach path.
- Direction did not transfer from pros to an amateur, so Tell evaluates on held-out *matches* and reports a majority-class baseline beside every number instead of headline accuracy.

## Badminton

**ShuttleNet (Wang et al., AAAI 2022) and ShuttleSet (Wang et al., KDD 2023).** Stroke forecasting from a rally's history, with separate predictions for shot type and landing area. ShuttleSet has over 36,000 expert-labeled strokes from top-level singles.
- RallyWorld has the same split: a direction head plus a landing head (a Gaussian mixture over landing position).
- These datasets are all pro play. That's the gap Tell's club-level data fills.

**TrackNet and RacketVision (AAAI 2026).** Ball tracking, racket pose and ball trajectory forecasting for badminton, table tennis and tennis. Tell doesn't track the ball yet. Adding a TrackNet-style ball tracker is the obvious next step, because it would label landing spots automatically and remove most of the manual tagging.

## Table tennis

**Wu et al. (2019, 2020)** predicted where a serve lands from the server's motion before contact, using pose estimation. It's the same "read the body before the ball" setup as Tell.

## Volleyball

- **Ibrahim et al., Volleyball dataset (CVPR 2016).** 55 YouTube match videos labeled for group activity recognition (right spike, left set, and so on). It recognizes actions after they happen and doesn't predict anything.
- **Suda et al. (2019)** predicted the set's trajectory 0.3 s ahead from the setter's 3D joints. It's the closest volleyball work, but it covers the setter, not the attacker.
- **Wenninger et al. (2020)** used RNNs on positional data to forecast beach volleyball tactics, with about 37% accuracy for attack direction.

## What's new in Tell

1. Attacker shot-direction prediction from pose, for volleyball, on a phone.
2. A per-player **tell score**: information gain of the model's pre-contact read over base rates, with a bootstrap range, rather than raw accuracy.
3. A **read curve**: how early before contact each hitter becomes readable.
4. Practice mode that turns the same model into a reading drill for liberos and blockers.
5. Club-level data collected as a side effect of people using the app.
