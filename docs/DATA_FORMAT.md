# Dataset format (version 1)

Film > Export dataset writes one JSON file. The Python pipeline in `ml/` reads the same file. Type definitions live in `src/lib/types.ts`.

```jsonc
{
  "format": "tell-dataset",
  "version": 1,
  "sampleHz": 30,          // pose frames per second
  "contactIndex": 36,      // frame index of ball contact (1.2 s in)
  "windowFrames": 46,      // 1.2 s before contact + contact + 0.3 s after
  "players": [{ "id": "p…", "name": "Sam", "number": "7", "hand": "R" }],
  "matches": [{ "id": "m…", "name": "vs Lambert", "date": "2026-11-06", "videoName": "IMG_4410.MOV" }],
  "attacks": [{
    "id": "a…",
    "matchId": "m…",
    "hitterId": "p…",
    "contactTime": 812.43,                 // seconds into the match video
    "hitterTap": { "x": 0.41, "y": 0.37 }, // where the hitter was tapped on the contact frame, 0..1
    "position": "left",                    // left | middle | right | back-left | back-middle | back-right
    "landing": { "x": 6.8, "y": 5.2 },     // meters on the opponent half (optional)
    "direction": "right",                  // left | straight | right, from the hitter's point of view
    "shotType": "hard",                    // hard | roll | tip
    "outcome": "dug",                      // kill | dug | blocked | error | unknown
    "frames": [ { "w": [99 numbers], "p": [99 numbers] } | null, … ]  // 46 entries, or null if pose not extracted yet
  }]
}
```

## Court frame

The opponent's half, seen from behind the hitter (the camera's view):

- `x` runs from 0 (the hitter's left sideline) to 9 (the right sideline), in meters.
- `y` runs from 0 (the net) to 9 (the opponent's end line). Your own side is negative `y`.

`direction` comes from the angle between the takeoff point (`TAKEOFF` in `src/lib/court.ts`) and the landing spot. Anything within 12° of straight down the court counts as `straight`. The app shows players the usual names (line, cross, wide line), but the model only ever sees left / straight / right.

## Pose frames

- `w`: MediaPipe world landmarks, 33 × (x, y, z), in meters, centered on the hips. x points to the camera's right, y points down, and z points away from the camera (towards the net).
- `p`: image landmarks, 33 × (x, y, visibility), normalized to 0..1.
- `null`: the hitter wasn't found in that frame. Feature extraction interpolates gaps and drops attacks with fewer than half the frames.

## Left-handed hitters

Before feature extraction, lefties are mirrored into a right-handed frame: negate x, swap left/right landmarks, and swap the left/right labels. One model then reads both hands. Predictions are mirrored back for display.

## Privacy

Exports hold skeletons and tags, never video or images. They still describe identifiable players by name and number, so get consent (from parents for players under 18) before sharing an export.
