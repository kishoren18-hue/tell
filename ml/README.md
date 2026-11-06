# Tell training pipeline

Desktop side of Tell. It reads the dataset the app exports and trains models that are too heavy for a phone.

```bash
cd ml
pip install -e ".[dev,extract,export]"   # CPU torch is fine

# try it on the synthetic demo data (written by the app repo)
npm --prefix .. run export-demo -- ml/data/demo.json
python -m tell.train data/demo.json

# your own data: Film > Export dataset in the app
python -m tell.train my-team.json --models logreg,lstm,rallyworld --out results.json
```

| Module | What it does |
| --- | --- |
| `tell/features.py` | Exact port of the app's features. `tests/test_parity.py` checks the two agree to 1e-6. |
| `tell/data.py` | Loads an export and canonicalizes lefties |
| `tell/models.py` | `PoseLSTM` (the tennis baseline) and `RallyWorld` (the causal transformer world model) |
| `tell/train.py` | Grouped-by-match cross-validation, temperature scaling, accuracy against the majority baseline, information gain, ECE, and pose-forecast error against "freeze" and "constant velocity" |
| `tell/extract.py` | Re-runs pose with the heavy MediaPipe model on a laptop, using the app's contact times and taps |
| `tell/export.py` | Trains RallyWorld on everything and exports ONNX, checked against onnxruntime |

RallyWorld predicts three things from every frame, using only frames before it: the next 0.5 s of the hitter's pose (no labels needed), a Gaussian mixture over the landing spot, and the shot direction. It trains at random cutoffs between 0 and 500 ms before contact, so one model reads at any horizon.
