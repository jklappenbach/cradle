# Cradle

A framework for demonstrating continuous, on-device learning and dynamic network growth using a perception model that starts from random weights, learns from natural-language teaching paired with a live
video stream, and is designed for the embedded GPU (NVIDIA Jetson Orin).  As the network learns and becomes more sophisticated, it can add more layers and nodes within layers.  With the forward pass SPELA training algorithm, training can occur out of phase of inference, and be conducted continuously.

Cradle is built on [SPELA](https://github.com/.../spela-training) — a single-
forward-pass training algorithm (no global backprop) where every layer has its
own local loss against fixed class embeddings on the unit sphere. SPELA's
properties make it unusually well-suited to this kind of work:

- **Cheap online updates.** Per-layer, local-loss training means a training
  step costs about as much as inference. We can `predict` on every frame and
  `train` whenever the teacher labels something, on the same hardware, in real
  time.
- **No global graph.** Layers can be added at runtime without rebuilding a
  backward pass. This is what makes the long-term "grow the network as it
  learns" goal tractable instead of fantasy.
- **Predict-from-any-layer.** Early-exit inference and per-layer accuracy come
  for free, which matters when the device has a latency budget.

## What v1 does

A "newborn" CNN (random init, ~100k params) sees the webcam. You teach it by
typing sentences in the terminal:

```
> this is a cup
  [teach] 'cup' (id=0) <- 8 frames  buffered=8/16
> this is a hand
  [teach] 'hand' (id=1) <- 8 frames  buffered=16/16  ** train step **
> yes
  [+] reinforced 'hand' with 8 frames
> list
  [classes] 2/256: cup, hand
> save
> quit
```

The OpenCV window shows the live frame with the predicted class + confidence
overlay. Predictions are noise until the buffer fills and the first SPELA step
runs (about two teaching utterances at default settings).

v1 deliberately uses **typed input + a rule-based parser** as the "oracle"
(text → label). That gets the basis working without dragging in an LLM or a
speech stack. The oracle interface is the seam: v1.1 swaps in a microphone +
Whisper STT, v1.2 swaps in a local LLM-backed parser. main.py never changes.

## Quickstart

```bash
git clone <this repo> cradle
cd cradle
python -m venv .venv
source .venv/bin/activate

# spela-training lives as a sibling directory by default — adjust the path
# dependency in pyproject.toml if yours is elsewhere.
pip install -e ../spela-training
pip install -e .

python -m cradle.main
```

Then in the same terminal, type teaching utterances. `q` in the OpenCV window
or `quit` in the terminal exits.

## Where to read next

- [docs/vision.md](docs/vision.md) — the long-term picture: what "newborn
  intellect" means here, what we're *not* claiming, where this is going.
- [docs/architecture.md](docs/architecture.md) — v1 component breakdown,
  threading model, data flow, key design choices and their costs.
- [docs/roadmap.md](docs/roadmap.md) — staged milestones from v1 (typed) to
  v1.1 (spoken) to v1.2 (LLM oracle) to v2 (network growth).

## Hardware

Target platform is an NVIDIA Jetson Orin (Nano or AGX). The v1 model is small
enough that it trains comfortably on CPU; the GPU mostly helps with throughput
on the video pipeline. Everything runs locally — no cloud dependencies at
runtime.

## License

MIT.
