# Architecture (v1)

## Components

```
┌──────────────────┐                ┌────────────────────┐
│   Camera thread  │  frame queue   │   Inference loop   │
│  cv2.VideoCapture├──────────────► │  predict()@10 Hz   │
│  preprocess once │                │  overlay & display │
└──────────────────┘                └─────────┬──────────┘
                                              │ last_pred
                                              ▼
┌──────────────────┐    text     ┌────────────────────┐
│   Stdin thread   ├────────────►│   handle_utterance │
│  (mic in v1.1)   │             │      dispatcher    │
└──────────────────┘             └─────────┬──────────┘
                                            │ Label
                                            ▼
                                  ┌────────────────────┐
                                  │      Oracle        │ <- swappable: rules | LLM
                                  │  parse(utt) -> Lbl │
                                  └─────────┬──────────┘
                                            │ teach(name)
                                            ▼
                                  ┌────────────────────┐
                                  │   LabelRegistry    │ <- name -> int id
                                  └─────────┬──────────┘
                                            │ class_id
                                            ▼
┌──────────────────┐  frames + y  ┌────────────────────┐
│   Camera buffer  ├──────────────►   TeacherRuntime   │
│  recent_tensors  │              │  buffer -> train   │
└──────────────────┘              │  SpelaTrainer step │
                                  └────────────────────┘
```

Five source modules in `src/cradle/`:

| module          | responsibility                                              |
|-----------------|-------------------------------------------------------------|
| `registry.py`   | name ↔ id mapping, dynamic allocation up to `capacity`.     |
| `oracle.py`     | utterance → `Label`. Rule-based default; LLM seam declared. |
| `perception.py` | webcam capture thread + preprocessed-frame ring buffer.     |
| `runtime.py`    | small CNN under `SpelaTrainer`, buffered online training.   |
| `main.py`       | wires it all, handles stdin and the cv2 display loop.       |

## Threading model

Three threads in v1:

- **Camera thread.** Reads from `cv2.VideoCapture` in a tight loop, runs
  preprocessing (BGR→RGB, resize, normalize, to CHW tensor), pushes into a
  bounded `deque`. Holding the lock for as little time as possible — the
  preprocessed tensor is computed before the lock is taken.
- **Stdin thread.** Blocking `sys.stdin.readline()` into a queue. In v1.1 the
  microphone capture thread + Whisper STT push into the same queue, so the
  main loop's dispatch logic doesn't change.
- **Main thread.** Drains the input queue, drives prediction at ~10 Hz, drives
  the OpenCV display. Training steps run inline on the main thread when the
  buffer fills, because they're cheap (a few ms on CPU for this model).

We deliberately do *not* put training on its own thread in v1. The buffered
batch is small (16 by default), the model is tiny, and inlining keeps the
mental model simple: each teaching utterance produces at most one training
step. v2 may revisit this if larger models or higher utterance rates make
training dominate frame latency.

## Data flow for a teaching utterance

1. User types `this is a cup` and hits enter.
2. Stdin thread enqueues the string.
3. Main thread dequeues it, passes to `RuleBasedOracle.parse()`.
4. Oracle returns `Label(action="teach", class_name="cup")`.
5. `LabelRegistry.get_or_create("cup")` returns an int (allocates on first
   sight).
6. `Camera.recent_tensors(frames_per_utterance=8)` returns the most recent 8
   preprocessed frames.
7. `TeacherRuntime.observe(frames, class_id)` appends each (frame, id) to its
   buffer. If the buffer is now ≥ batch_size, it calls
   `SpelaTrainer.train_epoch([(xs, ys)])` and clears the buffer.

## Model

```python
block1 = Sequential(
    Conv2d(3, 16, 3, stride=2, padding=1), BatchNorm2d(16), ReLU,
    Conv2d(16, 32, 3, stride=2, padding=1), BatchNorm2d(32), ReLU,
    Conv2d(32, 64, 3, stride=2, padding=1), BatchNorm2d(64), ReLU,
    AdaptiveAvgPool2d(1), Flatten,
)                                       # -> (B, 64)
block2 = Sequential(Linear(64, 128), ReLU)   # -> (B, 128)
head   = Linear(128, 64)                     # -> (B, 64)
```

SPELA sizes the per-layer class embedding tables lazily based on the actual
activation dim of each layer, so `head` outputs 64-d and the head's embeddings
are (num_classes=256, 64). Cosine similarity between the head's output and the
embeddings produces the prediction.

96×96×3 input is a deliberate choice — small enough that the full pipeline
fits comfortably in a few milliseconds on CPU, large enough that humans can
tell what's in the frame.

## Key design choices and their costs

**Fixed capacity (256 classes).** SPELA's class embeddings are sized at init.
Dynamic resizing would mean regenerating farthest-point embeddings, which
changes existing class vectors and invalidates learned weights. Over-
allocating is the trade: a 256×64 table is 64 KB, lost in the noise. The cost
is a hard cap; the registry raises `RegistryFull` when hit. v2 will address.

**Untrained-row masking at inference.** With 256 allocated embeddings but only
*N* registered classes, an unmasked nearest-neighbor lookup could let an
untrained random embedding win by chance. `runtime.predict` only scores the
first `len(registry)` rows. The cost is correctness logic to keep in sync; the
benefit is sensible predictions when only 2 classes are known.

**Last-N frames per utterance.** A teaching utterance labels the last 8 frames
as samples of that class. This assumes you don't move the object between
seeing it and naming it. The cost is label noise when you violate that
assumption; the benefit is decent sample efficiency without needing a
selection UI.

**Buffered training (size 16).** A teaching utterance accrues 8 samples; two
utterances trigger a SPELA step. Smaller is more responsive but more wasteful;
larger is more efficient but feels laggy. 16 is a working default — tune to
taste.

**Rule-based oracle.** No LLM in v1. The parser handles the handful of
patterns that come up in actual teaching ("this is a X", "yes", "no", "save",
"quit", "forget X", "list"). The cost is brittleness to phrasing; the benefit
is that v1 runs with zero ML-system dependencies beyond SPELA itself.

## What v1 deliberately leaves out

- Microphone + STT. (v1.1.)
- LLM-backed oracle. (v1.2.)
- Negative-feedback learning ("no, that's not a cup"). The current code
  acknowledges negatives but doesn't use them as a contrastive signal.
- Class forgetting / pruning. Acknowledged as a command, not implemented.
- Dynamic class growth past 256. (v2.)
- Depth growth. (v2.)
- Persistent label history (which frames led to which prediction). Useful for
  debugging; not needed to demo the loop.
