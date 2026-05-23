# Roadmap

Versions are sized to each fit in one focused sitting (a weekend or less) and
to leave the codebase fully working at the end. We avoid speculative scope —
the next version only addresses the part of the previous one that's actually
become a friction point.

## v1 — typed teaching (this release)

**Goal.** Get the basis working: webcam → SPELA trainer, taught by typed
utterances, predicted classes shown live on the video stream.

**Done when:**
- `python -m cradle.main` runs end-to-end with a USB webcam.
- Teaching a class with two utterances produces a visibly correct prediction.
- Save / load preserves both model weights and the registry.
- A pytest suite covers oracle parsing and registry semantics.

**Limitations carried forward.** Typed input only; rule-based parser; max 256
classes; no negative-feedback learning; no growth.

---

## v1.1 — spoken teaching

**Goal.** Replace stdin with a microphone + Whisper STT, so teaching is
hands-free. The oracle interface does not change.

**Plan.**
- `cradle/audio.py`: microphone capture (sounddevice or PyAudio) + simple
  energy-based VAD or webrtcvad.
- `cradle/stt.py`: whisper.cpp via subprocess, or
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CUDA. Wrap so
  the transcript is just a string going into the same queue stdin used.
- Main: replace `stdin_reader` with an audio reader; everything else identical.

**Done when:**
- Speaking "this is a cup" works the same as typing it.
- Transcription latency under 1 s on Jetson Orin AGX with `small.en`.
- Falls back to stdin with `--no-audio` for development on machines without a
  mic.

**Risks.** STT errors compound with parser brittleness. Acceptable for v1.1
because v1.2 fixes the parser; if it's painful in the meantime, expand the
rule set.

---

## v1.2 — LLM-backed oracle

**Goal.** Replace `RuleBasedOracle` with a local LLM (Llama 3.2 3B or similar)
using JSON-mode output. Any reasonable phrasing should parse.

**Plan.**
- Pick the model and runner. Default: [ollama](https://ollama.com) with
  `llama3.2:3b` (works on Orin Nano 8 GB).
- `cradle.oracle.LLMOracle`: prompt the LLM with the utterance + the current
  list of known classes; constrain output to the `Label` JSON schema; map to
  the dataclass.
- Cache the model in memory; warm it on startup.
- Add a `--oracle {rule,llm}` flag.

**Done when:**
- "look, that's the wrench over there" parses correctly.
- The LLM also handles disambiguation: when a known and a similar new class
  collide, it asks (via a `clarify` action) instead of misclassifying.
- Median utterance-to-label latency under 500 ms on the target device.

**Risks.** LLM startup latency, memory pressure on Orin Nano. Mitigation:
keep the rule-based oracle as a fallback for low-RAM deployments.

---

## v2 — dynamic class growth + depth growth

This is where the project starts doing something that backprop-based stacks
can't easily match.

### v2.0 — true dynamic class capacity

**Plan.**
- Replace the fixed-size class embedding table with a `nn.ParameterList` (or
  a thin wrapper around a growing tensor) per layer.
- `LabelRegistry.get_or_create` extends every layer's embedding table by one
  row, sampled to maximize distance from existing rows (online farthest-point).
- Keep `RegistryFull` only as a soft cap users can configure.

**Done when:** the model can be taught arbitrarily many classes within
hardware memory, with no `num_classes` flag.

### v2.1 — depth growth

**Plan.**
- A `GrowthMonitor` watches per-layer accuracy + loss. When the top layer's
  accuracy plateaus on the rolling teaching set, trigger `runtime.grow()`.
- `grow()` instantiates a new block (configurable architecture; default: a
  copy of the current top block), pushes it onto `trainer.layers`, allocates a
  fresh optimizer, and lazily initializes its embeddings on the next batch.
- Earlier layers continue training by default. A `--freeze-on-grow` flag pins
  them.

**Done when:**
- We can demo "teach 10 classes, watch network grow from 3 to 4 layers, teach
  10 more, watch accuracy rise where a fixed-depth baseline saturates."
- A simple plot in the docs comparing fixed-depth vs growing-depth over a
  long teaching session.

**Risks.** This is the genuinely novel part. The first growth event might
destabilize earlier layers via shifted activation statistics. Mitigations:
warmup the new block at higher LR while freezing predecessors for the first N
batches.

---

## v3+ — research items (no promises)

These are speculative, listed so we don't forget the threads.

- **Width growth.** Per-layer channel addition with function-preserving init.
  See Net2Net / Firefly / Cascade-Correlation. Hardest of the growth modes
  because the autograd / parameter-set assumptions are deepest here.
- **Negative-feedback learning.** "No, that's not a cup" as a contrastive
  signal — push the head's activation *away* from the cup embedding instead
  of just acknowledging.
- **Hierarchical class structure.** The registry currently has flat strings.
  Real taxonomies ("a coffee cup is a cup") could improve sample efficiency.
- **Multi-modal grounding.** Audio events as a second input stream alongside
  video, sharing the same teaching interface.
- **Lifelong evaluation.** Continual-learning benchmarks (Permuted MNIST,
  Split CIFAR, embodied streams) to characterize forgetting curves over long
  sessions.

## Non-goals (worth being explicit about)

- **Learning language from scratch.** The LLM teacher stays. Trying to make
  the trainee replace it would conflate two problems and almost certainly
  fail by the standards of either.
- **Matching SOTA on static benchmarks.** A small CNN trained online will not
  beat ImageNet-pretrained ResNets at ImageNet. That's fine; that's not what
  cradle is for.
- **Generality across tasks.** v1–v2 are about a single perceptual classifier
  with teaching. Generalizing to actions / control / multi-step plans is a
  different project.
