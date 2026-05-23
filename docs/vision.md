# Vision

## The picture in one paragraph

Cradle is an attempt to build a small, embodied perception system that starts
empty, learns continuously from a human teacher speaking in natural language,
and grows its own capacity over time. Three properties make it different from
a normal trained-and-deployed classifier: training and inference share the
same forward pass and run together on-device; new classes can appear at any
moment without retraining the whole model; and the network is designed to
*grow* — adding layers (and eventually neurons) as the teaching extends past
what its current capacity can absorb.

## Why "newborn intellect"

The framing is intentional but not literal. We do not claim this system is
intelligent in any general sense. What we mean by "newborn" is structural:

- It starts with **random weights and zero classes**. No pretraining, no seed
  vocabulary, no transferred features. The first information it gets about the
  world comes from a teacher.
- It learns **online and incrementally**. There is no train / deploy split.
  Every observation is potentially both an inference and a training step.
- It can be taught **in natural language**, by a human pointing a camera at
  something and saying what it is. An LLM (or for v1, a regex parser) does the
  symbolic work of converting an utterance into a class id.

The interesting question isn't "is this a mind?" — it's "how much of what we
normally bake into a model at training time can we instead teach to it, after
deployment, on small hardware?"

## The LLM-as-teacher pattern

We sit a language model between the human and the trainee. The human says
"this is a wrench"; the LLM emits `{"action":"teach", "class_name":"wrench"}`;
the trainee gets a `(frame, class_id)` pair. Feedback flows the same way —
"yes," "no," "the one on the left" — converted into structured signals the
trainee understands.

This is a real, working pattern (RLAIF / LLM-as-judge / supervisor-LLM). The
honest concerns:

- **Latency mismatch.** SPELA wants (x, y) pairs at frame rate. The LLM runs
  at "teaching moment" rate (seconds). The cradle handles this by buffering: a
  labeling utterance captures the last *N* frames as samples of that class.
- **Label noise.** LLM-derived labels are noisier than ground truth. SPELA's
  cosine / cosface loss tolerates this reasonably well, but it isn't magic.
- **Symbol grounding ≠ language.** Mapping "wrench" to an int is grounding a
  symbol. Learning what words *mean* — compositionally, recursively — is a
  much bigger problem that an LLM-supervised classifier does not solve.

The LLM is a permanent scaffolding choice, not a temporary one. We do not plan
to "remove the LLM" once the trainee "learns language" — that conflates two
very different problems and would set this project up to fail by the wrong
yardstick. The LLM is the part of the system that already knows English. Let
it keep that job.

## Network growth as a first-class concern

Most ML frameworks assume the network's structure is fixed at init time. That
assumption is baked into PyTorch's autograd, into checkpoint formats, into
optimizer state. SPELA mostly escapes it because there is no global backward
pass — each layer is an island with its own optimizer and its own loss. That
makes two kinds of growth tractable:

1. **Depth growth.** When accuracy plateaus, snap a new block onto the top of
   the stack. The new block gets its own optimizer + lazily-sized embedding
   row and starts training on the detached activation of the previous top.
   Earlier layers keep training (or get frozen — a knob, not a constraint).
2. **Width growth.** Inside an existing layer, add neurons / channels and
   initialize them so the function is approximately preserved (Net2Net,
   Firefly, Cascade-Correlation are the prior art). Harder than depth growth.

v1 does neither — we want the basis working first. v2 is depth growth. Width
growth is a v3+ research item.

This is, to my knowledge, an under-explored direction precisely because most
ML practitioners are using backprop frameworks that resist it. SPELA's local-
loss structure is a quiet enabler.

## What this is not

- **Not a path to AGI.** SPELA is a classifier trainer. It does not generate.
  It does not reason. It does not plan.
- **Not a language model.** The trainee doesn't learn language. The LLM
  provides language understanding from the start, on purpose.
- **Not a benchmark chaser.** A small CNN that learns 50 classes online from a
  human teacher will lose every academic vision benchmark by a wide margin.
  The interesting metric is something else: time-to-learn-a-new-class,
  catastrophic-forgetting curves, how growth affects capacity over weeks of
  use, how labeling noise from the LLM translates into accuracy.

## What success looks like

- v1: you can teach the model a handful of objects in a few minutes, and
  inference visibly tracks what's in front of the camera.
- v1.1: same, hands-free, by speaking.
- v1.2: the parser tolerates arbitrary phrasing because an LLM is doing the
  parsing.
- v2: you teach it well past the original network's capacity, watch a new
  layer get added autonomously, and accuracy keeps climbing instead of
  saturating.
- vN: a research write-up on continual + growing networks trained with local
  losses, with the Jetson demo as the existence proof.
