"""End-to-end runtime test on synthetic frames. Validates that:
  - SpelaTrainer wiring builds successfully and runs a training step,
  - the buffered-observe pattern triggers exactly one step per `batch_size`,
  - prediction returns a registered class name (not '?'),
  - save / load preserves the registry alongside the trainer.
"""
from __future__ import annotations

import torch

from cradle.registry import LabelRegistry
from cradle.runtime import RuntimeConfig, TeacherRuntime


def _fake_frame(class_name: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, 96, 96, generator=g) * 0.1
    if class_name == "cup":
        base[:, 20:60, 20:60] += 0.8
    elif class_name == "hand":
        base[:, 40:80, 30:70] += 0.6
    return base


def test_buffer_flush_runs_one_step():
    reg = LabelRegistry(capacity=16)
    rt = TeacherRuntime(reg, RuntimeConfig(num_classes=16, batch_size=8, device="cpu"))

    cid = reg.get_or_create("cup")
    # 7 frames -> no step
    assert rt.observe([_fake_frame("cup", i) for i in range(7)], cid) is False
    assert rt.steps == 0
    assert rt.buffered() == 7
    # 1 more frame -> step
    assert rt.observe([_fake_frame("cup", 99)], cid) is True
    assert rt.steps == 1
    assert rt.buffered() == 0


def test_predict_returns_named_class_after_training():
    reg = LabelRegistry(capacity=16)
    rt = TeacherRuntime(reg, RuntimeConfig(num_classes=16, batch_size=8,
                                            frames_per_utterance=4, device="cpu"))
    # Two classes, 8 frames each across two cycles -> 4 train steps.
    for cycle in range(2):
        for name in ("cup", "hand"):
            cid = reg.get_or_create(name)
            rt.observe([_fake_frame(name, cycle * 10 + i) for i in range(8)], cid)
    assert rt.steps >= 2

    cid, conf, name = rt.predict(_fake_frame("cup", 1000))
    assert name in {"cup", "hand"}  # at minimum, a registered class wins
    assert -1.0 <= conf <= 1.0


def test_save_load_round_trip(tmp_path):
    reg = LabelRegistry(capacity=16)
    rt = TeacherRuntime(reg, RuntimeConfig(num_classes=16, batch_size=4, device="cpu"))
    cid = reg.get_or_create("cup")
    rt.observe([_fake_frame("cup", i) for i in range(4)], cid)

    ckpt = str(tmp_path / "test.ckpt")
    rt.save(ckpt)

    reg2 = LabelRegistry(capacity=16)
    rt2 = TeacherRuntime(reg2, RuntimeConfig(num_classes=16, batch_size=4, device="cpu"))
    rt2.load(ckpt)
    assert reg2.known_names() == ["cup"]
