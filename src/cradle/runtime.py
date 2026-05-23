"""SPELA trainer wrapped for online, teacher-driven learning.

Architecture (deliberately tiny so it trains in milliseconds on CPU and
microseconds on the Jetson Orin GPU):

  block1  : 3 stride-2 conv stages -> AdaptiveAvgPool -> 64-d feature
  block2  : Linear 64 -> 128 + ReLU
  head    : Linear 128 -> 64   (per-layer SPELA embeddings sized lazily)

Online pattern:
  - observe(frames, class_id) appends each (frame, class_id) to a buffer.
  - When the buffer reaches `batch_size`, run one SpelaTrainer.train_epoch
    over the single buffered batch and clear.
  - predict(frame) returns (class_id, cosine_confidence, label_name).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from spela_train import SpelaConfig, SpelaTrainer

from .registry import LabelRegistry


def _build_layers(input_size: int) -> Tuple[List[nn.Module], List[Optional[nn.Module]]]:
    block1 = nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    block2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(inplace=True))
    head = nn.Linear(128, 64)
    layers: List[nn.Module] = [block1, block2, head]
    activations: List[Optional[nn.Module]] = [None, None, nn.Identity()]
    return layers, activations


@dataclass
class RuntimeConfig:
    input_size: int = 96
    num_classes: int = 256
    lr: float = 1e-3
    batch_size: int = 16
    frames_per_utterance: int = 8
    device: Optional[str] = None  # None -> auto cuda/cpu


class TeacherRuntime:
    def __init__(self, registry: LabelRegistry, cfg: Optional[RuntimeConfig] = None):
        self.cfg = cfg or RuntimeConfig()
        if self.cfg.num_classes != registry.capacity:
            raise ValueError(
                f"registry capacity {registry.capacity} must equal num_classes "
                f"{self.cfg.num_classes}"
            )
        self.registry = registry

        device = torch.device(self.cfg.device) if self.cfg.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        layers, activations = _build_layers(self.cfg.input_size)
        sp_cfg = SpelaConfig(
            num_classes=self.cfg.num_classes,
            lr=self.cfg.lr,
            optimizer="adamw",
            loss_type="cosface",
            cosine_scale=30.0,
            lr_schedule="constant",
            seed=0,
            device=device,
        )
        self.trainer = SpelaTrainer(nn.Sequential(*layers), layers, activations, sp_cfg)
        self.device = device

        self._buf_x: List[torch.Tensor] = []
        self._buf_y: List[int] = []
        self._lock = threading.Lock()
        self._steps = 0

    # ---- teaching ----------------------------------------------------------

    def observe(self, frames: List[torch.Tensor], class_id: int) -> bool:
        """Buffer (frame, class_id) pairs; flush to a training step when the
        buffer fills. Returns True if a step ran.
        """
        if not frames:
            return False
        with self._lock:
            for f in frames:
                self._buf_x.append(f)
                self._buf_y.append(class_id)
            if len(self._buf_x) < self.cfg.batch_size:
                return False
            xs = torch.stack(self._buf_x).to(self.device)
            ys = torch.tensor(self._buf_y, dtype=torch.long, device=self.device)
            self._buf_x.clear()
            self._buf_y.clear()
        self.trainer.train_epoch([(xs, ys)])
        self._steps += 1
        return True

    @property
    def steps(self) -> int:
        return self._steps

    def buffered(self) -> int:
        with self._lock:
            return len(self._buf_x)

    # ---- inference ---------------------------------------------------------

    @torch.no_grad()
    def predict(self, frame: torch.Tensor) -> Tuple[int, float, str]:
        """Returns (class_id, confidence, label_name).

        Only class ids actually allocated by the registry are considered —
        untrained rows of the embedding table would otherwise win by chance.
        """
        if len(self.registry) == 0:
            return -1, 0.0, "?"
        from spela_train import _flatten_features, _normalize  # type: ignore

        x = frame.unsqueeze(0).to(self.device)
        if self.trainer.cfg.auto_flatten and x.dim() > 2 and isinstance(self.trainer.layers[0], nn.Linear):
            x = _flatten_features(x)
        h = x
        for i in range(len(self.trainer.layers)):
            if self.trainer.cfg.normalize_layer_inputs and h.dim() >= 2:
                h = _normalize(h, dim=1)
            h = self.trainer._layer_forward(i, h)

        vecs = self.trainer.class_vectors[-1]
        if vecs.numel() == 0:
            return -1, 0.0, "?"
        n_known = len(self.registry)
        h_n = _normalize(h, dim=1)
        vecs_n = _normalize(vecs[:n_known], dim=1)
        cos = (h_n @ vecs_n.t()).squeeze(0)
        conf, cid = torch.max(cos, dim=0)
        cid_i = int(cid.item())
        return cid_i, float(conf.item()), self.registry.name(cid_i) or "?"

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        self.trainer.save(path)
        torch.save(self.registry.state_dict(), path + ".registry.pt")

    def load(self, path: str) -> None:
        self.trainer.load(path, strict=True)
        try:
            state = torch.load(path + ".registry.pt", map_location="cpu", weights_only=False)
            self.registry.load_state_dict(state)
        except FileNotFoundError:
            pass
