"""Webcam capture with a frame ring buffer.

cv2.VideoCapture runs on a daemon thread so the main loop never blocks on
I/O. Each captured frame is preprocessed once (BGR->RGB, resize, [0,1] CHW
float tensor) and pushed into a bounded deque. The runtime asks for the
latest N preprocessed tensors when a teaching utterance arrives; the display
loop asks for the latest raw BGR frame for the OpenCV window.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class CameraConfig:
    device: int = 0
    width: int = 640
    height: int = 480
    fps_target: int = 30
    input_size: int = 96            # CNN input HxW
    buffer_seconds: float = 2.0     # ring-buffer depth in seconds


class Camera:
    def __init__(self, cfg: CameraConfig):
        import cv2  # local import so this module is importable without opencv

        self.cv2 = cv2
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.device)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera device {cfg.device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps_target)

        depth = max(8, int(cfg.fps_target * cfg.buffer_seconds))
        self._buf: Deque[Tuple[float, np.ndarray, torch.Tensor]] = deque(maxlen=depth)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        cv2 = self.cv2
        s = self.cfg.input_size
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(rgb).float().div_(255.0).permute(2, 0, 1).contiguous()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            tensor = self._preprocess(frame)
            with self._lock:
                self._buf.append((time.time(), frame, tensor))

    def latest_raw(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._buf[-1][1].copy() if self._buf else None

    def latest_tensor(self) -> Optional[torch.Tensor]:
        with self._lock:
            return self._buf[-1][2] if self._buf else None

    def recent_tensors(self, n: int) -> List[torch.Tensor]:
        """Return up to the most recent n preprocessed tensors (oldest first)."""
        with self._lock:
            items = list(self._buf)[-n:]
        return [t for (_, _, t) in items]
